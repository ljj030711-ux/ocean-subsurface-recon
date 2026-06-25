"""Train HVCARefiner as a second-stage residual post-processor.

This script keeps the existing layer-wise Du_Unet checkpoints frozen. It first
generates or reads normalized T0 cache shards, then trains a single HVCARefiner
with the same masked RMSE loss family used by the first-stage Du_Unet.

Example:
    python -u eval_hvca_refiner.py \
        --data-dir /root/autodl-tmp/.autodl/raw \
        --checkpoint outputs/2dto2d/Du_Unet/Refined/hvca_refiner/best.pt \
        --t0-cache-dir outputs/2dto2d/Du_Unet/Refined/t0_cache_test \
        --base-checkpoint-dir checkpoints/2dto2d/Du_Unet \
        --target-var temperature \
        --eval-start 2023-01-01 \
        --eval-end 2023-12-31 \
        --batch-size 16 \
        --column-chunk-size 8192 \
        --num-workers 0 \
        --save-dir outputs/2dto2d/Du_Unet/Refined/hvca_eval
"""

import argparse
import bisect
import json
import os
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import (
    CHECKPOINTS_ROOT,
    DATA_DIR,
    DATA_START_DATE,
    DEPTH_LEVELS_25M,
    DU_UNET_CKPT_NAME_TEMPLATE,
    PARADIGM_2DTO2D,
    SEED,
    get_checkpoint_dir,
)
from datasets.dataset_2dto2d import Dataset2Dto2D
from datasets.date_utils import generate_date_list
from models.du_unet import Du_Unet
from models.hvca_refiner import HVCARefiner
from utils.losses import masked_rmse_loss


def get_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_depths(depths_arg):
    if not depths_arg:
        return list(DEPTH_LEVELS_25M)
    depths = []
    for item in depths_arg.split(","):
        text = item.strip()
        if text:
            depths.append(int(float(text)))
    if not depths:
        raise ValueError("--depths 不能为空")
    return depths


def resolve_depth_indices(depths):
    available = list(DEPTH_LEVELS_25M)
    missing = [d for d in depths if d not in available]
    if missing:
        raise ValueError(
            f"--depths 必须是 DEPTH_LEVELS_25M 的子集；缺失 checkpoint 的深度: {missing}"
        )
    return [available.index(d) for d in depths]


def get_year_split_indices(total_len, start_date_str):
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    train_days = (datetime(2021, 12, 31) - start).days + 1
    val_days = (datetime(2022, 12, 31) - datetime(2021, 12, 31)).days
    train_idx = list(range(min(train_days, total_len)))
    val_start = train_days
    val_idx = list(range(val_start, min(val_start + val_days, total_len)))
    test_start = val_start + val_days
    test_idx = list(range(test_start, total_len))
    return train_idx, val_idx, test_idx


def cache_shards(cache_dir):
    if not os.path.isdir(cache_dir):
        return []
    return sorted(
        os.path.join(cache_dir, name)
        for name in os.listdir(cache_dir)
        if name.startswith("shard_") and name.endswith(".pt")
    )


def cache_ready(cache_dir):
    return len(cache_shards(cache_dir)) > 0


def save_cache_metadata(cache_dir, payload):
    path = os.path.join(cache_dir, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_du_unet_subset(args, device, depth_indices):
    base_dir = args.base_checkpoint_dir or get_checkpoint_dir(
        PARADIGM_2DTO2D, "Du_Unet", base_dir=CHECKPOINTS_ROOT
    )
    if os.path.basename(os.path.normpath(base_dir)) == args.target_var:
        candidates = [base_dir, os.path.dirname(os.path.normpath(base_dir))]
    else:
        candidates = [os.path.join(base_dir, args.target_var), base_dir]

    models = []
    for depth_idx in depth_indices:
        depth_m = DEPTH_LEVELS_25M[depth_idx]
        ckpt_name = DU_UNET_CKPT_NAME_TEMPLATE.format(
            target_var=args.target_var, depth_m=depth_m
        )
        searched_paths = [os.path.join(path, ckpt_name) for path in candidates]
        ckpt_path = next((path for path in searched_paths if os.path.exists(path)), None)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"缺少 {args.target_var} 深度 {depth_m}m checkpoint；"
                f"已查找：{', '.join(searched_paths)}"
            )
        model = Du_Unet(out_channels=1).to(device)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        model.eval()
        models.append(model)
    return models


def stack_dataset_samples(dataset, indices, depth_indices):
    items = [dataset[int(i)] for i in indices]
    sst = torch.stack([item["sst"] for item in items], dim=0)
    ssh_sss = torch.stack([item["ssh_sss"] for item in items], dim=0)
    target = torch.stack([item["target"] for item in items], dim=0)[:, depth_indices]
    mask = torch.stack([item["target_mask"] for item in items], dim=0)[:, depth_indices]
    return sst, ssh_sss, target, mask


@torch.no_grad()
def infer_t0(models, sst, ssh_sss, device):
    sst = sst.to(device)
    ssh_sss = ssh_sss.to(device)
    pred_layers = [model(sst, ssh_sss) for model in models]
    return torch.cat(pred_layers, dim=1).cpu()


def generate_t0_cache(
    *,
    cache_dir,
    dataset,
    indices,
    models,
    depth_values,
    depth_indices,
    device,
    target_var,
    split_name,
    shard_size,
):
    """Generate normalized T0/T_true cache shards for one split."""
    os.makedirs(cache_dir, exist_ok=True)
    if cache_ready(cache_dir):
        print(f"[cache] 发现已有 {split_name} cache，跳过生成: {cache_dir}")
        return

    dates = generate_date_list(dataset.start_date, dataset.end_date)
    total = len(indices)
    if total == 0:
        raise ValueError(f"{split_name} split 为空，无法生成 cache")

    print(f"[cache] 生成 {split_name} cache: {cache_dir} | samples={total}")
    for model in models:
        model.eval()

    shard_id = 0
    for start in range(0, total, shard_size):
        batch_indices = indices[start : start + shard_size]
        sst, ssh_sss, target, mask = stack_dataset_samples(
            dataset, batch_indices, depth_indices
        )
        T0 = infer_t0(models, sst, ssh_sss, device)
        months = np.asarray(dataset.months[batch_indices], dtype=np.int64)
        sample_id = [dates[int(i)] for i in batch_indices]
        payload = {
            "sample_id": sample_id,
            "months": torch.as_tensor(months, dtype=torch.long),
            "T0": T0.to(torch.float32),
            "T_true": target.to(torch.float32),
            "target_mask": mask.to(torch.float32),
        }
        out_path = os.path.join(cache_dir, f"shard_{shard_id:05d}.pt")
        torch.save(payload, out_path)
        shard_id += 1
        print(f"[cache] {split_name}: {min(start + shard_size, total)}/{total}")

    save_cache_metadata(
        cache_dir,
        {
            "split": split_name,
            "target_var": target_var,
            "start_date": dataset.start_date,
            "end_date": dataset.end_date,
            "depth_m": list(map(int, depth_values)),
            "depth_indices": list(map(int, depth_indices)),
            "normalization_space": "monthly_climatology_layer_std",
            "contains_surface": False,
        },
    )


class HvcaCacheDataset(Dataset):
    """Read sharded normalized T0 cache."""

    def __init__(self, cache_dir, shard_cache_size=4):
        self.cache_dir = cache_dir
        self.shard_cache_size = max(int(shard_cache_size), 1)
        self.files = cache_shards(cache_dir)
        if not self.files:
            raise FileNotFoundError(f"未找到 cache shard: {cache_dir}")

        self.lengths = []
        for path in self.files:
            shard = torch.load(path, map_location="cpu")
            self.lengths.append(int(shard["T0"].shape[0]))
        self.cumulative = np.cumsum(self.lengths).tolist()
        self._shard_cache = OrderedDict()

    def __len__(self):
        return int(self.cumulative[-1])

    def _load_shard(self, shard_idx):
        path = self.files[shard_idx]
        if path in self._shard_cache:
            shard = self._shard_cache.pop(path)
            self._shard_cache[path] = shard
            return shard
        shard = torch.load(path, map_location="cpu")
        self._shard_cache[path] = shard
        while len(self._shard_cache) > self.shard_cache_size:
            self._shard_cache.popitem(last=False)
        return shard

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        shard_idx = bisect.bisect_right(self.cumulative, idx)
        prev = 0 if shard_idx == 0 else self.cumulative[shard_idx - 1]
        local_idx = idx - prev
        shard = self._load_shard(shard_idx)
        return {
            "sample_id": shard["sample_id"][local_idx],
            "months": shard["months"][local_idx],
            "T0": shard["T0"][local_idx],
            "T_true": shard["T_true"][local_idx],
            "target_mask": shard["target_mask"][local_idx],
        }


def prepare_train_val_cache(args, depth_values, depth_indices, device):
    if cache_ready(args.t0_cache_dir) and cache_ready(args.val_cache_dir):
        print("[cache] train/val cache 均已存在，直接读取")
        return

    dataset = Dataset2Dto2D(
        args.data_dir,
        normalize=True,
        target_var=args.target_var,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    train_idx, val_idx, _ = get_year_split_indices(len(dataset), dataset.start_date)
    if args.max_samples is not None:
        train_idx = train_idx[: args.max_samples]
        val_idx = val_idx[: args.max_samples]
    if not train_idx or not val_idx:
        raise ValueError("训练或验证 split 为空，请检查 --start-date/--end-date")

    models = load_du_unet_subset(args, device, depth_indices)
    generate_t0_cache(
        cache_dir=args.t0_cache_dir,
        dataset=dataset,
        indices=train_idx,
        models=models,
        depth_values=depth_values,
        depth_indices=depth_indices,
        device=device,
        target_var=args.target_var,
        split_name="train",
        shard_size=args.cache_shard_size,
    )
    generate_t0_cache(
        cache_dir=args.val_cache_dir,
        dataset=dataset,
        indices=val_idx,
        models=models,
        depth_values=depth_values,
        depth_indices=depth_indices,
        device=device,
        target_var=args.target_var,
        split_name="val",
        shard_size=args.cache_shard_size,
    )


def train_one_epoch(model, loader, optimizer, scaler, depth_tensor, device, use_amp):
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        T0 = batch["T0"].to(device)
        T_true = batch["T_true"].to(device)
        mask = batch["target_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(T0, depth_tensor)
            loss = masked_rmse_loss(pred, T_true, mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu())
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def validate(model, loader, depth_tensor, device, use_amp):
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        T0 = batch["T0"].to(device)
        T_true = batch["T_true"].to(device)
        mask = batch["target_mask"].to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(T0, depth_tensor)
            loss = masked_rmse_loss(pred, T_true, mask)
        total += float(loss.detach().cpu())
        count += 1
    return total / max(count, 1)


def dataloader_kwargs(args, device):
    kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def save_checkpoint(path, model, optimizer, args, depth_values, best_val, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": {
                "dim": args.dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "column_chunk_size": args.column_chunk_size,
            },
            "depth_values": list(map(float, depth_values)),
            "target_var": args.target_var,
            "best_val": float(best_val),
            "epoch": int(epoch),
        },
        path,
    )


def parse_args():
    default_ckpt_dir = get_checkpoint_dir(
        PARADIGM_2DTO2D, "Du_Unet", base_dir=CHECKPOINTS_ROOT
    )
    parser = argparse.ArgumentParser(description="Train HVCARefiner post-processor")
    parser.add_argument("--t0-cache-dir", required=True)
    parser.add_argument("--val-cache-dir", required=True)
    parser.add_argument("--base-checkpoint-dir", default=default_ckpt_dir)
    parser.add_argument("--target-var", choices=["temperature", "salinity"], default="temperature")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--start-date", default=DATA_START_DATE)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--depths", default=None, help="逗号分隔深度米数，默认 DEPTH_LEVELS_25M")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cache-shard-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--column-chunk-size",
        type=int,
        default=8192,
        help="每次送入垂向/交叉 attention 的水平柱数量，减小可降低显存和 CUDA kernel 压力",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="cache DataLoader worker 数")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--shard-cache-size", type=int, default=4, help="每个 worker 内缓存的 shard 数")
    parser.add_argument("--save-dir", default="outputs/hvca_train")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.cache_shard_size <= 0:
        raise ValueError("--batch-size 和 --cache-shard-size 必须大于 0")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.device)
    use_amp = device.type == "cuda"
    depth_values = parse_depths(args.depths)
    depth_indices = resolve_depth_indices(depth_values)
    depth_tensor = torch.as_tensor(depth_values, dtype=torch.float32, device=device)

    print("=" * 60)
    print("HVCARefiner 训练配置")
    print("=" * 60)
    print(f"target_var: {args.target_var}")
    print(f"device: {device}")
    print(f"depths: {depth_values}")
    print(f"train cache: {args.t0_cache_dir}")
    print(f"val cache: {args.val_cache_dir}")
    print(f"Du_Unet checkpoint dir: {args.base_checkpoint_dir}")
    print("=" * 60)

    prepare_train_val_cache(args, depth_values, depth_indices, device)

    train_ds = HvcaCacheDataset(args.t0_cache_dir, shard_cache_size=args.shard_cache_size)
    val_ds = HvcaCacheDataset(args.val_cache_dir, shard_cache_size=args.shard_cache_size)
    loader_kwargs = dataloader_kwargs(args, device)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = HVCARefiner(
        dim=args.dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        column_chunk_size=args.column_chunk_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "best.pt")
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, depth_tensor, device, use_amp
        )
        val_loss = validate(model, val_loader, depth_tensor, device, use_amp)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch}/{args.epochs}] | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr_now:.2e}"
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_path, model, optimizer, args, depth_values, best_val, epoch)
            print(f"  -> 保存最佳 HVCARefiner: {best_path} (val={best_val:.6f})")

    np.savez(
        os.path.join(args.save_dir, "training_history_HVCARefiner.npz"),
        train_loss=np.asarray(history["train_loss"], dtype=np.float32),
        val_loss=np.asarray(history["val_loss"], dtype=np.float32),
    )
    print(f"\n训练完成。最佳验证 RMSE loss: {best_val:.6f}")
    print(f"最佳权重: {best_path}")


if __name__ == "__main__":
    main()
