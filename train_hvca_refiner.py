"""训练基于 Du_Unet 初猜场的 HVCARefiner 二阶段残差修正模型。"""

import argparse
import bisect
import json
import os
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from config import (
    CHECKPOINTS_ROOT,
    DATA_DIR,
    DATA_START_DATE,
    DEFAULT_DATASET_PROFILE,
    DU_UNET_CKPT_NAME_TEMPLATE,
    OUTPUTS_ROOT,
    PARADIGM_2DTO2D,
    SEED,
    dataset_profile_names,
    get_2dto2d_depth_levels,
    get_checkpoint_dir,
)
from datasets.dataset_2dto2d import Dataset2Dto2D
from datasets.date_utils import generate_date_list
from models.du_unet import Du_Unet
from models.hvca_refiner import HVCARefiner
from utils.losses import hvca_refiner_loss
from utils.run_config import (
    count_model_parameters,
    print_and_save_run_config,
)


def get_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def use_lazy_2dto2d_data(args):
    return args.lazy_data or args.dataset_profile == "glorys12v1_all_size"


def _depth_equal(a, b, tol=1e-4):
    return abs(float(a) - float(b)) <= tol


def _format_depth_list(depths):
    return ", ".join(f"{float(d):g}" for d in depths)


def parse_depths(depths_arg, depth_levels):
    if not depths_arg:
        return list(depth_levels)
    depths = []
    for item in depths_arg.split(","):
        text = item.strip()
        if text:
            depths.append(float(text))
    if not depths:
        raise ValueError("--depths 不能为空")
    return depths


def resolve_depth_indices(depths, depth_levels):
    available = list(depth_levels)
    missing = [
        d for d in depths
        if not any(_depth_equal(d, candidate) for candidate in available)
    ]
    if missing:
        raise ValueError(
            f"--depths 必须是当前 dataset profile 深度层的子集；"
            f"缺失 checkpoint 的深度: {_format_depth_list(missing)}；"
            f"可选深度: {_format_depth_list(available)}"
        )
    return [
        next(i for i, candidate in enumerate(available) if _depth_equal(d, candidate))
        for d in depths
    ]


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


def default_hvca_output_dir():
    return os.path.join(OUTPUTS_ROOT, PARADIGM_2DTO2D, "HVCARefiner")


def run_config_path(save_dir, stage):
    return os.path.join(save_dir, f"HVCARefiner_{stage}_config.json")


def save_loss_curve(train_losses, val_losses, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked RMSE Loss")
    ax.set_title("HVCARefiner Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"HVCA loss 曲线已保存至: {output_path}")


def save_cache_metadata(cache_dir, payload):
    path = os.path.join(cache_dir, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_cache_metadata(cache_dir):
    path = os.path.join(cache_dir, "metadata.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_du_unet_subset(args, device, depth_indices, depth_levels):
    base_dir = args.base_checkpoint_dir or get_checkpoint_dir(
        PARADIGM_2DTO2D, "Du_Unet", base_dir=CHECKPOINTS_ROOT
    )
    if os.path.basename(os.path.normpath(base_dir)) == args.target_var:
        candidates = [base_dir, os.path.dirname(os.path.normpath(base_dir))]
    else:
        candidates = [os.path.join(base_dir, args.target_var), base_dir]

    models = []
    for depth_idx in depth_indices:
        depth_m = depth_levels[depth_idx]
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
def infer_t0(models, sst, ssh_sss, device, output_size=None):
    sst = sst.to(device)
    ssh_sss = ssh_sss.to(device)
    pred_layers = [model(sst, ssh_sss, output_size=output_size) for model in models]
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
        cached_reference = load_cache_metadata(cache_dir).get(
            "normalization_stats"
        )
        current_reference = dataset.get_normalization_reference()
        if (
            cached_reference
            and cached_reference.get("fingerprint")
            and cached_reference.get("fingerprint")
            != current_reference.get("fingerprint")
        ):
            raise ValueError(
                f"已有 {split_name} cache 与当前归一化统计量不一致: {cache_dir}；"
                "请换一个空 cache 目录重新生成。"
            )
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
        T0 = infer_t0(models, sst, ssh_sss, device, output_size=target.shape[-2:])
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
            "depth_m": list(map(float, depth_values)),
            "depth_indices": list(map(int, depth_indices)),
            "normalization_space": "monthly_climatology_layer_std",
            "normalization_stats": dataset.get_normalization_reference(),
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


class ShardLocalBatchSampler(Sampler):
    """Shuffle shards and samples while keeping each batch inside one shard."""

    def __init__(self, dataset, batch_size, shuffle, seed=SEED, drop_last=False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        shard_order = np.arange(len(self.dataset.lengths), dtype=np.int64)
        if self.shuffle:
            rng.shuffle(shard_order)

        shard_starts = [0] + self.dataset.cumulative[:-1]
        for shard_idx in shard_order:
            start = int(shard_starts[int(shard_idx)])
            length = int(self.dataset.lengths[int(shard_idx)])
            indices = np.arange(start, start + length, dtype=np.int64)
            if self.shuffle:
                rng.shuffle(indices)
            for batch_start in range(0, length, self.batch_size):
                batch = indices[batch_start : batch_start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch.tolist()

    def __len__(self):
        if self.drop_last:
            return sum(length // self.batch_size for length in self.dataset.lengths)
        return sum(
            (length + self.batch_size - 1) // self.batch_size
            for length in self.dataset.lengths
        )


def prepare_train_val_cache(args, depth_values, depth_indices, depth_levels, device):
    if cache_ready(args.t0_cache_dir) and cache_ready(args.val_cache_dir):
        print("[cache] train/val cache 均已存在，直接读取")
        cache_metadata = load_cache_metadata(args.t0_cache_dir)
        val_metadata = load_cache_metadata(args.val_cache_dir)
        reference = cache_metadata.get("normalization_stats")
        val_reference = val_metadata.get("normalization_stats")
        if (
            reference
            and val_reference
            and reference.get("fingerprint") != val_reference.get("fingerprint")
        ):
            raise ValueError("train/val cache 使用了不同的归一化统计量，请重新生成 cache")
        if reference:
            args._normalization_stats_reference = reference
            if args.normalization_stats_path is None:
                args.normalization_stats_path = reference.get("path")
        else:
            print("[cache] 警告：旧 cache 未记录归一化统计文件引用")
        return

    dataset = Dataset2Dto2D(
        args.data_dir,
        normalize=True,
        target_var=args.target_var,
        start_date=args.start_date,
        end_date=args.end_date,
        dataset_profile=args.dataset_profile,
        lazy=use_lazy_2dto2d_data(args),
        normalization_chunk_size=args.normalization_chunk_size,
        normalization_stats_path=args.normalization_stats_path,
        rebuild_normalization_stats=args.rebuild_normalization_stats,
    )
    args.normalization_stats_path = dataset.normalization_stats_path
    args._normalization_stats_reference = dataset.get_normalization_reference()
    train_idx, val_idx, _ = get_year_split_indices(len(dataset), dataset.start_date)
    if args.max_samples is not None:
        train_idx = train_idx[: args.max_samples]
        val_idx = val_idx[: args.max_samples]
    if not train_idx or not val_idx:
        raise ValueError("训练或验证 split 为空，请检查 --start-date/--end-date")

    if len(depth_levels) != dataset.num_depths:
        raise ValueError(
            f"dataset_profile={args.dataset_profile} 深度层数 {len(depth_levels)} "
            f"与 target 数据层数 {dataset.num_depths} 不一致"
        )
    models = load_du_unet_subset(args, device, depth_indices, depth_levels)
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


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    depth_tensor,
    device,
    use_amp,
    lambda_delta_z,
):
    model.train()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    count = 0
    for batch in loader:
        T0 = batch["T0"].to(device, non_blocking=True)
        T_true = batch["T_true"].to(device, non_blocking=True)
        mask = batch["target_mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(T0, depth_tensor)
            loss_parts = hvca_refiner_loss(
                pred,
                T0,
                T_true,
                mask,
                depth_tensor,
                lambda_delta_z=lambda_delta_z,
            )
            loss = loss_parts["loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        totals += torch.stack(
            [loss, loss_parts["rmse"], loss_parts["delta_z"]]
        ).detach().to(torch.float64)
        count += 1
    values = (totals / max(count, 1)).cpu().tolist()
    return {
        "loss": values[0],
        "rmse": values[1],
        "delta_z": values[2],
    }


@torch.no_grad()
def validate(model, loader, depth_tensor, device, use_amp, lambda_delta_z):
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    count = 0
    for batch in loader:
        T0 = batch["T0"].to(device, non_blocking=True)
        T_true = batch["T_true"].to(device, non_blocking=True)
        mask = batch["target_mask"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(T0, depth_tensor)
            loss_parts = hvca_refiner_loss(
                pred,
                T0,
                T_true,
                mask,
                depth_tensor,
                lambda_delta_z=lambda_delta_z,
            )
            loss = loss_parts["loss"]
        totals += torch.stack(
            [loss, loss_parts["rmse"], loss_parts["delta_z"]]
        ).detach().to(torch.float64)
        count += 1
    values = (totals / max(count, 1)).cpu().tolist()
    return {
        "loss": values[0],
        "rmse": values[1],
        "delta_z": values[2],
    }


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
            "loss_config": {
                "primary": "masked_rmse_loss",
                "lambda_delta_z": float(args.lambda_delta_z),
                "delta_z": "mean_abs_depth_scaled_delta_layer_difference",
            },
            "parameter_count": count_model_parameters(model),
            "depth_values": list(map(float, depth_values)),
            "target_var": args.target_var,
            "dataset_profile": args.dataset_profile,
            "normalization_stats": getattr(
                args, "_normalization_stats_reference", None
            ),
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
    parser.add_argument(
        "--dataset-profile",
        choices=dataset_profile_names(),
        default=DEFAULT_DATASET_PROFILE,
        help="2dto2d 数据集配置；旧实验用 legacy_small，新大区域用 glorys12v1_all_size",
    )
    parser.add_argument("--start-date", default=DATA_START_DATE)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--depths", default=None, help="逗号分隔深度米数，默认当前 dataset profile 的全部深度")
    parser.add_argument(
        "--lazy-data",
        action="store_true",
        help="按样本延迟清洗/归一化；glorys12v1_all_size 会自动启用",
    )
    parser.add_argument(
        "--normalization-chunk-size",
        type=int,
        default=32,
        help="lazy 归一化统计量拟合时的时间分块大小",
    )
    parser.add_argument(
        "--normalization-stats-path",
        default=None,
        help="归一化统计量 .npz；默认保存到 data-dir/normalization_stats/",
    )
    parser.add_argument(
        "--rebuild-normalization-stats",
        action="store_true",
        help="生成 T0 cache 前显式重算并覆盖归一化统计量",
    )
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
    parser.add_argument(
        "--lambda-delta-z",
        type=float,
        default=0.0,
        help="delta 垂向平滑约束权重；0 表示关闭，v2 建议先试 0.01",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="cache DataLoader worker 数")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--shard-cache-size", type=int, default=4, help="每个 worker 内缓存的 shard 数")
    parser.add_argument("--save-dir", default=default_hvca_output_dir())
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="验证 RMSE 连续多少轮不下降后早停；0 表示关闭早停",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.cache_shard_size <= 0:
        raise ValueError("--batch-size 和 --cache-shard-size 必须大于 0")
    if args.lambda_delta_z < 0:
        raise ValueError("--lambda-delta-z 不能为负数")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience 不能为负数")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.device)
    use_amp = device.type == "cuda"
    depth_levels = get_2dto2d_depth_levels(args.dataset_profile)
    depth_values = parse_depths(args.depths, depth_levels)
    depth_indices = resolve_depth_indices(depth_values, depth_levels)
    depth_tensor = torch.as_tensor(depth_values, dtype=torch.float32, device=device)

    print("=" * 60)
    print("HVCARefiner 训练配置")
    print("=" * 60)
    print(f"target_var: {args.target_var}")
    print(f"dataset_profile: {args.dataset_profile}")
    print(f"device: {device}")
    print(f"depths: {depth_values}")
    print(f"train cache: {args.t0_cache_dir}")
    print(f"val cache: {args.val_cache_dir}")
    print(f"Du_Unet checkpoint dir: {args.base_checkpoint_dir}")
    print(f"lambda_delta_z: {args.lambda_delta_z}")
    print(f"early_stopping_patience: {args.early_stopping_patience}")
    print("=" * 60)
    prepare_train_val_cache(args, depth_values, depth_indices, depth_levels, device)

    train_ds = HvcaCacheDataset(args.t0_cache_dir, shard_cache_size=args.shard_cache_size)
    val_ds = HvcaCacheDataset(args.val_cache_dir, shard_cache_size=args.shard_cache_size)
    loader_kwargs = dataloader_kwargs(args, device)
    train_batch_sampler = ShardLocalBatchSampler(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_batch_sampler = ShardLocalBatchSampler(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_batch_sampler,
        **loader_kwargs,
    )

    model = HVCARefiner(
        dim=args.dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        column_chunk_size=args.column_chunk_size,
    ).to(device)
    scheduler_factor = 0.5
    scheduler_patience = 5
    training_strategy = {
        "epoch": {
            "planned": args.epochs,
            "indexing": "1_based",
        },
        "early_stopping": {
            "enabled": args.early_stopping_patience > 0,
            "monitor": "val_rmse",
            "mode": "min",
            "patience": args.early_stopping_patience,
        },
        "optimizer": {
            "name": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "mode": "min",
            "factor": scheduler_factor,
            "patience": scheduler_patience,
            "monitor": "val_loss",
            "step_on": "epoch_end",
        },
        "warmup": {
            "enabled": False,
            "epochs": 0,
        },
    }
    print_and_save_run_config(
        "HVCARefiner 训练超参数",
        {
            "stage": "train",
            "model": "HVCARefiner",
            "paradigm": PARADIGM_2DTO2D,
            "target_var": args.target_var,
            "device": device,
            "seed": args.seed,
            "data": {
                "data_dir": args.data_dir,
                "dataset_profile": args.dataset_profile,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "max_samples": args.max_samples,
                "lazy_data": use_lazy_2dto2d_data(args),
                "normalization_chunk_size": args.normalization_chunk_size,
                "normalization_stats_path": args.normalization_stats_path,
                "normalization_stats": getattr(
                    args, "_normalization_stats_reference", None
                ),
                "rebuild_normalization_stats": args.rebuild_normalization_stats,
            },
            "hyperparameters": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "cache_shard_size": args.cache_shard_size,
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
                "shard_cache_size": args.shard_cache_size,
                "batch_sampling": "shard_local_random",
                "gpu_metric_sync": "epoch_end",
                "non_blocking_transfer": device.type == "cuda",
                "depths": depth_values,
                "depth_indices": depth_indices,
                "use_amp": use_amp,
                "lambda_delta_z": args.lambda_delta_z,
                "early_stopping_patience": args.early_stopping_patience,
            },
            "training_strategy": training_strategy,
            "model_config": {
                "dim": args.dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "column_chunk_size": args.column_chunk_size,
            },
            "loss_config": {
                "primary": "masked_rmse_loss",
                "lambda_delta_z": args.lambda_delta_z,
                "delta_z": "mean_abs_depth_scaled_delta_layer_difference",
                "checkpoint_monitor": "val_rmse",
            },
            "parameter_count": count_model_parameters(model),
            "paths": {
                "save_dir": args.save_dir,
                "t0_cache_dir": args.t0_cache_dir,
                "val_cache_dir": args.val_cache_dir,
                "base_checkpoint_dir": args.base_checkpoint_dir,
            },
        },
        run_config_path(args.save_dir, "train"),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "best.pt")
    best_val = float("inf")
    epochs_without_improvement = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_total_loss": [],
        "val_total_loss": [],
        "train_delta_z_loss": [],
        "val_delta_z_loss": [],
    }

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            depth_tensor,
            device,
            use_amp,
            args.lambda_delta_z,
        )
        val_stats = validate(
            model, val_loader, depth_tensor, device, use_amp, args.lambda_delta_z
        )
        train_loss = train_stats["rmse"]
        val_loss = val_stats["rmse"]
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_total_loss"].append(train_stats["loss"])
        history["val_total_loss"].append(val_stats["loss"])
        history["train_delta_z_loss"].append(train_stats["delta_z"])
        history["val_delta_z_loss"].append(val_stats["delta_z"])
        lr_now = optimizer.param_groups[0]["lr"]
        if args.lambda_delta_z > 0:
            print(
                f"Epoch [{epoch}/{args.epochs}] | "
                f"Train RMSE: {train_loss:.6f} | Val RMSE: {val_loss:.6f} | "
                f"Train Total: {train_stats['loss']:.6f} | "
                f"Val Total: {val_stats['loss']:.6f} | "
                f"DeltaZ: {train_stats['delta_z']:.6f}/{val_stats['delta_z']:.6f} | "
                f"LR: {lr_now:.2e}"
            )
        else:
            print(
                f"Epoch [{epoch}/{args.epochs}] | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr_now:.2e}"
            )
        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, optimizer, args, depth_values, best_val, epoch)
            print(f"  -> 保存最佳 HVCARefiner: {best_path} (val={best_val:.6f})")
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0:
                print(
                    f"  -> Val 未刷新 best: "
                    f"{epochs_without_improvement}/{args.early_stopping_patience}"
                )
                if epochs_without_improvement >= args.early_stopping_patience:
                    print(
                        f"  -> 早停触发：验证 RMSE 连续 "
                        f"{args.early_stopping_patience} 轮未下降"
                    )
                    break

    np.savez(
        os.path.join(args.save_dir, "training_history_HVCARefiner.npz"),
        train_loss=np.asarray(history["train_loss"], dtype=np.float32),
        val_loss=np.asarray(history["val_loss"], dtype=np.float32),
        train_total_loss=np.asarray(history["train_total_loss"], dtype=np.float32),
        val_total_loss=np.asarray(history["val_total_loss"], dtype=np.float32),
        train_delta_z_loss=np.asarray(history["train_delta_z_loss"], dtype=np.float32),
        val_delta_z_loss=np.asarray(history["val_delta_z_loss"], dtype=np.float32),
        lambda_delta_z=np.asarray(args.lambda_delta_z, dtype=np.float32),
    )
    save_loss_curve(
        history["train_loss"],
        history["val_loss"],
        os.path.join(args.save_dir, "loss_HVCARefiner.png"),
    )
    print(f"\n训练完成。最佳验证 RMSE loss: {best_val:.6f}")
    print(f"最佳权重: {best_path}")


if __name__ == "__main__":
    main()
