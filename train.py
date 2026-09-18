"""统一训练入口，支持逐层、联合深度和 ocean_transformer 训练。"""

import argparse
import os
import time
from datetime import datetime

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    B3_DEPTHFORMER_BASE_GRID_SIZE,
    B3_DEPTHFORMER_BATCH_SIZE,
    B3_DEPTHFORMER_CKPT_NAME_TEMPLATE,
    B3_DEPTHFORMER_DECODER_LAYERS,
    B3_DEPTHFORMER_DIM_FF,
    B3_DEPTHFORMER_D_MODEL,
    B3_DEPTHFORMER_DROPOUT,
    B3_DEPTHFORMER_ENCODER_LAYERS,
    B3_DEPTHFORMER_EPOCHS,
    B3_DEPTHFORMER_HISTORY_NAME_TEMPLATE,
    B3_DEPTHFORMER_LOSS_CURVE_TEMPLATE,
    B3_DEPTHFORMER_LR,
    B3_DEPTHFORMER_NHEAD,
    B3_DEPTHFORMER_OUTPUT_PATCH_SIZE,
    B3_DEPTHFORMER_PATIENCE,
    B3_DEPTHFORMER_REFINEMENT_CHANNELS,
    B3_DEPTHFORMER_SST_PATCH_SIZE,
    B3_DEPTHFORMER_SURFACE_PATCH_SIZE,
    B3_DEPTHFORMER_TEACHER_FORCING_DECAY_EPOCHS,
    B3_DEPTHFORMER_TEACHER_FORCING_END,
    B3_DEPTHFORMER_TEACHER_FORCING_START,
    B3_DEPTHFORMER_WEIGHT_DECAY,
    CHECKPOINTS_ROOT,
    DATA_DIR,
    DATA_START_DATE,
    DEFAULT_DATASET_PROFILE,
    DX,
    DY,
    DU_UNET_BATCH_SIZE,
    DU_UNET_CKPT_NAME_TEMPLATE,
    DU_UNET_EPOCHS,
    DU_UNET_HISTORY_NAME_TEMPLATE,
    DU_UNET_LOSS_CURVE_TEMPLATE,
    DU_UNET_LR,
    DU_UNET_PATIENCE,
    DU_UNET_WEIGHT_DECAY,
    JOINT_DU_UNET_CKPT_NAME_TEMPLATE,
    JOINT_DU_UNET_HISTORY_NAME_TEMPLATE,
    JOINT_DU_UNET_LOSS_CURVE_TEMPLATE,
    JOINT_DUVIT_CKPT_NAME_TEMPLATE,
    JOINT_DUVIT_DROPOUT,
    JOINT_DUVIT_HISTORY_NAME_TEMPLATE,
    JOINT_DUVIT_LOSS_CURVE_TEMPLATE,
    JOINT_DUVIT_MLP_RATIO,
    JOINT_DUVIT_SST_EMBED_DIM,
    JOINT_DUVIT_SST_NUM_HEADS,
    JOINT_DUVIT_SST_NUM_LAYERS,
    JOINT_DUVIT_SST_MAX_TOKENS,
    JOINT_DUVIT_SST_PATCH_SIZE,
    JOINT_DUVIT_SURFACE_EMBED_DIM,
    JOINT_DUVIT_SURFACE_NUM_HEADS,
    JOINT_DUVIT_SURFACE_NUM_LAYERS,
    JOINT_DUVIT_SURFACE_MAX_TOKENS,
    JOINT_DUVIT_SURFACE_PATCH_SIZE,
    TWODTO3D_DATA_START_DATE,
    TWODTO3D_BATCH_SIZE,
    TWODTO3D_D_MODEL,
    TWODTO3D_DEPTH_LAYERS,
    TWODTO3D_DEPTH_LEVELS,
    TWODTO3D_DIM_FF,
    TWODTO3D_EPOCHS,
    TWODTO3D_IN_CHANNELS,
    TWODTO3D_LAMBDA_HYDRO,
    TWODTO3D_LAMBDA_STRAT,
    TWODTO3D_LR,
    TWODTO3D_NHEAD,
    TWODTO3D_NUM_DEPTHS,
    TWODTO3D_OUT_VARS,
    TWODTO3D_PATIENCE,
    TWODTO3D_SPATIAL_LAYERS,
    TWODTO3D_WEIGHT_DECAY,
    OUTPUTS_ROOT,
    PARADIGM_2DTO2D,
    PARADIGM_2DTO2D_METHODS,
    PARADIGM_2DTO3D,
    PARADIGM_2DTO3D_METHODS,
    SEED,
    dataset_profile_names,
    get_2dto2d_depth_levels,
    get_checkpoint_dir,
    get_output_dir,
    joint_du_unet_depth_tag,
)
from datasets.dataset_2dto2d import Dataset2Dto2D
from datasets.dataset_2dto3d import DummyTwoDto3DDataset
from models.b3_depthformer import B3DepthFormer
from models.du_unet import Du_Unet
from models.joint_du_unet import JointDuUnet
from models.joint_duvit import JointDUViT
from models.ocean_transformer import OceanTransformer
from utils.losses import masked_depth_mean_rmse_loss, masked_rmse_loss
from utils.physics import compute_eke, compute_grad_ssh
from utils.physics_loss import PhysicsLoss
from utils.run_config import (
    count_model_parameters,
    model_config_path,
    print_and_save_run_config,
)
from utils.reproducibility import (
    build_reproducibility_report,
    configure_reproducibility,
    make_dataloader_generator,
    print_reproducibility_report,
    seed_dataloader_worker,
)

TRAINABLE_METHODS = {
    "b3_depthformer",
    "du_unet",
    "joint_du_unet",
    "joint_duvit",
    "ocean_transformer",
}


class DepthTargetDataset(Dataset):
    """Expose one target depth while reusing the surface inputs and normalization."""

    def __init__(self, dataset, depth_idx):
        self.dataset = dataset
        self.depth_idx = int(depth_idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.get_depth_item(idx, self.depth_idx)


class DepthIndicesTargetDataset(Dataset):
    """Expose an ordered target-depth subset for one joint-output model."""

    def __init__(self, dataset, depth_indices):
        self.dataset = dataset
        self.depth_indices = tuple(int(idx) for idx in depth_indices)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset.get_depth_indices_item(idx, self.depth_indices)


def use_lazy_2dto2d_data(args):
    return args.lazy_data or args.dataset_profile == "glorys12v1_all_size"


def dataloader_kwargs(args, device, seed_offset=0):
    kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory and device.type == "cuda"),
        "generator": make_dataloader_generator(args.seed + seed_offset),
        "worker_init_fn": seed_dataloader_worker,
    }
    if args.num_workers > 0:
        kwargs.update(
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
        )
    return kwargs


def build_model(method, **kwargs):
    """根据 method 构建模型。"""
    name = method.strip().lower()
    out_ch = kwargs.get("out_channels", 1)
    if name in {"du_unet", "du-unet"}:
        return Du_Unet(out_channels=out_ch)
    if name in {"joint_du_unet", "joint-du-unet"}:
        num_depths = kwargs.get("num_depths", kwargs.get("out_channels"))
        if num_depths is None:
            raise ValueError("Joint_Du_Unet 构建时必须提供 num_depths")
        return JointDuUnet(num_depths=num_depths)
    if name in {"joint_duvit", "joint-duvit"}:
        num_depths = kwargs.get("num_depths", kwargs.get("out_channels"))
        if num_depths is None:
            raise ValueError("Joint_DUViT 构建时必须提供 num_depths")
        return JointDUViT(
            num_depths=num_depths,
            sst_embed_dim=kwargs.get(
                "sst_embed_dim", JOINT_DUVIT_SST_EMBED_DIM
            ),
            sst_patch_size=kwargs.get(
                "sst_patch_size", JOINT_DUVIT_SST_PATCH_SIZE
            ),
            sst_num_heads=kwargs.get(
                "sst_num_heads", JOINT_DUVIT_SST_NUM_HEADS
            ),
            sst_num_layers=kwargs.get(
                "sst_num_layers", JOINT_DUVIT_SST_NUM_LAYERS
            ),
            sst_max_tokens=kwargs.get(
                "sst_max_tokens", JOINT_DUVIT_SST_MAX_TOKENS
            ),
            surface_embed_dim=kwargs.get(
                "surface_embed_dim", JOINT_DUVIT_SURFACE_EMBED_DIM
            ),
            surface_patch_size=kwargs.get(
                "surface_patch_size", JOINT_DUVIT_SURFACE_PATCH_SIZE
            ),
            surface_num_heads=kwargs.get(
                "surface_num_heads", JOINT_DUVIT_SURFACE_NUM_HEADS
            ),
            surface_num_layers=kwargs.get(
                "surface_num_layers", JOINT_DUVIT_SURFACE_NUM_LAYERS
            ),
            surface_max_tokens=kwargs.get(
                "surface_max_tokens", JOINT_DUVIT_SURFACE_MAX_TOKENS
            ),
            mlp_ratio=kwargs.get("mlp_ratio", JOINT_DUVIT_MLP_RATIO),
            dropout=kwargs.get("dropout", JOINT_DUVIT_DROPOUT),
        )
    if name in {"b3_depthformer", "b3-depthformer", "b3"}:
        depth_values = kwargs.get("depth_values")
        if depth_values is None:
            raise ValueError("B3_DepthFormer 构建时必须提供 depth_values")
        return B3DepthFormer(
            depth_values=depth_values,
            d_model=kwargs.get("d_model", B3_DEPTHFORMER_D_MODEL),
            nhead=kwargs.get("nhead", B3_DEPTHFORMER_NHEAD),
            encoder_layers=kwargs.get(
                "encoder_layers", B3_DEPTHFORMER_ENCODER_LAYERS
            ),
            decoder_layers=kwargs.get(
                "decoder_layers", B3_DEPTHFORMER_DECODER_LAYERS
            ),
            dim_feedforward=kwargs.get(
                "dim_feedforward", B3_DEPTHFORMER_DIM_FF
            ),
            dropout=kwargs.get("dropout", B3_DEPTHFORMER_DROPOUT),
            sst_patch_size=kwargs.get(
                "sst_patch_size", B3_DEPTHFORMER_SST_PATCH_SIZE
            ),
            surface_patch_size=kwargs.get(
                "surface_patch_size", B3_DEPTHFORMER_SURFACE_PATCH_SIZE
            ),
            output_patch_size=kwargs.get(
                "output_patch_size", B3_DEPTHFORMER_OUTPUT_PATCH_SIZE
            ),
            refinement_channels=kwargs.get(
                "refinement_channels", B3_DEPTHFORMER_REFINEMENT_CHANNELS
            ),
            base_grid_size=kwargs.get(
                "base_grid_size", B3_DEPTHFORMER_BASE_GRID_SIZE
            ),
            depth_attention=kwargs.get("depth_attention", "causal"),
        )
    if name == "ocean_transformer":
        return OceanTransformer(
            in_channels=kwargs.get("in_channels", TWODTO3D_IN_CHANNELS),
            d_model=kwargs.get("d_model", TWODTO3D_D_MODEL),
            nhead=kwargs.get("nhead", TWODTO3D_NHEAD),
            spatial_layers=kwargs.get("spatial_layers", TWODTO3D_SPATIAL_LAYERS),
            depth_layers=kwargs.get("depth_layers", TWODTO3D_DEPTH_LAYERS),
            dim_ff=kwargs.get("dim_ff", TWODTO3D_DIM_FF),
            num_depths=kwargs.get("num_depths", TWODTO3D_NUM_DEPTHS),
            out_vars=kwargs.get("out_vars", TWODTO3D_OUT_VARS),
        )
    raise ValueError(f"未识别的方法: {method}")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_year_split_indices(total_len, start_date_str="2019-01-01"):
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    train_days = (datetime(2021, 12, 31) - start).days + 1
    val_days = (datetime(2022, 12, 31) - datetime(2021, 12, 31)).days
    train_idx = list(range(min(train_days, total_len)))
    val_start = train_days
    val_idx = list(range(val_start, min(val_start + val_days, total_len)))
    test_start = val_start + val_days
    test_idx = list(range(test_start, total_len))
    return train_idx, val_idx, test_idx


def save_loss_curve(train_losses, val_losses, output_path):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"损失曲线已保存至：{output_path}")


def select_target_depth(target, depth_idx=None):
    if depth_idx is None:
        return target
    if target.ndim != 4:
        raise ValueError(f"target 需为 (B,C,H,W)，实际：{tuple(target.shape)}")
    return target[:, depth_idx : depth_idx + 1]


def train_2dto2d_epoch(
    model,
    loader,
    optimizer,
    device,
    depth_idx=None,
    equal_depth_weighting=False,
):
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        sst = batch["sst"].to(device, non_blocking=True)
        ssh_sss = batch["ssh_sss"].to(device, non_blocking=True)
        target = select_target_depth(
            batch["target"].to(device, non_blocking=True), depth_idx=depth_idx
        )
        target_mask = select_target_depth(
            batch["target_mask"].to(device, non_blocking=True), depth_idx=depth_idx
        )
        pred = model(sst, ssh_sss, output_size=target.shape[-2:])
        loss_fn = (
            masked_depth_mean_rmse_loss
            if equal_depth_weighting
            else masked_rmse_loss
        )
        loss = loss_fn(pred, target, target_mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
        count += 1
    return total / max(count, 1)


def validate_2dto2d(
    model,
    loader,
    device,
    depth_idx=None,
    equal_depth_weighting=False,
):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            sst = batch["sst"].to(device, non_blocking=True)
            ssh_sss = batch["ssh_sss"].to(device, non_blocking=True)
            target = select_target_depth(
                batch["target"].to(device, non_blocking=True), depth_idx=depth_idx
            )
            target_mask = select_target_depth(
                batch["target_mask"].to(device, non_blocking=True), depth_idx=depth_idx
            )
            pred = model(sst, ssh_sss, output_size=target.shape[-2:])
            loss_fn = (
                masked_depth_mean_rmse_loss
                if equal_depth_weighting
                else masked_rmse_loss
            )
            loss = loss_fn(pred, target, target_mask)
            total += loss.item()
            count += 1
    return total / max(count, 1)


def teacher_forcing_ratio_for_epoch(epoch, start, end, decay_epochs):
    """Linearly decay the probability of feeding the previous true layer."""
    start = float(start)
    end = float(end)
    decay_epochs = int(decay_epochs)
    if not 0.0 <= start <= 1.0 or not 0.0 <= end <= 1.0:
        raise ValueError("teacher forcing start/end 必须位于 [0,1]")
    if decay_epochs <= 0:
        raise ValueError("teacher forcing decay epochs 必须大于0")
    progress = min(max(int(epoch), 0) / max(decay_epochs - 1, 1), 1.0)
    return start + (end - start) * progress


def train_b3_depthformer_epoch(
    model,
    loader,
    optimizer,
    device,
    teacher_forcing_ratio,
):
    """Train B3 with selected normalized depths and scheduled sampling."""
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        sst = batch["sst"].to(device, non_blocking=True)
        ssh_sss = batch["ssh_sss"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        target_mask = batch["target_mask"].to(device, non_blocking=True)
        pred = model(
            sst,
            ssh_sss,
            target=target,
            output_size=target.shape[-2:],
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        loss = masked_depth_mean_rmse_loss(pred, target, target_mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
        count += 1
    return total / max(count, 1)


def validate_b3_depthformer(model, loader, device):
    """Validate under the same fully autoregressive condition used at test time."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            sst = batch["sst"].to(device, non_blocking=True)
            ssh_sss = batch["ssh_sss"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            target_mask = batch["target_mask"].to(device, non_blocking=True)
            pred = model(
                sst,
                ssh_sss,
                output_size=target.shape[-2:],
            )
            loss = masked_depth_mean_rmse_loss(pred, target, target_mask)
            total += loss.item()
            count += 1
    return total / max(count, 1)


def build_2dto3d_features(surface_raw):
    ssh = surface_raw[:, 0:1]
    sss = surface_raw[:, 1:2]
    grad = compute_grad_ssh(ssh)
    eke = compute_eke(ssh, DX, DY)
    return torch.cat([ssh, sss, grad, eke], dim=1)


def train_2dto3d_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        surface_raw = batch["surface_raw"].to(device)
        target = batch["target"].to(device)
        sla = batch.get("sla")
        if sla is not None:
            sla = sla.to(device)
        pred = model(build_2dto3d_features(surface_raw))
        loss, _ = criterion(pred, target, sla)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
        count += 1
    return total / max(count, 1)


def validate_2dto3d(model, loader, device, criterion):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            surface_raw = batch["surface_raw"].to(device)
            target = batch["target"].to(device)
            sla = batch.get("sla")
            if sla is not None:
                sla = sla.to(device)
            loss, _ = criterion(model(build_2dto3d_features(surface_raw)), target, sla)
            total += loss.item()
            count += 1
    return total / max(count, 1)


def paradigm_of_method(method):
    if method in PARADIGM_2DTO2D_METHODS:
        return PARADIGM_2DTO2D
    if method in PARADIGM_2DTO3D_METHODS:
        return PARADIGM_2DTO3D
    return PARADIGM_2DTO3D


def train_one_model(
    method,
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    patience,
    checkpoint_path,
    depth_idx=None,
    criterion=None,
    equal_depth_weighting=False,
    checkpoint_metadata=None,
    teacher_forcing_schedule=None,
):
    best_val = float("inf")
    patience_cnt = 0
    train_losses, val_losses, epoch_times_seconds = [], [], []
    training_started = time.perf_counter()

    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        teacher_forcing_ratio = None
        if method == "ocean_transformer":
            t_loss = train_2dto3d_epoch(model, train_loader, optimizer, device, criterion)
            v_loss = validate_2dto3d(model, val_loader, device, criterion)
        elif method == "b3_depthformer":
            if teacher_forcing_schedule is None:
                raise ValueError("B3 训练缺少 teacher_forcing_schedule")
            teacher_forcing_ratio = teacher_forcing_ratio_for_epoch(
                epoch, *teacher_forcing_schedule
            )
            t_loss = train_b3_depthformer_epoch(
                model,
                train_loader,
                optimizer,
                device,
                teacher_forcing_ratio,
            )
            v_loss = validate_b3_depthformer(model, val_loader, device)
        else:
            t_loss = train_2dto2d_epoch(
                model,
                train_loader,
                optimizer,
                device,
                depth_idx=depth_idx,
                equal_depth_weighting=equal_depth_weighting,
            )
            v_loss = validate_2dto2d(
                model,
                val_loader,
                device,
                depth_idx=depth_idx,
                equal_depth_weighting=equal_depth_weighting,
            )

        scheduler.step()
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        lr_now = scheduler.get_last_lr()[0]
        teacher_text = (
            f" | Teacher forcing: {teacher_forcing_ratio:.3f}"
            if teacher_forcing_ratio is not None
            else ""
        )
        epoch_elapsed = time.perf_counter() - epoch_started
        epoch_times_seconds.append(epoch_elapsed)
        print(
            f"Epoch [{epoch + 1}/{epochs}] | Train: {t_loss:.6f} | "
            f"Val: {v_loss:.6f} | LR: {lr_now:.2e}{teacher_text} | "
            f"Time: {epoch_elapsed:.2f}s"
        )

        if v_loss < best_val:
            best_val = v_loss
            if checkpoint_metadata is None:
                checkpoint_payload = model.state_dict()
            else:
                checkpoint_payload = {
                    "format_version": 1,
                    "state_dict": model.state_dict(),
                    "metadata": dict(checkpoint_metadata),
                    "best_epoch": int(epoch + 1),
                    "best_val_loss": float(v_loss),
                }
            torch.save(checkpoint_payload, checkpoint_path)
            patience_cnt = 0
            print(f"  -> 保存最佳模型 (val={v_loss:.6f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  -> 早停触发 (patience={patience})")
                break

    total_training_seconds = time.perf_counter() - training_started
    mean_epoch_seconds = (
        float(np.mean(epoch_times_seconds)) if epoch_times_seconds else 0.0
    )
    print(
        "训练计时 | "
        f"总耗时: {total_training_seconds:.2f}s "
        f"({total_training_seconds / 60.0:.2f}min, "
        f"{total_training_seconds / 3600.0:.2f}h) | "
        f"平均每轮: {mean_epoch_seconds:.2f}s | "
        f"实际轮数: {len(epoch_times_seconds)}"
    )
    return (
        best_val,
        train_losses,
        val_losses,
        epoch_times_seconds,
        total_training_seconds,
    )


def parse_depth_indices(depth_indices_arg, depth_levels):
    # 如果没传，训练该 profile 的全部深度层。
    if not depth_indices_arg:
        return list(range(len(depth_levels)))
    out = []
    for item in depth_indices_arg.split(","):
        text = item.strip()
        if not text:
            raise ValueError("--depth-indices 不能包含空项")
        idx = int(text)
        if idx < 0 or idx >= len(depth_levels):
            raise ValueError(f"depth index 越界: {idx}")
        if idx in out:
            raise ValueError(f"depth index 重复: {idx}")
        out.append(idx)
    if not out:
        raise ValueError("--depth-indices 不能为空")
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="统一训练入口（2dto2d / 2dto3d）")
    parser.add_argument("--method", required=True)
    parser.add_argument("--target-var", choices=["temperature", "salinity"], default="temperature",
                        help="Du_Unet/Joint_Du_Unet 训练目标变量")
    parser.add_argument("--start-date", default=DATA_START_DATE,
                        help="Du_Unet 使用的数据起始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", default=None,
                        help="Du_Unet 使用的数据结束日期 YYYY-MM-DD，默认使用配置中的结束日期")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader CPU worker 数；0 表示由训练主进程读取数据",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="每个 DataLoader worker 预取的 batch 数，仅 num-workers>0 时生效",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="CUDA 训练时使用锁页内存，加快 CPU 到 GPU 的异步传输",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="跨 epoch 保留 DataLoader worker，仅 num-workers>0 时生效",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument(
        "--dataset-profile",
        choices=dataset_profile_names(),
        default=DEFAULT_DATASET_PROFILE,
        help="2dto2d 数据集配置；旧实验用 legacy_small，新大区域用 glorys12v1_all_size",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="2dto2d 调试用：跳过月气候态归一化，避免 smoke test 卡在统计量拟合",
    )
    parser.add_argument(
        "--lazy-data",
        action="store_true",
        help="2dto2d 按样本延迟清洗/归一化；glorys12v1_all_size 会自动启用",
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
        help="忽略已有统计文件并用训练时段重新拟合后覆盖",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="2dto2d 调试用：限制 train/val 各自最多使用多少样本",
    )
    parser.add_argument("--output-dir", default=OUTPUTS_ROOT)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINTS_ROOT)
    parser.add_argument("--dummy", action="store_true", help="2dto3d(ocean_transformer) 使用合成数据")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--quality-report", action="store_true",
        help="2dto2d模型加载数据时生成缺失率质检图")
    parser.add_argument(
        "--depth-indices",
        default=None,
        help=(
            "原始数据中的目标深度索引；Du_Unet逐层训练，"
            "联合模型和B3用一个模型输出这些层；"
            "默认使用全部深度"
        ),
    )
    b3 = parser.add_argument_group("B3 DepthFormer 模型参数")
    b3.add_argument("--b3-d-model", type=int, default=B3_DEPTHFORMER_D_MODEL)
    b3.add_argument("--b3-nhead", type=int, default=B3_DEPTHFORMER_NHEAD)
    b3.add_argument(
        "--b3-encoder-layers",
        type=int,
        default=B3_DEPTHFORMER_ENCODER_LAYERS,
    )
    b3.add_argument(
        "--b3-decoder-layers",
        type=int,
        default=B3_DEPTHFORMER_DECODER_LAYERS,
    )
    b3.add_argument(
        "--b3-dim-feedforward",
        type=int,
        default=B3_DEPTHFORMER_DIM_FF,
    )
    b3.add_argument("--b3-dropout", type=float, default=B3_DEPTHFORMER_DROPOUT)
    b3.add_argument(
        "--b3-sst-patch-size",
        type=int,
        default=B3_DEPTHFORMER_SST_PATCH_SIZE,
    )
    b3.add_argument(
        "--b3-surface-patch-size",
        type=int,
        default=B3_DEPTHFORMER_SURFACE_PATCH_SIZE,
    )
    b3.add_argument(
        "--b3-output-patch-size",
        type=int,
        default=B3_DEPTHFORMER_OUTPUT_PATCH_SIZE,
    )
    b3.add_argument(
        "--b3-refinement-channels",
        type=int,
        default=B3_DEPTHFORMER_REFINEMENT_CHANNELS,
    )
    b3.add_argument(
        "--b3-base-grid-height",
        type=int,
        default=B3_DEPTHFORMER_BASE_GRID_SIZE[0],
    )
    b3.add_argument(
        "--b3-base-grid-width",
        type=int,
        default=B3_DEPTHFORMER_BASE_GRID_SIZE[1],
    )
    b3.add_argument(
        "--b3-teacher-forcing-start",
        type=float,
        default=B3_DEPTHFORMER_TEACHER_FORCING_START,
    )
    b3.add_argument(
        "--b3-teacher-forcing-end",
        type=float,
        default=B3_DEPTHFORMER_TEACHER_FORCING_END,
    )
    b3.add_argument(
        "--b3-teacher-forcing-decay-epochs",
        type=int,
        default=B3_DEPTHFORMER_TEACHER_FORCING_DECAY_EPOCHS,
    )
    b3.add_argument(
        "--b3-depth-attention",
        choices=["causal", "independent"],
        default="causal",
        help=(
            "causal 使用原始B3因果深度注意力；independent仅允许每个深度"
            "关注自身，模型参数量保持不变"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers 不能小于 0")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor 必须大于 0")
    if args.persistent_workers and args.num_workers == 0:
        raise ValueError("--persistent-workers 要求 --num-workers 大于 0")
    method = args.method.strip().lower()
    if method == "du-unet":
        method = "du_unet"
    if method in {"joint-du-unet", "joint_unet", "joint-unet"}:
        method = "joint_du_unet"
    if method in {"joint-duvit", "joint_vit", "joint-vit"}:
        method = "joint_duvit"
    if method in {"b3-depthformer", "b3"}:
        method = "b3_depthformer"
    if method not in TRAINABLE_METHODS:
        raise ValueError(f"未识别的方法: {args.method}")
    paradigm = paradigm_of_method(method)
    method_dir = {
        "du_unet": "Du_Unet",
        "joint_du_unet": "Joint_Du_Unet",
        "joint_duvit": "Joint_DUViT",
        "b3_depthformer": "B3_DepthFormer",
    }.get(method, method)

    configure_reproducibility(args.seed)
    device = get_device()
    reproducibility = build_reproducibility_report(args.seed, device)

    if method in {
        "du_unet",
        "joint_du_unet",
        "joint_duvit",
        "b3_depthformer",
    }:
        out_dir = get_output_dir(
            paradigm, method_dir, base_dir=args.output_dir, target_var=args.target_var
        )
        requested_ckpt_dir = os.path.normpath(args.checkpoint_dir)
        if os.path.basename(requested_ckpt_dir) == args.target_var:
            ckpt_dir = requested_ckpt_dir
        else:
            ckpt_dir = get_checkpoint_dir(
                paradigm,
                method_dir,
                base_dir=args.checkpoint_dir,
                target_var=args.target_var,
            )
    else:
        out_dir = get_output_dir(paradigm, method_dir, base_dir=args.output_dir)
        ckpt_dir = get_checkpoint_dir(paradigm, method_dir, base_dir=args.checkpoint_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print("=" * 60)
    print("训练配置")
    print("=" * 60)
    print(f"方法:       {method_dir}")
    print(f"范式:       {paradigm}")
    print(f"设备:       {device}")
    print_reproducibility_report(reproducibility)
    print(f"输出目录:   {out_dir}")
    print(f"权重目录:   {ckpt_dir}")
    print("=" * 60)

    if method == "ocean_transformer":
        from datasets.dataset_2dto3d import TwoDto3DDataset

        epochs = args.epochs or TWODTO3D_EPOCHS
        batch_size = args.batch_size or TWODTO3D_BATCH_SIZE
        lr = args.lr or TWODTO3D_LR
        patience = args.patience or TWODTO3D_PATIENCE

        if args.dummy:
            dataset = DummyTwoDto3DDataset(num_samples=200, H=32, W=32)
            train_idx, val_idx = list(range(140)), list(range(140, 170))
        else:
            dataset = TwoDto3DDataset(args.data_dir, normalize=True)
            train_idx, val_idx, _ = get_year_split_indices(len(dataset), TWODTO3D_DATA_START_DATE)

        train_loader_kwargs = dataloader_kwargs(args, device, seed_offset=0)
        val_loader_kwargs = dataloader_kwargs(args, device, seed_offset=1)
        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=batch_size,
            shuffle=True,
            **train_loader_kwargs,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=batch_size,
            shuffle=False,
            **val_loader_kwargs,
        )

        model = build_model(method).to(device)
        criterion = PhysicsLoss(
            depth_levels=TWODTO3D_DEPTH_LEVELS,
            lambda_hydro=TWODTO3D_LAMBDA_HYDRO,
            lambda_strat=TWODTO3D_LAMBDA_STRAT,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=TWODTO3D_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.5)

        best_ckpt = os.path.join(ckpt_dir, f"{method}_best.pth")
        (
            best_val,
            train_losses,
            val_losses,
            epoch_times_seconds,
            total_training_seconds,
        ) = train_one_model(
            method, model, train_loader, val_loader, optimizer, scheduler, device,
            epochs, patience, best_ckpt, criterion=criterion
        )
        np.savez(
            os.path.join(out_dir, f"training_history_{method}.npz"),
            train_losses=train_losses,
            val_losses=val_losses,
            epoch_times_seconds=np.asarray(epoch_times_seconds, dtype=np.float64),
            total_training_seconds=np.asarray(total_training_seconds, dtype=np.float64),
        )
        save_loss_curve(train_losses, val_losses, os.path.join(out_dir, f"loss_{method}.png"))
        print(f"\n训练完成。最佳验证损失: {best_val:.6f}")
        print(f"最佳权重: {best_ckpt}")
        return

    dataset = Dataset2Dto2D(
        args.data_dir,
        normalize=not args.no_normalize,
        target_var=args.target_var,
        start_date=args.start_date,
        end_date=args.end_date,
        quality_report=args.quality_report,
        dataset_profile=args.dataset_profile,
        lazy=use_lazy_2dto2d_data(args),
        normalization_chunk_size=args.normalization_chunk_size,
        normalization_stats_path=args.normalization_stats_path,
        rebuild_normalization_stats=args.rebuild_normalization_stats,
    )
    depth_levels = get_2dto2d_depth_levels(args.dataset_profile)
    if len(depth_levels) != dataset.num_depths:
        raise ValueError(
            f"dataset_profile={args.dataset_profile} 深度层数 {len(depth_levels)} "
            f"与 target 数据层数 {dataset.num_depths} 不一致"
        )
    train_idx, val_idx, _ = get_year_split_indices(len(dataset), dataset.start_date)
    if args.max_samples is not None:
        train_idx = train_idx[: args.max_samples]
        val_idx = val_idx[: args.max_samples]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(
            f"当前日期范围 [{dataset.start_date}, {dataset.end_date}] 无法形成训练/验证集；"
            "建议本地调试使用 --start-date 2021-01-01 --end-date 2023-12-31"
        )

    if method == "du_unet":
        depth_indices = parse_depth_indices(args.depth_indices, depth_levels)
        epochs = args.epochs or DU_UNET_EPOCHS
        batch_size = args.batch_size or DU_UNET_BATCH_SIZE
        lr = args.lr or DU_UNET_LR
        patience = args.patience or DU_UNET_PATIENCE
        param_stats = count_model_parameters(build_model("du_unet", out_channels=1))
        scheduler_step_size = max(epochs // 3, 1)
        training_strategy = {
            "epoch": {
                "planned": epochs,
                "indexing": "1_based",
            },
            "early_stopping": {
                "enabled": True,
                "monitor": "val_loss",
                "mode": "min",
                "patience": patience,
            },
            "optimizer": {
                "name": "Adam",
                "lr": lr,
                "weight_decay": DU_UNET_WEIGHT_DECAY,
            },
            "scheduler": {
                "name": "StepLR",
                "step_size": scheduler_step_size,
                "gamma": 0.5,
                "step_on": "epoch",
            },
            "warmup": {
                "enabled": False,
                "epochs": 0,
            },
        }

        loader_kwargs = dataloader_kwargs(args, device)
        print_and_save_run_config(
            "Du_Unet 训练超参数",
            {
                "stage": "train",
                "model": "Du_Unet",
                "method": method,
                "paradigm": paradigm,
                "target_var": args.target_var,
                "device": device,
                "seed": args.seed,
                "reproducibility": reproducibility,
                "data": {
                    "data_dir": args.data_dir,
                    "dataset_profile": args.dataset_profile,
                    "start_date": dataset.start_date,
                    "end_date": dataset.end_date,
                    "train_samples": len(train_idx),
                    "val_samples": len(val_idx),
                    "shapes": dataset.get_data_info(),
                    "normalize": not args.no_normalize,
                    "lazy_data": use_lazy_2dto2d_data(args),
                    "normalization_chunk_size": args.normalization_chunk_size,
                    "num_workers": args.num_workers,
                    "prefetch_factor": (
                        args.prefetch_factor if args.num_workers > 0 else None
                    ),
                    "pin_memory": loader_kwargs["pin_memory"],
                    "persistent_workers": (
                        args.persistent_workers if args.num_workers > 0 else False
                    ),
                    "dataloader_seed_rule": (
                        "train=seed+2*depth_index; val=train+1"
                    ),
                    "target_loading": "selected_depth_only",
                    "normalization_stats_path": dataset.normalization_stats_path,
                    "normalization_stats": dataset.get_normalization_reference(),
                    "normalization_stats_loaded": dataset.normalization_stats_loaded,
                    "rebuild_normalization_stats": args.rebuild_normalization_stats,
                    "max_samples": args.max_samples,
                },
                "hyperparameters": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "learning_rate": lr,
                    "weight_decay": DU_UNET_WEIGHT_DECAY,
                    "patience": patience,
                    "depth_indices": depth_indices,
                    "depth_m": [depth_levels[i] for i in depth_indices],
                },
                "training_strategy": training_strategy,
                "model_config": {
                    "out_channels": 1,
                },
                "parameter_count": {
                    "per_depth_model": param_stats,
                    "num_depth_models": len(depth_indices),
                    "total_if_all_depth_models_loaded": {
                        name: value * len(depth_indices)
                        for name, value in param_stats.items()
                    },
                },
                "paths": {
                    "output_dir": out_dir,
                    "checkpoint_dir": ckpt_dir,
                },
            },
            model_config_path(args.output_dir, "Du_Unet", "train"),
        )

        for depth_idx in depth_indices:
            depth_m = depth_levels[depth_idx]
            print("\n" + "-" * 60)
            print(f"训练变量: {args.target_var} | 深度层: idx={depth_idx}, depth={depth_m}m")
            print("-" * 60)

            depth_dataset = DepthTargetDataset(dataset, depth_idx)
            train_loader_kwargs = dataloader_kwargs(
                args, device, seed_offset=2 * depth_idx
            )
            val_loader_kwargs = dataloader_kwargs(
                args, device, seed_offset=2 * depth_idx + 1
            )
            train_loader = DataLoader(
                Subset(depth_dataset, train_idx),
                batch_size=batch_size,
                shuffle=True,
                **train_loader_kwargs,
            )
            val_loader = DataLoader(
                Subset(depth_dataset, val_idx),
                batch_size=batch_size,
                shuffle=False,
                **val_loader_kwargs,
            )
            model = build_model("du_unet", out_channels=1).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=DU_UNET_WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=scheduler_step_size, gamma=0.5
            )
            ckpt_name = DU_UNET_CKPT_NAME_TEMPLATE.format(
                target_var=args.target_var, depth_m=depth_m
            )
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)

            (
                best_val,
                train_losses,
                val_losses,
                epoch_times_seconds,
                total_training_seconds,
            ) = train_one_model(
                "du_unet",
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                device,
                epochs,
                patience,
                ckpt_path,
                depth_idx=None,
            )

            hist_name = DU_UNET_HISTORY_NAME_TEMPLATE.format(
                target_var=args.target_var, depth_m=depth_m
            )
            curve_name = DU_UNET_LOSS_CURVE_TEMPLATE.format(
                target_var=args.target_var, depth_m=depth_m
            )
            np.savez(
                os.path.join(out_dir, hist_name),
                train_losses=train_losses,
                val_losses=val_losses,
                epoch_times_seconds=np.asarray(epoch_times_seconds, dtype=np.float64),
                total_training_seconds=np.asarray(
                    total_training_seconds, dtype=np.float64
                ),
            )
            save_loss_curve(train_losses, val_losses, os.path.join(out_dir, curve_name))
            print(f"深度 {depth_m}m 训练完成，best val={best_val:.6f}")
        return

    if method == "b3_depthformer":
        epochs = args.epochs or B3_DEPTHFORMER_EPOCHS
        batch_size = args.batch_size or B3_DEPTHFORMER_BATCH_SIZE
        lr = args.lr or B3_DEPTHFORMER_LR
        patience = args.patience or B3_DEPTHFORMER_PATIENCE
        depth_indices = parse_depth_indices(args.depth_indices, depth_levels)
        if depth_indices != sorted(depth_indices):
            raise ValueError(
                "B3 --depth-indices 必须按从浅到深递增，例如 0,4,9,17,24,27"
            )
        selected_depth_levels = [depth_levels[idx] for idx in depth_indices]
        num_outputs = len(depth_indices)
        depth_tag = joint_du_unet_depth_tag(
            depth_indices, total_num_depths=dataset.num_depths
        )
        model_kwargs = {
            "depth_values": selected_depth_levels,
            "d_model": args.b3_d_model,
            "nhead": args.b3_nhead,
            "encoder_layers": args.b3_encoder_layers,
            "decoder_layers": args.b3_decoder_layers,
            "dim_feedforward": args.b3_dim_feedforward,
            "dropout": args.b3_dropout,
            "sst_patch_size": args.b3_sst_patch_size,
            "surface_patch_size": args.b3_surface_patch_size,
            "output_patch_size": args.b3_output_patch_size,
            "refinement_channels": args.b3_refinement_channels,
            "base_grid_size": (
                args.b3_base_grid_height,
                args.b3_base_grid_width,
            ),
            "depth_attention": args.b3_depth_attention,
        }
        model = build_model(method, **model_kwargs).to(device)
        architecture_config = model.model_config()
        architecture_config["source_depth_indices"] = list(depth_indices)
        param_stats = count_model_parameters(model)
        b3_dataset = DepthIndicesTargetDataset(dataset, depth_indices)
        train_loader_kwargs = dataloader_kwargs(args, device, seed_offset=0)
        val_loader_kwargs = dataloader_kwargs(args, device, seed_offset=1)
        train_loader = DataLoader(
            Subset(b3_dataset, train_idx),
            batch_size=batch_size,
            shuffle=True,
            **train_loader_kwargs,
        )
        val_loader = DataLoader(
            Subset(b3_dataset, val_idx),
            batch_size=batch_size,
            shuffle=False,
            **val_loader_kwargs,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=B3_DEPTHFORMER_WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
            eta_min=lr * 0.05,
        )
        teacher_forcing_schedule = (
            args.b3_teacher_forcing_start,
            args.b3_teacher_forcing_end,
            args.b3_teacher_forcing_decay_epochs,
        )
        # Validate the complete schedule before creating any artifacts.
        teacher_forcing_ratio_for_epoch(0, *teacher_forcing_schedule)

        ckpt_name = B3_DEPTHFORMER_CKPT_NAME_TEMPLATE.format(
            target_var=args.target_var,
            depth_tag=depth_tag,
        )
        history_name = B3_DEPTHFORMER_HISTORY_NAME_TEMPLATE.format(
            target_var=args.target_var,
            depth_tag=depth_tag,
        )
        curve_name = B3_DEPTHFORMER_LOSS_CURVE_TEMPLATE.format(
            target_var=args.target_var,
            depth_tag=depth_tag,
        )
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        history_path = os.path.join(out_dir, history_name)
        curve_path = os.path.join(out_dir, curve_name)
        training_strategy = {
            "experiment": "B3_BOS_depth_autoregressive",
            "epoch": {"planned": epochs, "indexing": "1_based"},
            "early_stopping": {
                "enabled": True,
                "monitor": "autoregressive_val_loss",
                "mode": "min",
                "patience": patience,
            },
            "optimizer": {
                "name": "AdamW",
                "lr": lr,
                "weight_decay": B3_DEPTHFORMER_WEIGHT_DECAY,
            },
            "scheduler": {
                "name": "CosineAnnealingLR",
                "t_max_epochs": epochs,
                "eta_min": lr * 0.05,
                "step_on": "epoch",
            },
            "teacher_forcing": {
                "start_ratio": args.b3_teacher_forcing_start,
                "end_ratio": args.b3_teacher_forcing_end,
                "linear_decay_epochs": args.b3_teacher_forcing_decay_epochs,
                "selection": "per_sample_Bernoulli",
                "validation_ratio": 0.0,
            },
        }
        config_path = model_config_path(
            args.output_dir,
            "B3_DepthFormer",
            "train",
            target_var=args.target_var,
            run_tag=depth_tag,
        )
        checkpoint_metadata = {
            "checkpoint_type": "B3_DepthFormer",
            "target_var": args.target_var,
            "dataset_profile": args.dataset_profile,
            "dataset_num_depths": dataset.num_depths,
            "normalize": not args.no_normalize,
            "depth_indices": depth_indices,
            "depth_m": selected_depth_levels,
            "num_outputs": num_outputs,
            "depth_tag": depth_tag,
            "normalization_stats": dataset.get_normalization_reference(),
            "model_config": architecture_config,
            "reproducibility": reproducibility,
        }
        print_and_save_run_config(
            "B3 B3_DepthFormer 训练超参数",
            {
                "stage": "train",
                "experiment": "B3_BOS_depth_autoregressive",
                "model": "B3_DepthFormer",
                "method": method,
                "paradigm": paradigm,
                "target_var": args.target_var,
                "device": device,
                "seed": args.seed,
                "reproducibility": reproducibility,
                "data": {
                    "data_dir": args.data_dir,
                    "dataset_profile": args.dataset_profile,
                    "start_date": dataset.start_date,
                    "end_date": dataset.end_date,
                    "train_samples": len(train_idx),
                    "val_samples": len(val_idx),
                    "shapes": dataset.get_data_info(),
                    "normalize": not args.no_normalize,
                    "lazy_data": use_lazy_2dto2d_data(args),
                    "normalization_chunk_size": args.normalization_chunk_size,
                    "normalization_stats_path": dataset.normalization_stats_path,
                    "normalization_stats": dataset.get_normalization_reference(),
                    "normalization_stats_loaded": dataset.normalization_stats_loaded,
                    "rebuild_normalization_stats": args.rebuild_normalization_stats,
                    "num_workers": args.num_workers,
                    "prefetch_factor": (
                        args.prefetch_factor if args.num_workers > 0 else None
                    ),
                    "pin_memory": train_loader_kwargs["pin_memory"],
                    "persistent_workers": (
                        args.persistent_workers if args.num_workers > 0 else False
                    ),
                    "train_dataloader_seed": args.seed,
                    "val_dataloader_seed": args.seed + 1,
                    "dataset_num_depths": dataset.num_depths,
                    "target_loading": "selected_depths_shallow_to_deep",
                    "max_samples": args.max_samples,
                },
                "hyperparameters": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "learning_rate": lr,
                    "weight_decay": B3_DEPTHFORMER_WEIGHT_DECAY,
                    "patience": patience,
                    "depth_indices": depth_indices,
                    "depth_m": selected_depth_levels,
                    "teacher_forcing_start": args.b3_teacher_forcing_start,
                    "teacher_forcing_end": args.b3_teacher_forcing_end,
                    "teacher_forcing_decay_epochs": (
                        args.b3_teacher_forcing_decay_epochs
                    ),
                    "depth_attention": args.b3_depth_attention,
                },
                "training_strategy": training_strategy,
                "model_config": architecture_config,
                "loss_config": {
                    "name": "masked_depth_mean_rmse_loss",
                    "description": "mean of per-depth masked RMSE",
                    "depth_weighting": "equal",
                    "vertical_regularization": False,
                },
                "parameter_count": param_stats,
                "paths": {
                    "output_dir": out_dir,
                    "checkpoint_dir": ckpt_dir,
                    "checkpoint": ckpt_path,
                    "history": history_path,
                    "loss_curve": curve_path,
                },
            },
            config_path,
        )

        (
            best_val,
            train_losses,
            val_losses,
            epoch_times_seconds,
            total_training_seconds,
        ) = train_one_model(
            method,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            device,
            epochs,
            patience,
            ckpt_path,
            equal_depth_weighting=True,
            checkpoint_metadata=checkpoint_metadata,
            teacher_forcing_schedule=teacher_forcing_schedule,
        )
        teacher_forcing_ratios = [
            teacher_forcing_ratio_for_epoch(epoch, *teacher_forcing_schedule)
            for epoch in range(len(train_losses))
        ]
        np.savez(
            history_path,
            train_losses=np.asarray(train_losses, dtype=np.float32),
            val_losses=np.asarray(val_losses, dtype=np.float32),
            teacher_forcing_ratios=np.asarray(
                teacher_forcing_ratios, dtype=np.float32
            ),
            epoch_times_seconds=np.asarray(epoch_times_seconds, dtype=np.float64),
            total_training_seconds=np.asarray(
                total_training_seconds, dtype=np.float64
            ),
            seed=np.asarray(args.seed, dtype=np.int64),
            depth_indices=np.asarray(depth_indices, dtype=np.int64),
            depth_m=np.asarray(selected_depth_levels, dtype=np.float32),
        )
        save_loss_curve(train_losses, val_losses, curve_path)
        print(
            "\nB3 深度自回归训练完成。"
            f"最佳自回归验证损失: {best_val:.6f}"
        )
        print(f"最佳权重: {ckpt_path}")
        return

    if method in {"joint_du_unet", "joint_duvit"}:
        is_duvit = method == "joint_duvit"
        model_name = "Joint_DUViT" if is_duvit else "Joint_Du_Unet"
        experiment_name = (
            "B2_spatial_vit_bottlenecks"
            if is_duvit
            else "B1_joint_output_baseline"
        )
        ckpt_template = (
            JOINT_DUVIT_CKPT_NAME_TEMPLATE
            if is_duvit
            else JOINT_DU_UNET_CKPT_NAME_TEMPLATE
        )
        history_template = (
            JOINT_DUVIT_HISTORY_NAME_TEMPLATE
            if is_duvit
            else JOINT_DU_UNET_HISTORY_NAME_TEMPLATE
        )
        curve_template = (
            JOINT_DUVIT_LOSS_CURVE_TEMPLATE
            if is_duvit
            else JOINT_DU_UNET_LOSS_CURVE_TEMPLATE
        )
        epochs = args.epochs or DU_UNET_EPOCHS
        batch_size = args.batch_size or DU_UNET_BATCH_SIZE
        lr = args.lr or DU_UNET_LR
        patience = args.patience or DU_UNET_PATIENCE
        depth_indices = parse_depth_indices(args.depth_indices, depth_levels)
        selected_depth_levels = [depth_levels[idx] for idx in depth_indices]
        num_outputs = len(depth_indices)
        depth_tag = joint_du_unet_depth_tag(
            depth_indices, total_num_depths=dataset.num_depths
        )
        joint_dataset = DepthIndicesTargetDataset(dataset, depth_indices)
        model = build_model(method, num_depths=num_outputs).to(device)
        architecture_config = (
            model.model_config()
            if is_duvit
            else {
                "backbone": "Du_Unet",
                "out_channels": num_outputs,
                "joint_depth_output": True,
                "attention": False,
                "transformer": False,
            }
        )
        architecture_config["source_depth_indices"] = list(depth_indices)
        param_stats = count_model_parameters(model)
        scheduler_step_size = max(epochs // 3, 1)
        train_loader_kwargs = dataloader_kwargs(args, device, seed_offset=0)
        val_loader_kwargs = dataloader_kwargs(args, device, seed_offset=1)

        train_loader = DataLoader(
            Subset(joint_dataset, train_idx),
            batch_size=batch_size,
            shuffle=True,
            **train_loader_kwargs,
        )
        val_loader = DataLoader(
            Subset(joint_dataset, val_idx),
            batch_size=batch_size,
            shuffle=False,
            **val_loader_kwargs,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=DU_UNET_WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_step_size, gamma=0.5
        )

        ckpt_name = ckpt_template.format(
            target_var=args.target_var,
            depth_tag=depth_tag,
        )
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        history_name = history_template.format(
            target_var=args.target_var,
            depth_tag=depth_tag,
        )
        curve_name = curve_template.format(
            target_var=args.target_var,
            depth_tag=depth_tag,
        )
        training_strategy = {
            "experiment": experiment_name,
            "epoch": {
                "planned": epochs,
                "indexing": "1_based",
            },
            "early_stopping": {
                "enabled": True,
                "monitor": "val_loss",
                "mode": "min",
                "patience": patience,
            },
            "optimizer": {
                "name": "Adam",
                "lr": lr,
                "weight_decay": DU_UNET_WEIGHT_DECAY,
            },
            "scheduler": {
                "name": "StepLR",
                "step_size": scheduler_step_size,
                "gamma": 0.5,
                "step_on": "epoch",
            },
            "warmup": {
                "enabled": False,
                "epochs": 0,
            },
        }
        config_path = model_config_path(
            args.output_dir,
            model_name,
            "train",
            target_var=args.target_var,
            run_tag=depth_tag,
        )
        print_and_save_run_config(
            f"{'B2' if is_duvit else 'B1'} {model_name} 训练超参数",
            {
                "stage": "train",
                "experiment": experiment_name,
                "model": model_name,
                "method": method,
                "paradigm": paradigm,
                "target_var": args.target_var,
                "device": device,
                "seed": args.seed,
                "reproducibility": reproducibility,
                "data": {
                    "data_dir": args.data_dir,
                    "dataset_profile": args.dataset_profile,
                    "start_date": dataset.start_date,
                    "end_date": dataset.end_date,
                    "train_samples": len(train_idx),
                    "val_samples": len(val_idx),
                    "shapes": dataset.get_data_info(),
                    "normalize": not args.no_normalize,
                    "lazy_data": use_lazy_2dto2d_data(args),
                    "normalization_chunk_size": args.normalization_chunk_size,
                    "normalization_stats_path": dataset.normalization_stats_path,
                    "normalization_stats": dataset.get_normalization_reference(),
                    "normalization_stats_loaded": dataset.normalization_stats_loaded,
                    "rebuild_normalization_stats": args.rebuild_normalization_stats,
                    "num_workers": args.num_workers,
                    "prefetch_factor": (
                        args.prefetch_factor if args.num_workers > 0 else None
                    ),
                    "pin_memory": train_loader_kwargs["pin_memory"],
                    "persistent_workers": (
                        args.persistent_workers if args.num_workers > 0 else False
                    ),
                    "train_dataloader_seed": args.seed,
                    "val_dataloader_seed": args.seed + 1,
                    "dataset_num_depths": dataset.num_depths,
                    "target_loading": "selected_depths_jointly",
                    "max_samples": args.max_samples,
                },
                "hyperparameters": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "learning_rate": lr,
                    "weight_decay": DU_UNET_WEIGHT_DECAY,
                    "patience": patience,
                    "depth_indices": depth_indices,
                    "depth_m": selected_depth_levels,
                },
                "training_strategy": training_strategy,
                "model_config": architecture_config,
                "loss_config": {
                    "name": "masked_depth_mean_rmse_loss",
                    "description": "mean of per-depth masked RMSE",
                    "vertical_regularization": False,
                },
                "parameter_count": param_stats,
                "paths": {
                    "output_dir": out_dir,
                    "checkpoint_dir": ckpt_dir,
                    "checkpoint": ckpt_path,
                    "history": os.path.join(out_dir, history_name),
                    "loss_curve": os.path.join(out_dir, curve_name),
                },
            },
            config_path,
        )

        (
            best_val,
            train_losses,
            val_losses,
            epoch_times_seconds,
            total_training_seconds,
        ) = train_one_model(
            method,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            device,
            epochs,
            patience,
            ckpt_path,
            depth_idx=None,
            equal_depth_weighting=True,
            checkpoint_metadata={
                "checkpoint_type": model_name,
                "target_var": args.target_var,
                "dataset_profile": args.dataset_profile,
                "dataset_num_depths": dataset.num_depths,
                "normalize": not args.no_normalize,
                "depth_indices": depth_indices,
                "depth_m": selected_depth_levels,
                "num_outputs": num_outputs,
                "depth_tag": depth_tag,
                "normalization_stats": dataset.get_normalization_reference(),
                "model_config": architecture_config,
                "reproducibility": reproducibility,
            },
        )
        np.savez(
            os.path.join(out_dir, history_name),
            train_losses=np.asarray(train_losses, dtype=np.float32),
            val_losses=np.asarray(val_losses, dtype=np.float32),
            epoch_times_seconds=np.asarray(epoch_times_seconds, dtype=np.float64),
            total_training_seconds=np.asarray(
                total_training_seconds, dtype=np.float64
            ),
            seed=np.asarray(args.seed, dtype=np.int64),
            depth_indices=np.asarray(depth_indices, dtype=np.int64),
            depth_m=np.asarray(selected_depth_levels, dtype=np.float32),
        )
        save_loss_curve(
            train_losses,
            val_losses,
            os.path.join(out_dir, curve_name),
        )
        print(
            f"\n{'B2 空间ViT' if is_duvit else 'B1 联合深度'}训练完成。"
            f"最佳验证损失: {best_val:.6f}"
        )
        print(f"最佳权重: {ckpt_path}")
        return

    raise ValueError(f"未识别的方法: {method}")


if __name__ == "__main__":
    main()
