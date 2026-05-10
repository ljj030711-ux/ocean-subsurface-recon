"""统一训练入口（2dto2d / 2dto3d）。"""

import argparse
import os
from datetime import datetime

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    CHECKPOINTS_ROOT,
    DATA_DIR,
    DATA_START_DATE,
    DEPTH_LEVELS_25M,
    DX,
    DY,
    DU_UNET_BATCH_SIZE,
    DU_UNET_CKPT_NAME_TEMPLATE,
    DU_UNET_EPOCHS,
    DU_UNET_HISTORY_NAME_TEMPLATE,
    DU_UNET_LAMBDA_SMOOTH,
    DU_UNET_LOSS_CURVE_TEMPLATE,
    DU_UNET_LR,
    DU_UNET_PATIENCE,
    DU_UNET_WEIGHT_DECAY,
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
    get_checkpoint_dir,
    get_output_dir,
)
from datasets.dataset_2dto2d import Dataset2Dto2D
from datasets.dataset_2dto3d import DummyTwoDto3DDataset
from models.du_unet import Du_Unet
from models.ocean_transformer import OceanTransformer
from utils.physics import compute_eke, compute_grad_ssh
from utils.physics_loss import PhysicsLoss

TRAINABLE_METHODS = {"du_unet", "ocean_transformer"}


def build_model(method, **kwargs):
    """根据 method 构建模型。"""
    name = method.strip().lower()
    out_ch = kwargs.get("out_channels", 1)
    if name in {"du_unet", "du-unet"}:
        return Du_Unet(out_channels=out_ch)
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


def smooth_loss(x):
    loss_x = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    loss_y = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    return loss_x + loss_y


def select_target_depth(target, depth_idx=None):
    if depth_idx is None:
        return target
    if target.ndim != 4:
        raise ValueError(f"target 需为 (B,C,H,W)，实际：{tuple(target.shape)}")
    return target[:, depth_idx : depth_idx + 1]


def train_2dto2d_epoch(model, loader, optimizer, device, lambda_smooth, depth_idx=None):
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        sst = batch["sst"].to(device)
        ssh_sss = batch["ssh_sss"].to(device)
        target = select_target_depth(batch["target"].to(device), depth_idx=depth_idx)
        pred = model(sst, ssh_sss)
        loss = F.mse_loss(pred, target) + lambda_smooth * smooth_loss(pred)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
        count += 1
    return total / max(count, 1)


def validate_2dto2d(model, loader, device, depth_idx=None):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            sst = batch["sst"].to(device)
            ssh_sss = batch["ssh_sss"].to(device)
            target = select_target_depth(batch["target"].to(device), depth_idx=depth_idx)
            loss = F.mse_loss(model(sst, ssh_sss), target)
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
):
    best_val = float("inf")
    patience_cnt = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        if method == "ocean_transformer":
            t_loss = train_2dto3d_epoch(model, train_loader, optimizer, device, criterion)
            v_loss = validate_2dto3d(model, val_loader, device, criterion)
        else:
            t_loss = train_2dto2d_epoch(
                model, train_loader, optimizer, device,
                DU_UNET_LAMBDA_SMOOTH, depth_idx=depth_idx
            )
            v_loss = validate_2dto2d(model, val_loader, device, depth_idx=depth_idx)

        scheduler.step()
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch + 1}/{epochs}] | Train: {t_loss:.6f} | Val: {v_loss:.6f} | LR: {lr_now:.2e}")

        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), checkpoint_path)
            patience_cnt = 0
            print(f"  -> 保存最佳模型 (val={v_loss:.6f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  -> 早停触发 (patience={patience})")
                break

    return best_val, train_losses, val_losses


def parse_depth_indices(depth_indices_arg):
    # 如果没穿，训练该变量的全部 25 个深度层
    if not depth_indices_arg:
        return list(range(len(DEPTH_LEVELS_25M)))
    out = []
    for item in depth_indices_arg.split(","):
        idx = int(item.strip())
        if idx < 0 or idx >= len(DEPTH_LEVELS_25M):
            raise ValueError(f"depth index 越界: {idx}")
        out.append(idx)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="统一训练入口（2dto2d / 2dto3d）")
    parser.add_argument("--method", required=True)
    parser.add_argument("--target-var", choices=["temperature", "salinity"], default="temperature",
                        help="Du_Unet 训练目标变量")
    parser.add_argument("--start-date", default=DATA_START_DATE,
                        help="Du_Unet 使用的数据起始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", default=None,
                        help="Du_Unet 使用的数据结束日期 YYYY-MM-DD，默认使用配置中的结束日期")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-dir", default=OUTPUTS_ROOT)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINTS_ROOT)
    parser.add_argument("--dummy", action="store_true", help="2dto3d(ocean_transformer) 使用合成数据")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--depth-indices",
        default=None,
        help="2dto2d(Du_Unet) 调试开关，仅训练指定深度索引，逗号分隔",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    method = args.method.strip().lower()
    if method in {"du-unet"}:
        method = "du_unet"
    if method not in TRAINABLE_METHODS:
        raise ValueError(f"未识别的方法: {args.method}")
    paradigm = paradigm_of_method(method)
    method_dir = "Du_Unet" if method == "du_unet" else method

    device = get_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

        model = build_model(method).to(device)
        criterion = PhysicsLoss(
            depth_levels=TWODTO3D_DEPTH_LEVELS,
            lambda_hydro=TWODTO3D_LAMBDA_HYDRO,
            lambda_strat=TWODTO3D_LAMBDA_STRAT,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=TWODTO3D_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.5)

        best_ckpt = os.path.join(ckpt_dir, f"{method}_best.pth")
        best_val, train_losses, val_losses = train_one_model(
            method, model, train_loader, val_loader, optimizer, scheduler, device,
            epochs, patience, best_ckpt, criterion=criterion
        )
        np.savez(os.path.join(out_dir, f"training_history_{method}.npz"), train_losses=train_losses, val_losses=val_losses)
        save_loss_curve(train_losses, val_losses, os.path.join(out_dir, f"loss_{method}.png"))
        print(f"\n训练完成。最佳验证损失: {best_val:.6f}")
        print(f"最佳权重: {best_ckpt}")
        return

    dataset = Dataset2Dto2D(
        args.data_dir,
        normalize=True,
        target_var=args.target_var,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    train_idx, val_idx, _ = get_year_split_indices(len(dataset), dataset.start_date)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(
            f"当前日期范围 [{dataset.start_date}, {dataset.end_date}] 无法形成训练/验证集；"
            "建议本地调试使用 --start-date 2021-01-01 --end-date 2023-12-31"
        )

    if method == "du_unet":
        depth_indices = parse_depth_indices(args.depth_indices)
        epochs = args.epochs or DU_UNET_EPOCHS
        batch_size = args.batch_size or DU_UNET_BATCH_SIZE
        lr = args.lr or DU_UNET_LR
        patience = args.patience or DU_UNET_PATIENCE

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)

        for depth_idx in depth_indices:
            depth_m = DEPTH_LEVELS_25M[depth_idx]
            print("\n" + "-" * 60)
            print(f"训练变量: {args.target_var} | 深度层: idx={depth_idx}, depth={depth_m}m")
            print("-" * 60)

            model = build_model("du_unet", out_channels=1).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=DU_UNET_WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(epochs // 3, 1), gamma=0.5
            )
            ckpt_name = DU_UNET_CKPT_NAME_TEMPLATE.format(
                target_var=args.target_var, depth_m=depth_m
            )
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)

            best_val, train_losses, val_losses = train_one_model(
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
                depth_idx=depth_idx,
            )

            hist_name = DU_UNET_HISTORY_NAME_TEMPLATE.format(
                target_var=args.target_var, depth_m=depth_m
            )
            curve_name = DU_UNET_LOSS_CURVE_TEMPLATE.format(
                target_var=args.target_var, depth_m=depth_m
            )
            np.savez(os.path.join(out_dir, hist_name), train_losses=train_losses, val_losses=val_losses)
            save_loss_curve(train_losses, val_losses, os.path.join(out_dir, curve_name))
            print(f"深度 {depth_m}m 训练完成，best val={best_val:.6f}")
        return

    raise ValueError(f"未识别的方法: {method}")


if __name__ == "__main__":
    main()
