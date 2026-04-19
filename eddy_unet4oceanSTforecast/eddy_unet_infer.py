"""EddyUNet 26 层推理函数（独立模块，零侵入）。"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
import torch

from config import (
    DATA_END_DATE,
    DATA_START_DATE,
    DEPTH_LEVELS_26M,
    EDDY_UNET_CKPT_NAME_TEMPLATE,
)
from datasets.date_utils import date_to_index
from datasets.eddy_dataset import EddyDataset
from train import build_2dto2d_features, build_model
from utils.metrics import mae, mse, r2, rmse


METRIC_UNITS = {
    "mse": "psu^2",
    "rmse": "psu",
    "mae": "psu",
    "r2": "dimensionless",
}


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _scalar_metrics(y_true_flat: np.ndarray, y_pred_flat: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    yt, yp = y_true_flat[mask], y_pred_flat[mask]
    return {
        "mse": float(mse(yt, yp)),
        "rmse": float(rmse(yt, yp)),
        "mae": float(mae(yt, yp)),
        "r2": float(r2(yt, yp)),
    }


def _validate_level(target_level: int, num_depths: int) -> None:
    if target_level < 0 or target_level >= num_depths:
        raise ValueError(
            f"target_level 越界：{target_level}，应在 [0, {num_depths - 1}]。"
        )


def infer_eddy_unet_26layers(
    select_day: str,
    data_dir: str,
    checkpoint_dir: str,
    use_physics_features: bool = False,
    target_level: int = 10,
    checkpoint_policy: str = "strict",
) -> dict[str, Any]:
    """
    函数化推理入口：加载 26 个深度模型并拼装预测结果。

    Args:
        select_day: 预测日期，格式 YYYY-MM-DD。
        data_dir: 数据目录。
        checkpoint_dir: 26 层 checkpoint 所在目录。
        use_physics_features: 是否启用物理特征（EKE/grad）。
        target_level: 用于结果摘要的目标层索引，仅做校验与回传。
        checkpoint_policy:
            - strict: 缺失任一层 checkpoint 直接报错（默认）
            - warn: 仅告警，缺失层使用随机初始化权重

    Returns:
        {
            "pred": np.ndarray (1, 26, H, W),
            "target": np.ndarray (1, 26, H, W),
            "summary": {"mse","rmse","mae","r2"},
            "meta": {
                "metric_space": "physical",
                "metric_units": {...},
                "pred_shape": [1,26,H,W],
                "target_shape": [1,26,H,W],
                "device": "...",
                "select_day": "...",
                "target_level": int,
                "checkpoint_policy": "...",
                "missing_checkpoints": [...]
            }
        }
    """
    if checkpoint_policy not in {"strict", "warn"}:
        raise ValueError(
            f"checkpoint_policy 仅支持 strict/warn，收到：{checkpoint_policy}"
        )

    device = _get_device()
    dataset = EddyDataset(data_dir, normalize=True)

    t = date_to_index(select_day, DATA_START_DATE, DATA_END_DATE)
    if t < 0 or t >= len(dataset):
        raise IndexError(
            f"日期 {select_day} 映射索引 t={t} 超出数据集长度 {len(dataset)}。"
        )

    sample = dataset[t]
    y_true = sample["target"].unsqueeze(0).numpy()  # (1, 26, H, W)
    if y_true.ndim != 4:
        raise ValueError(f"目标 shape 非法，期望 (1,C,H,W)，实际：{y_true.shape}")
    if y_true.shape[1] != len(DEPTH_LEVELS_26M):
        raise ValueError(
            f"目标深度层数不匹配，期望 {len(DEPTH_LEVELS_26M)}，实际：{y_true.shape[1]}"
        )

    _validate_level(target_level, y_true.shape[1])

    sss = sample["sss"].unsqueeze(0).to(device)
    ssh = sample["ssh"].unsqueeze(0).to(device)
    x = build_2dto2d_features(sss, ssh, use_physics_features)

    in_ch = 4 if use_physics_features else 2
    pred_list: list[np.ndarray] = []
    missing_checkpoints: list[str] = []

    for depth_m in DEPTH_LEVELS_26M:
        ckpt_name = EDDY_UNET_CKPT_NAME_TEMPLATE.format(depth_m=depth_m)
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        model = build_model("eddy_unet", in_channels=in_ch, out_channels=1).to(device)

        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
        else:
            msg = f"缺失深度 {depth_m}m checkpoint: {ckpt_path}"
            missing_checkpoints.append(msg)
            if checkpoint_policy == "strict":
                raise FileNotFoundError(msg)
            warnings.warn(msg + "；该层将使用随机初始化权重。", RuntimeWarning)

        model.eval()
        with torch.no_grad():
            pred_layer = model(x).cpu().numpy()

        if pred_layer.shape != (1, 1, y_true.shape[2], y_true.shape[3]):
            raise ValueError(
                "单层预测 shape 不匹配，"
                f"期望 {(1, 1, y_true.shape[2], y_true.shape[3])}，实际 {pred_layer.shape}。"
            )
        pred_list.append(pred_layer)

    y_pred = np.concatenate(pred_list, axis=1)  # (1,26,H,W)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"预测与真值 shape 不一致：pred={y_pred.shape}, target={y_true.shape}")

    norm_stats = dataset.get_norm_stats()
    target_mean = float(norm_stats["target_mean"])
    target_std = float(norm_stats["target_std"])
    y_pred = y_pred * target_std + target_mean
    y_true = y_true * target_std + target_mean

    summary = _scalar_metrics(y_true.flatten(), y_pred.flatten())
    meta = {
        "metric_space": "physical",
        "metric_units": dict(METRIC_UNITS),
        "pred_shape": list(y_pred.shape),
        "target_shape": list(y_true.shape),
        "device": str(device),
        "select_day": select_day,
        "target_level": int(target_level),
        "checkpoint_policy": checkpoint_policy,
        "missing_checkpoints": missing_checkpoints,
    }
    return {
        "pred": y_pred,
        "target": y_true,
        "summary": summary,
        "meta": meta,
    }
