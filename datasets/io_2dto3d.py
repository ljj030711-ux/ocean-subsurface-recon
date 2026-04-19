"""2dto3d 数据读取/校验/清洗工具。"""

import os

import numpy as np

from config import TWODTO3D_SURFACE_FILENAME, TWODTO3D_TARGET_FILENAME
from utils.data_quality import report_missing_values, sanitize_with_value


def load_2dto3d_raw(data_dir):
    """读取 2dto3d 海表输入与真值。"""
    surface_path = os.path.join(data_dir, TWODTO3D_SURFACE_FILENAME)
    target_path = os.path.join(data_dir, TWODTO3D_TARGET_FILENAME)
    if not os.path.exists(surface_path):
        raise FileNotFoundError(f"未找到海表输入：{surface_path}")
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"未找到 2dto3d 真值：{target_path}")

    surface_raw = np.load(surface_path).astype(np.float32)
    target_data = np.load(target_path).astype(np.float32)
    return surface_raw, target_data


def validate_2dto3d_shapes(surface_raw, target_data):
    """校验 2dto3d 数据 shape。"""
    if surface_raw.ndim != 4 or surface_raw.shape[1] != 2:
        raise ValueError(f"surface_raw 需要 (T,2,H,W)，实际: {surface_raw.shape}")
    if target_data.ndim != 4:
        raise ValueError(f"target_data 需要 (T,D,H,W)，实际: {target_data.shape}")
    if surface_raw.shape[0] != target_data.shape[0]:
        raise ValueError("海表与水下真值时间步数不一致")
    if surface_raw.shape[2:] != target_data.shape[2:]:
        raise ValueError("海表与水下真值空间分辨率不一致")


def clean_2dto3d(surface_raw, target_data, normalize=False):
    """缺失值处理与可选通道标准化。"""
    report_missing_values("2dto3d.surface_raw", surface_raw)
    report_missing_values("2dto3d.target_data", target_data)

    surface_raw = sanitize_with_value(surface_raw, fill_value=0.0)
    target_data = sanitize_with_value(target_data, fill_value=0.0)

    if normalize:
        for c in range(surface_raw.shape[1]):
            ch = surface_raw[:, c]
            surface_raw[:, c] = (ch - ch.mean()) / (ch.std() + 1e-6)
    return surface_raw, target_data
