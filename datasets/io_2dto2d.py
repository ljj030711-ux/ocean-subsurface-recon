"""2dto2d 数据读取/校验/清洗工具。"""

import os

import numpy as np

from config import TWODTO2D_SURFACE_FILENAME, TWODTO2D_TARGET_FILENAME
from datasets.climatology_normalizer import MonthlyClimatologyLayerStdNormalizer
from utils.data_quality import report_missing_values, sanitize_with_value


def load_2dto2d_raw(data_dir, target_filename=None):
    """读取 2dto2d 所需的海表输入与水下目标。"""
    surface_path = os.path.join(data_dir, TWODTO2D_SURFACE_FILENAME)
    if not os.path.exists(surface_path):
        raise FileNotFoundError(f"未找到海表输入：{surface_path}")

    surface = np.load(surface_path).astype(np.float32)
    if surface.ndim != 4 or surface.shape[1] != 2:
        raise ValueError(
            f"{TWODTO2D_SURFACE_FILENAME} 需要 (T,2,H,W)，实际：{surface.shape}"
        )
    ssh = surface[:, 0]
    sss = surface[:, 1]

    target_name = target_filename or TWODTO2D_TARGET_FILENAME
    target_path = os.path.join(data_dir, target_name)
    if not os.path.exists(target_path):
        legacy_path = os.path.join(data_dir, "subsurface.npy")
        if os.path.exists(legacy_path):
            target_path = legacy_path
        else:
            raise FileNotFoundError(
                f"未找到目标文件: {target_path}（也未找到 subsurface.npy）"
            )
    target = np.load(target_path).astype(np.float32)
    return sss, ssh, target


def validate_2dto2d_shapes(sss, ssh, target):
    """校验 2dto2d 输入输出 shape。"""
    if sss.ndim != 3 or ssh.ndim != 3:
        raise ValueError(f"海表输入需为 (T,H,W)，实际 sss={sss.shape}, ssh={ssh.shape}")
    if sss.shape != ssh.shape:
        raise ValueError(f"SSS/SSH shape 不一致：{sss.shape} vs {ssh.shape}")
    if target.ndim not in (3, 4):
        raise ValueError(f"target 需要 (T,H,W) 或 (T,D,H,W)，实际：{target.shape}")

    t, h, w = sss.shape
    if target.ndim == 3:
        if target.shape != (t, h, w):
            raise ValueError(f"单层 target 需与海表同形，实际：{target.shape} vs {(t, h, w)}")
        num_depths = 1
    else:
        if target.shape[0] != t or target.shape[2:] != (h, w):
            raise ValueError(
                f"多层 target 需为 (T,D,H,W) 且与海表时空一致，实际：{target.shape} vs {(t, h, w)}"
            )
        num_depths = int(target.shape[1])
    return num_depths


def clean_and_normalize_2dto2d(
    sss,
    ssh,
    target,
    normalize=False,
    months=None,
    fit_indices=None,
):
    """缺失值处理与可选月气候态距平归一化，并返回统计量。"""
    report_missing_values("2dto2d.sss", sss)
    report_missing_values("2dto2d.ssh", ssh)
    report_missing_values("2dto2d.target", target)

    if normalize:
        if months is None:
            raise ValueError("normalize=True 时必须提供 months")
        sss_norm = MonthlyClimatologyLayerStdNormalizer().fit(
            sss, months, fit_indices=fit_indices
        )
        ssh_norm = MonthlyClimatologyLayerStdNormalizer().fit(
            ssh, months, fit_indices=fit_indices
        )
        target_norm = MonthlyClimatologyLayerStdNormalizer().fit(
            target, months, fit_indices=fit_indices
        )
        stats = {
            "normalization": "monthly_climatology_layer_std",
            "sss": sss_norm.to_stats(),
            "ssh": ssh_norm.to_stats(),
            "target": target_norm.to_stats(),
        }
        return (
            sss_norm.transform(sss, months),
            ssh_norm.transform(ssh, months),
            target_norm.transform(target, months),
            stats,
        )

    stats = {"normalization": "none"}
    sss = sanitize_with_value(sss, fill_value=0.0)
    ssh = sanitize_with_value(ssh, fill_value=0.0)
    target = sanitize_with_value(target, fill_value=0.0)
    return sss, ssh, target, stats
