"""2dto2d 数据读取、校验与归一化工具。"""

import os

import numpy as np

from config import (
    TWODTO2D_SSH_FILENAME,
    TWODTO2D_SSS_FILENAME,
    TWODTO2D_SST_FILENAME,
    TWODTO2D_TARGET_FILENAMES,
)
from datasets.climatology_normalizer import MonthlyClimatologyLayerStdNormalizer
from utils.data_quality import report_missing_values, sanitize_with_value


def _load_npy(data_dir, filename, label):
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 {label} 文件：{path}")
    return np.load(path, mmap_mode="r")


def load_2dto2d_raw(data_dir, target_var):
    """读取 Du_Unet 需要的 SST、SSH、SSS 与单变量水下目标。"""
    if target_var not in TWODTO2D_TARGET_FILENAMES:
        raise ValueError(f"target_var 必须是 temperature/salinity，实际：{target_var}")

    sst = _load_npy(data_dir, TWODTO2D_SST_FILENAME, "SST")
    ssh = _load_npy(data_dir, TWODTO2D_SSH_FILENAME, "SSH")
    sss = _load_npy(data_dir, TWODTO2D_SSS_FILENAME, "SSS")
    target_name = TWODTO2D_TARGET_FILENAMES[target_var]
    target_path = os.path.join(data_dir, target_name)
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"未找到 {target_var} 目标文件：{target_path}")
    target = np.load(target_path).astype(np.float32)

    if sss.ndim == 4 and sss.shape[1] == 1:
        sss = sss[:, 0]
    return sst, ssh, sss, target


def validate_2dto2d_shapes(sst, ssh, sss, target):
    """校验双分支输入与目标 shape。"""
    if sst.ndim != 3:
        raise ValueError(f"SST 需要 (T,160,160)，实际：{sst.shape}")
    if ssh.ndim != 3 or sss.ndim != 3:
        raise ValueError(f"SSH/SSS 需要 (T,64,64)，实际 ssh={ssh.shape}, sss={sss.shape}")
    if sss.shape != ssh.shape:
        raise ValueError(f"SSH/SSS shape 不一致：{ssh.shape} vs {sss.shape}")
    if target.ndim != 4:
        raise ValueError(f"target 需要 (T,D,64,64)，实际：{target.shape}")

    t, h_low, w_low = ssh.shape
    if sst.shape[0] != t or target.shape[0] != t:
        raise ValueError(f"时间维不一致：sst={sst.shape}, ssh={ssh.shape}, target={target.shape}")
    if target.shape[2:] != (h_low, w_low):
        raise ValueError(f"target 空间维需等于 SSH/SSS，实际：{target.shape[2:]} vs {(h_low, w_low)}")
    num_depths = int(target.shape[1])
    return num_depths


def clean_and_normalize_2dto2d(
    sst,
    ssh,
    sss,
    target,
    normalize=False,
    months=None,
    fit_indices=None,
    start_date=None,
):
    """缺失值处理与可选月气候态距平归一化，并返回统计量。"""
    report_missing_values("2dto2d.sst", sst, start_date=start_date)
    report_missing_values("2dto2d.ssh", ssh, start_date=start_date)
    report_missing_values("2dto2d.sss", sss, start_date=start_date)
    report_missing_values("2dto2d.target", target, start_date=start_date, depth_dim=1)

    if normalize:
        if months is None:
            raise ValueError("normalize=True 时必须提供 months")
        sst_norm = MonthlyClimatologyLayerStdNormalizer().fit(
            sst, months, fit_indices=fit_indices
        )
        ssh_norm = MonthlyClimatologyLayerStdNormalizer().fit(
            ssh, months, fit_indices=fit_indices
        )
        sss_norm = MonthlyClimatologyLayerStdNormalizer().fit(
            sss, months, fit_indices=fit_indices
        )
        target_norm = MonthlyClimatologyLayerStdNormalizer().fit(
            target, months, fit_indices=fit_indices
        )
        stats = {
            "normalization": "monthly_climatology_layer_std",
            "sst": sst_norm.to_stats(),
            "ssh": ssh_norm.to_stats(),
            "sss": sss_norm.to_stats(),
            "target": target_norm.to_stats(),
        }
        return (
            sst_norm.transform(sst, months),
            ssh_norm.transform(ssh, months),
            sss_norm.transform(sss, months),
            target_norm.transform(target, months),
            stats,
        )

    stats = {"normalization": "none"}
    sst = sanitize_with_value(sst, fill_value=0.0)
    ssh = sanitize_with_value(ssh, fill_value=0.0)
    sss = sanitize_with_value(sss, fill_value=0.0)
    target = sanitize_with_value(target, fill_value=0.0)
    return sst, ssh, sss, target, stats
