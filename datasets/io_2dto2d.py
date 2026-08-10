"""2dto2d 数据读取、校验与归一化工具。"""

import os

import numpy as np

from config import (
    DEFAULT_DATASET_PROFILE,
    get_2dto2d_filenames,
)
from datasets.climatology_normalizer import MonthlyClimatologyLayerStdNormalizer
from utils.data_quality import report_missing_values, sanitize_with_value


SST_KELVIN_THRESHOLD = 100.0
KELVIN_TO_CELSIUS = 273.15


def _load_npy(data_dir, filename, label):
    path = filename if os.path.isabs(filename) else os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 {label} 文件：{path}")
    return np.load(path, mmap_mode="r")


def _finite_sample(array):
    arr = np.asarray(array)
    if arr.ndim == 3:
        t_step = max(arr.shape[0] // 32, 1)
        return np.asarray(arr[::t_step, ::8, ::8])
    return arr


def _ensure_sst_celsius(sst, conversion=None):
    conversion = conversion or detect_sst_unit_conversion(sst)
    if conversion == "kelvin_to_celsius":
        return (
            np.asarray(sst, dtype=np.float32) - np.float32(KELVIN_TO_CELSIUS),
            conversion,
        )
    return np.asarray(sst, dtype=np.float32), conversion


def detect_sst_unit_conversion(sst):
    sample = _finite_sample(sst)
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return "unknown_no_finite_values"

    median = float(np.nanmedian(finite))
    if median > SST_KELVIN_THRESHOLD:
        print(
            f"SST 单位检测：median={median:.3f}，疑似 Kelvin，已转换为 Celsius。"
        )
        return "kelvin_to_celsius"
    return "already_celsius"


def apply_sst_unit_conversion(sst, conversion):
    sst = np.asarray(sst, dtype=np.float32)
    if conversion == "kelvin_to_celsius":
        return sst - np.float32(KELVIN_TO_CELSIUS)
    return sst


def _squeeze_single_channel(array, label):
    if array.ndim == 4 and array.shape[1] == 1:
        return array[:, 0]
    if array.ndim == 3:
        return array
    raise ValueError(f"{label} 需要 (T,H,W) 或 (T,1,H,W)，实际：{array.shape}")


def load_2dto2d_raw(data_dir, target_var, dataset_profile=DEFAULT_DATASET_PROFILE):
    """读取 Du_Unet 需要的 SST、SSH、SSS 与单变量水下目标。"""
    filenames = get_2dto2d_filenames(target_var, dataset_profile)
    sst = _squeeze_single_channel(_load_npy(data_dir, filenames["sst"], "SST"), "SST")
    ssh = _squeeze_single_channel(_load_npy(data_dir, filenames["ssh"], "SSH"), "SSH")
    sss = _squeeze_single_channel(_load_npy(data_dir, filenames["sss"], "SSS"), "SSS")
    target = _load_npy(data_dir, filenames["target"], f"{target_var} 目标")
    if target.dtype != np.float32:
        target = target.astype(np.float32)
    return sst, ssh, sss, target


def validate_2dto2d_shapes(sst, ssh, sss, target):
    """校验双分支输入与目标 shape。"""
    if sst.ndim != 3:
        raise ValueError(f"SST 需要 (T,H,W)，实际：{sst.shape}")
    if ssh.ndim != 3 or sss.ndim != 3:
        raise ValueError(f"SSH/SSS 需要 (T,H,W)，实际 ssh={ssh.shape}, sss={sss.shape}")
    if sss.shape != ssh.shape:
        raise ValueError(f"SSH/SSS shape 不一致：{ssh.shape} vs {sss.shape}")
    if target.ndim != 4:
        raise ValueError(f"target 需要 (T,D,H,W)，实际：{target.shape}")

    t = ssh.shape[0]
    if sst.shape[0] != t or target.shape[0] != t:
        raise ValueError(f"时间维不一致：sst={sst.shape}, ssh={ssh.shape}, target={target.shape}")
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
    quality_report=False,
    normalization_stats=None,
    sst_unit_conversion=None,
):
    """缺失值处理与可选月气候态距平归一化，并返回统计量。"""
    sst, sst_unit_conversion = _ensure_sst_celsius(
        sst, conversion=sst_unit_conversion
    )

    if quality_report:
        report_missing_values("2dto2d.sst", sst, start_date=start_date)
        report_missing_values("2dto2d.ssh", ssh, start_date=start_date)
        report_missing_values("2dto2d.sss", sss, start_date=start_date)
        report_missing_values("2dto2d.target", target, start_date=start_date, depth_dim=1)
    target_mask = np.isfinite(target).astype(np.float32)

    if normalize:
        if months is None:
            raise ValueError("normalize=True 时必须提供 months")
        if normalization_stats is None:
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
                "sst_unit": "degree_C",
                "sst_unit_conversion": sst_unit_conversion,
                "sst": sst_norm.to_stats(),
                "ssh": ssh_norm.to_stats(),
                "sss": sss_norm.to_stats(),
                "target": target_norm.to_stats(),
            }
        else:
            stats = dict(normalization_stats)
            sst_norm = MonthlyClimatologyLayerStdNormalizer.from_stats(stats["sst"])
            ssh_norm = MonthlyClimatologyLayerStdNormalizer.from_stats(stats["ssh"])
            sss_norm = MonthlyClimatologyLayerStdNormalizer.from_stats(stats["sss"])
            target_norm = MonthlyClimatologyLayerStdNormalizer.from_stats(stats["target"])
        return (
            sst_norm.transform(sst, months),
            ssh_norm.transform(ssh, months),
            sss_norm.transform(sss, months),
            target_norm.transform(target, months),
            target_mask,
            stats,
        )

    stats = {
        "normalization": "none",
        "sst_unit": "degree_C",
        "sst_unit_conversion": sst_unit_conversion,
    }
    sst = sanitize_with_value(sst, fill_value=0.0)
    ssh = sanitize_with_value(ssh, fill_value=0.0)
    sss = sanitize_with_value(sss, fill_value=0.0)
    target = sanitize_with_value(target, fill_value=0.0)
    return sst, ssh, sss, target, target_mask, stats
