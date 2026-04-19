"""2DVar/MODAS 前处理（从 maths_preprocessing 迁移）。"""

import numpy as np

from datasets.date_utils import generate_date_list
from utils.data_quality import report_missing_values, sanitize_with_value


def load_sla_sss(path):
    """加载海表数据并校验维度 (T, 2, H, W)。"""
    data = np.load(path).astype(np.float32)
    if data.ndim != 4 or data.shape[1] != 2:
        raise ValueError(f"sla_sss 需要 (T, 2, H, W)，实际：{data.shape}")
    report_missing_values("sla_sss", data)
    return sanitize_with_value(data, fill_value=0.0)


def get_dataset_split(
    start_date="2019-01-01",
    end_date="2023-12-31",
    train_end="2021-12-31",
    val_start="2022-01-01",
    val_end="2022-12-31",
    test_start="2023-01-01",
    test_end="2023-12-31",
):
    """返回固定时段的 train/val/test 切片。"""
    full = generate_date_list(start_date, end_date)
    train_dates = generate_date_list(start_date, train_end)
    val_dates = generate_date_list(val_start, val_end)
    test_dates = generate_date_list(test_start, test_end)

    t_train_end = full.index(train_dates[-1]) + 1
    t_val_start = full.index(val_dates[0])
    t_val_end = full.index(val_dates[-1]) + 1
    t_test_start = full.index(test_dates[0])
    t_test_end = full.index(test_dates[-1]) + 1

    return {
        "full_dates": full,
        "train_slice": slice(0, t_train_end),
        "val_slice": slice(t_val_start, t_val_end),
        "test_slice": slice(t_test_start, t_test_end),
        "train_dates": train_dates,
        "val_dates": val_dates,
        "test_dates": test_dates,
    }


def load_and_validate(sla_sss_path, sws_true_path):
    """加载并校验海表与水下真值的一致性。"""
    sla_sss = np.load(sla_sss_path).astype(np.float32)
    sws_true = np.load(sws_true_path).astype(np.float32)

    if sla_sss.ndim != 4 or sla_sss.shape[1] != 2:
        raise ValueError(f"sla_sss 需要 (T, 2, H, W)，实际：{sla_sss.shape}")
    if sws_true.ndim != 4:
        raise ValueError(f"sws_true 需要 (T, D, H, W)，实际：{sws_true.shape}")
    if sla_sss.shape[0] != sws_true.shape[0]:
        raise ValueError("海表与真值时间步数不一致")
    if sla_sss.shape[2:] != sws_true.shape[2:]:
        raise ValueError("海表与真值空间分辨率不一致")

    report_missing_values("sla_sss", sla_sss)
    report_missing_values("sws_true", sws_true)
    sla_sss = sanitize_with_value(sla_sss, fill_value=0.0)
    sws_true = sanitize_with_value(sws_true, fill_value=0.0)

    t, c_surface, h, w = sla_sss.shape
    _, c_depth, _, _ = sws_true.shape
    print(
        f"MODAS 数据加载完成：T={t}, 海表通道={c_surface}, 深度层={c_depth}, H={h}, W={w}"
    )
    return sla_sss, sws_true
