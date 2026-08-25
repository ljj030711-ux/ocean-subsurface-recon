"""
评估指标模块
用于模型性能评估和结果验证
"""

import numpy as np
from scipy.ndimage import gaussian_filter


class RegressionMetricAccumulator:
    """分批累计 MAE、RMSE、R² 和相关系数所需统计量。"""

    def __init__(self):
        self.count = None
        self.sum_abs_error = None
        self.sum_sq_error = None
        self.sum_true = None
        self.sum_pred = None
        self.sum_true_sq = None
        self.sum_pred_sq = None
        self.sum_cross = None

    def update(self, y_true, y_pred, mask=None, axis=None):
        """累计一个 batch；axis 指定被聚合的维度。"""
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        valid = _valid_mask(y_true, y_pred, mask=mask)
        true_valid = np.where(valid, y_true, 0.0)
        pred_valid = np.where(valid, y_pred, 0.0)
        error = pred_valid - true_valid

        batch_stats = {
            "count": np.sum(valid, axis=axis, dtype=np.int64),
            "sum_abs_error": np.sum(np.abs(error), axis=axis, dtype=np.float64),
            "sum_sq_error": np.sum(error * error, axis=axis, dtype=np.float64),
            "sum_true": np.sum(true_valid, axis=axis, dtype=np.float64),
            "sum_pred": np.sum(pred_valid, axis=axis, dtype=np.float64),
            "sum_true_sq": np.sum(true_valid * true_valid, axis=axis, dtype=np.float64),
            "sum_pred_sq": np.sum(pred_valid * pred_valid, axis=axis, dtype=np.float64),
            "sum_cross": np.sum(true_valid * pred_valid, axis=axis, dtype=np.float64),
        }
        for name, value in batch_stats.items():
            current = getattr(self, name)
            value = np.asarray(value)
            setattr(self, name, value if current is None else current + value)

    def compute(self):
        """返回与 scalar_metrics 相同口径的四项指标及有效样本数。"""
        if self.count is None:
            raise RuntimeError("尚未累计任何数据")

        count = np.asarray(self.count, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            mae_value = self.sum_abs_error / count
            rmse_value = np.sqrt(self.sum_sq_error / count)

            true_centered_sq = self.sum_true_sq - np.square(self.sum_true) / count
            pred_centered_sq = self.sum_pred_sq - np.square(self.sum_pred) / count
            cross_centered = self.sum_cross - self.sum_true * self.sum_pred / count

            r2_value = 1.0 - self.sum_sq_error / true_centered_sq
            correlation_value = cross_centered / np.sqrt(
                true_centered_sq * pred_centered_sq
            )

        invalid = count <= 0
        return {
            "mae": _with_nan(mae_value, invalid),
            "rmse": _with_nan(rmse_value, invalid),
            "r2": _with_nan(r2_value, invalid | (true_centered_sq <= 0)),
            "correlation": _with_nan(
                correlation_value,
                invalid | (true_centered_sq <= 0) | (pred_centered_sq <= 0),
            ),
            "valid_count": np.asarray(self.count, dtype=np.int64),
        }


def _with_nan(value, invalid):
    result = np.where(invalid, np.nan, np.asarray(value, dtype=np.float64))
    if result.ndim == 0:
        return float(result)
    return result


def mse(y_true, y_pred):
    """均方误差"""
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """均方根误差"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """平均绝对误差"""
    return np.mean(np.abs(y_true - y_pred))


def r2(y_true, y_pred):
    """R²决定系数"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)


def correlation(y_true, y_pred):
    """皮尔逊相关系数"""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    if y_true_flat.size < 2 or np.std(y_true_flat) == 0 or np.std(y_pred_flat) == 0:
        return np.nan
    return np.corrcoef(y_true_flat, y_pred_flat)[0, 1]


def _valid_mask(y_true, y_pred, mask=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true/y_pred 形状必须一致: {y_true.shape} vs {y_pred.shape}"
        )
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != y_true.shape:
            raise ValueError(
                f"mask 形状必须与目标一致: {mask.shape} vs {y_true.shape}"
            )
        valid = valid & mask.astype(bool)
    return valid


def scalar_metrics(y_true, y_pred, mask=None):
    """计算标量评估指标，可选只统计 mask=1 的位置。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid = _valid_mask(y_true, y_pred, mask=mask)
    if not np.any(valid):
        return {
            "mse": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "correlation": np.nan,
        }

    yt = y_true[valid]
    yp = y_pred[valid]
    return {
        "mse": float(mse(yt, yp)),
        "rmse": float(rmse(yt, yp)),
        "mae": float(mae(yt, yp)),
        "r2": float(r2(yt, yp)),
        "correlation": float(correlation(yt, yp)),
    }


DAY_DEPTH_METRIC_NAMES = (
    "mae",
    "rmse",
    "r2",
    "correlation",
    "ssim",
    "acc",
)


def masked_pearson_correlation(y_true, y_pred, mask=None):
    """在共同有效区域计算 Pearson 相关系数和有效点数。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    valid = _valid_mask(y_true, y_pred, mask=mask)
    valid_count = int(np.count_nonzero(valid))
    if valid_count < 2:
        return np.nan, valid_count

    true_valid = y_true[valid]
    pred_valid = y_pred[valid]
    true_centered = true_valid - np.mean(true_valid)
    pred_centered = pred_valid - np.mean(pred_valid)
    denominator = np.sqrt(
        np.sum(true_centered * true_centered)
        * np.sum(pred_centered * pred_centered)
    )
    if not np.isfinite(denominator) or denominator <= 0:
        return np.nan, valid_count
    value = np.sum(true_centered * pred_centered) / denominator
    return float(np.clip(value, -1.0, 1.0)), valid_count


def masked_ssim(
    y_true,
    y_pred,
    mask,
    data_range,
    win_size=11,
    sigma=1.5,
    min_valid_fraction=0.8,
    k1=0.01,
    k2=0.03,
):
    """
    计算 mask-aware Gaussian SSIM。

    局部窗口有效海洋权重低于 ``min_valid_fraction`` 时排除，
    返回 ``(ssim, valid_window_count)``。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError(
            f"masked_ssim 需要二维场，实际: {y_true.shape}, {y_pred.shape}"
        )
    valid = _valid_mask(y_true, y_pred, mask=mask)
    win_size = int(win_size)
    sigma = float(sigma)
    data_range = float(data_range)
    min_valid_fraction = float(min_valid_fraction)
    if win_size < 3 or win_size % 2 == 0:
        raise ValueError(f"win_size 必须是不小于3的奇数，实际: {win_size}")
    if sigma <= 0 or data_range <= 0:
        raise ValueError("sigma 和 data_range 必须大于0")
    if min_valid_fraction < 0 or min_valid_fraction > 1:
        raise ValueError("min_valid_fraction 必须位于[0,1]")
    if min(y_true.shape) < win_size:
        return np.nan, 0

    radius = win_size // 2
    truncate = radius / sigma
    weight = valid.astype(np.float64)
    true_valid = np.where(valid, y_true, 0.0)
    pred_valid = np.where(valid, y_pred, 0.0)

    def local_sum(array):
        return gaussian_filter(
            array,
            sigma=sigma,
            mode="constant",
            cval=0.0,
            truncate=truncate,
        )

    local_weight = local_sum(weight)
    full_weight = local_sum(np.ones_like(weight, dtype=np.float64))
    eligible = local_weight >= min_valid_fraction * full_weight
    interior = np.zeros_like(eligible, dtype=bool)
    interior[radius:-radius, radius:-radius] = True
    eligible &= interior & (local_weight > 0)
    valid_windows = int(np.count_nonzero(eligible))
    if valid_windows == 0:
        return np.nan, 0

    with np.errstate(divide="ignore", invalid="ignore"):
        mu_true = local_sum(true_valid) / local_weight
        mu_pred = local_sum(pred_valid) / local_weight
        var_true = local_sum(true_valid * true_valid) / local_weight - mu_true ** 2
        var_pred = local_sum(pred_valid * pred_valid) / local_weight - mu_pred ** 2
        covariance = (
            local_sum(true_valid * pred_valid) / local_weight - mu_true * mu_pred
        )

    var_true = np.maximum(var_true, 0.0)
    var_pred = np.maximum(var_pred, 0.0)
    c1 = (float(k1) * data_range) ** 2
    c2 = (float(k2) * data_range) ** 2
    numerator = (2.0 * mu_true * mu_pred + c1) * (2.0 * covariance + c2)
    denominator = (
        (mu_true ** 2 + mu_pred ** 2 + c1)
        * (var_true + var_pred + c2)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ssim_map = numerator / denominator
    eligible &= np.isfinite(ssim_map)
    valid_windows = int(np.count_nonzero(eligible))
    if valid_windows == 0:
        return np.nan, 0
    return float(np.mean(ssim_map[eligible], dtype=np.float64)), valid_windows


def compute_day_depth_metrics(
    y_true,
    y_pred,
    mask,
    months,
    training_climatology,
    ssim_data_range,
    ssim_win_size=11,
    ssim_sigma=1.5,
    ssim_min_valid_fraction=0.8,
):
    """计算一个 batch 的每天×每层数值和水平结构指标。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.asarray(mask)
    if y_true.ndim != 4:
        raise ValueError(f"y_true/y_pred 需要 (B,D,H,W)，实际: {y_true.shape}")
    _valid_mask(y_true, y_pred, mask=mask)
    batch_size, num_depths, height, width = y_true.shape

    months = np.asarray(months, dtype=np.int64)
    if months.shape != (batch_size,) or np.any((months < 1) | (months > 12)):
        raise ValueError(f"months 应为长度B的1..12整数，实际: {months}")
    climatology = np.asarray(training_climatology, dtype=np.float64)
    if climatology.shape != (12, num_depths, height, width):
        raise ValueError(
            "training_climatology 形状错误: "
            f"expected={(12, num_depths, height, width)}, actual={climatology.shape}"
        )
    data_ranges = np.asarray(ssim_data_range, dtype=np.float64)
    if data_ranges.shape != (num_depths,) or np.any(data_ranges <= 0):
        raise ValueError(
            f"ssim_data_range 应为长度D的正数，实际: {data_ranges}"
        )

    result = {
        name: np.full((batch_size, num_depths), np.nan, dtype=np.float64)
        for name in DAY_DEPTH_METRIC_NAMES
    }
    result["valid_count"] = np.zeros(
        (batch_size, num_depths), dtype=np.int64
    )
    result["ssim_valid_windows"] = np.zeros(
        (batch_size, num_depths), dtype=np.int64
    )
    result["acc_valid_count"] = np.zeros(
        (batch_size, num_depths), dtype=np.int64
    )

    for day_index in range(batch_size):
        month_climatology = climatology[months[day_index] - 1]
        for depth_index in range(num_depths):
            valid = (
                mask[day_index, depth_index].astype(bool)
                & np.isfinite(y_true[day_index, depth_index])
                & np.isfinite(y_pred[day_index, depth_index])
            )
            metrics = scalar_metrics(
                y_true[day_index, depth_index],
                y_pred[day_index, depth_index],
                mask=valid,
            )
            for name in ("mae", "rmse", "r2", "correlation"):
                result[name][day_index, depth_index] = metrics[name]
            result["valid_count"][day_index, depth_index] = np.count_nonzero(valid)

            true_anomaly = (
                y_true[day_index, depth_index] - month_climatology[depth_index]
            )
            pred_anomaly = (
                y_pred[day_index, depth_index] - month_climatology[depth_index]
            )
            acc_value, acc_count = masked_pearson_correlation(
                true_anomaly, pred_anomaly, mask=valid
            )
            ssim_value, ssim_windows = masked_ssim(
                true_anomaly,
                pred_anomaly,
                mask=valid,
                data_range=data_ranges[depth_index],
                win_size=ssim_win_size,
                sigma=ssim_sigma,
                min_valid_fraction=ssim_min_valid_fraction,
            )
            result["acc"][day_index, depth_index] = acc_value
            result["ssim"][day_index, depth_index] = ssim_value
            result["acc_valid_count"][day_index, depth_index] = acc_count
            result["ssim_valid_windows"][day_index, depth_index] = ssim_windows
    return result


def summarize_horizontal_structure(day_depth_ssim, day_depth_acc):
    """SSIM/ACC 按天和深度等权汇总，并返回完整日标记。"""
    day_depth_ssim = np.asarray(day_depth_ssim, dtype=np.float64)
    day_depth_acc = np.asarray(day_depth_acc, dtype=np.float64)
    if day_depth_ssim.shape != day_depth_acc.shape or day_depth_ssim.ndim != 2:
        raise ValueError(
            "day_depth_ssim/day_depth_acc 必须是同形状 (T,D): "
            f"{day_depth_ssim.shape}, {day_depth_acc.shape}"
        )

    with np.errstate(invalid="ignore"):
        by_day_ssim = np.nanmean(day_depth_ssim, axis=1)
        by_day_acc = np.nanmean(day_depth_acc, axis=1)
        by_depth_ssim = np.nanmean(day_depth_ssim, axis=0)
        by_depth_acc = np.nanmean(day_depth_acc, axis=0)
        annual_ssim = float(np.nanmean(day_depth_ssim))
        annual_acc = float(np.nanmean(day_depth_acc))
    by_day_score = (by_day_ssim + by_day_acc) / 2.0
    by_depth_score = (by_depth_ssim + by_depth_acc) / 2.0
    complete_day_mask = np.all(
        np.isfinite(day_depth_ssim) & np.isfinite(day_depth_acc), axis=1
    )
    by_day_score = np.where(complete_day_mask, by_day_score, np.nan)
    return {
        "by_day_ssim": by_day_ssim,
        "by_day_acc": by_day_acc,
        "by_day_structure_score": by_day_score,
        "by_depth_ssim": by_depth_ssim,
        "by_depth_acc": by_depth_acc,
        "by_depth_structure_score": by_depth_score,
        "annual_mean_ssim": annual_ssim,
        "annual_mean_acc": annual_acc,
        "annual_mean_structure_score": (annual_ssim + annual_acc) / 2.0,
        "complete_day_mask": complete_day_mask,
        "by_depth_valid_days": np.sum(
            np.isfinite(day_depth_ssim) & np.isfinite(day_depth_acc), axis=0
        ).astype(np.int64),
    }


def select_structure_quantile_days(
    dates,
    structure_score,
    complete_day_mask,
    quantiles=(10.0, 50.0, 90.0),
):
    """选择最接近结构得分P10/P50/P90的不重复实际日期。"""
    dates = [str(value) for value in dates]
    score = np.asarray(structure_score, dtype=np.float64)
    complete = np.asarray(complete_day_mask, dtype=bool)
    if score.shape != (len(dates),) or complete.shape != score.shape:
        raise ValueError("dates/structure_score/complete_day_mask 长度不一致")
    eligible = np.flatnonzero(complete & np.isfinite(score))
    if eligible.size < len(tuple(quantiles)):
        raise ValueError(
            "可用的完整日期不足，无法选择不重复的代表日期: "
            f"available={eligible.size}, requested={len(tuple(quantiles))}"
        )

    selected = []
    used = set()
    eligible_scores = score[eligible]
    for quantile in quantiles:
        quantile = float(quantile)
        target = float(np.percentile(eligible_scores, quantile))
        candidates = sorted(
            (
                (abs(float(score[index]) - target), dates[index], int(index))
                for index in eligible
                if int(index) not in used
            ),
            key=lambda item: (item[0], item[1]),
        )
        _, _, index = candidates[0]
        used.add(index)
        selected.append(
            {
                "role": f"p{int(quantile):02d}",
                "quantile": quantile,
                "target_score": target,
                "date": dates[index],
                "day_index": index,
                "structure_score": float(score[index]),
            }
        )
    return selected


# ==================== 层反演网格指标 ====================


def compute_grid_metrics(y_true, y_pred, mask=None):
    """
    计算逐网格 MAE 和 RMSE。

    Args:
        y_true: (1, D, H, W) 或 (D, H, W)
        y_pred: (1, D, H, W) 或 (D, H, W)
        mask: 可选，同 y_true/y_pred，mask=1 表示有效标签点

    Returns:
        dict: {'mae': ndarray [D,H,W], 'rmse': ndarray [D,H,W]}
    """
    if y_true.ndim == 4:
        y_true = np.squeeze(y_true, axis=0)
    if y_pred.ndim == 4:
        y_pred = np.squeeze(y_pred, axis=0)
    if mask is not None:
        if mask.ndim == 4:
            mask = np.squeeze(mask, axis=0)
        mask = mask.astype(bool)

    mae_grid = np.abs(y_pred - y_true)
    rmse_grid = np.sqrt(np.square(y_pred - y_true))
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask is not None:
        finite &= mask
    mae_grid = np.where(finite, mae_grid, np.nan)
    rmse_grid = np.where(finite, rmse_grid, np.nan)

    return {'mae': mae_grid, 'rmse': rmse_grid}


def extract_level_map(metric_grid, level_idx):
    """
    从 [D, H, W] 指标数组中提取指定层 [H, W]。

    Args:
        metric_grid: (D, H, W)
        level_idx: 深度层索引

    Returns:
        (H, W) 二维数组
    """
    if level_idx < 0 or level_idx >= metric_grid.shape[0]:
        raise IndexError(
            f"level_idx={level_idx} 超出范围 [0, {metric_grid.shape[0] - 1}]"
        )
    return metric_grid[level_idx]


def save_grid_metrics(metrics, output_path):
    """保存网格指标为 npz（键名 rmse / mae）"""
    np.savez(output_path, **metrics)
    print(f"网格指标已保存至：{output_path}")
