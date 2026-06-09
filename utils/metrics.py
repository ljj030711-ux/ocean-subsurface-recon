"""
评估指标模块
用于模型性能评估和结果验证
"""

import numpy as np


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
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask is not None:
        mask = np.asarray(mask)
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
