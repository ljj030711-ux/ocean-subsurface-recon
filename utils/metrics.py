"""
评估指标模块
用于模型性能评估和结果验证
"""

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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
    return 1 - (ss_res / ss_tot)


def correlation(y_true, y_pred):
    """皮尔逊相关系数"""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    return np.corrcoef(y_true_flat, y_pred_flat)[0, 1]

# ==================== 层反演网格指标 ====================


def compute_grid_metrics(y_true, y_pred):
    """
    计算逐网格 MAE 和 RMSE。

    Args:
        y_true: (1, D, H, W) 或 (D, H, W)
        y_pred: (1, D, H, W) 或 (D, H, W)

    Returns:
        dict: {'mae': ndarray [D,H,W], 'rmse': ndarray [D,H,W]}
    """
    if y_true.ndim == 4:
        y_true = np.squeeze(y_true, axis=0)
    if y_pred.ndim == 4:
        y_pred = np.squeeze(y_pred, axis=0)

    mae_grid = np.abs(y_pred - y_true)
    rmse_grid = np.sqrt(np.square(y_pred - y_true))

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
