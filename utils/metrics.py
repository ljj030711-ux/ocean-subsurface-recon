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


def evaluate_prediction(y_true, y_pred):
    """
    综合评估预测结果
    
    Args:
        y_true: 真实值 [T, H, W] 或 [T, 1, H, W]
        y_pred: 预测值 [T, H, W] 或 [T, 1, H, W]
    
    Returns:
        dict: 包含各项指标的字典
    """
    # 展平数据
    if y_true.ndim == 4:
        y_true = y_true.squeeze(1)
    if y_pred.ndim == 4:
        y_pred = y_pred.squeeze(1)
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    metrics = {
        'mse': mse(y_true_flat, y_pred_flat),
        'rmse': rmse(y_true_flat, y_pred_flat),
        'mae': mae(y_true_flat, y_pred_flat),
        'r2': r2(y_true_flat, y_pred_flat),
        'correlation': np.corrcoef(y_true_flat, y_pred_flat)[0, 1],
    }
    
    return metrics


def spatial_gradient_consistency(y_true, y_pred):
    """
    评估梯度一致性（物理约束指标）
    
    Args:
        y_true: 真实值 [T, H, W]
        y_pred: 预测值 [T, H, W]
    
    Returns:
        float: 梯度差异 (越小越好)
    """
    # 计算梯度
    grad_true_x = np.diff(y_true, axis=2)
    grad_true_y = np.diff(y_true, axis=1)
    
    grad_pred_x = np.diff(y_pred, axis=2)
    grad_pred_y = np.diff(y_pred, axis=1)
    
    # 梯度差异
    diff_x = np.mean(np.abs(grad_true_x - grad_pred_x))
    diff_y = np.mean(np.abs(grad_true_y - grad_pred_y))
    
    return (diff_x + diff_y) / 2


def temporal_consistency(predictions):
    """
    评估时间连贯性
    
    Args:
        predictions: 预测序列 [T, H, W]
    
    Returns:
        float: 时间梯度 (越小越平滑)
    """
    temporal_diff = np.diff(predictions, axis=0)
    return np.mean(np.abs(temporal_diff))


def spatial_correlation_map(y_true, y_pred):
    """
    计算空间相关系数图
    
    Args:
        y_true: 真实值 [T, H, W]
        y_pred: 预测值 [T, H, W]
    
    Returns:
        corr_map: 空间相关系数 [H, W]
    """
    T, H, W = y_true.shape
    corr_map = np.zeros((H, W))
    
    for i in range(H):
        for j in range(W):
            corr_map[i, j] = np.corrcoef(y_true[:, i, j], y_pred[:, i, j])[0, 1]
    
    return corr_map


def bias_map(y_true, y_pred):
    """
    计算偏差图
    
    Args:
        y_true: 真实值 [T, H, W]
        y_pred: 预测值 [T, H, W]
    
    Returns:
        bias: 偏差 [H, W]
    """
    return np.mean(y_pred - y_true, axis=0)


def print_metrics(metrics, prefix=""):
    """
    打印评估指标
    
    Args:
        metrics: 指标字典
        prefix: 前缀字符串
    """
    print(f"\n{prefix}评估指标:")
    print(f"  MSE:         {metrics['mse']:.6f}")
    print(f"  RMSE:        {metrics['rmse']:.6f}")
    print(f"  MAE:         {metrics['mae']:.6f}")
    print(f"  R²:          {metrics['r2']:.6f}")
    print(f"  相关系数:     {metrics['correlation']:.6f}")
