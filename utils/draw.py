"""
可视化 persistence 和 eddyunet 推理结果对比
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 禁用交互模式
plt.ioff()

RESULTS_DIR = Path('/Users/lijunjie/Documents/上大硕士/data/eddy_inversion_results')
OUTPUT_DIR = Path('/Users/lijunjie/Documents/python/eddy_inversion/outputs/visualization')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_results(model_name):
    """加载推理结果"""
    path = RESULTS_DIR / f'inversion_results_2023-01-01_2023-12-31_{model_name}.npz'
    data = np.load(path)
    return data['targets'], data['predictions']


def plot_spatial_error():
    """绘制空间误差热力图"""
    targets_per, preds_per = load_results('persistence')
    targets_unet, preds_unet = load_results('eddyunet')
    
    # 计算平均误差（形状为 64x64，每个位置是时间和深度方向的平均）
    error_per = np.mean(np.abs(preds_per - targets_per), axis=(0, 1))
    error_unet = np.mean(np.abs(preds_unet - targets_unet), axis=(0, 1))
    
    # 获取统一的颜色范围
    vmin = np.min([np.min(error_per), np.min(error_unet)])
    vmax = np.max([np.max(error_per), np.max(error_unet)])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 设置 colormap
    cmap = plt.cm.RdYlBu_r
    
    # 绘制Persistence误差
    im1 = axes[0].imshow(error_per, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title('Persistence: Mean Absolute Error', fontsize=12)
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    cbar1 = plt.colorbar(im1, ax=axes[0], label='MAE')
    
    # 绘制EddyUNet误差
    im2 = axes[1].imshow(error_unet, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title('EddyUNet: Mean Absolute Error', fontsize=12)
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    cbar2 = plt.colorbar(im2, ax=axes[1], label='MAE')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'spatial_error.png', dpi=300, bbox_inches='tight')
    print('✓ 已保存: spatial_error.png')
    plt.close()


def plot_sample_fields():
    """绘制几个样本的空间场对比"""
    targets_per, preds_per = load_results('persistence')
    targets_unet, preds_unet = load_results('eddyunet')
    
    # 选择几个代表性的日期（开始、中间、结束）
    sample_indices = [10, 90, 180, 270, 364]
    
    fig, axes = plt.subplots(len(sample_indices), 3, figsize=(14, 14), dpi=150)
    
    for row, idx in enumerate(sample_indices):
        target = targets_per[idx, 0]  # 假设只有一个深度层
        pred_per = preds_per[idx, 0]
        pred_unet = preds_unet[idx, 0]
        
        date = datetime(2023, 1, 1) + timedelta(days=idx)
        
        # Target
        im0 = axes[row, 0].imshow(target, cmap='viridis')
        axes[row, 0].set_title(f'Target ({date.strftime("%Y-%m-%d")})', fontsize=10)
        axes[row, 0].set_xlabel('Lon')
        axes[row, 0].set_ylabel('Lat')
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)
        
        # Persistence
        im1 = axes[row, 1].imshow(pred_per, cmap='viridis')
        err_per = np.abs(pred_per - target)
        axes[row, 1].set_title(f'Persistence (MAE: {err_per.mean():.4f})', fontsize=10)
        axes[row, 1].set_xlabel('Lon')
        axes[row, 1].set_ylabel('Lat')
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)
        
        # EddyUNet
        im2 = axes[row, 2].imshow(pred_unet, cmap='viridis')
        err_unet = np.abs(pred_unet - target)
        axes[row, 2].set_title(f'EddyUNet (MAE: {err_unet.mean():.4f})', fontsize=10)
        axes[row, 2].set_xlabel('Lon')
        axes[row, 2].set_ylabel('Lat')
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)
    
    plt.suptitle('Sample Field Predictions: 2023', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sample_fields.png', dpi=300, bbox_inches='tight')
    print('✓ 已保存: sample_fields.png')
    plt.close()



# def print_summary():
#     """打印汇总统计"""
#     targets_per, preds_per = load_results('persistence')
#     targets_unet, preds_unet = load_results('eddyunet')
    
#     metrics_per = compute_metrics(targets_per, preds_per)
#     metrics_unet = compute_metrics(targets_unet, preds_unet)
    
#     print("\n" + "="*60)
#     print("推理结果对比总结 (2023年全年)")
#     print("="*60)
    
#     print("\n【Persistence 基准模型】")
#     for k, v in metrics_per.items():
#         print(f"  {k:12s}: {v:.6f}")
    
#     print("\n【EddyUNet 模型】")
#     for k, v in metrics_unet.items():
#         print(f"  {k:12s}: {v:.6f}")
    
#     print("\n【改进对比】")
#     for k in metrics_per.keys():
#         if k == 'Correlation':
#             improvement = ((metrics_unet[k] - metrics_per[k]) / metrics_per[k] * 100) if metrics_per[k] != 0 else 0
#             print(f"  {k:12s}: {improvement:+.2f}%")
#         else:
#             improvement = ((metrics_per[k] - metrics_unet[k]) / metrics_per[k] * 100) if metrics_per[k] != 0 else 0
#             print(f"  {k:12s}: {improvement:+.2f}% {'↓' if improvement > 0 else '↑'} (越低越好)")
    
#     print("\n✓ 所有图表已保存到:", OUTPUT_DIR)
#     print("="*60 + "\n")


if __name__ == '__main__':
    print("\n正在生成可视化图表...\n")
    
    try:
        plot_spatial_error()
    except Exception as e:
        print(f"✗ spatial_error 生成失败: {e}")
    
    try:
        plot_sample_fields()
    except Exception as e:
        print(f"✗ sample_fields 生成失败: {e}")
    
    # print_summary()
