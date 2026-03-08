"""
绘制训练曲线
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')

OUTPUT_DIR = Path('./outputs')

def plot_training_curve(history_file='./outputs/training_history.npz', title=None):
    """绘制训练和验证损失曲线"""
    
    if not Path(history_file).exists():
        print(f"✗ 错误：找不到训练历史文件 {history_file}")
        print("  请先运行 train.py 完成训练")
        return
    
    # 加载训练历史
    data = np.load(history_file)
    train_losses = data['train_losses']
    val_losses = data['val_losses']
    
    epochs = np.arange(1, len(train_losses) + 1)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制曲线
    ax.plot(epochs, train_losses, marker='o', linewidth=2.5, markersize=5, 
            label='Train Loss', color='#1f77b4', alpha=0.8)
    ax.plot(epochs, val_losses, marker='s', linewidth=2.5, markersize=5,
            label='Val Loss', color='#ff7f0e', alpha=0.8)
    
    # 设置标签和标题
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('MSE Loss', fontsize=13, fontweight='bold')
    
    if title is None:
        title = 'Train/Val MSE in Eddy Inversion'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 美化图表
    ax.legend(fontsize=12, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, len(train_losses) + 1)
    
    # 添加最小值标注
    best_train_epoch = np.argmin(train_losses) + 1
    best_train_loss = train_losses.min()
    best_val_epoch = np.argmin(val_losses) + 1
    best_val_loss = val_losses.min()
    
    ax.plot(best_train_epoch, best_train_loss, marker='*', markersize=20, 
            color='#1f77b4', markeredgecolor='darkblue', markeredgewidth=1.5, zorder=5)
    ax.plot(best_val_epoch, best_val_loss, marker='*', markersize=20,
            color='#ff7f0e', markeredgecolor='darkorange', markeredgewidth=1.5, zorder=5)
    
    # 添加文本注释
    ax.text(best_train_epoch, best_train_loss * 0.95, 
            f'  Best: {best_train_loss:.6f}\n  Epoch: {best_train_epoch}',
            fontsize=10, color='#1f77b4', fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    ax.text(best_val_epoch, best_val_loss * 1.05,
            f'  Best: {best_val_loss:.6f}\n  Epoch: {best_val_epoch}',
            fontsize=10, color='#ff7f0e', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图表
    output_path = OUTPUT_DIR / 'training_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {output_path}")
    
    # 打印统计信息
    print("\n" + "="*60)
    print("训练统计")
    print("="*60)
    print(f"总 Epoch 数: {len(train_losses)}")
    print(f"\n【训练损失】")
    print(f"  初始: {train_losses[0]:.6f}")
    print(f"  最终: {train_losses[-1]:.6f}")
    print(f"  最小: {best_train_loss:.6f} (Epoch {best_train_epoch})")
    print(f"  下降: {(train_losses[0] - train_losses[-1]) / train_losses[0] * 100:.2f}%")
    
    print(f"\n【验证损失】")
    print(f"  初始: {val_losses[0]:.6f}")
    print(f"  最终: {val_losses[-1]:.6f}")
    print(f"  最小: {best_val_loss:.6f} (Epoch {best_val_epoch})")
    print(f"  下降: {(val_losses[0] - val_losses[-1]) / val_losses[0] * 100:.2f}%")
    
    print("\n" + "="*60 + "\n")
    
    return fig


def plot_multiple_curves(history_files, labels, title=None):
    """绘制多个训练历史对比"""
    
    fig, ax = plt.subplots(figsize=(13, 6.5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, (history_file, label) in enumerate(zip(history_files, labels)):
        if not Path(history_file).exists():
            print(f"⚠ 警告：找不到 {history_file}，跳过")
            continue
        
        data = np.load(history_file)
        val_losses = data['val_losses']
        epochs = np.arange(1, len(val_losses) + 1)
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        ax.plot(epochs, val_losses, marker=marker, linewidth=2.5, markersize=5,
                label=label, color=color, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Val Loss (MSE)', fontsize=13, fontweight='bold')
    
    if title is None:
        title = 'Validation Loss Comparison'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'training_curves_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 已保存对比图: {output_path}")
    
    return fig


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        # 对比多个模型
        print("\n生成多模型训练曲线对比...\n")
        # 示例：可添加多个历史文件进行对比
        # plot_multiple_curves(['outputs/training_history_model1.npz', 'outputs/training_history_model2.npz'],
        #                      ['Model 1', 'Model 2'])
    else:
        # 绘制单个训练曲线
        print("\n生成训练曲线...\n")
        plot_training_curve()
