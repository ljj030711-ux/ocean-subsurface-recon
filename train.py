"""
训练脚本
用于模型的训练和优化
"""

import os
from datetime import datetime, timedelta
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from config import *
from datasets.eddy_dataset import DailySequenceEddyDataset, EddyDataset
from models.eddy_cnn import EddyAwareCNN, EddyResNet, EddyUNet
from utils.physics import compute_eke, compute_grad_ssh
from utils.metrics import evaluate_prediction


def smooth_loss(x):
    """
    计算光滑性正则化损失
    约束相邻像素的梯度平滑
    
    Args:
        x: 预测张量 [B, C, H, W]
    
    Returns:
        loss: 光滑损失
    """
    loss_x = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    loss_y = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    return loss_x + loss_y


def build_input_features(sss, ssh, use_physics_features):
    """
    构造模型输入特征

    False: [sss, ssh] (2通道)
    True : [sss, ssh, eke, grad] (4通道)
    """
    if use_physics_features:
        eke = compute_eke(ssh, DX, DY)
        grad = compute_grad_ssh(ssh)
        return torch.cat([sss, ssh, eke, grad], dim=1)
    return torch.cat([sss, ssh], dim=1)


def build_model(model_name, in_channels):
    """
    根据配置创建模型
    """
    name = model_name.strip().lower()
    if name in {"eddyawarecnn", "eddycnn", "cnn"}:
        return EddyAwareCNN(in_channels=in_channels, out_channels=1)
    if name in {"eddyunet", "unet"}:
        return EddyUNet(in_channels=in_channels, out_channels=1)
    if name in {"eddyresnet", "resnet"}:
        return EddyResNet(in_channels=in_channels, out_channels=1)
    raise ValueError(f"未识别的模型名称: {model_name}")


def train_epoch(model, loader, optimizer, device, lambda_smooth=0.1, use_physics_features=True):
    """
    训练一个 epoch
    
    Args:
        model: 神经网络模型
        loader: 数据加载器
        optimizer: 优化器
        device: 计算设备
        lambda_smooth: 光滑性正则化系数
    
    Returns:
        float: 该 epoch 的平均损失
    """
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    for batch in loader:
        sss = batch["sss"].to(device)
        ssh = batch["ssh"].to(device)
        target = batch["target"].to(device)
        
        # 拼接输入特征（2通道或4通道）
        x = build_input_features(sss, ssh, use_physics_features)
        
        # 前向传播
        pred = model(x)
        
        # 计算损失
        mse_loss = F.mse_loss(pred, target)
        smooth_reg = smooth_loss(pred)
        total_train_loss = mse_loss + lambda_smooth * smooth_reg
        
        # 反向传播和优化
        optimizer.zero_grad()
        total_train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += total_train_loss.item()
        batch_count += 1
    
    return total_loss / batch_count


def validate(model, loader, device, use_physics_features=True):
    """
    验证模型
    
    Args:
        model: 神经网络模型
        loader: 数据加载器
        device: 计算设备
    
    Returns:
        float: 验证集平均损失
    """
    model.eval()
    total_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch in loader:
            sss = batch["sss"].to(device)
            ssh = batch["ssh"].to(device)
            target = batch["target"].to(device)
            
            x = build_input_features(sss, ssh, use_physics_features)
            pred = model(x)
            
            loss = F.mse_loss(pred, target)
            total_loss += loss.item()
            batch_count += 1
    
    return total_loss / batch_count


def main():
    """主训练函数"""
    
    # 创建输出目录
    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 打印配置
    print("=" * 60)
    print("训练配置")
    print("=" * 60)
    print(f"批处理大小: {BATCH_SIZE}")
    print(f"训练轮数: {EPOCHS}")
    print(f"学习率: {LR}")
    print(f"模型: {MODEL_NAME}")
    print(f"使用物理特征: {USE_PHYSICS_FEATURES}")
    print(f"光滑正则化系数: {LAMBDA_SMOOTH}")
    print(f"计算设备: {DEVICE}")
    print("=" * 60)
    
    # 加载数据
    print("\n加载数据...")
    dataset = DailySequenceEddyDataset(
        DATA_DIR,
        window_days=WINDOW_DAYS,
        horizon_days=PREDICT_HORIZON_DAYS,
        normalize=True,
    )
    data_info = dataset.get_data_info()
    print(f"原始时间步: T={data_info['time_steps']}, H={data_info['height']}, W={data_info['width']}")
    print(
        f"序列参数: window_days={data_info['window_days']}, "
        f"horizon_days={data_info['horizon_days']}, "
        f"num_samples={data_info['num_samples']}"
    )
    print(f"SSH 范围: {data_info['ssh_range']}")
    print(f"目标范围: {data_info['target_range']}")
    
    # 按年份切分：训练2019-2021，验证2022，测试2023
    from datetime import datetime, timedelta
    start_date = datetime.strptime(DATA_START_DATE, "%Y-%m-%d")
    total_days = len(dataset)
    train_end_date = datetime(2021, 12, 31)
    val_end_date = datetime(2022, 12, 31)
    test_end_date = datetime(2023, 12, 31)

    train_days = (train_end_date - start_date).days + 1
    val_days = (val_end_date - train_end_date).days
    test_days = (test_end_date - val_end_date).days

    train_indices = list(range(0, min(train_days, total_days)))
    val_start = train_days
    val_indices = list(range(val_start, min(val_start + val_days, total_days)))
    test_start = val_start + val_days
    test_indices = list(range(test_start, min(test_start + test_days, total_days)))

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 打印训练/验证/测试目标日期范围（按天）
    series_start = datetime.strptime(DATA_START_DATE, "%Y-%m-%d")
    if train_indices:
        train_start_day = train_indices[0]
        train_end_day = train_indices[-1]
        print(
            f"训练目标日期: {(series_start + timedelta(days=train_start_day)).strftime('%Y-%m-%d')} "
            f"~ {(series_start + timedelta(days=train_end_day)).strftime('%Y-%m-%d')}"
        )
    if val_indices:
        val_start_day = val_indices[0]
        val_end_day = val_indices[-1]
        print(
            f"验证目标日期: {(series_start + timedelta(days=val_start_day)).strftime('%Y-%m-%d')} "
            f"~ {(series_start + timedelta(days=val_end_day)).strftime('%Y-%m-%d')}"
        )
    if test_indices:
        test_start_day = test_indices[0]
        test_end_day = test_indices[-1]
        print(
            f"测试目标日期: {(series_start + timedelta(days=test_start_day)).strftime('%Y-%m-%d')} "
            f"~ {(series_start + timedelta(days=test_end_day)).strftime('%Y-%m-%d')}"
        )

    # 创建模型
    print("\n初始化模型...")
    per_day_channels = 4 if USE_PHYSICS_FEATURES else 2
    in_channels = per_day_channels * WINDOW_DAYS
    model = build_model(MODEL_NAME, in_channels=in_channels).to(DEVICE)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建优化器和学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    
    # 训练循环
    print("\n开始训练...\n")
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, DEVICE, LAMBDA_SMOOTH, USE_PHYSICS_FEATURES
        )
        
        # 验证
        val_loss = validate(model, val_loader, DEVICE, USE_PHYSICS_FEATURES)
        
        # 学习率调度
        scheduler.step()
        
        # 打印进度
        if VERBOSE and (epoch + 1) % 1 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "./checkpoints/best_model.pth")
            model_tag = MODEL_NAME.strip().lower()
            torch.save(model.state_dict(), f"./checkpoints/{model_tag}_best_model.pth")
            patience_counter = 0
            if VERBOSE:
                print(f"  ✓ 保存最佳模型 (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停触发 (patience={patience})")
                break
    
    # 保存最终模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✓ 模型已保存到 {MODEL_SAVE_PATH}")
    print(f"✓ 最佳验证损失: {best_val_loss:.6f}")
    
    return model


def main_minimal_daily():
    """
    最简训练函数（非滑动）：
    当天输入 SST/SSS/SSH，预测当天 100m 温场。
    """
    os.makedirs("./checkpoints", exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("最简训练（当天->当天）")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print(f"使用物理特征: {USE_PHYSICS_FEATURES}")
    print(f"训练轮数: {EPOCHS}, 学习率: {LR}, 批处理: {BATCH_SIZE}")
    print(f"设备: {DEVICE}")

    dataset = EddyDataset(DATA_DIR, normalize=True)
    data_info = dataset.get_data_info()
    print(f"数据集: T={data_info['time_steps']}, H={data_info['height']}, W={data_info['width']}")

    # 按年份切分：训练2019-2021，验证2022，测试2023
    from datetime import datetime, timedelta
    start_date = datetime.strptime(DATA_START_DATE, "%Y-%m-%d")
    total_days = len(dataset)
    train_end_date = datetime(2021, 12, 31)
    val_end_date = datetime(2022, 12, 31)
    test_end_date = datetime(2023, 12, 31)

    train_days = (train_end_date - start_date).days + 1
    val_days = (val_end_date - train_end_date).days
    test_days = (test_end_date - val_end_date).days

    train_indices = list(range(0, min(train_days, total_days)))
    val_start = train_days
    val_indices = list(range(val_start, min(val_start + val_days, total_days)))
    test_start = val_start + val_days
    test_indices = list(range(test_start, min(test_start + test_days, total_days)))

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    series_start = datetime.strptime(DATA_START_DATE, "%Y-%m-%d")
    print(
        f"训练日期: {series_start.strftime('%Y-%m-%d')} ~ "
        f"{(series_start + timedelta(days=train_indices[-1] if train_indices else 0)).strftime('%Y-%m-%d')}"
    )
    print(
        f"验证日期: {(series_start + timedelta(days=val_indices[0] if val_indices else 0)).strftime('%Y-%m-%d')} ~ "
        f"{(series_start + timedelta(days=val_indices[-1] if val_indices else 0)).strftime('%Y-%m-%d')}"
    )
    print(
        f"测试日期: {(series_start + timedelta(days=test_indices[0] if test_indices else 0)).strftime('%Y-%m-%d')} ~ "
        f"{(series_start + timedelta(days=test_indices[-1] if test_indices else 0)).strftime('%Y-%m-%d')}"
    )

    in_channels = 4 if USE_PHYSICS_FEATURES else 2
    model = build_model(MODEL_NAME, in_channels=in_channels).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(
            model, train_loader, optimizer, DEVICE, lambda_smooth=0.0, use_physics_features=USE_PHYSICS_FEATURES
        )
        val_loss = validate(model, val_loader, DEVICE, use_physics_features=USE_PHYSICS_FEATURES)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "./checkpoints/minimal_daily_best_model.pth")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # 保存训练历史
    np.savez("./outputs/training_history.npz", train_losses=train_losses, val_losses=val_losses)
    
    print(f"✓ 最佳验证损失: {best_val_loss:.6f}")
    print(f"✓ 最佳模型: ./checkpoints/minimal_daily_best_model.pth")
    print(f"✓ 兼容模型: {MODEL_SAVE_PATH}")
    return model


