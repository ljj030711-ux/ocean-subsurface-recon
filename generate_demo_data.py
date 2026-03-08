"""
生成演示数据的脚本
创建 .npy 格式的模拟海洋数据
"""

import numpy as np
import os


def generate_demo_data(output_dir="./data/demo", T=30, H=64, W=64, random_seed=42):
    """
    生成演示数据（模拟海洋遥感数据）
    
    Args:
        output_dir: 输出目录
        T: 时间步数
        H: 空间高度（纬度）
        W: 空间宽度（经度）
        random_seed: 随机种子
    """
    
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"生成演示数据 [{T}, {H}, {W}]...")
    
    # ========== 1. 海表温度 (SST) ==========
    # 模型：高斯随机场 + 空间相关性
    sst = np.zeros((T, H, W), dtype=np.float32)
    for t in range(T):
        # 创建空间相关的高斯随机场
        x = np.linspace(-3, 3, H)
        y = np.linspace(-3, 3, W)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # 背景温度场（南北梯度）
        bg = 25 - 5 * X / 6  # 南（高）北（低）
        
        # 添加涡旋异常（Gaussian blob）
        center_x = H // 2 + np.random.randint(-10, 10)
        center_y = W // 2 + np.random.randint(-10, 10)
        eddy = 3 * np.exp(-((X - x[center_x])**2 + (Y - y[center_y])**2) / 4)
        
        # 随机噪声
        noise = np.random.normal(0, 0.3, (H, W))
        
        sst[t] = bg + eddy + noise
    
    sst = np.clip(sst, 10, 30)  # 物理范围
    
    # ========== 2. 海表盐度 (SSS) ==========
    # 与 SST 负相关（冷涡吸收淡水）
    sss = 35 + 0.3 * (25 - sst)
    sss = np.clip(sss, 34, 36)
    
    # ========== 3. 海面高度异常 (SSH) ==========
    # 与 SST 正相关（暖涡为正异常）
    # SSH = SST * 某个系数 + 噪声
    ssh = 0.02 * (sst - 20) + np.random.normal(0, 0.01, (T, H, W))
    ssh = np.clip(ssh, -0.2, 0.2)
    
    # ========== 4. 水下温度 (目标变量，100m深度) ==========
    # 物理关系：表层冷 → 100m也相对冷（但衰减）
    # 通过涡旋的几何结构和 SSH 推断
    subsurface = np.zeros((T, H, W), dtype=np.float32)
    
    for t in range(T):
        # 基础温度随深度降低
        base_temp = 20 - 2 * (sst[t] - 20)  # 与表层反相，深度更冷
        
        # SSH 梯度决定的地转流动
        dssh_dx = np.gradient(ssh[t], axis=1)
        dssh_dy = np.gradient(ssh[t], axis=0)
        
        # 流动强度影响水下混合
        flow_strength = np.sqrt(dssh_dx**2 + dssh_dy**2)
        
        # 冷涡中心（SST 冷）处，水下异常也冷
        temp_anomaly = -2 * (sst[t] - 22) + 1.5 * flow_strength * 50
        
        subsurface[t] = base_temp + temp_anomaly + np.random.normal(0, 0.5, (H, W))
    
    subsurface = np.clip(subsurface, 5, 25)
    
    # 保存为 .npy 文件
    np.save(f"{output_dir}/sst.npy", sst)
    np.save(f"{output_dir}/sss.npy", sss)
    np.save(f"{output_dir}/ssh.npy", ssh)
    np.save(f"{output_dir}/subsurface.npy", subsurface)
    
    print(f"✓ 演示数据已生成到 {output_dir}/")
    print(f"\n数据统计:")
    print(f"  SST:        min={sst.min():.3f}, max={sst.max():.3f}, mean={sst.mean():.3f}")
    print(f"  SSS:        min={sss.min():.3f}, max={sss.max():.3f}, mean={sss.mean():.3f}")
    print(f"  SSH:        min={ssh.min():.3f}, max={ssh.max():.3f}, mean={ssh.mean():.3f}")
    print(f"  Subsurface: min={subsurface.min():.3f}, max={subsurface.max():.3f}, mean={subsurface.mean():.3f}")
    print(f"\n文件列表:")
    for fname in ['sst.npy', 'sss.npy', 'ssh.npy', 'subsurface.npy']:
        fpath = f"{output_dir}/{fname}"
        size = os.path.getsize(fpath) / 1024  # KB
        print(f"  {fname:20s} {size:8.1f} KB")


if __name__ == "__main__":
    # 生成演示数据（30个时间步，64x64 网格）
    generate_demo_data(T=30, H=64, W=64)
