"""
数据集模块
处理数据加载和特征工程
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class EddyDataset(Dataset):
    """
    涡旋反演数据集
    
    从 .npy 文件加载多源海洋遥感数据，并组织为 PyTorch Dataset 格式
    """
    
    def __init__(self, data_dir, normalize=False):
        """
        初始化数据集
        
        Args:
            data_dir: 数据文件目录路径
            normalize: 是否进行标准化（均值0，方差1）
        """
        self.data_dir = data_dir
        self.normalize = normalize
        
        # 加载所有数据（只用SSS和SSH）
        self.sss = np.load(f"{data_dir}/sss.npy").astype(np.float32)
        self.ssh = np.load(f"{data_dir}/ssh.npy").astype(np.float32)
        self.target = np.load(f"{data_dir}/subsurface.npy").astype(np.float32)  # 假设这是指定深度的盐度
        
        # 挤压多余的维度（假设深度维度为1）
        if self.sss.ndim == 4 and self.sss.shape[1] == 1:
            self.sss = np.squeeze(self.sss, axis=1)
        if self.ssh.ndim == 4 and self.ssh.shape[1] == 1:
            self.ssh = np.squeeze(self.ssh, axis=1)
        if self.target.ndim == 4 and self.target.shape[1] == 1:
            self.target = np.squeeze(self.target, axis=1)

        if not (self.ssh.shape == self.sss.shape == self.target.shape):
            raise ValueError("SSH/SSS/subsurface 的 shape 必须一致")
        
        # 处理缺失值（用均值填充）
        sss_mean = np.nanmean(self.sss)
        self.sss = np.nan_to_num(self.sss, nan=sss_mean if not np.isnan(sss_mean) else 0.0)
        ssh_mean = np.nanmean(self.ssh)
        self.ssh = np.nan_to_num(self.ssh, nan=ssh_mean if not np.isnan(ssh_mean) else 0.0)
        target_mean = np.nanmean(self.target)
        self.target = np.nan_to_num(self.target, nan=target_mean if not np.isnan(target_mean) else 0.0)
        
        # 数据统计
        self.T, self.H, self.W = self.sss.shape
        
        # 可选的标准化
        if self.normalize:
            self.sss = (self.sss - self.sss.mean()) / (self.sss.std() + 1e-6)
            self.ssh = (self.ssh - self.ssh.mean()) / (self.ssh.std() + 1e-6)
            self.target = (self.target - self.target.mean()) / (self.target.std() + 1e-6)
    
    def __len__(self):
        """返回数据集大小"""
        return self.T
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 时间索引
        
        Returns:
            dict: 包含各个变量的字典
        """
        return {
            "sss": torch.tensor(self.sss[idx], dtype=torch.float32).unsqueeze(0),
            "ssh": torch.tensor(self.ssh[idx], dtype=torch.float32).unsqueeze(0),
            "target": torch.tensor(self.target[idx], dtype=torch.float32).unsqueeze(0),
        }
    
    def get_data_info(self):
        """返回数据集信息"""
        return {
            "time_steps": self.T,
            "height": self.H,
            "width": self.W,
            "sss_range": (self.sss.min(), self.sss.max()),
            "ssh_range": (self.ssh.min(), self.ssh.max()),
            "target_range": (self.target.min(), self.target.max()),
        }


class EddyDatasetAugmented(EddyDataset):
    """
    带数据增强的涡旋反演数据集
    支持随机翻转和旋转
    """
    
    def __init__(self, data_dir, normalize=False, augment=False):
        """
        初始化增强数据集
        
        Args:
            data_dir: 数据文件目录路径
            normalize: 是否进行标准化
            augment: 是否进行数据增强
        """
        super().__init__(data_dir, normalize)
        self.augment = augment
    
    def __getitem__(self, idx):
        """
        获取单个样本（支持增强）
        
        Args:
            idx: 时间索引
        
        Returns:
            dict: 包含各个变量的字典
        """
        sample = super().__getitem__(idx)
        
        if self.augment:
            sample = self._apply_augmentation(sample)
        
        return sample

    def _apply_augmentation(self, sample):
        """
        应用数据增强
        
        Args:
            sample: 单个样本
        
        Returns:
            dict: 增强后的样本
        """
        # 随机水平翻转
        if np.random.rand() > 0.5:
            for key in sample:
                sample[key] = torch.flip(sample[key], dims=[-1])
        
        # 随机垂直翻转
        if np.random.rand() > 0.5:
            for key in sample:
                sample[key] = torch.flip(sample[key], dims=[-2])
        
        # 随机旋转 (0, 90, 180, 270度)
        rot_k = np.random.randint(0, 4)
        if rot_k > 0:
            for key in sample:
                sample[key] = torch.rot90(sample[key], k=rot_k, dims=[-2, -1])
        
        return sample


class DailySequenceEddyDataset(EddyDataset):
    """
    按日序列构造样本的数据集。

    一个样本由连续 window_days 天的输入组成，目标为窗口末日+horizon_days 的 100m 温场。
    """

    def __init__(self, data_dir, window_days=7, horizon_days=0, normalize=False):
        super().__init__(data_dir, normalize=normalize)
        if window_days < 1:
            raise ValueError("window_days 必须 >= 1")
        if horizon_days < 0:
            raise ValueError("horizon_days 必须 >= 0")

        self.window_days = int(window_days)
        self.horizon_days = int(horizon_days)
        self.num_samples = self.T - self.window_days - self.horizon_days + 1
        if self.num_samples <= 0:
            raise ValueError(
                f"样本数 <= 0: T={self.T}, window_days={self.window_days}, horizon_days={self.horizon_days}"
            )

        # 每个样本对应目标日期在原时间轴的索引（按天）
        self.target_day_indices = np.arange(
            self.window_days - 1 + self.horizon_days,
            self.window_days - 1 + self.horizon_days + self.num_samples,
            dtype=np.int64,
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"样本索引越界: idx={idx}, num_samples={self.num_samples}")

        start = idx
        end = idx + self.window_days
        target_idx = end - 1 + self.horizon_days

        return {
            "sss": torch.tensor(self.sss[start:end], dtype=torch.float32),     # [D,H,W]
            "ssh": torch.tensor(self.ssh[start:end], dtype=torch.float32),     # [D,H,W]
            "target": torch.tensor(self.target[target_idx], dtype=torch.float32).unsqueeze(0),  # [1,H,W]
            "target_day_idx": torch.tensor(target_idx, dtype=torch.long),
        }

    def get_data_info(self):
        info = super().get_data_info()
        info.update(
            {
                "window_days": self.window_days,
                "horizon_days": self.horizon_days,
                "num_samples": self.num_samples,
                "first_target_day_idx": int(self.target_day_indices[0]),
                "last_target_day_idx": int(self.target_day_indices[-1]),
            }
        )
        return info
