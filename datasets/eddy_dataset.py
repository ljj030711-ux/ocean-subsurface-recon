"""
数据集模块
处理数据加载和特征工程
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from config import DATA_START_DATE, TRAIN_END_DATE
from datasets.date_utils import generate_month_numbers, indices_until_date
from datasets.io_2dto2d import (
    clean_and_normalize_2dto2d,
    load_2dto2d_raw,
    validate_2dto2d_shapes,
)


class EddyDataset(Dataset):
    """
    涡旋反演数据集
    
    从 .npy 文件加载多源海洋遥感数据，并组织为 PyTorch Dataset 格式
    """
    
    def __init__(self, data_dir, normalize=False, target_filename=None):
        """
        初始化数据集
        
        Args:
            data_dir: 数据文件目录路径
            normalize: 是否进行标准化（均值0，方差1）
        """
        self.data_dir = data_dir
        self.normalize = normalize
        self.sss, self.ssh, self.target = load_2dto2d_raw(
            data_dir, target_filename=target_filename
        )
        self.num_depths = validate_2dto2d_shapes(self.sss, self.ssh, self.target)
        self.months = np.asarray(
            generate_month_numbers(DATA_START_DATE, self.sss.shape[0]), dtype=np.int64
        )
        self.fit_indices = indices_until_date(
            self.sss.shape[0], DATA_START_DATE, TRAIN_END_DATE
        )
        self.sss, self.ssh, self.target, self.norm_stats = clean_and_normalize_2dto2d(
            self.sss,
            self.ssh,
            self.target,
            normalize=self.normalize,
            months=self.months,
            fit_indices=self.fit_indices,
        )
        
        # 数据统计
        self.T, self.H, self.W = self.sss.shape
    
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
        target = torch.tensor(self.target[idx], dtype=torch.float32)
        if target.ndim == 2:
            target = target.unsqueeze(0)
        return {
            "sss": torch.tensor(self.sss[idx], dtype=torch.float32).unsqueeze(0),
            "ssh": torch.tensor(self.ssh[idx], dtype=torch.float32).unsqueeze(0),
            "target": target,
        }
    
    def get_data_info(self):
        """返回数据集信息"""
        return {
            "time_steps": self.T,
            "height": self.H,
            "width": self.W,
            "num_depths": self.num_depths,
            "sss_range": (self.sss.min(), self.sss.max()),
            "ssh_range": (self.ssh.min(), self.ssh.max()),
            "target_range": (self.target.min(), self.target.max()),
        }

    def get_norm_stats(self):
        """返回标准化统计量（用于反标准化推理结果）。"""
        return dict(self.norm_stats)
