"""
2dto2d 数据集模块
读取 SST 高分辨率分支、SSH/SSS 低分辨率分支和单变量水下标签
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta
from config import DATA_START_DATE, TRAIN_END_DATE
from datasets.date_utils import generate_month_numbers, indices_until_date
from datasets.io_2dto2d import (
    clean_and_normalize_2dto2d,
    load_2dto2d_raw,
    validate_2dto2d_shapes,
)


def _date_slice(global_start_date, total_len, start_date=None, end_date=None):
    """把闭区间日期转换为数组切片。"""
    global_start = datetime.strptime(global_start_date, "%Y-%m-%d")
    global_end = global_start + timedelta(days=int(total_len) - 1)
    start_text = start_date or global_start_date
    end_text = end_date or global_end.strftime("%Y-%m-%d")
    start = datetime.strptime(start_text, "%Y-%m-%d")
    end = datetime.strptime(end_text, "%Y-%m-%d")
    if end < start:
        raise ValueError(f"end_date 不能早于 start_date：{start_text} > {end_text}")
    if start < global_start or end > global_end:
        raise ValueError(
            f"日期范围需位于 [{global_start_date}, {global_end.strftime('%Y-%m-%d')}] 内，"
            f"实际：[{start_text}, {end_text}]"
        )
    i0 = (start - global_start).days
    i1 = (end - global_start).days + 1
    return slice(i0, i1), start_text, end_text


class Dataset2Dto2D(Dataset):
    """
    Du_Unet 双分支数据集

    返回:
        sst: (1,160,160)
        ssh_sss: (2,64,64)
        target: (25,64,64)
        target_mask: (25,64,64)
    """
    
    def __init__(
        self,
        data_dir,
        normalize=False,
        target_var="temperature",
        start_date=None,
        end_date=None,
    ):
        """
        初始化数据集
        
        Args:
            data_dir: 数据文件目录路径
            normalize: 是否进行月气候态距平归一化
            target_var: temperature 或 salinity
            start_date: 可选数据起始日期
            end_date: 可选数据结束日期
        """
        self.data_dir = data_dir
        self.normalize = normalize
        self.target_var = target_var
        self.sst, self.ssh, self.sss, self.target = load_2dto2d_raw(data_dir, target_var)
        self.num_depths = validate_2dto2d_shapes(self.sst, self.ssh, self.sss, self.target)
        time_slice, self.start_date, self.end_date = _date_slice(
            DATA_START_DATE, self.sst.shape[0], start_date=start_date, end_date=end_date
        )
        self.sst = self.sst[time_slice]
        self.ssh = self.ssh[time_slice]
        self.sss = self.sss[time_slice]
        self.target = self.target[time_slice]
        self.months = np.asarray(
            generate_month_numbers(self.start_date, self.sst.shape[0]), dtype=np.int64
        )
        self.fit_indices = indices_until_date(
            self.sst.shape[0], self.start_date, TRAIN_END_DATE
        )
        if len(self.fit_indices) == 0:
            self.fit_indices = list(range(self.sst.shape[0]))
        (
            self.sst,
            self.ssh,
            self.sss,
            self.target,
            self.target_mask,
            self.norm_stats,
        ) = clean_and_normalize_2dto2d(
            self.sst,
            self.ssh,
            self.sss,
            self.target,
            normalize=self.normalize,
            months=self.months,
            fit_indices=self.fit_indices,
            start_date=self.start_date,
        )
        
        # 数据统计
        self.T, self.sst_H, self.sst_W = self.sst.shape
        _, self.H, self.W = self.ssh.shape
    
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
        ssh_sss = np.stack([self.ssh[idx], self.sss[idx]], axis=0)
        return {
            "sst": torch.tensor(self.sst[idx], dtype=torch.float32).unsqueeze(0),
            "ssh_sss": torch.tensor(ssh_sss, dtype=torch.float32),
            "target": target,
            "target_mask": torch.tensor(self.target_mask[idx], dtype=torch.float32),
        }
    
    def get_data_info(self):
        """返回数据集信息"""
        return {
            "time_steps": self.T,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "target_var": self.target_var,
            "sst_height": self.sst_H,
            "sst_width": self.sst_W,
            "height": self.H,
            "width": self.W,
            "num_depths": self.num_depths,
            "sst_range": (self.sst.min(), self.sst.max()),
            "ssh_range": (self.ssh.min(), self.ssh.max()),
            "sss_range": (self.sss.min(), self.sss.max()),
            "target_range": (self.target.min(), self.target.max()),
        }

    def get_norm_stats(self):
        """返回标准化统计量（用于反标准化推理结果）。"""
        return dict(self.norm_stats)
