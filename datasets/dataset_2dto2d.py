"""
2dto2d 数据集模块
读取 SST 高分辨率分支、SSH/SSS 低分辨率分支和单变量水下标签
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta
from config import DATA_START_DATE, DEFAULT_DATASET_PROFILE, TRAIN_END_DATE
from datasets.climatology_normalizer import MonthlyClimatologyLayerStdNormalizer
from datasets.date_utils import generate_month_numbers, indices_until_date
from datasets.io_2dto2d import (
    apply_sst_unit_conversion,
    clean_and_normalize_2dto2d,
    detect_sst_unit_conversion,
    load_2dto2d_raw,
    validate_2dto2d_shapes,
)
from datasets.normalization_stats import (
    NORMALIZATION_METHOD,
    build_normalization_metadata,
    default_normalization_stats_path,
    load_normalization_stats,
    normalization_reference,
    save_normalization_stats,
    validate_normalization_stats,
)
from utils.data_quality import sanitize_with_value


class _LazySSTCelsiusView:
    """按块把 Kelvin SST 视作 Celsius，避免生成完整副本。"""

    def __init__(self, array):
        self.array = array
        self.shape = array.shape

    def __getitem__(self, item):
        return np.asarray(self.array[item], dtype=np.float32) - np.float32(273.15)


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
        sst: (1,H_sst,W_sst)
        ssh_sss: (2,H_surface,W_surface)
        target: (D,H_target,W_target)
        target_mask: (D,H_target,W_target)
    """
    
    def __init__(
        self,
        data_dir,
        normalize=False,
        target_var="temperature",
        start_date=None,
        end_date=None,
        quality_report=False,
        dataset_profile=DEFAULT_DATASET_PROFILE,
        lazy=False,
        normalization_chunk_size=32,
        normalization_stats_path=None,
        rebuild_normalization_stats=False,
    ):
        """
        初始化数据集
        
        Args:
            data_dir: 数据文件目录路径
            normalize: 是否进行月气候态距平归一化
            target_var: temperature 或 salinity
            start_date: 可选数据起始日期
            end_date: 可选数据结束日期
            quality_report: 是否生成缺失率质检统计图
            dataset_profile: 数据集配置名，默认 legacy_small
            lazy: 是否按样本延迟清洗/归一化，适合大数据集
            normalization_chunk_size: lazy 归一化统计量拟合的时间分块大小
            normalization_stats_path: 可复用的归一化统计量 .npz 路径
            rebuild_normalization_stats: 忽略已有统计文件并显式重算
        """
        self.data_dir = data_dir
        self.normalize = normalize
        self.target_var = target_var
        self.dataset_profile = dataset_profile
        self.lazy = bool(lazy)
        self.normalization_chunk_size = max(int(normalization_chunk_size), 1)
        self.rebuild_normalization_stats = bool(rebuild_normalization_stats)
        self.normalization_stats_loaded = False
        self.normalization_metadata = None
        self.sst, self.ssh, self.sss, self.target = load_2dto2d_raw(
            data_dir, target_var, dataset_profile=dataset_profile
        )
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
        self.normalization_fit_start_date = self._date_for_index(
            int(self.fit_indices[0])
        )
        self.normalization_fit_end_date = self._date_for_index(
            int(self.fit_indices[-1])
        )
        if self.normalize:
            stats_path = normalization_stats_path or default_normalization_stats_path(
                self.data_dir,
                self.dataset_profile,
                self.target_var,
                self.normalization_fit_start_date,
                self.normalization_fit_end_date,
            )
            self.normalization_stats_path = os.path.abspath(stats_path)
        else:
            self.normalization_stats_path = None
        if self.lazy:
            self._init_lazy_normalization(quality_report=quality_report)
        else:
            sst_unit_conversion = detect_sst_unit_conversion(self.sst)
            cached_stats = self._load_cached_normalization_stats(
                sst_unit_conversion
            )
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
                quality_report=quality_report,
                normalization_stats=cached_stats,
                sst_unit_conversion=sst_unit_conversion,
            )
            if self.normalize:
                if cached_stats is None:
                    self._save_fitted_normalization_stats()
                self._attach_normalization_reference()
        
        # 数据统计
        self.T, self.sst_H, self.sst_W = self.sst.shape
        _, self.surface_H, self.surface_W = self.ssh.shape
        _, _, self.target_H, self.target_W = self.target.shape
        self.H, self.W = self.target_H, self.target_W
    
    def __len__(self):
        """返回数据集大小"""
        return self.T

    def _date_for_index(self, idx):
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        return (start + timedelta(days=int(idx))).strftime("%Y-%m-%d")

    def _expected_normalization_metadata(self, sst_unit_conversion):
        return build_normalization_metadata(
            dataset_profile=self.dataset_profile,
            target_var=self.target_var,
            fit_start_date=self.normalization_fit_start_date,
            fit_end_date=self.normalization_fit_end_date,
            fit_sample_count=len(self.fit_indices),
            arrays={
                "sst": self.sst,
                "ssh": self.ssh,
                "sss": self.sss,
                "target": self.target,
            },
            sst_unit_conversion=sst_unit_conversion,
        )

    def _load_cached_normalization_stats(self, sst_unit_conversion):
        if not self.normalize:
            return None
        path = self.normalization_stats_path
        expected = self._expected_normalization_metadata(sst_unit_conversion)
        self._normalization_expected_metadata = expected
        if self.rebuild_normalization_stats:
            if os.path.exists(path):
                print(f"[normalization] 显式重算，将覆盖已有统计文件: {path}")
            return None
        if not os.path.exists(path):
            print(f"[normalization] 未找到统计文件，首次拟合后将保存: {path}")
            return None

        stats, metadata = load_normalization_stats(path)
        validate_normalization_stats(stats, metadata, expected, path)
        self.normalization_stats_loaded = True
        self.normalization_metadata = metadata
        print(f"[normalization] 已加载统计文件，跳过拟合: {path}")
        return stats

    def _save_fitted_normalization_stats(self):
        metadata = getattr(self, "_normalization_expected_metadata", None)
        if metadata is None:
            metadata = self._expected_normalization_metadata(
                self.norm_stats["sst_unit_conversion"]
            )
        self.normalization_metadata = save_normalization_stats(
            self.normalization_stats_path,
            self.norm_stats,
            metadata,
        )
        self.normalization_stats_loaded = False
        print(f"[normalization] 统计文件已保存: {self.normalization_stats_path}")

    def _attach_normalization_reference(self):
        if not self.normalize or self.normalization_metadata is None:
            return
        self.norm_stats["artifact"] = normalization_reference(
            self.normalization_stats_path, self.normalization_metadata
        )

    def _set_normalizers_from_stats(self, stats):
        self._sst_norm = MonthlyClimatologyLayerStdNormalizer.from_stats(stats["sst"])
        self._ssh_norm = MonthlyClimatologyLayerStdNormalizer.from_stats(stats["ssh"])
        self._sss_norm = MonthlyClimatologyLayerStdNormalizer.from_stats(stats["sss"])
        self._target_norm = MonthlyClimatologyLayerStdNormalizer.from_stats(
            stats["target"]
        )

    def _init_lazy_normalization(self, quality_report=False):
        if quality_report:
            raise ValueError("lazy=True 时暂不支持 quality_report；请关闭 --quality-report")

        self.sst_unit_conversion = detect_sst_unit_conversion(self.sst)
        if not self.normalize:
            self.norm_stats = {
                "normalization": "none",
                "sst_unit": "degree_C",
                "sst_unit_conversion": self.sst_unit_conversion,
                "lazy": True,
            }
            self._sst_norm = None
            self._ssh_norm = None
            self._sss_norm = None
            self._target_norm = None
            return

        cached_stats = self._load_cached_normalization_stats(
            self.sst_unit_conversion
        )
        if cached_stats is not None:
            self._set_normalizers_from_stats(cached_stats)
            self.norm_stats = dict(cached_stats)
            self.norm_stats["lazy"] = True
            self._attach_normalization_reference()
            return

        self._sst_norm = MonthlyClimatologyLayerStdNormalizer().fit_streaming(
            self._sst_for_lazy_fit(),
            self.months,
            fit_indices=self.fit_indices,
            chunk_size=self.normalization_chunk_size,
            label="sst",
        )
        self._ssh_norm = MonthlyClimatologyLayerStdNormalizer().fit_streaming(
            self.ssh,
            self.months,
            fit_indices=self.fit_indices,
            chunk_size=self.normalization_chunk_size,
            label="ssh",
        )
        self._sss_norm = MonthlyClimatologyLayerStdNormalizer().fit_streaming(
            self.sss,
            self.months,
            fit_indices=self.fit_indices,
            chunk_size=self.normalization_chunk_size,
            label="sss",
        )
        self._target_norm = MonthlyClimatologyLayerStdNormalizer().fit_streaming(
            self.target,
            self.months,
            fit_indices=self.fit_indices,
            chunk_size=self.normalization_chunk_size,
            label="target",
        )
        self.norm_stats = {
            "normalization": NORMALIZATION_METHOD,
            "sst_unit": "degree_C",
            "sst_unit_conversion": self.sst_unit_conversion,
            "lazy": True,
            "sst": self._sst_norm.to_stats(),
            "ssh": self._ssh_norm.to_stats(),
            "sss": self._sss_norm.to_stats(),
            "target": self._target_norm.to_stats(),
        }
        self._save_fitted_normalization_stats()
        self._attach_normalization_reference()

    def _sst_for_lazy_fit(self):
        if self.sst_unit_conversion != "kelvin_to_celsius":
            return self.sst
        return _LazySSTCelsiusView(self.sst)

    def _prepare_lazy_sample(self, idx, depth_idx=None):
        month = np.asarray([self.months[idx]], dtype=np.int64)
        sst = apply_sst_unit_conversion(self.sst[idx], self.sst_unit_conversion)
        ssh = np.asarray(self.ssh[idx], dtype=np.float32)
        sss = np.asarray(self.sss[idx], dtype=np.float32)
        target_selector = slice(None) if depth_idx is None else slice(depth_idx, depth_idx + 1)
        target = np.asarray(self.target[idx, target_selector], dtype=np.float32)
        target_mask = np.isfinite(target).astype(np.float32)

        if self.normalize:
            sst = self._sst_norm.transform(sst[np.newaxis, ...], month)[0]
            ssh = self._ssh_norm.transform(ssh[np.newaxis, ...], month)[0]
            sss = self._sss_norm.transform(sss[np.newaxis, ...], month)[0]
            if depth_idx is None:
                target = self._target_norm.transform(target[np.newaxis, ...], month)[0]
            else:
                climatology = self._target_norm.climatology[
                    month[0] - 1, target_selector
                ]
                layer_std = np.asarray(
                    self._target_norm.layer_std[target_selector], dtype=np.float32
                ).reshape(-1, 1, 1)
                target = np.nan_to_num(
                    (target - climatology) / layer_std,
                    nan=self._target_norm.fill_value,
                    posinf=self._target_norm.fill_value,
                    neginf=self._target_norm.fill_value,
                ).astype(np.float32)
        else:
            sst = sanitize_with_value(sst, fill_value=0.0)
            ssh = sanitize_with_value(ssh, fill_value=0.0)
            sss = sanitize_with_value(sss, fill_value=0.0)
            target = sanitize_with_value(target, fill_value=0.0)
        return sst, ssh, sss, target, target_mask
    
    @staticmethod
    def _to_tensor_sample(sst, ssh, sss, target, target_mask):
        target = torch.tensor(target, dtype=torch.float32)
        ssh_sss = np.stack([ssh, sss], axis=0)
        return {
            "sst": torch.tensor(sst, dtype=torch.float32).unsqueeze(0),
            "ssh_sss": torch.tensor(ssh_sss, dtype=torch.float32),
            "target": target,
            "target_mask": torch.tensor(target_mask, dtype=torch.float32),
        }

    def get_depth_item(self, idx, depth_idx):
        """读取单个深度目标，避免逐层训练时处理完整三维 target。"""
        if depth_idx < 0 or depth_idx >= self.num_depths:
            raise IndexError(f"depth_idx 越界: {depth_idx}")
        if self.lazy:
            sample = self._prepare_lazy_sample(idx, depth_idx=depth_idx)
        else:
            sample = (
                self.sst[idx],
                self.ssh[idx],
                self.sss[idx],
                self.target[idx, depth_idx : depth_idx + 1],
                self.target_mask[idx, depth_idx : depth_idx + 1],
            )
        return self._to_tensor_sample(*sample)

    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 时间索引
        
        Returns:
            dict: 包含各个变量的字典
        """
        if self.lazy:
            sst, ssh, sss, target, target_mask = self._prepare_lazy_sample(idx)
        else:
            sst = self.sst[idx]
            ssh = self.ssh[idx]
            sss = self.sss[idx]
            target = self.target[idx]
            target_mask = self.target_mask[idx]
        return self._to_tensor_sample(sst, ssh, sss, target, target_mask)
    
    def get_data_info(self, include_ranges=False):
        """返回数据集信息"""
        info = {
            "time_steps": self.T,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "target_var": self.target_var,
            "dataset_profile": self.dataset_profile,
            "lazy": self.lazy,
            "normalize": self.normalize,
            "normalization_stats_path": self.normalization_stats_path,
            "normalization_stats_loaded": self.normalization_stats_loaded,
            "sst_height": self.sst_H,
            "sst_width": self.sst_W,
            "surface_height": self.surface_H,
            "surface_width": self.surface_W,
            "target_height": self.target_H,
            "target_width": self.target_W,
            "height": self.H,
            "width": self.W,
            "num_depths": self.num_depths,
            "surface_target_same_grid": (
                self.surface_H == self.target_H and self.surface_W == self.target_W
            ),
        }
        if include_ranges:
            info.update(
                {
                    "sst_range": (self.sst.min(), self.sst.max()),
                    "ssh_range": (self.ssh.min(), self.ssh.max()),
                    "sss_range": (self.sss.min(), self.sss.max()),
                    "target_range": (self.target.min(), self.target.max()),
                }
            )
        return info

    def get_norm_stats(self):
        """返回标准化统计量（用于反标准化推理结果）。"""
        return dict(self.norm_stats)

    def get_normalization_reference(self):
        """返回 checkpoint/cache 可记录的统计文件引用。"""
        if not self.normalize or self.normalization_metadata is None:
            return None
        return normalization_reference(
            self.normalization_stats_path, self.normalization_metadata
        )
