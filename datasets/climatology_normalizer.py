"""
月气候态距平归一化工具
用于把海表输入和水下目标统一处理为：减去对应月份的气候态均值，再除以训练集距平的全局标准差。
"""

import warnings

import numpy as np


class MonthlyClimatologyLayerStdNormalizer:
    """
    月气候态均值 / 全局标准差归一化器

    支持形状：
      - (T, H, W)：单个二维变量，如 SST、SSH、SSS
      - (T, D, H, W)：按深度组织的目标变量，如温度或盐度
    """

    def __init__(self, eps=1e-6, fill_value=0.0):
        self.eps = float(eps)
        self.fill_value = float(fill_value)
        self.climatology = None
        self.layer_std = None
        self.input_ndim = None

    def fit(self, array, months, fit_indices=None):
        """使用训练时段拟合月气候态均值和距平标准差。"""
        arr = self._as_float_with_nan(array)
        months = self._validate_months(months, arr.shape[0])
        self.input_ndim = arr.ndim
        if arr.ndim not in (3, 4):
            raise ValueError(f"仅支持 3D/4D 数据，实际 ndim={arr.ndim}")

        if fit_indices is None:
            fit_indices = np.arange(arr.shape[0])
        fit_indices = np.asarray(fit_indices, dtype=np.int64)
        if fit_indices.size == 0:
            raise ValueError("fit_indices 不能为空，无法拟合归一化统计量")

        fit_arr = arr[fit_indices]
        fit_months = months[fit_indices]

        # 月气候态均值按月份和空间位置计算。变换时先减掉该日所属月份的均值，
        # 这样模型学习的是相对季节背景的距平，而不是直接拟合季节循环。
        climatology = np.empty((12,) + arr.shape[1:], dtype=np.float32)
        for month in range(1, 13):
            month_data = fit_arr[fit_months == month]
            if month_data.size == 0:
                climatology[month - 1] = np.nan
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    climatology[month - 1] = np.nanmean(month_data, axis=0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fallback = np.nanmean(fit_arr, axis=0)
        missing_months = ~np.isfinite(climatology)
        climatology = np.where(missing_months, fallback[np.newaxis, ...], climatology)
        climatology = np.nan_to_num(
            climatology, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)

        anomalies = fit_arr - climatology[fit_months - 1]

        # 标准差只在训练时段的距平上计算，避免验证/测试信息泄漏。
        # 3D 输入变量使用一个全局标准差；4D 目标按深度层分别使用全局标准差，
        # 每一层的方差统计都覆盖该层所有训练日期和全部空间网格。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if arr.ndim == 3:
                layer_std = np.array([np.nanstd(anomalies)], dtype=np.float32)
            elif arr.ndim == 4:
                layer_std = np.nanstd(anomalies, axis=(0, 2, 3)).astype(np.float32)

        layer_std = np.where(
            np.isfinite(layer_std) & (layer_std >= self.eps), layer_std, 1.0
        ).astype(np.float32)

        self.climatology = climatology
        self.layer_std = layer_std
        return self

    def fit_streaming(self, array, months, fit_indices=None, chunk_size=32, label=None):
        """分块拟合月气候态和标准差，避免把大数组一次性展开到内存。"""
        months = self._validate_months(months, array.shape[0])
        self.input_ndim = len(array.shape)
        if self.input_ndim not in (3, 4):
            raise ValueError(f"仅支持 3D/4D 数据，实际 ndim={self.input_ndim}")

        if fit_indices is None:
            fit_indices = np.arange(array.shape[0])
        fit_indices = np.asarray(fit_indices, dtype=np.int64)
        if fit_indices.size == 0:
            raise ValueError("fit_indices 不能为空，无法拟合归一化统计量")

        chunk_size = max(int(chunk_size), 1)
        spatial_shape = tuple(array.shape[1:])
        sum_by_month = np.zeros((12,) + spatial_shape, dtype=np.float64)
        count_by_month = np.zeros((12,) + spatial_shape, dtype=np.float32)
        total_sum = np.zeros(spatial_shape, dtype=np.float64)
        total_count = np.zeros(spatial_shape, dtype=np.float32)

        tag = f" {label}" if label else ""
        total = int(fit_indices.size)
        print(f"[normalization]{tag} fit climatology streaming: samples={total}, chunk={chunk_size}")
        for start in range(0, total, chunk_size):
            idx = fit_indices[start : start + chunk_size]
            data = np.asarray(array[idx], dtype=np.float32)
            month_chunk = months[idx]
            valid = np.isfinite(data)
            data_valid = np.where(valid, data, 0.0)
            total_sum += data_valid.sum(axis=0, dtype=np.float64)
            total_count += valid.sum(axis=0, dtype=np.float32)
            for month in range(1, 13):
                selector = month_chunk == month
                if not np.any(selector):
                    continue
                data_m = data_valid[selector]
                valid_m = valid[selector]
                sum_by_month[month - 1] += data_m.sum(axis=0, dtype=np.float64)
                count_by_month[month - 1] += valid_m.sum(axis=0, dtype=np.float32)
            if start == 0 or start + chunk_size >= total or ((start // chunk_size + 1) % 20 == 0):
                print(f"[normalization]{tag} climatology: {min(start + chunk_size, total)}/{total}")

        fallback = np.divide(
            total_sum,
            total_count,
            out=np.zeros_like(total_sum, dtype=np.float64),
            where=total_count > 0,
        )
        climatology = np.divide(
            sum_by_month,
            count_by_month,
            out=np.full_like(sum_by_month, np.nan, dtype=np.float64),
            where=count_by_month > 0,
        )
        missing_months = ~np.isfinite(climatology)
        climatology = np.where(missing_months, fallback[np.newaxis, ...], climatology)
        climatology = np.nan_to_num(
            climatology, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)

        if self.input_ndim == 3:
            anomaly_sum = 0.0
            anomaly_sumsq = 0.0
            anomaly_count = 0.0
        else:
            depth_count = int(array.shape[1])
            anomaly_sum = np.zeros(depth_count, dtype=np.float64)
            anomaly_sumsq = np.zeros(depth_count, dtype=np.float64)
            anomaly_count = np.zeros(depth_count, dtype=np.float64)

        print(f"[normalization]{tag} fit std streaming: samples={total}, chunk={chunk_size}")
        for start in range(0, total, chunk_size):
            idx = fit_indices[start : start + chunk_size]
            data = np.asarray(array[idx], dtype=np.float32)
            month_chunk = months[idx]
            valid = np.isfinite(data)
            anomalies = data - climatology[month_chunk - 1]
            anomalies = np.where(valid, anomalies, 0.0)
            if self.input_ndim == 3:
                anomaly_sum += float(anomalies.sum(dtype=np.float64))
                anomaly_sumsq += float((anomalies * anomalies).sum(dtype=np.float64))
                anomaly_count += float(valid.sum(dtype=np.float64))
            else:
                anomaly_sum += anomalies.sum(axis=(0, 2, 3), dtype=np.float64)
                anomaly_sumsq += (anomalies * anomalies).sum(
                    axis=(0, 2, 3), dtype=np.float64
                )
                anomaly_count += valid.sum(axis=(0, 2, 3), dtype=np.float64)
            if start == 0 or start + chunk_size >= total or ((start // chunk_size + 1) % 20 == 0):
                print(f"[normalization]{tag} std: {min(start + chunk_size, total)}/{total}")

        mean = np.divide(
            anomaly_sum,
            anomaly_count,
            out=np.zeros_like(np.asarray(anomaly_sum), dtype=np.float64),
            where=np.asarray(anomaly_count) > 0,
        )
        second = np.divide(
            anomaly_sumsq,
            anomaly_count,
            out=np.zeros_like(np.asarray(anomaly_sumsq), dtype=np.float64),
            where=np.asarray(anomaly_count) > 0,
        )
        variance = np.maximum(second - mean * mean, 0.0)
        layer_std = np.sqrt(variance).astype(np.float32)
        if self.input_ndim == 3:
            layer_std = np.asarray([float(layer_std)], dtype=np.float32)
        layer_std = np.where(
            np.isfinite(layer_std) & (layer_std >= self.eps), layer_std, 1.0
        ).astype(np.float32)

        self.climatology = climatology
        self.layer_std = layer_std
        return self

    def transform(self, array, months):
        """执行距平归一化，并把无效值填成模型可接收的数值。"""
        self._check_fitted()
        arr = self._as_float_with_nan(array)
        months = self._validate_months(months, arr.shape[0])
        normalized = (arr - self.climatology[months - 1]) / self._std_broadcast(arr.ndim)
        return np.nan_to_num(
            normalized,
            nan=self.fill_value,
            posinf=self.fill_value,
            neginf=self.fill_value,
        ).astype(np.float32)

    def inverse_transform(self, array, months):
        """把归一化结果恢复到原始物理量单位。"""
        self._check_fitted()
        arr = np.asarray(array, dtype=np.float32)
        months = self._validate_months(months, arr.shape[0])
        restored = arr * self._std_broadcast(arr.ndim) + self.climatology[months - 1]
        return restored.astype(np.float32)

    def to_stats(self):
        """返回可复用的归一化统计量。"""
        self._check_fitted()
        return {
            "climatology": self.climatology.copy(),
            "layer_std": self.layer_std.copy(),
            "eps": self.eps,
            "fill_value": self.fill_value,
        }

    @classmethod
    def from_stats(cls, stats):
        """由保存的统计量重建归一化器。"""
        obj = cls(
            eps=stats.get("eps", 1e-6),
            fill_value=stats.get("fill_value", 0.0),
        )
        obj.climatology = np.asarray(stats["climatology"], dtype=np.float32)
        obj.layer_std = np.asarray(stats["layer_std"], dtype=np.float32)
        return obj

    @staticmethod
    def _as_float_with_nan(array):
        arr = np.asarray(array, dtype=np.float32)
        arr = arr.copy()
        arr[~np.isfinite(arr)] = np.nan
        return arr

    @staticmethod
    def _validate_months(months, time_len):
        months = np.asarray(months, dtype=np.int64)
        if months.shape[0] != time_len:
            raise ValueError(f"months 长度 {months.shape[0]} 与时间维 {time_len} 不一致")
        if np.any((months < 1) | (months > 12)):
            raise ValueError("months 必须是 1..12")
        return months

    def _std_broadcast(self, ndim):
        if ndim == 3:
            return float(self.layer_std.reshape(-1)[0])
        if ndim == 4:
            return self.layer_std.reshape(1, -1, 1, 1)
        raise ValueError(f"仅支持 3D/4D 数据，实际 ndim={ndim}")

    def _check_fitted(self):
        if self.climatology is None or self.layer_std is None:
            raise RuntimeError("MonthlyClimatologyLayerStdNormalizer 尚未 fit")
