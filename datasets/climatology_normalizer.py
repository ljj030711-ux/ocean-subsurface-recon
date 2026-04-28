"""Monthly climatology anomaly normalization utilities."""

import warnings

import numpy as np


class MonthlyClimatologyLayerStdNormalizer:
    """
    Normalize ocean fields by monthly climatology and layer-wise global std.

    Supported shapes:
      - (T, H, W): one 2D variable
      - (T, D, H, W): depth/channel-wise field
      - (T, D, H, W, V): depth and variable-wise field
    """

    def __init__(self, eps=1e-6, fill_value=0.0):
        self.eps = float(eps)
        self.fill_value = float(fill_value)
        self.climatology = None
        self.layer_std = None
        self.input_ndim = None

    def fit(self, array, months, fit_indices=None):
        """Fit monthly climatology and layer-wise std from selected times."""
        arr = self._as_float_with_nan(array)
        months = self._validate_months(months, arr.shape[0])
        self.input_ndim = arr.ndim

        if fit_indices is None:
            fit_indices = np.arange(arr.shape[0])
        fit_indices = np.asarray(fit_indices, dtype=np.int64)
        if fit_indices.size == 0:
            raise ValueError("fit_indices 不能为空，无法拟合归一化统计量")

        fit_arr = arr[fit_indices]
        fit_months = months[fit_indices]

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if arr.ndim == 3:
                layer_std = np.array([np.nanstd(anomalies)], dtype=np.float32)
            elif arr.ndim == 4:
                layer_std = np.nanstd(anomalies, axis=(0, 2, 3)).astype(np.float32)
            elif arr.ndim == 5:
                layer_std = np.nanstd(anomalies, axis=(0, 2, 3)).astype(np.float32)
            else:
                raise ValueError(f"仅支持 3D/4D/5D 数据，实际 ndim={arr.ndim}")

        layer_std = np.where(
            np.isfinite(layer_std) & (layer_std >= self.eps), layer_std, 1.0
        ).astype(np.float32)

        self.climatology = climatology
        self.layer_std = layer_std
        return self

    def transform(self, array, months):
        """Apply anomaly normalization and fill invalid values for model input."""
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
        """Restore normalized data to original physical units."""
        self._check_fitted()
        arr = np.asarray(array, dtype=np.float32)
        months = self._validate_months(months, arr.shape[0])
        restored = arr * self._std_broadcast(arr.ndim) + self.climatology[months - 1]
        return restored.astype(np.float32)

    def to_stats(self):
        """Return serializable normalization statistics."""
        self._check_fitted()
        return {
            "climatology": self.climatology.copy(),
            "layer_std": self.layer_std.copy(),
            "eps": self.eps,
            "fill_value": self.fill_value,
        }

    @classmethod
    def from_stats(cls, stats):
        """Build a normalizer from statistics returned by to_stats()."""
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
        if ndim == 5:
            return self.layer_std.reshape(1, self.layer_std.shape[0], 1, 1, -1)
        raise ValueError(f"仅支持 3D/4D/5D 数据，实际 ndim={ndim}")

    def _check_fitted(self):
        if self.climatology is None or self.layer_std is None:
            raise RuntimeError("MonthlyClimatologyLayerStdNormalizer 尚未 fit")
