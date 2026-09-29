"""Coordinate-aware feature regridding for the fixed regional grids."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateFeatureRegrid(nn.Module):
    """Map a rectilinear source feature grid to a target point grid.

    The satellite products are treated as cell-centred grids, while GLORYS is
    represented by coordinates including the configured regional endpoints.
    The sampling grid is computed from physical latitude/longitude rather than
    from array-size ratios. It has no trainable parameters and remains fully
    differentiable with respect to the source features.
    """

    def __init__(
        self,
        lat_range=(0.0, 24.0),
        lon_range=(105.0, 121.0),
        *,
        source_cell_centered=True,
        source_lat_descending=False,
    ):
        super().__init__()
        self.lat_range = tuple(float(value) for value in lat_range)
        self.lon_range = tuple(float(value) for value in lon_range)
        self.source_cell_centered = bool(source_cell_centered)
        self.source_lat_descending = bool(source_lat_descending)
        if self.lat_range[1] <= self.lat_range[0]:
            raise ValueError("lat_range 必须严格递增")
        if self.lon_range[1] <= self.lon_range[0]:
            raise ValueError("lon_range 必须严格递增")

    @staticmethod
    def _source_limits(value_range, size, cell_centered):
        lower, upper = value_range
        if not cell_centered:
            return lower, upper
        spacing = (upper - lower) / float(size)
        return lower + spacing / 2.0, upper - spacing / 2.0

    @staticmethod
    def _normalize(values, lower, upper):
        if upper <= lower:
            return torch.zeros_like(values)
        return 2.0 * (values - lower) / (upper - lower) - 1.0

    def forward(self, feature, output_size):
        if feature.ndim != 4:
            raise ValueError(
                f"重网格输入应为 (B,C,H,W)，实际 {tuple(feature.shape)}"
            )
        output_h, output_w = (int(value) for value in output_size)
        if min(output_h, output_w) <= 0:
            raise ValueError(f"output_size 必须为正数，实际 {output_size}")
        source_h, source_w = feature.shape[-2:]
        dtype = feature.dtype
        device = feature.device

        target_lat = torch.linspace(
            self.lat_range[0],
            self.lat_range[1],
            output_h,
            dtype=dtype,
            device=device,
        )
        target_lon = torch.linspace(
            self.lon_range[0],
            self.lon_range[1],
            output_w,
            dtype=dtype,
            device=device,
        )
        source_lat_min, source_lat_max = self._source_limits(
            self.lat_range, source_h, self.source_cell_centered
        )
        source_lon_min, source_lon_max = self._source_limits(
            self.lon_range, source_w, self.source_cell_centered
        )
        normalized_y = self._normalize(
            target_lat, source_lat_min, source_lat_max
        )
        if self.source_lat_descending:
            normalized_y = -normalized_y
        normalized_x = self._normalize(
            target_lon, source_lon_min, source_lon_max
        )
        grid_y, grid_x = torch.meshgrid(
            normalized_y, normalized_x, indexing="ij"
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        grid = grid.expand(feature.shape[0], output_h, output_w, 2)
        return F.grid_sample(
            feature,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def config_dict(self):
        return {
            "lat_range": list(self.lat_range),
            "lon_range": list(self.lon_range),
            "source_cell_centered": self.source_cell_centered,
            "source_lat_descending": self.source_lat_descending,
            "target_coordinates": "regular_GLORYS_endpoints",
            "mode": "bilinear_grid_sample",
            "padding_mode": "border",
        }
