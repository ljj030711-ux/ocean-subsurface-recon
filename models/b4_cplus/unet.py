"""C+ native-grid encoders and shared multi-scale U-Net decoder."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.b4_cplus.regrid import CoordinateFeatureRegrid


def _group_count(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.body(x) + self.shortcut(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size not in {3, 4}:
            raise ValueError("DownBlock kernel_size 仅支持3或4")
        padding = 1
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class CPlusSurfaceEncoder(nn.Module):
    """Encode native grids, regrid once at 1/2 target scale, then fuse."""

    def __init__(
        self,
        embed_dim=128,
        native_base_channels=32,
        fusion_channels=64,
        lat_range=(0.0, 24.0),
        lon_range=(105.0, 121.0),
        source_lat_descending=False,
    ):
        super().__init__()
        embed_dim = int(embed_dim)
        native_base_channels = int(native_base_channels)
        fusion_channels = int(fusion_channels)
        if min(embed_dim, native_base_channels, fusion_channels) <= 0:
            raise ValueError("C+ Encoder 通道数必须为正数")
        if fusion_channels > embed_dim:
            raise ValueError("fusion_channels 不能大于 embed_dim")
        self.embed_dim = embed_dim
        self.native_base_channels = native_base_channels
        self.fusion_channels = fusion_channels

        # A 4x4/stride-2 convolution centres the 240x160 features at the
        # midpoint of each adjacent pair of 0.05-degree source cells. Their
        # coordinates therefore match a regular 0.1-degree cell-centred grid.
        self.sst_native = nn.Sequential(
            DownBlock(1, native_base_channels, kernel_size=4),
            ResidualConvBlock(native_base_channels, fusion_channels),
        )
        # SSH/SSS are already closer to the 1/2 target grid and retain their
        # 0.125-degree native resolution before the single coordinate regrid.
        self.surface_native = nn.Sequential(
            ResidualConvBlock(2, native_base_channels),
            ResidualConvBlock(native_base_channels, fusion_channels),
        )
        self.sst_regrid = CoordinateFeatureRegrid(
            lat_range,
            lon_range,
            source_cell_centered=True,
            source_lat_descending=source_lat_descending,
        )
        self.surface_regrid = CoordinateFeatureRegrid(
            lat_range,
            lon_range,
            source_cell_centered=True,
            source_lat_descending=source_lat_descending,
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(
                fusion_channels * 2,
                fusion_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(fusion_channels), fusion_channels),
            nn.GELU(),
            ResidualConvBlock(fusion_channels, fusion_channels),
        )
        self.shared_down = nn.ModuleList(
            [
                DownBlock(fusion_channels, embed_dim),
                DownBlock(embed_dim, embed_dim),
            ]
        )
        self.skip_channels = (fusion_channels, embed_dim)

    @staticmethod
    def fusion_size(output_size):
        return tuple(math.ceil(int(value) / 2) for value in output_size)

    def forward(self, sst, ssh_sss, output_size):
        fusion_size = self.fusion_size(output_size)
        sst_feature = self.sst_regrid(self.sst_native(sst), fusion_size)
        surface_feature = self.surface_regrid(
            self.surface_native(ssh_sss), fusion_size
        )
        feature = self.fusion(torch.cat([sst_feature, surface_feature], dim=1))
        skips = [feature]
        for index, block in enumerate(self.shared_down):
            feature = block(feature)
            if index < len(self.shared_down) - 1:
                skips.append(feature)
        return feature, skips


class CPlusUNetDecoder(nn.Module):
    """Two skip-connected stages plus one skip-free full-resolution stage."""

    def __init__(self, embed_dim, skip_channels, output_channels):
        super().__init__()
        reversed_skips = tuple(reversed(tuple(int(v) for v in skip_channels)))
        self.skip_channels = tuple(int(v) for v in skip_channels)
        self.up_projections = nn.ModuleList()
        self.decode_blocks = nn.ModuleList()
        self.scale_heads = nn.ModuleList()
        current_channels = int(embed_dim)
        for skip_channels_at_level in reversed_skips:
            self.up_projections.append(
                nn.Conv2d(
                    current_channels,
                    skip_channels_at_level,
                    kernel_size=1,
                    bias=False,
                )
            )
            self.decode_blocks.append(
                ResidualConvBlock(
                    skip_channels_at_level * 2, skip_channels_at_level
                )
            )
            self.scale_heads.append(
                nn.Conv2d(skip_channels_at_level, 1, kernel_size=1)
            )
            current_channels = skip_channels_at_level

        self.final_projection = nn.Conv2d(
            current_channels, int(output_channels), kernel_size=1, bias=False
        )
        self.final_block = ResidualConvBlock(
            int(output_channels), int(output_channels)
        )
        self.final_head = nn.Conv2d(int(output_channels), 1, kernel_size=1)
        self.scale_logits = nn.Parameter(torch.zeros(len(skip_channels) + 1))

    @staticmethod
    def _repeat_for_depth(feature, num_depths):
        batch, channels, height, width = feature.shape
        return (
            feature.unsqueeze(1)
            .expand(batch, num_depths, channels, height, width)
            .reshape(batch * num_depths, channels, height, width)
        )

    def forward(self, bottleneck, skips, output_size):
        if bottleneck.ndim != 5:
            raise ValueError(
                "C+ bottleneck 应为 (B,D,C,H,W)，"
                f"实际 {tuple(bottleneck.shape)}"
            )
        if len(skips) != len(self.skip_channels):
            raise ValueError(
                f"C+ 跳连数量应为 {len(self.skip_channels)}，实际 {len(skips)}"
            )
        batch, num_depths, channels, height, width = bottleneck.shape
        feature = bottleneck.reshape(batch * num_depths, channels, height, width)
        predictions = []
        for projection, block, head, skip in zip(
            self.up_projections,
            self.decode_blocks,
            self.scale_heads,
            reversed(skips),
        ):
            feature = F.interpolate(
                feature,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            feature = projection(feature)
            feature = block(
                torch.cat(
                    [feature, self._repeat_for_depth(skip, num_depths)], dim=1
                )
            )
            predictions.append(head(feature))

        feature = F.interpolate(
            feature,
            size=tuple(output_size),
            mode="bilinear",
            align_corners=False,
        )
        feature = self.final_block(self.final_projection(feature))
        predictions.append(self.final_head(feature))

        weights = torch.softmax(self.scale_logits, dim=0)
        fused = 0.0
        for weight, prediction in zip(weights, predictions):
            if prediction.shape[-2:] != tuple(output_size):
                prediction = F.interpolate(
                    prediction,
                    size=tuple(output_size),
                    mode="bilinear",
                    align_corners=False,
                )
            fused = fused + weight * prediction
        return fused.reshape(batch, num_depths, *tuple(output_size))
