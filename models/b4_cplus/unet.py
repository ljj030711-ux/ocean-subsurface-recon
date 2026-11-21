"""C+ native-grid encoders and shared multi-scale U-Net decoder."""

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
    """Extract native-grid features and build a full-resolution skip pyramid."""

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

        # Keep both branches at native resolution before coordinate regridding.
        self.sst_native = nn.Sequential(
            ResidualConvBlock(1, native_base_channels),
            ResidualConvBlock(native_base_channels, fusion_channels),
        )
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
                DownBlock(fusion_channels, fusion_channels),
                DownBlock(fusion_channels, embed_dim),
                DownBlock(embed_dim, embed_dim),
            ]
        )
        self.skip_channels = (fusion_channels, fusion_channels, embed_dim)

    def forward(self, sst, ssh_sss, output_size):
        output_size = tuple(int(value) for value in output_size)
        sst_feature = self.sst_regrid(self.sst_native(sst), output_size)
        surface_feature = self.surface_regrid(
            self.surface_native(ssh_sss), output_size
        )
        feature = self.fusion(torch.cat([sst_feature, surface_feature], dim=1))
        skips = [feature]
        for index, block in enumerate(self.shared_down):
            feature = block(feature)
            if index < len(self.shared_down) - 1:
                skips.append(feature)
        return feature, skips


class CPlusUNetDecoder(nn.Module):
    """Fuse three pyramid skips before one full-resolution temperature head."""

    def __init__(self, embed_dim, skip_channels, output_channels):
        super().__init__()
        reversed_skips = tuple(reversed(tuple(int(v) for v in skip_channels)))
        self.skip_channels = tuple(int(v) for v in skip_channels)
        self.up_projections = nn.ModuleList()
        self.decode_blocks = nn.ModuleList()
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
            current_channels = skip_channels_at_level

        if len(reversed_skips) != 3:
            raise ValueError("C+ 全分辨率解码器需要三条 skip")
        self.final_projection = nn.Conv2d(
            current_channels, int(output_channels), kernel_size=1, bias=False
        )
        self.final_block = ResidualConvBlock(
            int(output_channels), int(output_channels)
        )
        self.final_head = nn.Conv2d(int(output_channels), 1, kernel_size=1)

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
        for projection, block, skip in zip(
            self.up_projections, self.decode_blocks, reversed(skips)
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
        if tuple(feature.shape[-2:]) != tuple(output_size):
            raise ValueError("最终 skip 尺寸与 output_size 不一致")
        feature = self.final_block(self.final_projection(feature))
        return self.final_head(feature).reshape(
            batch, num_depths, *tuple(output_size)
        )
