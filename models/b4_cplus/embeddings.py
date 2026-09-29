"""Embeddings owned by the standalone B4 C+ model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding2D(nn.Module):
    """Convert a field into a padded non-overlapping patch grid."""

    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        if min(self.in_channels, self.embed_dim, self.patch_size) <= 0:
            raise ValueError("PatchEmbedding2D 参数必须为正数")
        self.projection = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def grid_size(self, height, width):
        return (
            math.ceil(int(height) / self.patch_size),
            math.ceil(int(width) / self.patch_size),
        )

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"PatchEmbedding2D 输入应为 (B,{self.in_channels},H,W)，"
                f"实际为 {tuple(x.shape)}"
            )
        height, width = x.shape[-2:]
        grid_h, grid_w = self.grid_size(height, width)
        pad_h = grid_h * self.patch_size - height
        pad_w = grid_w * self.patch_size - width
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return self.projection(x)


class Interpolated2DPositionEmbedding(nn.Module):
    """Learnable 2-D position map interpolated to the runtime grid."""

    def __init__(self, embed_dim, base_grid_size=(16, 16)):
        super().__init__()
        base_h, base_w = (int(value) for value in base_grid_size)
        if min(int(embed_dim), base_h, base_w) <= 0:
            raise ValueError("位置编码维度和基础网格必须为正数")
        self.embedding = nn.Parameter(
            torch.zeros(1, int(embed_dim), base_h, base_w)
        )
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def forward(self, height, width, *, dtype, device):
        position = self.embedding
        if position.shape[-2:] != (height, width):
            position = F.interpolate(
                position,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )
        return position.to(device=device, dtype=dtype).flatten(2).transpose(1, 2)


class ContinuousDepthEmbedding(nn.Module):
    """Embed the real, irregular GLORYS depth coordinates."""

    def __init__(self, depth_values, embed_dim):
        super().__init__()
        depths = torch.as_tensor(depth_values, dtype=torch.float32).reshape(-1)
        if depths.numel() == 0:
            raise ValueError("depth_values 不能为空")
        if not torch.isfinite(depths).all() or torch.any(depths < 0):
            raise ValueError("depth_values 必须是有限的非负数")
        if depths.numel() > 1 and not torch.all(depths[1:] > depths[:-1]):
            raise ValueError("B4 自回归深度必须按从浅到深严格递增")
        self.register_buffer("depth_values", depths, persistent=True)
        self.network = nn.Sequential(
            nn.Linear(1, int(embed_dim)),
            nn.GELU(),
            nn.Linear(int(embed_dim), int(embed_dim)),
        )

    def forward(self, *, dtype, device):
        depth = self.depth_values.to(device=device, dtype=dtype)
        scale = torch.log1p(depth.max()).clamp_min(torch.finfo(dtype).eps)
        normalized = (torch.log1p(depth) / scale).unsqueeze(-1)
        return self.network(normalized)
