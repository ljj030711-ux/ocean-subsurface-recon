"""Spatial Vision Transformer bottleneck for dense ocean-field reconstruction."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialViTBottleneck(nn.Module):
    """Encode a 2-D feature map as spatial tokens and restore its original shape.

    The module intentionally has no class token.  A learnable 2-D positional map
    is interpolated to the runtime patch grid, so the same checkpoint can handle
    both dataset profiles.  The restored Transformer feature is fused with the
    original CNN feature through a residual 3x3 convolution.
    """

    def __init__(
        self,
        in_channels,
        embed_dim,
        patch_size,
        num_heads,
        num_layers,
        mlp_ratio=4,
        dropout=0.1,
        base_grid_size=(16, 16),
        max_tokens=None,
    ):
        super().__init__()
        in_channels = int(in_channels)
        embed_dim = int(embed_dim)
        patch_size = int(patch_size)
        num_heads = int(num_heads)
        num_layers = int(num_layers)
        mlp_ratio = int(mlp_ratio)
        if min(in_channels, embed_dim, patch_size, num_heads, num_layers, mlp_ratio) <= 0:
            raise ValueError("SpatialViTBottleneck 的通道数、Patch和层数参数必须为正数")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} 必须能被 num_heads={num_heads} 整除"
            )

        base_h, base_w = (int(value) for value in base_grid_size)
        if base_h <= 0 or base_w <= 0:
            raise ValueError("base_grid_size 必须为正数")

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = mlp_ratio
        self.dropout = float(dropout)
        self.base_grid_size = (base_h, base_w)
        self.max_tokens = None if max_tokens is None else int(max_tokens)
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens 必须为正数或None")

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, embed_dim, base_h, base_w)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )
        self.restore = nn.ConvTranspose2d(
            embed_dim,
            in_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.local_fusion = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
        )
        self.activation = nn.ReLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _position_for_grid(self, height, width, dtype, device):
        position = self.position_embedding
        if position.shape[-2:] != (height, width):
            position = F.interpolate(
                position,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )
        return position.to(device=device, dtype=dtype)

    def _limit_token_grid(self, tokens_2d):
        """Bound global-attention memory while preserving the grid aspect ratio."""
        height, width = tokens_2d.shape[-2:]
        if self.max_tokens is None or height * width <= self.max_tokens:
            return tokens_2d
        scale = math.sqrt(self.max_tokens / float(height * width))
        target_h = max(1, int(math.floor(height * scale)))
        target_w = max(1, int(math.floor(width * scale)))
        while target_h * target_w > self.max_tokens:
            if target_h >= target_w and target_h > 1:
                target_h -= 1
            elif target_w > 1:
                target_w -= 1
            else:
                break
        return F.adaptive_avg_pool2d(tokens_2d, (target_h, target_w))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                "SpatialViTBottleneck 输入应为 "
                f"(B,{self.in_channels},H,W)，实际为 {tuple(x.shape)}"
            )
        input_size = x.shape[-2:]
        tokens_2d = self._limit_token_grid(self.patch_embed(x))
        patch_h, patch_w = tokens_2d.shape[-2:]
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError(
                f"输入尺寸 {input_size} 小于 patch_size={self.patch_size}"
            )
        tokens_2d = tokens_2d + self._position_for_grid(
            patch_h,
            patch_w,
            tokens_2d.dtype,
            tokens_2d.device,
        )
        tokens = tokens_2d.flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens)
        encoded = tokens.transpose(1, 2).reshape(
            x.shape[0], self.embed_dim, patch_h, patch_w
        )
        restored = self.restore(encoded)
        if restored.shape[-2:] != input_size:
            restored = F.interpolate(
                restored,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
        return self.activation(x + self.local_fusion(restored))

    def config_dict(self):
        return {
            "in_channels": self.in_channels,
            "embed_dim": self.embed_dim,
            "patch_size": self.patch_size,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "base_grid_size": list(self.base_grid_size),
            "max_tokens": self.max_tokens,
            "position_embedding": "learnable_2d_interpolated",
            "class_token": False,
            "norm": "pre_layer_norm",
            "activation": "gelu",
            "fusion": "cnn_residual_3x3",
        }
