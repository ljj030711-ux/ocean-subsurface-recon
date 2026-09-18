"""Causal depth decoder and patch-fold reconstruction for B3DepthFormer."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDepthDecoderLayer(nn.Module):
    """One pre-norm decoder layer operating on depth and surface memory.

    Depth self-attention is applied independently at every spatial patch. Cross
    attention is then applied independently at every generated depth, with all
    surface patches available as memory.
    """

    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout):
        super().__init__()
        self.depth_norm = nn.LayerNorm(embed_dim)
        self.depth_attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(embed_dim)
        self.memory_norm = nn.LayerNorm(embed_dim)
        self.surface_attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory, causal_mask):
        """
        Args:
            query: ``(B,D,N,d)`` decoder tokens.
            memory: ``(B,N,d)`` encoded SST/SSH/SSS surface memory.
            causal_mask: bool ``(D,D)`` mask, where True entries are blocked.
        """
        if query.ndim != 4 or memory.ndim != 3:
            raise ValueError("decoder query/memory 应分别为 (B,D,N,d)/(B,N,d)")
        batch, num_depths, num_patches, embed_dim = query.shape
        if memory.shape != (batch, num_patches, embed_dim):
            raise ValueError(
                f"surface memory 与 query 网格不一致: {tuple(memory.shape)} vs "
                f"{(batch, num_patches, embed_dim)}"
            )

        # Every horizontal patch owns one shallow-to-deep causal sequence.
        depth_tokens = (
            query.permute(0, 2, 1, 3)
            .reshape(batch * num_patches, num_depths, embed_dim)
        )
        normalized = self.depth_norm(depth_tokens)
        attended, _ = self.depth_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        depth_tokens = depth_tokens + self.dropout(attended)
        query = (
            depth_tokens.reshape(batch, num_patches, num_depths, embed_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        # Each depth reads the same surface memory, including remote patches.
        spatial_query = query.reshape(
            batch * num_depths, num_patches, embed_dim
        )
        spatial_memory = (
            memory.unsqueeze(1)
            .expand(batch, num_depths, num_patches, embed_dim)
            .reshape(batch * num_depths, num_patches, embed_dim)
        )
        attended, _ = self.surface_attention(
            self.cross_norm(spatial_query),
            self.memory_norm(spatial_memory),
            self.memory_norm(spatial_memory),
            need_weights=False,
        )
        spatial_query = spatial_query + self.dropout(attended)
        spatial_query = spatial_query + self.dropout(
            self.ffn(self.ffn_norm(spatial_query))
        )
        return spatial_query.reshape(
            batch, num_depths, num_patches, embed_dim
        )


class CausalDepthDecoder(nn.Module):
    """Stack depth decoder layers with a final normalization.

    ``causal`` is the original B3 path. ``independent`` keeps exactly the
    same modules and parameters, but restricts depth self-attention to the
    diagonal so tokens cannot exchange information across depth positions.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_layers,
        dim_feedforward,
        dropout,
        depth_attention="causal",
    ):
        super().__init__()
        if int(num_layers) <= 0:
            raise ValueError("decoder num_layers 必须大于0")
        if depth_attention not in {"causal", "independent"}:
            raise ValueError(
                "depth_attention 必须是 causal 或 independent，"
                f"实际为 {depth_attention!r}"
            )
        self.depth_attention_mode = depth_attention
        layer = CausalDepthDecoderLayer(
            embed_dim,
            num_heads,
            dim_feedforward,
            dropout,
        )
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(int(num_layers))]
        )
        self.norm = nn.LayerNorm(embed_dim)

    @staticmethod
    def causal_mask(length, device):
        return torch.triu(
            torch.ones(length, length, device=device, dtype=torch.bool),
            diagonal=1,
        )

    @staticmethod
    def independent_mask(length, device):
        """Block every off-diagonal depth pair while retaining self attention."""
        return ~torch.eye(length, device=device, dtype=torch.bool)

    def forward(self, query, memory):
        if self.depth_attention_mode == "causal":
            mask = self.causal_mask(query.shape[1], query.device)
        else:
            mask = self.independent_mask(query.shape[1], query.device)
        for layer in self.layers:
            query = layer(query, memory, mask)
        return self.norm(query)


class PatchFoldReconstructionHead(nn.Module):
    """Project tokens back to non-overlapping patches and fold them into maps."""

    def __init__(self, embed_dim, patch_size, refinement_channels=32):
        super().__init__()
        self.patch_size = int(patch_size)
        refinement_channels = int(refinement_channels)
        if min(self.patch_size, refinement_channels) <= 0:
            raise ValueError("输出 patch 和 refinement_channels 必须为正数")
        self.patch_projection = nn.Linear(
            int(embed_dim), self.patch_size * self.patch_size
        )
        self.refinement = nn.Sequential(
            nn.Conv2d(1, refinement_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(refinement_channels, 1, kernel_size=3, padding=1),
        )
        # Start from the official patch-fold path; local refinement is learned.
        nn.init.zeros_(self.refinement[-1].weight)
        nn.init.zeros_(self.refinement[-1].bias)

    def forward(self, tokens, grid_size, output_size):
        if tokens.ndim != 4:
            raise ValueError(f"输出 tokens 应为 (B,D,N,d)，实际 {tuple(tokens.shape)}")
        batch, num_depths, num_patches, _ = tokens.shape
        grid_h, grid_w = (int(value) for value in grid_size)
        output_h, output_w = (int(value) for value in output_size)
        if num_patches != grid_h * grid_w:
            raise ValueError(
                f"token 数 {num_patches} 与网格 {grid_h}x{grid_w} 不一致"
            )

        patches = self.patch_projection(tokens).reshape(
            batch * num_depths,
            num_patches,
            self.patch_size * self.patch_size,
        )
        padded_h = grid_h * self.patch_size
        padded_w = grid_w * self.patch_size
        fields = F.fold(
            patches.transpose(1, 2),
            output_size=(padded_h, padded_w),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        fields = fields[..., :output_h, :output_w]
        fields = fields + self.refinement(fields)
        return fields.reshape(batch, num_depths, output_h, output_w)
