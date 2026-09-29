"""B4-owned causal depth decoder with surface-memory cross-attention."""

import copy

import torch
import torch.nn as nn


class CausalDepthDecoderLayer(nn.Module):
    """Depth self-attention, memory cross-attention, then feed-forward."""

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
        self.memory_cross_attention = nn.MultiheadAttention(
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
        if query.ndim != 4 or memory.ndim != 3:
            raise ValueError("decoder query/memory 应分别为 (B,D,N,d)/(B,N,d)")
        batch, num_depths, num_patches, embed_dim = query.shape
        if memory.shape != (batch, num_patches, embed_dim):
            raise ValueError(
                f"surface memory 与 query 网格不一致: {tuple(memory.shape)} vs "
                f"{(batch, num_patches, embed_dim)}"
            )

        # At each horizontal patch, shallow-to-deep tokens form one sequence.
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

        # Each depth reads all encoded surface-memory patches.
        spatial_query = query.reshape(
            batch * num_depths, num_patches, embed_dim
        )
        spatial_memory = (
            memory.unsqueeze(1)
            .expand(batch, num_depths, num_patches, embed_dim)
            .reshape(batch * num_depths, num_patches, embed_dim)
        )
        normalized_memory = self.memory_norm(spatial_memory)
        attended, _ = self.memory_cross_attention(
            self.cross_norm(spatial_query),
            normalized_memory,
            normalized_memory,
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
    """Stack B4 causal depth-decoder layers and apply final normalization."""

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
        return ~torch.eye(length, device=device, dtype=torch.bool)

    def forward(self, query, memory):
        if self.depth_attention_mode == "causal":
            mask = self.causal_mask(query.shape[1], query.device)
        else:
            mask = self.independent_mask(query.shape[1], query.device)
        for layer in self.layers:
            query = layer(query, memory, mask)
        return self.norm(query)
