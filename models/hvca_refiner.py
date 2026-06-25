"""HVCA residual refiner for stacked layer-wise Du_Unet predictions.

The refiner takes an initial full-depth temperature field T0 with shape
(B, D, H, W), fuses horizontal layer features with vertical profile features,
and returns T_final = T0 + delta. It does not consume surface variables in this
first version, keeping the second-stage experiment focused on the HVCA module.
"""

import torch
import torch.nn as nn


class Horizontal2DFeatureEncoder(nn.Module):
    """Extract per-depth horizontal features with shared 2D convolutions."""

    def __init__(self, dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, T0):
        """
        Args:
            T0: (B, D, H, W)

        Returns:
            Fh: (B, dim, D, H, W)
        """
        assert T0.ndim == 4, f"T0 must be (B,D,H,W), got {tuple(T0.shape)}"
        B, D, H, W = T0.shape
        x = T0.unsqueeze(2).reshape(B * D, 1, H, W)
        feat = self.encoder(x)  # (B*D, dim, H, W)
        dim = feat.shape[1]
        return feat.reshape(B, D, dim, H, W).permute(0, 2, 1, 3, 4).contiguous()


class VerticalProfileEncoder(nn.Module):
    """Encode each horizontal grid column as a depth sequence."""

    def __init__(self, dim=64, num_heads=4, num_layers=2, dropout=0.1, chunk_size=8192):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.temp_embed = nn.Linear(1, dim)
        self.depth_embed = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    @staticmethod
    def _normalize_depth(depth_values, device, dtype):
        depth = torch.as_tensor(depth_values, device=device, dtype=dtype).reshape(-1, 1)
        span = depth.max() - depth.min()
        if torch.isclose(span, torch.zeros((), device=device, dtype=dtype)):
            return torch.zeros_like(depth)
        return (depth - depth.min()) / span

    def forward(self, T0, depth_values):
        """
        Args:
            T0: (B, D, H, W)
            depth_values: (D,)

        Returns:
            Fv: (B, dim, D, H, W)
        """
        assert T0.ndim == 4, f"T0 must be (B,D,H,W), got {tuple(T0.shape)}"
        B, D, H, W = T0.shape
        assert len(depth_values) == D, (
            f"len(depth_values) must equal D={D}, got {len(depth_values)}"
        )

        profile = T0.permute(0, 2, 3, 1).reshape(B * H * W, D, 1)
        depth_norm = self._normalize_depth(depth_values, T0.device, T0.dtype)
        depth_token = self.depth_embed(depth_norm).unsqueeze(0)  # (1, D, dim)
        encoded_parts = []
        chunk_size = max(self.chunk_size, 1)
        for start in range(0, profile.shape[0], chunk_size):
            profile_chunk = profile[start : start + chunk_size]
            token = self.temp_embed(profile_chunk) + depth_token
            encoded_parts.append(self.encoder(token))
        encoded = torch.cat(encoded_parts, dim=0)  # (B*H*W, D, dim)
        dim = encoded.shape[-1]
        return (
            encoded.reshape(B, H, W, D, dim)
            .permute(0, 4, 3, 1, 2)
            .contiguous()
        )


class ColumnCrossAttention(nn.Module):
    """Fuse horizontal and vertical features within each horizontal column."""

    def __init__(self, dim=64, num_heads=4, dropout=0.1, chunk_size=8192):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, Fh, Fv):
        """
        Args:
            Fh: (B, dim, D, H, W)
            Fv: (B, dim, D, H, W)

        Returns:
            F: (B, dim, D, H, W)
        """
        assert Fh.ndim == 5 and Fv.ndim == 5, "Fh/Fv must be (B,dim,D,H,W)"
        assert Fh.shape == Fv.shape, f"Fh and Fv shapes differ: {Fh.shape} vs {Fv.shape}"
        B, dim, D, H, W = Fh.shape
        q = Fh.permute(0, 3, 4, 2, 1).reshape(B * H * W, D, dim)
        kv = Fv.permute(0, 3, 4, 2, 1).reshape(B * H * W, D, dim)
        out_parts = []
        chunk_size = max(self.chunk_size, 1)
        for start in range(0, q.shape[0], chunk_size):
            q_chunk = q[start : start + chunk_size]
            kv_chunk = kv[start : start + chunk_size]
            out_chunk, _ = self.attn(q_chunk, kv_chunk, kv_chunk, need_weights=False)
            out_parts.append(out_chunk)
        out = torch.cat(out_parts, dim=0)
        out = out.reshape(B, H, W, D, dim).permute(0, 4, 3, 1, 2).contiguous()
        return Fh + self.gamma * out


class ResidualHead(nn.Module):
    """Predict residual delta from fused features."""

    def __init__(self, dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GELU(),
            nn.Conv3d(dim, 1, kernel_size=1),
        )
        last = self.conv[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, F):
        """
        Args:
            F: (B, dim, D, H, W)

        Returns:
            delta: (B, D, H, W)
        """
        assert F.ndim == 5, f"F must be (B,dim,D,H,W), got {tuple(F.shape)}"
        return self.conv(F).squeeze(1)


class HVCARefiner(nn.Module):
    """Horizontal-Vertical Cross-Attention residual refiner."""

    def __init__(
        self,
        dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        column_chunk_size=8192,
    ):
        super().__init__()
        self.horizontal = Horizontal2DFeatureEncoder(dim=dim)
        self.vertical = VerticalProfileEncoder(
            dim=dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            chunk_size=column_chunk_size,
        )
        self.cross = ColumnCrossAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            chunk_size=column_chunk_size,
        )
        self.head = ResidualHead(dim=dim)

    def forward(self, T0, depth_values):
        """
        Args:
            T0: (B, D, H, W), stacked layer-wise Du_Unet prediction.
            depth_values: (D,), real depth values in meters.

        Returns:
            T_final: (B, D, H, W), where T_final = T0 + delta.
        """
        assert T0.ndim == 4, f"T0 must be (B,D,H,W), got {tuple(T0.shape)}"
        D = T0.shape[1]
        assert len(depth_values) == D, (
            f"len(depth_values) must equal D={D}, got {len(depth_values)}"
        )
        Fh = self.horizontal(T0)
        Fv = self.vertical(T0, depth_values)
        F = self.cross(Fh, Fv)
        delta = self.head(F)
        assert delta.shape == T0.shape, f"delta shape {delta.shape} != T0 shape {T0.shape}"
        return T0 + delta


__all__ = [
    "Horizontal2DFeatureEncoder",
    "VerticalProfileEncoder",
    "ColumnCrossAttention",
    "ResidualHead",
    "HVCARefiner",
]
