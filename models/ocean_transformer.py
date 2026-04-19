"""
三维温盐场反演 Transformer 模型

架构：
    海表 4 通道 (SLA, SSS, SSH梯度, EKE)
    → 轻量 CNN 提取空间特征
    → Spatial Transformer Encoder 学习水平结构
    → Depth-wise Transformer Encoder 学习垂向关联（温跃层下压/托举）
    → 输出 10 层深度的温度与盐度

数据流：
    (B,4,H,W) ─CNN→ (B,d,H,W) ─flatten→ (B,N,d)
    ─spatial_PE+encoder→ (B,N,d)
    ─expand×10+depth_PE→ (B·N,10,d) ─depth_encoder→ (B·N,10,d)
    ─reshape→ (B,10,H,W,d) ─linear→ (B,10,H,W,2)
"""

import math
import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """轻量级 CNN：从海表多源遥感提取空间特征，不做下采样"""

    def __init__(self, in_channels=4, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, d_model, 3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class Spatial2DPositionalEncoding(nn.Module):
    """
    可学习的二维空间位置编码

    将行坐标与列坐标分别编码后拼接，为展平的空间序列注入位置信息。
    """

    def __init__(self, d_model, max_h=256, max_w=256):
        super().__init__()
        d_row = d_model // 2
        d_col = d_model - d_row
        self.row_embed = nn.Embedding(max_h, d_row)
        self.col_embed = nn.Embedding(max_w, d_col)
        nn.init.normal_(self.row_embed.weight, std=0.02)
        nn.init.normal_(self.col_embed.weight, std=0.02)

    def forward(self, x, H, W):
        """
        Args:
            x: (B, H*W, d_model)
        """
        rows = torch.arange(H, device=x.device)
        cols = torch.arange(W, device=x.device)
        row_pe = self.row_embed(rows).unsqueeze(1).expand(-1, W, -1)
        col_pe = self.col_embed(cols).unsqueeze(0).expand(H, -1, -1)
        pe = torch.cat([row_pe, col_pe], dim=-1).reshape(H * W, -1).unsqueeze(0)
        return x + pe


class OceanTransformer(nn.Module):
    """
    三维温盐场反演 Transformer

    Args:
        in_channels:    海表输入通道数（默认 4）
        d_model:        Transformer 隐藏维度
        nhead:          多头注意力头数
        spatial_layers: 空间 Transformer 层数
        depth_layers:   深度 Transformer 层数
        dim_ff:         前馈网络维度
        num_depths:     预测深度层数（默认 10）
        out_vars:       输出变量数（默认 2：温度+盐度）
        dropout:        Dropout 比率
    """

    def __init__(self, in_channels=4, d_model=128, nhead=8,
                 spatial_layers=4, depth_layers=2, dim_ff=512,
                 num_depths=10, out_vars=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_depths = num_depths

        self.feature_extractor = FeatureExtractor(in_channels, d_model)
        self.spatial_pe = Spatial2DPositionalEncoding(d_model)

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(
            spatial_layer, num_layers=spatial_layers,
        )

        # 深度位置编码（可学习），让模型区分温跃层 50-250m 与深层
        self.depth_embedding = nn.Embedding(num_depths, d_model)

        depth_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.depth_encoder = nn.TransformerEncoder(
            depth_layer, num_layers=depth_layers,
        )

        self.output_head = nn.Linear(d_model, out_vars)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 海表特征图
        Returns:
            (B, num_depths, H, W, out_vars)
        """
        B, _, H, W = x.shape
        N = H * W

        feat = self.feature_extractor(x)              # (B, d, H, W)
        feat = feat.flatten(2).permute(0, 2, 1)       # (B, N, d)
        feat = self.spatial_pe(feat, H, W)
        feat = self.spatial_encoder(feat)              # (B, N, d)

        # 每个空间位置扩展到 num_depths 层，加深度编码
        feat = feat.reshape(B * N, 1, self.d_model)
        feat = feat.expand(-1, self.num_depths, -1).clone()

        depth_ids = torch.arange(self.num_depths, device=x.device)
        feat = feat + self.depth_embedding(depth_ids).unsqueeze(0)

        feat = self.depth_encoder(feat)                # (B*N, D, d)

        out = self.output_head(feat)                   # (B*N, D, 2)
        out = out.reshape(B, N, self.num_depths, -1)
        out = out.permute(0, 2, 1, 3)                 # (B, D, N, 2)
        out = out.reshape(B, self.num_depths, H, W, -1)

        return out

