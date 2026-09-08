"""BOS-started depth-autoregressive Transformer for 3-D reconstruction."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.b3_depthformer.decoder import CausalDepthDecoder, PatchFoldReconstructionHead
from models.b3_depthformer.embeddings import (
    ContinuousDepthEmbedding,
    Interpolated2DPositionEmbedding,
    PatchEmbedding2D,
)


class B3DepthFormer(nn.Module):
    """Reconstruct selected subsurface layers from SST, SSH and SSS.

    The surface encoder runs once. During training, ground-truth previous layers
    can be supplied through teacher forcing. During inference, a learned BOS
    token starts generation and each predicted layer becomes the next input.
    """

    def __init__(
        self,
        depth_values,
        d_model=128,
        nhead=8,
        encoder_layers=4,
        decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        sst_patch_size=16,
        surface_patch_size=4,
        output_patch_size=8,
        refinement_channels=32,
        base_grid_size=(16, 16),
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.encoder_layers = int(encoder_layers)
        self.decoder_layers = int(decoder_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.sst_patch_size = int(sst_patch_size)
        self.surface_patch_size = int(surface_patch_size)
        self.output_patch_size = int(output_patch_size)
        self.refinement_channels = int(refinement_channels)
        self.base_grid_size = tuple(int(value) for value in base_grid_size)
        positive_values = (
            self.d_model,
            self.nhead,
            self.encoder_layers,
            self.decoder_layers,
            self.dim_feedforward,
            self.sst_patch_size,
            self.surface_patch_size,
            self.output_patch_size,
            self.refinement_channels,
        )
        if min(positive_values) <= 0:
            raise ValueError("B3DepthFormer 的维度、层数和 patch 参数必须为正数")
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model={self.d_model} 必须能被 nhead={self.nhead} 整除"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0,1)")

        # Separate stems preserve native resolutions until their feature grids
        # are aligned to the decoder's common spatial grid.
        self.sst_patch_embedding = PatchEmbedding2D(
            1, self.d_model, self.sst_patch_size
        )
        self.ssh_patch_embedding = PatchEmbedding2D(
            1, self.d_model, self.surface_patch_size
        )
        self.sss_patch_embedding = PatchEmbedding2D(
            1, self.d_model, self.surface_patch_size
        )
        self.modality_embedding = nn.Parameter(
            torch.zeros(1, 3, self.d_model, 1, 1)
        )
        self.surface_fusion = nn.Sequential(
            nn.Conv2d(self.d_model * 3, self.d_model, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, padding=1),
        )
        self.spatial_position = Interpolated2DPositionEmbedding(
            self.d_model, self.base_grid_size
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.surface_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.encoder_layers,
            norm=nn.LayerNorm(self.d_model),
        )

        self.previous_field_embedding = PatchEmbedding2D(
            1, self.d_model, self.output_patch_size
        )
        self.depth_embedding = ContinuousDepthEmbedding(depth_values, self.d_model)
        self.bos_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.depth_decoder = CausalDepthDecoder(
            embed_dim=self.d_model,
            num_heads=self.nhead,
            num_layers=self.decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )
        self.reconstruction_head = PatchFoldReconstructionHead(
            self.d_model,
            self.output_patch_size,
            refinement_channels=self.refinement_channels,
        )
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)
        nn.init.trunc_normal_(self.bos_token, std=0.02)

    @property
    def num_depths(self):
        return int(self.depth_embedding.depth_values.numel())

    @staticmethod
    def _align_feature(feature, grid_size):
        if feature.shape[-2:] == tuple(grid_size):
            return feature
        return F.interpolate(
            feature,
            size=tuple(grid_size),
            mode="bilinear",
            align_corners=False,
        )

    def encode_surface(self, sst, ssh_sss, output_size):
        """Encode differently resolved surface sources into one memory grid."""
        if sst.ndim != 4 or sst.shape[1] != 1:
            raise ValueError(f"sst 应为 (B,1,H,W)，实际 {tuple(sst.shape)}")
        if ssh_sss.ndim != 4 or ssh_sss.shape[1] != 2:
            raise ValueError(
                f"ssh_sss 应为 (B,2,H,W)，实际 {tuple(ssh_sss.shape)}"
            )
        if sst.shape[0] != ssh_sss.shape[0]:
            raise ValueError("SST 与 SSH/SSS 的 batch size 不一致")
        output_h, output_w = (int(value) for value in output_size)
        if min(output_h, output_w) <= 0:
            raise ValueError(f"output_size 必须为正数，实际 {output_size}")
        grid_size = self.previous_field_embedding.grid_size(output_h, output_w)

        source_features = [
            self.sst_patch_embedding(sst),
            self.ssh_patch_embedding(ssh_sss[:, 0:1]),
            self.sss_patch_embedding(ssh_sss[:, 1:2]),
        ]
        aligned = []
        for index, feature in enumerate(source_features):
            feature = self._align_feature(feature, grid_size)
            aligned.append(feature + self.modality_embedding[:, index])
        fused = self.surface_fusion(torch.cat(aligned, dim=1))
        tokens = fused.flatten(2).transpose(1, 2)
        position = self.spatial_position(
            *grid_size,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        memory = self.surface_encoder(tokens + position)
        return memory, grid_size, position

    def _embed_previous_fields(self, fields, grid_size):
        """Embed one or more previous temperature fields as spatial tokens."""
        if fields.ndim != 4:
            raise ValueError(
                f"previous fields 应为 (B,D,H,W)，实际 {tuple(fields.shape)}"
            )
        batch, num_fields, height, width = fields.shape
        embedded = self.previous_field_embedding(
            fields.reshape(batch * num_fields, 1, height, width)
        )
        embedded = self._align_feature(embedded, grid_size)
        grid_h, grid_w = grid_size
        return (
            embedded.flatten(2)
            .transpose(1, 2)
            .reshape(batch, num_fields, grid_h * grid_w, self.d_model)
        )

    def _parallel_teacher_forced(
        self,
        memory,
        grid_size,
        spatial_position,
        target,
        output_size,
    ):
        """Decode all depths together from BOS and right-shifted truths."""
        batch, target_depths = target.shape[:2]
        if target_depths != self.num_depths:
            raise ValueError(
                f"target 深度数应为 {self.num_depths}，实际 {target_depths}"
            )
        num_patches = grid_size[0] * grid_size[1]
        bos = self.bos_token.expand(batch, num_patches, self.d_model).unsqueeze(1)
        if self.num_depths > 1:
            previous = self._embed_previous_fields(target[:, :-1], grid_size)
            query = torch.cat([bos, previous], dim=1)
        else:
            query = bos
        depth_position = self.depth_embedding(
            dtype=query.dtype, device=query.device
        ).view(1, self.num_depths, 1, self.d_model)
        query = query + spatial_position.unsqueeze(1) + depth_position
        decoded = self.depth_decoder(query, memory)
        return self.reconstruction_head(
            decoded,
            grid_size=grid_size,
            output_size=output_size,
        )

    def _autoregressive_decode(
        self,
        memory,
        grid_size,
        spatial_position,
        output_size,
        *,
        target=None,
        teacher_forcing_ratio=0.0,
    ):
        """Generate shallow-to-deep fields, optionally with scheduled sampling."""
        batch = memory.shape[0]
        num_patches = grid_size[0] * grid_size[1]
        depth_position = self.depth_embedding(
            dtype=memory.dtype, device=memory.device
        )
        query_history = []
        predictions = []
        previous_field = None

        for depth_index in range(self.num_depths):
            if depth_index == 0:
                content = self.bos_token.expand(
                    batch, num_patches, self.d_model
                )
            else:
                content = self._embed_previous_fields(
                    previous_field.unsqueeze(1), grid_size
                )[:, 0]
            query = (
                content
                + spatial_position
                + depth_position[depth_index].view(1, 1, self.d_model)
            )
            query_history.append(query.unsqueeze(1))
            decoded = self.depth_decoder(torch.cat(query_history, dim=1), memory)
            predicted = self.reconstruction_head(
                decoded[:, -1:],
                grid_size=grid_size,
                output_size=output_size,
            )[:, 0]
            predictions.append(predicted.unsqueeze(1))

            if target is None or depth_index == self.num_depths - 1:
                previous_field = predicted
                continue
            if teacher_forcing_ratio <= 0.0:
                previous_field = predicted
            elif teacher_forcing_ratio >= 1.0:
                previous_field = target[:, depth_index]
            else:
                use_truth = torch.rand(
                    batch, 1, 1, device=predicted.device
                ) < teacher_forcing_ratio
                previous_field = torch.where(
                    use_truth, target[:, depth_index], predicted
                )
        return torch.cat(predictions, dim=1)

    def forward(
        self,
        sst,
        ssh_sss,
        *,
        target=None,
        output_size=None,
        teacher_forcing_ratio=0.0,
    ):
        """Run teacher-forced training or fully autoregressive inference."""
        teacher_forcing_ratio = float(teacher_forcing_ratio)
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio 必须位于 [0,1]")
        if target is not None:
            if target.ndim != 4 or target.shape[1] != self.num_depths:
                raise ValueError(
                    f"target 应为 (B,{self.num_depths},H,W)，"
                    f"实际 {tuple(target.shape)}"
                )
            if target.shape[0] != sst.shape[0]:
                raise ValueError("target 与 surface inputs 的 batch size 不一致")
        if output_size is None:
            output_size = (
                target.shape[-2:]
                if target is not None
                else ssh_sss.shape[-2:]
            )
        output_size = tuple(int(value) for value in output_size)
        memory, grid_size, spatial_position = self.encode_surface(
            sst, ssh_sss, output_size
        )

        # Pure teacher forcing is parallel across depths and much faster.
        if target is not None and teacher_forcing_ratio >= 1.0:
            return self._parallel_teacher_forced(
                memory,
                grid_size,
                spatial_position,
                target,
                output_size,
            )
        return self._autoregressive_decode(
            memory,
            grid_size,
            spatial_position,
            output_size,
            target=target,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

    def generate(self, sst, ssh_sss, output_size=None):
        """Explicit inference helper for external callers and smoke tests."""
        return self.forward(
            sst,
            ssh_sss,
            target=None,
            output_size=output_size,
            teacher_forcing_ratio=0.0,
        )

    def model_config(self):
        """Return serializable architecture metadata stored in checkpoints."""
        return {
            "architecture": "BOS_depth_autoregressive_encoder_decoder",
            "surface_inputs": ["SST", "SSH", "SSS"],
            "depth_values": [
                float(value) for value in self.depth_embedding.depth_values.tolist()
            ],
            "num_depths": self.num_depths,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "sst_patch_size": self.sst_patch_size,
            "surface_patch_size": self.surface_patch_size,
            "output_patch_size": self.output_patch_size,
            "refinement_channels": self.refinement_channels,
            "base_grid_size": list(self.base_grid_size),
            "depth_embedding": "learned_mlp_log1p_real_depth",
            "surface_position_embedding": "learnable_2d_interpolated",
            "decoder_start": "learned_BOS",
            "decoder_attention": "causal_depth_then_surface_cross_attention",
            "reconstruction": "linear_patch_projection_and_fold_with_local_refinement",
        }
