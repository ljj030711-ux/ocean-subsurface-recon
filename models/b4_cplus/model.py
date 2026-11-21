"""B4 C+: native-grid CNNs, full U-Net pyramid and B3 depth Transformer."""

import torch
import torch.nn as nn

from models.b4_cplus.decoder import CausalDepthDecoder
from models.b4_cplus.embeddings import (
    ContinuousDepthEmbedding,
    Interpolated2DPositionEmbedding,
    PatchEmbedding2D,
)
from models.b4_cplus.unet import CPlusSurfaceEncoder, CPlusUNetDecoder


class B4CPlusDepthFormer(nn.Module):
    """C+ spatial backbone retaining the B3 causal depth Transformer."""

    def __init__(
        self,
        depth_values,
        d_model=128,
        nhead=8,
        encoder_layers=4,
        decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        native_base_channels=32,
        fusion_channels=64,
        output_patch_size=8,
        base_grid_size=(16, 16),
        depth_attention="causal",
        lat_range=(0.0, 24.0),
        lon_range=(105.0, 121.0),
        source_lat_descending=False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.encoder_layers = int(encoder_layers)
        self.decoder_layers = int(decoder_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.native_base_channels = int(native_base_channels)
        self.fusion_channels = int(fusion_channels)
        self.output_patch_size = int(output_patch_size)
        self.base_grid_size = tuple(int(value) for value in base_grid_size)
        self.depth_attention = str(depth_attention)
        self.lat_range = tuple(float(value) for value in lat_range)
        self.lon_range = tuple(float(value) for value in lon_range)
        self.source_lat_descending = bool(source_lat_descending)
        if self.output_patch_size != 8:
            raise ValueError("B4 C+ 当前固定使用 output_patch_size=8")
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model={self.d_model} 必须能被 nhead={self.nhead} 整除"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0,1)")

        self.spatial_encoder = CPlusSurfaceEncoder(
            embed_dim=self.d_model,
            native_base_channels=self.native_base_channels,
            fusion_channels=self.fusion_channels,
            lat_range=self.lat_range,
            lon_range=self.lon_range,
            source_lat_descending=self.source_lat_descending,
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
            depth_attention=self.depth_attention,
        )
        self.spatial_decoder = CPlusUNetDecoder(
            self.d_model,
            self.spatial_encoder.skip_channels,
            output_channels=self.native_base_channels,
        )
        nn.init.trunc_normal_(self.bos_token, std=0.02)

    @property
    def num_depths(self):
        return int(self.depth_embedding.depth_values.numel())

    def encode_surface(self, sst, ssh_sss, output_size):
        if sst.ndim != 4 or sst.shape[1] != 1:
            raise ValueError(f"sst 应为 (B,1,H,W)，实际 {tuple(sst.shape)}")
        if ssh_sss.ndim != 4 or ssh_sss.shape[1] != 2:
            raise ValueError(
                f"ssh_sss 应为 (B,2,H,W)，实际 {tuple(ssh_sss.shape)}"
            )
        if sst.shape[0] != ssh_sss.shape[0]:
            raise ValueError("SST 与 SSH/SSS 的 batch size 不一致")
        bottleneck, skips = self.spatial_encoder(sst, ssh_sss, output_size)
        grid_size = bottleneck.shape[-2:]
        expected_grid = self.previous_field_embedding.grid_size(*output_size)
        if tuple(grid_size) != tuple(expected_grid):
            raise RuntimeError(
                "C+ bottleneck 与 previous-field 网格不一致: "
                f"{tuple(grid_size)} vs {tuple(expected_grid)}"
            )
        tokens = bottleneck.flatten(2).transpose(1, 2)
        position = self.spatial_position(
            *grid_size, dtype=tokens.dtype, device=tokens.device
        )
        return self.surface_encoder(tokens + position), grid_size, position, skips

    def _embed_previous_fields(self, fields, grid_size):
        batch, num_fields, height, width = fields.shape
        embedded = self.previous_field_embedding(
            fields.reshape(batch * num_fields, 1, height, width)
        )
        grid_h, grid_w = grid_size
        if embedded.shape[-2:] != tuple(grid_size):
            raise RuntimeError(
                "previous-field embedding 与 C+ bottleneck 网格不一致"
            )
        return (
            embedded.flatten(2)
            .transpose(1, 2)
            .reshape(batch, num_fields, grid_h * grid_w, self.d_model)
        )

    def _decode_spatial(self, decoded, grid_size, skips, output_size):
        batch, num_depths, num_patches, channels = decoded.shape
        grid_h, grid_w = grid_size
        if num_patches != grid_h * grid_w:
            raise ValueError("decoder token 数与 C+ bottleneck 网格不一致")
        bottleneck = decoded.transpose(2, 3).reshape(
            batch, num_depths, channels, grid_h, grid_w
        )
        return self.spatial_decoder(bottleneck, skips, output_size)

    def _parallel_teacher_forced(
        self, memory, grid_size, position, skips, target, output_size
    ):
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
        decoded = self.depth_decoder(
            query + position.unsqueeze(1) + depth_position, memory
        )
        return self._decode_spatial(decoded, grid_size, skips, output_size)

    def _autoregressive_decode(
        self,
        memory,
        grid_size,
        position,
        skips,
        output_size,
        *,
        target=None,
        teacher_forcing_ratio=0.0,
    ):
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
                content = self.bos_token.expand(batch, num_patches, self.d_model)
            else:
                content = self._embed_previous_fields(
                    previous_field.unsqueeze(1), grid_size
                )[:, 0]
            query = (
                content
                + position
                + depth_position[depth_index].view(1, 1, self.d_model)
            )
            query_history.append(query.unsqueeze(1))
            decoded = self.depth_decoder(torch.cat(query_history, dim=1), memory)
            predicted = self._decode_spatial(
                decoded[:, -1:], grid_size, skips, output_size
            )[:, 0]
            predictions.append(predicted.unsqueeze(1))
            if target is None or depth_index == self.num_depths - 1:
                previous_field = predicted
            elif teacher_forcing_ratio <= 0.0:
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
                raise ValueError("target 与海表输入的 batch size 不一致")
        if output_size is None:
            output_size = target.shape[-2:] if target is not None else ssh_sss.shape[-2:]
        output_size = tuple(int(value) for value in output_size)
        if len(output_size) != 2 or min(output_size) <= 0:
            raise ValueError(f"output_size 应为两个正整数，实际 {output_size}")
        if target is not None and tuple(target.shape[-2:]) != output_size:
            raise ValueError(
                "target 空间尺寸与 output_size 不一致: "
                f"{tuple(target.shape[-2:])} vs {output_size}"
            )
        memory, grid_size, position, skips = self.encode_surface(
            sst, ssh_sss, output_size
        )
        if target is not None and teacher_forcing_ratio >= 1.0:
            return self._parallel_teacher_forced(
                memory, grid_size, position, skips, target, output_size
            )
        return self._autoregressive_decode(
            memory,
            grid_size,
            position,
            skips,
            output_size,
            target=target,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

    def generate(self, sst, ssh_sss, output_size=None):
        return self.forward(sst, ssh_sss, output_size=output_size)

    def model_config(self):
        return {
            "architecture": "B4_CPlus_DepthFormer",
            "architecture_version": 5,
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
            "native_base_channels": self.native_base_channels,
            "fusion_channels": self.fusion_channels,
            "output_patch_size": self.output_patch_size,
            "base_grid_size": list(self.base_grid_size),
            "depth_attention": self.depth_attention,
            "lat_range": list(self.lat_range),
            "lon_range": list(self.lon_range),
            "source_lat_descending": self.source_lat_descending,
            "native_branches": {
                "sst": "480x320 native feature extraction",
                "ssh_sss": "192x128 native feature extraction",
            },
            "single_regrid_target": "output_size, actual 289x193",
            "shared_pyramid": ["289x193", "145x97", "73x49", "37x25"],
            "skip_connections": ["289x193", "145x97", "73x49"],
            "full_resolution_skip": True,
            "surface_memory": "4-layer spatial Transformer encoder",
            "depth_decoder": "causal self-attention plus memory cross-attention",
            "reconstruction": "single 289x193 head after three full-pyramid skip fusions",
            "regrid": self.spatial_encoder.sst_regrid.config_dict(),
        }

    def parameter_count_by_component(self):
        modules = {
            "cplus_native_and_shared_encoder": self.spatial_encoder,
            "surface_transformer": self.surface_encoder,
            "previous_field_embedding": self.previous_field_embedding,
            "depth_embedding": self.depth_embedding,
            "depth_transformer": self.depth_decoder,
            "cplus_single_head_decoder": self.spatial_decoder,
        }
        counts = {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in modules.items()
        }
        counts["position_and_bos"] = (
            self.spatial_position.embedding.numel() + self.bos_token.numel()
        )
        counts["total"] = sum(counts.values())
        return {name: int(value) for name, value in counts.items()}
