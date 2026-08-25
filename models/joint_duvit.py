"""B2 joint-depth DUViT with two spatial Transformer bottlenecks."""

import torch.nn as nn

from models.du_unet import Du_Unet
from models.spatial_vit import SpatialViTBottleneck


class JointDUViT(Du_Unet):
    """Joint-depth Du_Unet enhanced only by two spatial ViT bottlenecks.

    B1 remains untouched.  This B2 model keeps the same inputs, convolutional
    encoders/decoders, skip concatenations, fusion head and dynamic output-depth
    contract.  Skip Attention Gates and vertical attention are deliberately not
    included in this version.
    """

    def __init__(
        self,
        num_depths,
        sst_channels=1,
        surface_channels=2,
        base_channels=32,
        sst_embed_dim=256,
        sst_patch_size=1,
        sst_num_heads=8,
        sst_num_layers=2,
        sst_max_tokens=256,
        surface_embed_dim=128,
        surface_patch_size=2,
        surface_num_heads=4,
        surface_num_layers=2,
        surface_max_tokens=512,
        mlp_ratio=4,
        dropout=0.1,
    ):
        num_depths = int(num_depths)
        if num_depths <= 0:
            raise ValueError(f"num_depths 必须大于0，实际为 {num_depths}")
        base_channels = int(base_channels)
        super().__init__(
            out_channels=num_depths,
            sst_channels=sst_channels,
            surface_channels=surface_channels,
            base_channels=base_channels,
        )

        original_sst_bridge = self.sst_bridge
        original_surface_bridge = self.surface_bridge
        self.sst_bridge = nn.Sequential(
            original_sst_bridge,
            SpatialViTBottleneck(
                in_channels=base_channels * 8,
                embed_dim=sst_embed_dim,
                patch_size=sst_patch_size,
                num_heads=sst_num_heads,
                num_layers=sst_num_layers,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                base_grid_size=(10, 10),
                max_tokens=sst_max_tokens,
            ),
        )
        self.surface_bridge = nn.Sequential(
            original_surface_bridge,
            SpatialViTBottleneck(
                in_channels=base_channels * 2,
                embed_dim=surface_embed_dim,
                patch_size=surface_patch_size,
                num_heads=surface_num_heads,
                num_layers=surface_num_layers,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                base_grid_size=(16, 16),
                max_tokens=surface_max_tokens,
            ),
        )
        self.num_depths = num_depths
        self.use_skip_attention = False
        self.use_vertical_transformer = False

    def model_config(self):
        return {
            "backbone": "Du_Unet",
            "out_channels": self.num_depths,
            "joint_depth_output": True,
            "spatial_transformer": True,
            "skip_attention": False,
            "vertical_transformer": False,
            "sst_vit": self.sst_bridge[1].config_dict(),
            "surface_vit": self.surface_bridge[1].config_dict(),
        }
