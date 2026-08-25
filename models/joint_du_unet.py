"""B1 joint-depth baseline built from the unchanged Du_Unet backbone."""

from models.du_unet import Du_Unet


class JointDuUnet(Du_Unet):
    """Predict the selected depth set in one forward pass.

    This wrapper intentionally adds no attention, Transformer, depth embedding,
    or vertical post-processing.  Relative to the layer-wise B0 Du_Unet
    backbone, the only architectural change is the number of channels in the
    final 1x1 convolution.
    """

    def __init__(
        self,
        num_depths,
        sst_channels=1,
        surface_channels=2,
        base_channels=32,
    ):
        num_depths = int(num_depths)
        if num_depths <= 0:
            raise ValueError(f"num_depths 必须大于 0，实际为 {num_depths}")
        super().__init__(
            out_channels=num_depths,
            sst_channels=sst_channels,
            surface_channels=surface_channels,
            base_channels=base_channels,
        )
        self.num_depths = num_depths
