"""B4 C+ native-grid, regrid, U-Net and depth-attention tests."""

import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from config import (
    B4_CPLUS_CKPT_NAME_TEMPLATE,
    GLORYS12V1_DEPTH_LEVELS_28M,
    PARADIGM_2DTO2D,
    get_checkpoint_dir,
    joint_du_unet_depth_tag,
)
from models.b4_cplus import B4CPlusDepthFormer
from models.b4_cplus.regrid import CoordinateFeatureRegrid
from test import _load_joint_du_unet_model
from utils.losses import masked_depth_mean_rmse_loss


DEPTH_INDICES = [0, 4, 9]
DEPTH_VALUES = [GLORYS12V1_DEPTH_LEVELS_28M[index] for index in DEPTH_INDICES]


def tiny_model():
    return B4CPlusDepthFormer(
        depth_values=DEPTH_VALUES,
        d_model=16,
        nhead=4,
        encoder_layers=1,
        decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        native_base_channels=4,
        fusion_channels=8,
        output_patch_size=8,
        base_grid_size=(2, 2),
    )


def synthetic_batch(batch_size=1):
    generator = torch.Generator().manual_seed(42)
    return {
        "sst": torch.randn(batch_size, 1, 32, 24, generator=generator),
        "ssh_sss": torch.randn(batch_size, 2, 13, 11, generator=generator),
        "target": torch.randn(
            batch_size, len(DEPTH_VALUES), 17, 19, generator=generator
        ),
        "target_mask": torch.ones(
            batch_size, len(DEPTH_VALUES), 17, 19
        ),
    }


class B4CPlusTest(unittest.TestCase):
    def test_coordinate_regrid_uses_physical_endpoints_and_backpropagates(self):
        feature = torch.arange(12.0).reshape(1, 1, 3, 4).requires_grad_()
        regrid = CoordinateFeatureRegrid(
            lat_range=(0.0, 3.0),
            lon_range=(10.0, 14.0),
            source_cell_centered=True,
        )
        output = regrid(feature, (4, 5))
        self.assertEqual(output.shape, (1, 1, 4, 5))
        # Target endpoints lie outside the source cell centres and therefore
        # use border values instead of silently shifting the physical domain.
        torch.testing.assert_close(output[0, 0, 0, 0], feature[0, 0, 0, 0])
        torch.testing.assert_close(output[0, 0, -1, -1], feature[0, 0, -1, -1])
        output.sum().backward()
        self.assertIsNotNone(feature.grad)
        self.assertGreater(feature.grad.abs().sum().item(), 0.0)

    def test_native_encoder_actual_dataset_shapes(self):
        model = B4CPlusDepthFormer(
            depth_values=DEPTH_VALUES,
            d_model=16,
            nhead=4,
            encoder_layers=1,
            decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            native_base_channels=4,
            fusion_channels=8,
            output_patch_size=8,
            base_grid_size=(2, 2),
        ).eval()
        with torch.no_grad():
            memory, grid_size, _, skips = model.encode_surface(
                torch.randn(1, 1, 480, 320),
                torch.randn(1, 2, 192, 128),
                (289, 193),
            )
        self.assertEqual(grid_size, torch.Size([37, 25]))
        self.assertEqual(memory.shape, (1, 925, 16))
        self.assertEqual(
            [tuple(skip.shape[-2:]) for skip in skips],
            [(145, 97), (73, 49)],
        )

    def test_forward_backward_and_component_parameter_sum(self):
        batch = synthetic_batch(batch_size=2)
        model = tiny_model().train()
        prediction = model(
            batch["sst"],
            batch["ssh_sss"],
            target=batch["target"],
            teacher_forcing_ratio=1.0,
        )
        self.assertEqual(prediction.shape, batch["target"].shape)
        self.assertTrue(torch.isfinite(prediction).all())
        loss = masked_depth_mean_rmse_loss(
            prediction, batch["target"], batch["target_mask"]
        )
        loss.backward()
        self.assertGreater(
            sum(
                parameter.grad.abs().sum().item()
                for parameter in model.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )
        counts = model.parameter_count_by_component()
        self.assertEqual(
            counts["total"],
            sum(parameter.numel() for parameter in model.parameters()),
        )

    def test_first_depth_cannot_see_future_teacher_forcing_fields(self):
        batch = synthetic_batch()
        model = tiny_model().eval()
        changed_target = batch["target"].clone()
        changed_target[:, 1:] += 100.0
        with torch.no_grad():
            original = model(
                batch["sst"],
                batch["ssh_sss"],
                target=batch["target"],
                teacher_forcing_ratio=1.0,
            )
            changed = model(
                batch["sst"],
                batch["ssh_sss"],
                target=changed_target,
                teacher_forcing_ratio=1.0,
            )
        torch.testing.assert_close(original[:, 0], changed[:, 0])

    def test_depth_decoder_contains_memory_cross_attention(self):
        model = tiny_model()
        self.assertEqual(
            model.depth_decoder.__class__.__module__,
            "models.b4_cplus.decoder",
        )
        self.assertEqual(
            model.depth_embedding.__class__.__module__,
            "models.b4_cplus.embeddings",
        )
        self.assertTrue(model.depth_decoder.layers)
        for layer in model.depth_decoder.layers:
            self.assertIsInstance(
                layer.memory_cross_attention, torch.nn.MultiheadAttention
            )

    def test_structured_checkpoint_round_trip(self):
        model = tiny_model().eval()
        depth_tag = joint_du_unet_depth_tag(DEPTH_INDICES, 28)
        with tempfile.TemporaryDirectory() as temporary_root:
            checkpoint_dir = get_checkpoint_dir(
                PARADIGM_2DTO2D,
                "B4_CPlus",
                base_dir=temporary_root,
                target_var="temperature",
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                checkpoint_dir,
                B4_CPLUS_CKPT_NAME_TEMPLATE.format(
                    target_var="temperature", depth_tag=depth_tag
                ),
            )
            metadata = {
                "checkpoint_type": "B4_CPlus",
                "target_var": "temperature",
                "dataset_profile": "glorys12v1_all_size",
                "dataset_num_depths": 28,
                "normalize": True,
                "depth_indices": DEPTH_INDICES,
                "depth_m": DEPTH_VALUES,
                "num_outputs": len(DEPTH_INDICES),
                "depth_tag": depth_tag,
                "normalization_stats": None,
                "model_config": {
                    **model.model_config(),
                    "source_depth_indices": DEPTH_INDICES,
                },
            }
            torch.save(
                {
                    "format_version": 1,
                    "state_dict": model.state_dict(),
                    "metadata": metadata,
                },
                checkpoint_path,
            )
            args = SimpleNamespace(
                method="B4_CPlus",
                checkpoint=checkpoint_path,
                checkpoint_dir=checkpoint_dir,
                output_dir=temporary_root,
                target_var="temperature",
                dataset_profile="glorys12v1_all_size",
                depth_indices="0,4,9",
            )
            dataset = SimpleNamespace(
                num_depths=28,
                get_normalization_reference=lambda: None,
            )
            loaded, loaded_path, indices, loaded_metadata = (
                _load_joint_du_unet_model(
                    args, torch.device("cpu"), dataset
                )
            )
        self.assertEqual(loaded_path, checkpoint_path)
        self.assertEqual(indices, DEPTH_INDICES)
        self.assertEqual(loaded_metadata["checkpoint_type"], "B4_CPlus")
        self.assertEqual(loaded.model_config(), model.model_config())


if __name__ == "__main__":
    unittest.main()
