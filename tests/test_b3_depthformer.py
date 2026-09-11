"""Smoke tests for the B3 BOS-started depth-autoregressive pipeline."""

import os
import random
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from config import (
    B3_DEPTHFORMER_CKPT_NAME_TEMPLATE,
    GLORYS12V1_DEPTH_LEVELS_28M,
    PARADIGM_2DTO2D,
    get_checkpoint_dir,
    get_output_dir,
    joint_du_unet_depth_tag,
)
from models.b3_depthformer import B3DepthFormer
from test import _load_joint_du_unet_model
from train import teacher_forcing_ratio_for_epoch
from utils.losses import masked_depth_mean_rmse_loss
from utils.reproducibility import (
    build_reproducibility_report,
    configure_reproducibility,
    make_dataloader_generator,
)


DEPTH_INDICES = [0, 4, 9, 17, 24, 27]
DEPTH_VALUES = [GLORYS12V1_DEPTH_LEVELS_28M[index] for index in DEPTH_INDICES]


def tiny_model():
    return B3DepthFormer(
        depth_values=DEPTH_VALUES,
        d_model=16,
        nhead=4,
        encoder_layers=1,
        decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        sst_patch_size=4,
        surface_patch_size=2,
        output_patch_size=4,
        refinement_channels=4,
        base_grid_size=(4, 4),
    )


def synthetic_batch(batch_size=1):
    generator = torch.Generator().manual_seed(42)
    return {
        "sst": torch.randn(batch_size, 1, 20, 24, generator=generator),
        "ssh_sss": torch.randn(batch_size, 2, 9, 11, generator=generator),
        "target": torch.randn(
            batch_size, len(DEPTH_VALUES), 9, 11, generator=generator
        ),
        "target_mask": torch.ones(batch_size, len(DEPTH_VALUES), 9, 11),
    }


class B3DepthFormerSmokeTest(unittest.TestCase):
    def test_fixed_seed_repeats_rngs_and_dataloader_order(self):
        configure_reproducibility(2026)
        first_python = random.random()
        first_numpy = np.random.random(4)
        first_torch = torch.rand(4)
        first_order = torch.randperm(
            12, generator=make_dataloader_generator(2026)
        )

        configure_reproducibility(2026)
        report = build_reproducibility_report(2026, torch.device("cpu"))
        self.assertEqual(first_python, random.random())
        np.testing.assert_array_equal(first_numpy, np.random.random(4))
        torch.testing.assert_close(first_torch, torch.rand(4), rtol=0, atol=0)
        torch.testing.assert_close(
            first_order,
            torch.randperm(12, generator=make_dataloader_generator(2026)),
            rtol=0,
            atol=0,
        )
        self.assertEqual(report["seed"], 2026)
        self.assertEqual(report["device"]["type"], "cpu")
        self.assertTrue(report["deterministic_algorithms"])

    def test_scheduled_sampling_is_repeatable(self):
        batch = synthetic_batch()
        configure_reproducibility(31415)
        first_model = tiny_model().train()
        first = first_model(
            batch["sst"],
            batch["ssh_sss"],
            target=batch["target"],
            teacher_forcing_ratio=0.5,
        )

        configure_reproducibility(31415)
        second_model = tiny_model().train()
        second = second_model(
            batch["sst"],
            batch["ssh_sss"],
            target=batch["target"],
            teacher_forcing_ratio=0.5,
        )
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_different_input_resolutions_and_backward(self):
        batch = synthetic_batch(batch_size=2)
        model = tiny_model().train()
        prediction = model(
            batch["sst"],
            batch["ssh_sss"],
            target=batch["target"],
            output_size=batch["target"].shape[-2:],
            teacher_forcing_ratio=1.0,
        )
        self.assertEqual(prediction.shape, batch["target"].shape)
        self.assertTrue(torch.isfinite(prediction).all())
        loss = masked_depth_mean_rmse_loss(
            prediction, batch["target"], batch["target_mask"]
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(
            sum(
                parameter.grad.abs().sum().item()
                for parameter in model.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )

    def test_bos_output_cannot_see_future_teacher_forcing_layers(self):
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

    def test_autoregressive_generate_and_teacher_forcing_schedule(self):
        batch = synthetic_batch()
        model = tiny_model().eval()
        with torch.no_grad():
            prediction = model.generate(
                batch["sst"],
                batch["ssh_sss"],
                output_size=batch["target"].shape[-2:],
            )
            scheduled = model(
                batch["sst"],
                batch["ssh_sss"],
                target=batch["target"],
                teacher_forcing_ratio=0.5,
            )
        self.assertEqual(prediction.shape, batch["target"].shape)
        self.assertEqual(scheduled.shape, batch["target"].shape)
        self.assertTrue(torch.isfinite(prediction).all())
        ratios = [
            teacher_forcing_ratio_for_epoch(epoch, 1.0, 0.5, 3)
            for epoch in range(4)
        ]
        self.assertEqual(ratios, [1.0, 0.75, 0.5, 0.5])

    def test_parallel_and_rolling_teacher_forcing_are_equivalent(self):
        batch = synthetic_batch()
        model = tiny_model().eval()
        output_size = batch["target"].shape[-2:]
        with torch.no_grad():
            memory, grid_size, spatial_position = model.encode_surface(
                batch["sst"], batch["ssh_sss"], output_size
            )
            parallel = model._parallel_teacher_forced(
                memory,
                grid_size,
                spatial_position,
                batch["target"],
                output_size,
            )
            rolling = model._autoregressive_decode(
                memory,
                grid_size,
                spatial_position,
                output_size,
                target=batch["target"],
                teacher_forcing_ratio=1.0,
            )
        torch.testing.assert_close(parallel, rolling, atol=1e-6, rtol=1e-5)

    def test_b2_aligned_checkpoint_can_be_loaded(self):
        model = tiny_model().eval()
        depth_tag = joint_du_unet_depth_tag(DEPTH_INDICES, 28)
        with tempfile.TemporaryDirectory() as temporary_root:
            checkpoint_dir = get_checkpoint_dir(
                PARADIGM_2DTO2D,
                "B3_DepthFormer",
                base_dir=temporary_root,
                target_var="temperature",
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_name = B3_DEPTHFORMER_CKPT_NAME_TEMPLATE.format(
                target_var="temperature",
                depth_tag=depth_tag,
            )
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            metadata = {
                "checkpoint_type": "B3_DepthFormer",
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
                    "best_epoch": 1,
                    "best_val_loss": 1.0,
                },
                checkpoint_path,
            )

            args = SimpleNamespace(
                method="B3_DepthFormer",
                checkpoint=checkpoint_path,
                checkpoint_dir=checkpoint_dir,
                output_dir=temporary_root,
                target_var="temperature",
                dataset_profile="glorys12v1_all_size",
                depth_indices="0,4,9,17,24,27",
            )
            dataset = SimpleNamespace(
                num_depths=28,
                get_normalization_reference=lambda: None,
            )
            loaded, loaded_path, indices, loaded_metadata = (
                _load_joint_du_unet_model(args, torch.device("cpu"), dataset)
            )
            self.assertEqual(loaded_path, checkpoint_path)
            self.assertEqual(indices, DEPTH_INDICES)
            self.assertEqual(loaded_metadata["depth_tag"], depth_tag)
            output_dir = get_output_dir(
                PARADIGM_2DTO2D,
                "B3_DepthFormer",
                base_dir=temporary_root,
                target_var="temperature",
            )
            self.assertTrue(
                output_dir.endswith("2dto2d/B3_DepthFormer/temperature")
            )
            self.assertEqual(loaded.num_depths, len(DEPTH_INDICES))


if __name__ == "__main__":
    unittest.main()
