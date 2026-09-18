"""Structural checks for the parameter-matched B3 depth-attention ablation."""

import unittest
from pathlib import Path

import torch

from config import B3_DEPTHFORMER_BATCH_SIZE
from models.b3_depthformer import B3DepthFormer
from models.b3_depthformer.decoder import CausalDepthDecoder
from scripts.run_depth_fusion_ablation import (
    TRAIN_BATCH_SIZE,
    build_commands,
    build_parser,
)


def tiny_model(mode):
    torch.manual_seed(42)
    return B3DepthFormer(
        depth_values=[0.5, 5.0, 50.0],
        d_model=8,
        nhead=2,
        encoder_layers=1,
        decoder_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        sst_patch_size=2,
        surface_patch_size=2,
        output_patch_size=2,
        refinement_channels=4,
        base_grid_size=(2, 2),
        depth_attention=mode,
    )


class B3DepthAttentionAblationTest(unittest.TestCase):
    def test_reference_batch_size_is_sixteen_everywhere(self):
        self.assertEqual(B3_DEPTHFORMER_BATCH_SIZE, 16)
        self.assertEqual(TRAIN_BATCH_SIZE, 16)
        args = build_parser().parse_args([])
        self.assertTrue(args.experiment_root.endswith("_bs16"))
        train, evaluate, _ = build_commands(
            args, Path("/tmp/b3-depth-attention-test"), "causal"
        )
        self.assertEqual(train[train.index("--batch-size") + 1], "16")
        self.assertEqual(evaluate[evaluate.index("--eval-batch-size") + 1], "1")

    def test_masks(self):
        causal = CausalDepthDecoder.causal_mask(3, torch.device("cpu"))
        independent = CausalDepthDecoder.independent_mask(
            3, torch.device("cpu")
        )
        torch.testing.assert_close(
            causal,
            torch.tensor(
                [[False, True, True], [False, False, True], [False, False, False]]
            ),
        )
        torch.testing.assert_close(
            independent,
            torch.tensor(
                [[False, True, True], [True, False, True], [True, True, False]]
            ),
        )

    def test_parameter_count_and_initialization_match(self):
        causal = tiny_model("causal")
        independent = tiny_model("independent")
        self.assertEqual(
            sum(parameter.numel() for parameter in causal.parameters()),
            sum(parameter.numel() for parameter in independent.parameters()),
        )
        for name, value in causal.state_dict().items():
            torch.testing.assert_close(
                value, independent.state_dict()[name], rtol=0, atol=0
            )

    def test_independent_attention_blocks_cross_depth_tokens(self):
        torch.manual_seed(7)
        query = torch.randn(1, 3, 2, 8)
        changed = query.clone()
        changed[:, 0] += torch.arange(8, dtype=query.dtype).view(1, 1, 8)
        memory = torch.randn(1, 2, 8)
        causal = CausalDepthDecoder(8, 2, 1, 16, 0.0, "causal").eval()
        independent = CausalDepthDecoder(
            8, 2, 1, 16, 0.0, "independent"
        ).eval()
        independent.load_state_dict(causal.state_dict())
        with torch.no_grad():
            causal_delta = (causal(query, memory) - causal(changed, memory))[:, 2]
            independent_delta = (
                independent(query, memory) - independent(changed, memory)
            )[:, 2]
        self.assertGreater(causal_delta.abs().max().item(), 1e-6)
        torch.testing.assert_close(
            independent_delta,
            torch.zeros_like(independent_delta),
            rtol=0,
            atol=1e-6,
        )

    def test_legacy_default_is_causal(self):
        model = B3DepthFormer(
            depth_values=[0.5],
            d_model=8,
            nhead=2,
            encoder_layers=1,
            decoder_layers=1,
            dim_feedforward=16,
            dropout=0.0,
            sst_patch_size=2,
            surface_patch_size=2,
            output_patch_size=2,
            refinement_channels=4,
            base_grid_size=(2, 2),
        )
        self.assertEqual(model.depth_attention, "causal")
        self.assertEqual(model.model_config()["depth_attention"], "causal")


if __name__ == "__main__":
    unittest.main()
