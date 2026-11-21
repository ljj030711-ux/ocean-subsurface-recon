"""Case exports, strict cross-run matching, and independent comparison plots."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import test as evaluation
from config import GLORYS12V1_DEPTH_LEVELS_28M
from utils.case_fields import load_comparison, read_case_archive, write_case_archive
from utils.metrics import scalar_metrics
from utils.viz_layer_2d import plot_horizontal_comparison
from utils.viz_layer_profile import plot_section_comparison


DATES = ["2023-05-26", "2023-06-05"]
INDICES = [17, 27]


def case_fixture(method="B3_DepthFormer", reverse=False):
    dates, indices = (DATES[::-1], INDICES[::-1]) if reverse else (DATES, INDICES)
    y, x = np.mgrid[:13, :12]
    truth = np.stack([
        np.stack([25 - index * 0.4 + x * 0.03 + y * 0.04 + DATES.index(day) * 0.2
                  for index in indices]) for day in dates
    ]).astype(np.float32)
    prediction = truth + (0.2 if method == "B3_DepthFormer" else -0.35)
    mask = np.ones(truth.shape, dtype=bool)
    mask[:, :, 0, 0] = False
    payload = {
        "format_version": np.asarray(1), "method": np.asarray(method),
        "target_var": np.asarray("temperature"), "dataset_profile": np.asarray("glorys12v1_all_size"),
        "dates": np.asarray(dates), "depth_indices": np.asarray(indices),
        "depth_m": np.asarray([GLORYS12V1_DEPTH_LEVELS_28M[index] for index in indices], dtype=np.float32),
        "longitude": np.linspace(105, 121, 12), "latitude": np.linspace(0, 24, 13),
        "prediction": prediction, "truth": truth, "valid_mask": mask,
        "months": np.asarray([int(day[5:7]) for day in dates]), "roles": np.asarray(["manual"] * len(dates)),
    }
    metadata = {
        "stage": "test", "method": method, "target_var": "temperature",
        "data": {"dataset_profile": "glorys12v1_all_size", "eval_start": min(dates), "eval_end": max(dates)},
        "hyperparameters": {"depth_indices": indices, "depth_m": payload["depth_m"].tolist()},
        "model_config": {"depth_attention": "causal", "output_patch_size": 8},
        "paths": {"loaded_checkpoint": "/server/best.pth"},
    }
    if method == "B4_CPlus":
        metadata["model_config"]["architecture_version"] = 2
    return payload, metadata


class CaseMatchingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.a, self.b = self.root / "B3", self.root / "B4"
        self.a.mkdir()
        self.b.mkdir()
        self.path_a, self.path_b = self.a / "case_fields_a.npz", self.b / "case_fields_b.npz"
        self.data_a, self.meta_a = case_fixture()
        self.data_b, self.meta_b = case_fixture("B4_CPlus", reverse=True)
        self.save()

    def save(self):
        write_case_archive(self.path_a, self.data_a, self.meta_a)
        write_case_archive(self.path_b, self.data_b, self.meta_b)

    def load(self, dates=DATES, indices=INDICES, **kwargs):
        return load_comparison(self.a, self.b, dates, indices, **kwargs)

    def test_content_alignment_and_adaptive_labels(self):
        result = self.load()
        np.testing.assert_allclose((result.prediction_a - result.truth)[result.valid_mask], 0.2, atol=1e-6)
        np.testing.assert_allclose((result.prediction_b - result.truth)[result.valid_mask], -0.35, atol=1e-6)
        self.assertIn("causal", result.label_a)
        self.assertIn("v2", result.label_b)
        self.assertIn("ps8", result.label_b)
        self.assertFalse(result.valid_mask[:, :, 0, 0].any())

    def test_missing_layer_requires_new_inference(self):
        with self.assertRaisesRegex(ValueError, "没有储存原始索引 9 层的推理结果，需要重新推理"):
            self.load(indices=[9])

    def test_missing_date_and_path(self):
        with self.assertRaisesRegex(ValueError, "2023-05-27 日期.*需要重新推理"):
            self.load(dates=["2023-05-27"])
        with self.assertRaisesRegex(ValueError, "结果路径不存在"):
            load_comparison(self.root / "absent", self.b, DATES, INDICES)

    def test_metrics_only_directory(self):
        self.path_a.unlink()
        np.savez(self.a / "evaluation_metrics_a.npz", by_depth_mae=[1, 2])
        with self.assertRaisesRegex(ValueError, "没有预测场案例 NPZ"):
            self.load()

    def test_incomplete_npz_reports_path_and_redownload(self):
        # An interrupted transfer keeps the ZIP header but loses its directory.
        self.path_b.write_bytes(self.path_b.read_bytes()[:-22])
        with self.assertRaisesRegex(ValueError, "NPZ 文件损坏或下载不完整.*重新下载") as caught:
            self.load()
        self.assertIn(str(self.path_b), str(caught.exception))
        for module in ("utils.viz_layer_2d", "utils.viz_layer_profile"):
            process = subprocess.run(
                [sys.executable, "-m", module, "--run-a", str(self.a), "--run-b", str(self.b),
                 "--dates", DATES[0], "--depth-indices", "17,27"],
                capture_output=True, text=True,
            )
            self.assertEqual(process.returncode, 2, process.stderr)
            self.assertIn(str(self.path_b), process.stderr)
            self.assertIn("重新下载", process.stderr)
            self.assertNotIn("Traceback", process.stderr)

    def test_duplicate_sources(self):
        shutil.copy(self.path_a, self.a / "case_fields_duplicate.npz")
        with self.assertRaisesRegex(ValueError, "重复案例来源"):
            self.load()

    def test_coordinate_and_depth_mismatch(self):
        self.data_b["longitude"] += 0.1
        self.save()
        with self.assertRaisesRegex(ValueError, "longitude 不对应"):
            self.load()
        self.data_b, self.meta_b = case_fixture("B4_CPlus", reverse=True)
        self.data_b["depth_m"] += 1
        self.meta_b["hyperparameters"]["depth_m"] = self.data_b["depth_m"].tolist()
        self.save()
        with self.assertRaisesRegex(ValueError, "depth_m 不对应"):
            self.load()

    def test_truth_and_empty_common_mask(self):
        self.data_b["truth"] += 1
        self.save()
        with self.assertRaisesRegex(ValueError, "真值不一致"):
            self.load()
        self.data_b["valid_mask"][:] = False
        self.save()
        with self.assertRaisesRegex(ValueError, "没有共同有效海洋区域"):
            self.load()

    def test_v1_log_metadata_and_missing_metadata(self):
        np.savez(self.path_a, **self.data_a)
        with self.assertRaisesRegex(ValueError, "无法唯一确定模型版本"):
            self.load()
        log = self.root / "original_test.log"
        log.write_text("测试配置\n" + json.dumps(self.meta_a, indent=2) + "\n评估完成", encoding="utf-8")
        result = self.load(metadata_a=log)
        self.assertIn("causal", result.label_a)

    def test_v1_automatic_json_lookup_and_conflict(self):
        np.savez(self.path_a, **self.data_a)
        config = self.a / "B3_DepthFormer_test_config.json"
        config.write_text(json.dumps(self.meta_a), encoding="utf-8")
        self.load()
        conflicting = dict(self.meta_a, model_config={"depth_attention": "independent", "output_patch_size": 8})
        (self.a / "B3_DepthFormer_test_config_other.json").write_text(json.dumps(conflicting), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "匹配到 2 种配置"):
            self.load()

    def test_embedded_metadata_conflict_and_wrong_date(self):
        conflicting = dict(self.meta_a, model_config={"depth_attention": "independent", "output_patch_size": 8})
        config = self.root / "wrong.json"
        config.write_text(json.dumps(conflicting), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "版本或权重来源冲突"):
            self.load(metadata_a=config)
        self.meta_a["data"]["eval_start"] = "2024-01-01"
        self.save()
        with self.assertRaisesRegex(ValueError, "元数据日期、深度或数据集不对应"):
            self.load()

    def test_invalid_month_and_shape(self):
        self.data_a["months"][:] = 1
        with self.assertRaisesRegex(ValueError, "months 与 dates 不对应"):
            self.save()
        self.data_a, self.meta_a = case_fixture()
        self.data_a["truth"] = self.data_a["truth"][:, :1]
        with self.assertRaisesRegex(ValueError, "truth 的形状"):
            self.save()

    def test_five_panel_shared_scales(self):
        comparison = self.load()
        figure = plot_horizontal_comparison(comparison, DATES[0], self.root / "horizontal.png")
        self.assertEqual(len(figure.axes), 14)
        for start in (0, 7):
            panels = figure.axes[start:start + 5]
            self.assertIs(panels[0].collections[0].norm, panels[2].collections[0].norm)
            self.assertIs(panels[3].collections[0].norm, panels[4].collections[0].norm)
            self.assertEqual(panels[3].collections[0].norm.vmin, 0)
        self.assertTrue((self.root / "horizontal.png").is_file())

    def test_section_uses_shared_mask_before_band_average(self):
        # Exclude a large value only in B's mask: A's band must exclude it too.
        self.data_a["truth"][0, :, 5, 2] = 1000
        self.data_a["prediction"][0, :, 5, 2] = 2000
        self.data_b["valid_mask"][1, :, 5, 2] = False
        self.save()
        comparison = self.load()
        figure = plot_section_comparison(comparison, DATES[0], "lat", 5, 3, self.root / "section.png")
        self.assertEqual(len(figure.axes), 7)
        self.assertTrue(figure.axes[0].yaxis_inverted())
        np.testing.assert_array_equal(figure.axes[0].get_ylim(), comparison.depth_m[::-1])
        plotted = np.asarray(figure.axes[0].collections[0].get_array()).reshape(2, 12)
        expected = np.nanmean(comparison.truth[0, :, 4:7], axis=1)
        np.testing.assert_allclose(plotted, expected)
        with self.assertRaisesRegex(ValueError, "至少需要两个"):
            plot_section_comparison(self.load(indices=[17]), DATES[0], "lat", 5, 1, self.root / "invalid.png")

    def test_local_modules_do_not_import_torch(self):
        for module in ("utils.viz_layer_2d", "utils.viz_layer_profile"):
            command = (
                "import sys, runpy; sys.modules['torch'] = None; "
                f"sys.argv = ['{module}', '--help']; runpy.run_module('{module}', run_name='__main__')"
            )
            process = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)


class SmallDataset(torch.utils.data.Dataset):
    """Deterministic 4-day normalized data, with the real original depth mapping."""
    def __init__(self, data_dir, **kwargs):
        self.start_date, self.end_date = "2002-01-01", "2002-01-04"
        self.num_depths = 28
        self.months = np.ones(4, dtype=np.int64)
        self.normalization_stats_path = "fixed-normalization.npz"
        self.normalization_stats_loaded = True
        self.quality_report = kwargs.get("quality_report")
        self.climatology = np.broadcast_to(np.arange(28, dtype=np.float32)[None, :, None, None], (12, 28, 13, 12)).copy()

    def __len__(self):
        return 4

    def __getitem__(self, index):
        y, x = np.mgrid[:13, :12]
        target = np.broadcast_to((x * 0.05 + y * 0.07 + index * 0.2)[None], (28, 13, 12)).copy().astype(np.float32)
        mask = np.ones(target.shape, dtype=np.float32)
        mask[:, 0, 0] = 0
        return {"sst": torch.full((1, 13, 12), float(index)), "ssh_sss": torch.zeros(2, 13, 12),
                "target": torch.from_numpy(target), "target_mask": torch.from_numpy(mask)}

    def get_norm_stats(self):
        return {"normalization": "monthly_climatology_layer_std", "target": {
            "climatology": self.climatology, "layer_std": np.ones(28, dtype=np.float32)}}

    def get_normalization_reference(self):
        return {"fixed_fixture": True}


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_calls = []

    def forward(self, sst, ssh_sss, output_size=None):
        self.batch_calls.append(sst[:, 0, 0, 0].tolist())
        y, x = np.mgrid[:13, :12]
        spatial = torch.from_numpy((x * 0.05 + y * 0.07).astype(np.float32))
        days = sst[:, 0, 0, 0, None, None, None]
        return (spatial[None, None] + days * 0.2 + 0.1).expand(-1, 6, -1, -1)

    def model_config(self):
        return {"depth_attention": "causal", "output_patch_size": 8}


def evaluation_args(output_root, automatic=False, single=False):
    command = ["test.py", "--method", "B3_DepthFormer", "--dataset-profile", "glorys12v1_all_size",
               "--data-dir", "unused", "--output-dir", str(output_root), "--checkpoint", "/server/best.pth",
               "--depth-indices", "0,4,9,17,24,27", "--target-levels", "3,5", "--quality-report"]
    if single:
        command += ["--select-day", "2002-01-02"]
    else:
        command += ["--eval-start", "2002-01-01", "--eval-end", "2002-01-04", "--eval-batch-size", "2"]
        if not automatic:
            command += ["--panel-days", "2002-01-02,2002-01-04"]
    with mock.patch.object(sys, "argv", command):
        args = evaluation.parse_args()
    args.reproducibility = {"fixed_fixture": True}
    return args


@contextlib.contextmanager
def fake_evaluation(module, model):
    with mock.patch("datasets.dataset_2dto2d.Dataset2Dto2D", side_effect=SmallDataset) as constructor, \
         mock.patch.object(module, "_get_device", return_value=torch.device("cpu")), \
         mock.patch.object(module, "_load_joint_du_unet_model", return_value=(
             model, "/server/best.pth", [0, 4, 9, 17, 24, 27], {"model_config": model.model_config()}
         )), mock.patch("matplotlib.figure.Figure.savefig", side_effect=AssertionError("测试端不能绘图")), \
         contextlib.redirect_stdout(io.StringIO()):
        yield constructor


class EvaluationExportTest(unittest.TestCase):
    def test_manual_period_exports_evaluated_values_without_extra_forward(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, model = evaluation_args(root), SmallModel()
            with fake_evaluation(evaluation, model) as constructor:
                evaluation.evaluate_2dto2d_period(args, str(root))
            self.assertFalse(constructor.call_args.kwargs["quality_report"])
            self.assertEqual(model.batch_calls, [[0, 1], [2, 3]])
            self.assertFalse(list(root.rglob("*.png")))
            archive = read_case_archive(next(root.glob("case_fields/*.npz")))
            self.assertEqual(archive["dates"].tolist(), ["2002-01-02", "2002-01-04"])
            self.assertEqual(archive["depth_indices"].tolist(), INDICES)
            np.testing.assert_allclose((archive["prediction"] - archive["truth"])[archive["valid_mask"]], 0.1, atol=2e-6)
            with np.load(next(root.glob("evaluation_metrics*.npz"))) as metrics:
                self.assertAlmostEqual(float(metrics["overall_mae"]), 0.1, places=5)
            summary = json.loads(next(root.glob("summary*.json")).read_text())
            self.assertFalse(summary["spatial_panels"]["enabled"])
            self.assertTrue(summary["case_fields"]["file"].startswith("case_fields/"))

    def test_automatic_selection_preserves_batches_and_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, model = evaluation_args(root, automatic=True), SmallModel()
            with fake_evaluation(evaluation, model):
                evaluation.evaluate_2dto2d_period(args, str(root))
            self.assertEqual(model.batch_calls[:2], [[0, 1], [2, 3]])
            self.assertLessEqual(len(model.batch_calls), 4)
            archive = read_case_archive(next(root.glob("case_fields/*.npz")))
            self.assertEqual(len(archive["dates"]), 3)
            self.assertEqual(archive["roles"].tolist(), ["p10", "p50", "p90"])
            self.assertFalse(list(root.rglob("*.png")))

    def test_single_day_export_preserves_prediction_and_scalar_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, model = evaluation_args(root, single=True), SmallModel()
            with fake_evaluation(evaluation, model):
                prediction, truth, mask = evaluation.predict_2dto2d(args)
                reference = scalar_metrics(truth, prediction, mask)
                evaluation.save_outputs(args, prediction, truth, str(root), mask)
            self.assertEqual(model.batch_calls, [[1]])
            archive = read_case_archive(next(root.glob("case_fields/*.npz")))
            np.testing.assert_array_equal(archive["prediction"][archive["valid_mask"]], prediction[:, [3, 5]][archive["valid_mask"]])
            np.testing.assert_array_equal(np.load(next(root.glob("pred*.npy"))), prediction)
            summary = json.loads(next(root.glob("summary*.json")).read_text())
            self.assertEqual(summary["variables"]["temperature"], reference)
            self.assertFalse(list(root.rglob("*.png")))


if __name__ == "__main__":
    unittest.main()
