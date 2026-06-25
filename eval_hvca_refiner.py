"""Evaluate HVCARefiner against the frozen Du_Unet baseline T0.

Outputs are aligned with the existing Du_Unet period-evaluation artifacts:
evaluation_metrics_*.npz, summary_*.json, metrics_by_depth_*.png, and
metrics_by_day_*.png. Each plot compares baseline_T0 and HVCA_refined.
"""

import argparse
import json
import os

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    CHECKPOINTS_ROOT,
    DATA_DIR,
    DATA_START_DATE,
    OUTPUTS_ROOT,
    PARADIGM_2DTO2D,
    TEST_END_DATE,
    TEST_START_DATE,
    get_checkpoint_dir,
)
from datasets.dataset_2dto2d import Dataset2Dto2D
from datasets.date_utils import date_to_index
from models.hvca_refiner import HVCARefiner
from train_hvca_refiner import (
    HvcaCacheDataset,
    cache_ready,
    dataloader_kwargs,
    generate_t0_cache,
    get_device,
    load_du_unet_subset,
    parse_depths,
    resolve_depth_indices,
)
from utils.metrics import RegressionMetricAccumulator, scalar_metrics


PERIOD_METRIC_NAMES = ("mae", "rmse", "r2", "correlation")


def finite_json_value(value):
    value = float(value)
    return value if np.isfinite(value) else None


def period_metrics_to_json(metrics):
    out = {name: finite_json_value(metrics[name]) for name in PERIOD_METRIC_NAMES}
    out["valid_count"] = int(np.asarray(metrics["valid_count"]).item())
    return out


def grouped_metrics_to_records(metrics, labels, label_name, extra_values=None):
    records = []
    for i, label in enumerate(labels):
        record = {label_name: label}
        if extra_values:
            record.update({name: values[i] for name, values in extra_values.items()})
        for metric_name in PERIOD_METRIC_NAMES:
            record[metric_name] = finite_json_value(np.asarray(metrics[metric_name])[i])
        record["valid_count"] = int(np.asarray(metrics["valid_count"])[i])
        records.append(record)
    return records


def inverse_target_subset(array, months, target_stats, depth_indices):
    """Inverse-transform normalized target subset back to physical units."""
    arr = np.asarray(array, dtype=np.float32)
    months = np.asarray(months, dtype=np.int64)
    climatology = np.asarray(target_stats["climatology"], dtype=np.float32)[:, depth_indices]
    layer_std = np.asarray(target_stats["layer_std"], dtype=np.float32)[depth_indices]
    return arr * layer_std.reshape(1, -1, 1, 1) + climatology[months - 1]


def plot_comparison_metrics(x, baseline, refined, xlabel, title, output_path, date_labels=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150)
    titles = {
        "mae": "MAE",
        "rmse": "RMSE",
        "r2": "R2",
        "correlation": "Correlation",
    }
    for ax, metric_name in zip(axes.flat, PERIOD_METRIC_NAMES):
        ax.plot(x, baseline[metric_name], label="baseline_T0", linewidth=1.5)
        ax.plot(x, refined[metric_name], label="HVCA_refined", linewidth=1.5)
        ax.set_title(titles[metric_name])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(titles[metric_name])
        ax.grid(True, alpha=0.3)
        ax.legend()
        if date_labels is not None:
            tick_step = max(len(x) // 8, 1)
            tick_indices = np.arange(0, len(x), tick_step)
            ax.set_xticks(tick_indices)
            ax.set_xticklabels(
                [date_labels[index] for index in tick_indices],
                rotation=35,
                ha="right",
            )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"指标对比图已保存至：{output_path}")


def compute_grad_arrays(y_true, y_pred, mask, depth_values):
    depth = np.asarray(depth_values, dtype=np.float32)
    dz = np.diff(depth).reshape(1, -1, 1, 1)
    grad_true = (y_true[:, 1:] - y_true[:, :-1]) / dz
    grad_pred = (y_pred[:, 1:] - y_pred[:, :-1]) / dz
    grad_mask = mask[:, 1:].astype(bool) & mask[:, :-1].astype(bool)
    return grad_true, grad_pred, grad_mask


def ensure_eval_cache(args, depth_values, depth_indices, device):
    if cache_ready(args.t0_cache_dir):
        print(f"[cache] 发现已有 test cache，直接读取: {args.t0_cache_dir}")
        return

    dataset = Dataset2Dto2D(
        args.data_dir,
        normalize=True,
        target_var=args.target_var,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    start_idx = date_to_index(args.eval_start, dataset.start_date, dataset.end_date)
    end_idx = date_to_index(args.eval_end, dataset.start_date, dataset.end_date)
    if end_idx < start_idx:
        raise ValueError(f"eval-end 不能早于 eval-start: {args.eval_end} < {args.eval_start}")
    eval_indices = list(range(start_idx, end_idx + 1))
    if args.max_samples is not None:
        eval_indices = eval_indices[: args.max_samples]
    models = load_du_unet_subset(args, device, depth_indices)
    generate_t0_cache(
        cache_dir=args.t0_cache_dir,
        dataset=dataset,
        indices=eval_indices,
        models=models,
        depth_values=depth_values,
        depth_indices=depth_indices,
        device=device,
        target_var=args.target_var,
        split_name="test",
        shard_size=args.cache_shard_size,
    )


def load_checkpoint(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 HVCA checkpoint: {path}")
    return torch.load(path, map_location=device)


@torch.no_grad()
def evaluate(args):
    device = get_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    depth_values = ckpt.get("depth_values") or parse_depths(args.depths)
    depth_values = [int(float(d)) for d in depth_values]
    depth_indices = resolve_depth_indices(depth_values)
    depth_tensor = torch.as_tensor(depth_values, dtype=torch.float32, device=device)

    ensure_eval_cache(args, depth_values, depth_indices, device)

    dataset = Dataset2Dto2D(
        args.data_dir,
        normalize=True,
        target_var=args.target_var,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    target_stats = dataset.get_norm_stats()["target"]

    model_config = ckpt.get(
        "model_config",
        {
            "dim": args.dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "column_chunk_size": args.column_chunk_size,
        },
    )
    model = HVCARefiner(**model_config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader = DataLoader(
        HvcaCacheDataset(args.t0_cache_dir, shard_cache_size=args.shard_cache_size),
        batch_size=args.batch_size,
        shuffle=False,
        **dataloader_kwargs(args, device),
    )

    baseline_overall = RegressionMetricAccumulator()
    refined_overall = RegressionMetricAccumulator()
    baseline_depth = RegressionMetricAccumulator()
    refined_depth = RegressionMetricAccumulator()
    baseline_thermo = RegressionMetricAccumulator()
    refined_thermo = RegressionMetricAccumulator()
    baseline_grad = RegressionMetricAccumulator()
    refined_grad = RegressionMetricAccumulator()

    baseline_daily_records = []
    refined_daily_records = []
    eval_dates = []
    thermo_mask = (
        (np.asarray(depth_values) >= args.thermocline_min)
        & (np.asarray(depth_values) <= args.thermocline_max)
    )

    use_amp = device.type == "cuda"
    for batch in loader:
        T0_norm_t = batch["T0"].to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            refined_norm_t = model(T0_norm_t, depth_tensor)

        T0_norm = batch["T0"].numpy()
        refined_norm = refined_norm_t.cpu().numpy()
        true_norm = batch["T_true"].numpy()
        mask = batch["target_mask"].numpy()
        months = batch["months"].numpy()
        sample_ids = list(batch["sample_id"])

        y_base = inverse_target_subset(T0_norm, months, target_stats, depth_indices)
        y_refined = inverse_target_subset(refined_norm, months, target_stats, depth_indices)
        y_true = inverse_target_subset(true_norm, months, target_stats, depth_indices)

        baseline_overall.update(y_true, y_base, mask=mask, axis=None)
        refined_overall.update(y_true, y_refined, mask=mask, axis=None)
        baseline_depth.update(y_true, y_base, mask=mask, axis=(0, 2, 3))
        refined_depth.update(y_true, y_refined, mask=mask, axis=(0, 2, 3))

        if np.any(thermo_mask):
            baseline_thermo.update(
                y_true[:, thermo_mask],
                y_base[:, thermo_mask],
                mask=mask[:, thermo_mask],
                axis=None,
            )
            refined_thermo.update(
                y_true[:, thermo_mask],
                y_refined[:, thermo_mask],
                mask=mask[:, thermo_mask],
                axis=None,
            )

        grad_true, grad_base, grad_mask = compute_grad_arrays(
            y_true, y_base, mask, depth_values
        )
        _, grad_refined, _ = compute_grad_arrays(y_true, y_refined, mask, depth_values)
        baseline_grad.update(grad_true, grad_base, mask=grad_mask, axis=None)
        refined_grad.update(grad_true, grad_refined, mask=grad_mask, axis=None)

        for i, sample_id in enumerate(sample_ids):
            base_daily = scalar_metrics(y_true[i], y_base[i], mask=mask[i])
            refined_daily = scalar_metrics(y_true[i], y_refined[i], mask=mask[i])
            valid = (
                mask[i].astype(bool)
                & np.isfinite(y_true[i])
                & np.isfinite(y_base[i])
                & np.isfinite(y_refined[i])
            )
            base_record = {
                "date": sample_id,
                **{name: finite_json_value(base_daily[name]) for name in PERIOD_METRIC_NAMES},
                "valid_count": int(np.count_nonzero(valid)),
            }
            refined_record = {
                "date": sample_id,
                **{
                    name: finite_json_value(refined_daily[name])
                    for name in PERIOD_METRIC_NAMES
                },
                "valid_count": int(np.count_nonzero(valid)),
            }
            baseline_daily_records.append(base_record)
            refined_daily_records.append(refined_record)
            eval_dates.append(sample_id)

        print(f"评估进度: {len(eval_dates)} samples")

    baseline_overall_metrics = baseline_overall.compute()
    refined_overall_metrics = refined_overall.compute()
    baseline_depth_metrics = baseline_depth.compute()
    refined_depth_metrics = refined_depth.compute()
    baseline_daily_metrics = {
        name: np.asarray([record[name] for record in baseline_daily_records], dtype=np.float64)
        for name in PERIOD_METRIC_NAMES
    }
    refined_daily_metrics = {
        name: np.asarray([record[name] for record in refined_daily_records], dtype=np.float64)
        for name in PERIOD_METRIC_NAMES
    }
    baseline_daily_metrics["valid_count"] = np.asarray(
        [record["valid_count"] for record in baseline_daily_records], dtype=np.int64
    )
    refined_daily_metrics["valid_count"] = np.asarray(
        [record["valid_count"] for record in refined_daily_records], dtype=np.int64
    )

    period_tag = f"{args.eval_start}_{args.eval_end}"
    os.makedirs(args.save_dir, exist_ok=True)
    metrics_name = f"evaluation_metrics_HVCARefiner_{args.target_var}_{period_tag}.npz"
    summary_name = f"summary_HVCARefiner_{args.target_var}_{period_tag}.json"
    depth_plot_name = f"metrics_by_depth_HVCARefiner_{args.target_var}_{period_tag}.png"
    day_plot_name = f"metrics_by_day_HVCARefiner_{args.target_var}_{period_tag}.png"

    payload = {
        "dates": np.asarray(eval_dates),
        "depth_m": np.asarray(depth_values, dtype=np.int32),
        "baseline_overall_valid_count": baseline_overall_metrics["valid_count"],
        "baseline_by_depth_valid_count": baseline_depth_metrics["valid_count"],
        "baseline_by_day_valid_count": baseline_daily_metrics["valid_count"],
        "refined_overall_valid_count": refined_overall_metrics["valid_count"],
        "refined_by_depth_valid_count": refined_depth_metrics["valid_count"],
        "refined_by_day_valid_count": refined_daily_metrics["valid_count"],
    }
    for name in PERIOD_METRIC_NAMES:
        payload[f"baseline_overall_{name}"] = baseline_overall_metrics[name]
        payload[f"baseline_by_depth_{name}"] = baseline_depth_metrics[name]
        payload[f"baseline_by_day_{name}"] = baseline_daily_metrics[name]
        payload[f"refined_overall_{name}"] = refined_overall_metrics[name]
        payload[f"refined_by_depth_{name}"] = refined_depth_metrics[name]
        payload[f"refined_by_day_{name}"] = refined_daily_metrics[name]

    if np.any(thermo_mask):
        payload["baseline_thermocline_rmse"] = baseline_thermo.compute()["rmse"]
        payload["refined_thermocline_rmse"] = refined_thermo.compute()["rmse"]
    payload["baseline_gradz_rmse"] = baseline_grad.compute()["rmse"]
    payload["refined_gradz_rmse"] = refined_grad.compute()["rmse"]

    np.savez(os.path.join(args.save_dir, metrics_name), **payload)

    baseline_depth_records = grouped_metrics_to_records(
        baseline_depth_metrics,
        list(range(len(depth_values))),
        "depth_index",
        extra_values={"depth_m": depth_values},
    )
    refined_depth_records = grouped_metrics_to_records(
        refined_depth_metrics,
        list(range(len(depth_values))),
        "depth_index",
        extra_values={"depth_m": depth_values},
    )
    summary = {
        "mode": "period",
        "method": "HVCARefiner",
        "target_var": args.target_var,
        "eval_start": args.eval_start,
        "eval_end": args.eval_end,
        "num_days": len(eval_dates),
        "metric_units": {
            "mae": "degC" if args.target_var == "temperature" else "psu",
            "rmse": "degC" if args.target_var == "temperature" else "psu",
            "r2": "dimensionless",
            "correlation": "dimensionless",
        },
        "variables": {
            args.target_var: {
                "baseline_T0": {
                    "overall": period_metrics_to_json(baseline_overall_metrics),
                    "by_depth": baseline_depth_records,
                    "by_day": baseline_daily_records,
                },
                "HVCA_refined": {
                    "overall": period_metrics_to_json(refined_overall_metrics),
                    "by_depth": refined_depth_records,
                    "by_day": refined_daily_records,
                },
            }
        },
        "extra_metrics": {
            "thermocline_range_m": [args.thermocline_min, args.thermocline_max],
            "baseline_thermocline_rmse": finite_json_value(
                payload.get("baseline_thermocline_rmse", np.nan)
            ),
            "refined_thermocline_rmse": finite_json_value(
                payload.get("refined_thermocline_rmse", np.nan)
            ),
            "baseline_gradz_rmse": finite_json_value(payload["baseline_gradz_rmse"]),
            "refined_gradz_rmse": finite_json_value(payload["refined_gradz_rmse"]),
        },
    }
    with open(os.path.join(args.save_dir, summary_name), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_comparison_metrics(
        np.asarray(depth_values),
        baseline_depth_metrics,
        refined_depth_metrics,
        "Depth / m",
        f"HVCARefiner {args.target_var} metrics by depth",
        os.path.join(args.save_dir, depth_plot_name),
    )
    plot_comparison_metrics(
        np.arange(len(eval_dates)),
        baseline_daily_metrics,
        refined_daily_metrics,
        "Date",
        f"HVCARefiner {args.target_var} metrics by day",
        os.path.join(args.save_dir, day_plot_name),
        date_labels=eval_dates,
    )

    print(f"指标文件: {metrics_name}")
    print(f"汇总文件: {summary_name}")
    print("评估完成。")


def parse_args():
    default_ckpt_dir = get_checkpoint_dir(
        PARADIGM_2DTO2D, "Du_Unet", base_dir=CHECKPOINTS_ROOT
    )
    parser = argparse.ArgumentParser(description="Evaluate HVCARefiner")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--t0-cache-dir", required=True)
    parser.add_argument("--base-checkpoint-dir", default=default_ckpt_dir)
    parser.add_argument("--target-var", choices=["temperature", "salinity"], default="temperature")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--start-date", default=DATA_START_DATE)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--eval-start", default=TEST_START_DATE)
    parser.add_argument("--eval-end", default=TEST_END_DATE)
    parser.add_argument("--depths", default=None, help="仅用于旧 checkpoint 无 depth_values 时")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cache-shard-size", type=int, default=64)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--column-chunk-size",
        type=int,
        default=8192,
        help="仅用于旧 checkpoint 无 model_config 时的 attention 分块大小",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="cache DataLoader worker 数")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--shard-cache-size", type=int, default=4, help="每个 worker 内缓存的 shard 数")
    parser.add_argument("--thermocline-min", type=float, default=50.0)
    parser.add_argument("--thermocline-max", type=float, default=200.0)
    parser.add_argument("--save-dir", default=os.path.join(OUTPUTS_ROOT, "hvca_eval"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.cache_shard_size <= 0:
        raise ValueError("--batch-size 和 --cache-shard-size 必须大于 0")
    evaluate(args)


if __name__ == "__main__":
    main()
