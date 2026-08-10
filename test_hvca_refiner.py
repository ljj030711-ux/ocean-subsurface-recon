"""测试 HVCARefiner，并与冻结 Du_Unet 生成的 T0 初猜场进行指标对比。"""

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
    DEFAULT_DATASET_PROFILE,
    OUTPUTS_ROOT,
    PARADIGM_2DTO2D,
    TEST_END_DATE,
    TEST_START_DATE,
    dataset_profile_names,
    get_2dto2d_depth_levels,
    get_2dto2d_lon_lat_range,
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
    load_cache_metadata,
    load_du_unet_subset,
    parse_depths,
    resolve_depth_indices,
)
from utils.metrics import RegressionMetricAccumulator, scalar_metrics
from utils.run_config import (
    count_model_parameters,
    load_run_config,
    print_and_save_run_config,
)
from utils.viz_layer_2d import plot_hvca_refiner_panel
from utils.viz_layer_profile import plot_hvca_refiner_section_panel


PERIOD_METRIC_NAMES = ("mae", "rmse", "r2", "correlation")
DEFAULT_PANEL_LEVELS = "0,4,9,17,24"
DEFAULT_SECTION_PERCENTILES = "10,25,50,75,90"


def default_hvca_output_dir():
    return os.path.join(OUTPUTS_ROOT, PARADIGM_2DTO2D, "HVCARefiner")


def active_lon_lat_range(args):
    return get_2dto2d_lon_lat_range(args.dataset_profile)


def use_lazy_2dto2d_data(args):
    return args.lazy_data or args.dataset_profile == "glorys12v1_all_size"


def run_config_path(save_dir, stage):
    return os.path.join(save_dir, f"HVCARefiner_{stage}_config.json")


def legacy_run_config_path(stage):
    return os.path.join(
        OUTPUTS_ROOT,
        PARADIGM_2DTO2D,
        "HVCARefiner",
        f"HVCARefiner_{stage}_config.json",
    )


def checkpoint_train_config_path(checkpoint_path):
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    preferred = run_config_path(checkpoint_dir, "train")
    if os.path.exists(preferred):
        return preferred
    return legacy_run_config_path("train")


def load_training_strategy_reference(train_config_path):
    train_config = load_run_config(train_config_path)
    if train_config and train_config.get("training_strategy"):
        strategy = {
            "source": "train_config",
            "config_path": train_config_path,
        }
        strategy.update(train_config["training_strategy"])
        return strategy
    return {
        "source": "not_available_in_test",
        "config_path": train_config_path,
        "epoch": {
            "planned": None,
            "indexing": None,
        },
        "early_stopping": {
            "enabled": None,
            "monitor": None,
            "mode": None,
            "patience": None,
        },
        "optimizer": {
            "name": None,
            "lr": None,
            "weight_decay": None,
        },
        "scheduler": {
            "name": None,
        },
        "warmup": {
            "enabled": None,
            "epochs": None,
        },
    }


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


def parse_index_list(value, name):
    if value is None:
        return []
    indices = []
    for item in str(value).split(","):
        text = item.strip()
        if not text:
            continue
        try:
            indices.append(int(text))
        except ValueError as exc:
            raise ValueError(f"{name} 必须是逗号分隔的整数索引: {value}") from exc
    return indices


def parse_percentile_list(value, name):
    percentiles = []
    for item in str(value).split(","):
        text = item.strip()
        if not text:
            continue
        try:
            pct = float(text)
        except ValueError as exc:
            raise ValueError(f"{name} 包含非法百分位: {text}") from exc
        if pct < 0.0 or pct > 100.0:
            raise ValueError(f"{name} 百分位需位于 [0,100]，实际: {pct}")
        percentiles.append(pct)
    if not percentiles:
        raise ValueError(f"{name} 不能为空")
    return percentiles


def percentile_indices(percentiles, size):
    if size <= 0:
        raise ValueError("size 必须大于 0")
    ordered = []
    for pct in percentiles:
        idx = int(round((pct / 100.0) * (size - 1)))
        idx = min(max(idx, 0), size - 1)
        if idx not in ordered:
            ordered.append(idx)
    return ordered


def build_section_specs(args, h, w):
    if args.multi_section_panels:
        percentiles = parse_percentile_list(
            args.section_percentiles, "--section-percentiles"
        )
        specs = []
        for axis, size in (("lat", h), ("lon", w)):
            seen = set()
            for pct in percentiles:
                idx = percentile_indices([pct], size)[0]
                if idx in seen:
                    continue
                seen.add(idx)
                specs.append(
                    {
                        "axis": axis,
                        "index": int(idx),
                        "percentile": float(pct),
                        "mode": "multi",
                    }
                )
        return specs

    section_axis = args.section_axis
    section_size = h if section_axis == "lat" else w
    section_index = resolve_section_index(args.section_index, section_size)
    return [
        {
            "axis": section_axis,
            "index": int(section_index),
            "percentile": None,
            "mode": "single",
        }
    ]


def parse_panel_days(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def validate_target_levels(levels, depth_values):
    if not levels:
        raise ValueError("--target-levels 不能为空")
    max_index = len(depth_values) - 1
    invalid = [level for level in levels if level < 0 or level > max_index]
    if invalid:
        raise ValueError(
            f"--target-levels 超出当前 HVCA 深度范围 [0,{max_index}]: {invalid}"
        )
    ordered = []
    for level in levels:
        if level not in ordered:
            ordered.append(level)
    return ordered


def select_representative_days(baseline_records, refined_records):
    if len(baseline_records) != len(refined_records):
        raise ValueError("baseline/refined daily record 数量不一致")

    candidates = []
    for base, refined in zip(baseline_records, refined_records):
        if base["date"] != refined["date"]:
            raise ValueError(
                f"baseline/refined 日期不一致: {base['date']} vs {refined['date']}"
            )
        base_rmse = base.get("rmse")
        refined_rmse = refined.get("rmse")
        if base_rmse is None or refined_rmse is None:
            continue
        delta = float(refined_rmse) - float(base_rmse)
        if not np.isfinite(delta):
            continue
        candidates.append(
            {
                "date": base["date"],
                "baseline_rmse": float(base_rmse),
                "refined_rmse": float(refined_rmse),
                "rmse_delta": delta,
            }
        )

    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda item: item["rmse_delta"])
    selected = [
        ("best_improved", ordered[0]),
        ("median_change", ordered[len(ordered) // 2]),
        ("worst_degraded", ordered[-1]),
    ]

    records = []
    for role, item in selected:
        record = {"role": role, **item}
        records.append(record)
    return records


def resolve_panel_day_records(args, eval_dates, baseline_records, refined_records):
    eval_date_set = set(eval_dates)
    explicit_days = parse_panel_days(args.panel_days)
    if explicit_days:
        missing = [day for day in explicit_days if day not in eval_date_set]
        if missing:
            raise ValueError(
                f"--panel-days 中日期不在评估区间或 cache 中: {missing}"
            )
        return [
            {
                "role": "manual",
                "date": day,
                "baseline_rmse": None,
                "refined_rmse": None,
                "rmse_delta": None,
            }
            for day in explicit_days
        ]

    return select_representative_days(baseline_records, refined_records)


def resolve_section_index(value, size):
    text = str(value).strip().lower()
    if text in {"middle", "mid", "center", "centre"}:
        return size // 2
    try:
        index = int(text)
    except ValueError as exc:
        raise ValueError(f"--section-index 必须是整数或 middle，实际: {value}") from exc
    if index < 0 or index >= size:
        raise ValueError(f"--section-index 超出范围 [0,{size - 1}]: {index}")
    return index


@torch.no_grad()
def render_spatial_panels(
    args,
    model,
    loader,
    depth_tensor,
    target_stats,
    depth_indices,
    depth_values,
    target_levels,
    panel_day_records,
    device,
    use_amp,
):
    if not panel_day_records:
        print("[spatial-panels] 没有可绘制的代表日期，跳过二维解释图")
        return []

    panel_days = {record["date"] for record in panel_day_records}
    rendered = []
    model.eval()
    lon_range, lat_range = active_lon_lat_range(args)

    for batch in loader:
        sample_ids = list(batch["sample_id"])
        if not any(sample_id in panel_days for sample_id in sample_ids):
            continue

        T0_norm_t = batch["T0"].to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            refined_norm_t = model(T0_norm_t, depth_tensor)

        T0_norm = batch["T0"].numpy()
        refined_norm = refined_norm_t.cpu().numpy()
        true_norm = batch["T_true"].numpy()
        mask = batch["target_mask"].numpy()
        months = batch["months"].numpy()

        y_base = inverse_target_subset(T0_norm, months, target_stats, depth_indices)
        y_refined = inverse_target_subset(refined_norm, months, target_stats, depth_indices)
        y_true = inverse_target_subset(true_norm, months, target_stats, depth_indices)

        for i, sample_id in enumerate(sample_ids):
            if sample_id not in panel_days:
                continue
            for level in target_levels:
                valid = mask[i, level].astype(bool)
                baseline_2d = np.where(valid, y_base[i, level], np.nan)
                refined_2d = np.where(valid, y_refined[i, level], np.nan)
                true_2d = np.where(valid, y_true[i, level], np.nan)
                date_tag = sample_id.replace("-", "")
                filename = (
                    f"map_hvca_panel_{args.target_var}_lvl{level}_{date_tag}.png"
                )
                output_path = os.path.join(args.save_dir, filename)
                plot_hvca_refiner_panel(
                    baseline_2d,
                    refined_2d,
                    true_2d,
                    args.target_var,
                    level,
                    sample_id,
                    output_path,
                    lon_range=lon_range,
                    lat_range=lat_range,
                    depth_values=depth_values,
                )
                rendered.append(
                    {
                        "date": sample_id,
                        "depth_index": int(level),
                        "depth_m": float(depth_values[level]),
                        "file": filename,
                    }
                )

    missing = sorted(panel_days - {item["date"] for item in rendered})
    if missing:
        raise RuntimeError(f"以下 panel 日期未成功绘制: {missing}")
    return rendered


@torch.no_grad()
def render_section_panels(
    args,
    model,
    loader,
    depth_tensor,
    target_stats,
    depth_indices,
    depth_values,
    panel_day_records,
    device,
    use_amp,
):
    if not panel_day_records:
        print("[section-panels] 没有可绘制的代表日期，跳过垂向剖面图")
        return []

    panel_days = {record["date"] for record in panel_day_records}
    rendered = []
    model.eval()
    section_width = max(int(args.section_width), 1)
    lon_range, lat_range = active_lon_lat_range(args)

    for batch in loader:
        sample_ids = list(batch["sample_id"])
        if not any(sample_id in panel_days for sample_id in sample_ids):
            continue

        T0_norm_t = batch["T0"].to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            refined_norm_t = model(T0_norm_t, depth_tensor)

        T0_norm = batch["T0"].numpy()
        refined_norm = refined_norm_t.cpu().numpy()
        true_norm = batch["T_true"].numpy()
        mask = batch["target_mask"].numpy()
        months = batch["months"].numpy()

        y_base = inverse_target_subset(T0_norm, months, target_stats, depth_indices)
        y_refined = inverse_target_subset(refined_norm, months, target_stats, depth_indices)
        y_true = inverse_target_subset(true_norm, months, target_stats, depth_indices)

        _, h, w = y_base.shape[1:]
        section_specs = build_section_specs(args, h, w)

        for i, sample_id in enumerate(sample_ids):
            if sample_id not in panel_days:
                continue
            valid = mask[i].astype(bool)
            baseline_3d = np.where(valid, y_base[i], np.nan)
            refined_3d = np.where(valid, y_refined[i], np.nan)
            true_3d = np.where(valid, y_true[i], np.nan)
            date_tag = sample_id.replace("-", "")
            for spec in section_specs:
                section_axis = spec["axis"]
                section_index = spec["index"]
                if spec["mode"] == "multi":
                    pct_tag = f"p{spec['percentile']:g}".replace(".", "p")
                    section_tag = f"{section_axis}_{pct_tag}_idx{section_index}"
                else:
                    section_tag = f"{section_axis}{section_index}"
                filename = (
                    f"section_hvca_panel_{args.target_var}_"
                    f"{section_tag}_{date_tag}.png"
                )
                output_path = os.path.join(args.save_dir, filename)
                plot_hvca_refiner_section_panel(
                    baseline_3d,
                    refined_3d,
                    true_3d,
                    args.target_var,
                    section_axis,
                    section_index,
                    section_width,
                    sample_id,
                    output_path,
                    depth_values=depth_values,
                    lon_range=lon_range,
                    lat_range=lat_range,
                )
                rendered.append(
                    {
                        "date": sample_id,
                        "section_axis": section_axis,
                        "section_index": int(section_index),
                        "section_percentile": spec["percentile"],
                        "section_width": int(section_width),
                        "file": filename,
                    }
                )

    missing = sorted(panel_days - {item["date"] for item in rendered})
    if missing:
        raise RuntimeError(f"以下 section panel 日期未成功绘制: {missing}")
    return rendered


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


def ensure_eval_cache(
    args, dataset, depth_values, depth_indices, depth_levels, device
):
    if cache_ready(args.t0_cache_dir):
        print(f"[cache] 发现已有 test cache，直接读取: {args.t0_cache_dir}")
        return

    start_idx = date_to_index(args.eval_start, dataset.start_date, dataset.end_date)
    end_idx = date_to_index(args.eval_end, dataset.start_date, dataset.end_date)
    if end_idx < start_idx:
        raise ValueError(f"eval-end 不能早于 eval-start: {args.eval_end} < {args.eval_start}")
    eval_indices = list(range(start_idx, end_idx + 1))
    if args.max_samples is not None:
        eval_indices = eval_indices[: args.max_samples]
    if len(depth_levels) != dataset.num_depths:
        raise ValueError(
            f"dataset_profile={args.dataset_profile} 深度层数 {len(depth_levels)} "
            f"与 target 数据层数 {dataset.num_depths} 不一致"
        )
    models = load_du_unet_subset(args, device, depth_indices, depth_levels)
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
    profile_from_ckpt = ckpt.get("dataset_profile")
    if profile_from_ckpt and args.dataset_profile == DEFAULT_DATASET_PROFILE:
        args.dataset_profile = profile_from_ckpt
    checkpoint_norm_ref = ckpt.get("normalization_stats")
    if checkpoint_norm_ref and args.normalization_stats_path is None:
        checkpoint_stats_path = checkpoint_norm_ref.get("path")
        if checkpoint_stats_path and os.path.exists(checkpoint_stats_path):
            args.normalization_stats_path = checkpoint_stats_path
        elif checkpoint_stats_path:
            print(
                "[normalization] checkpoint 记录的统计文件不存在，将使用 data-dir 默认路径: "
                f"{checkpoint_stats_path}"
            )
    depth_levels = get_2dto2d_depth_levels(args.dataset_profile)
    depth_values = ckpt.get("depth_values") or parse_depths(args.depths, depth_levels)
    depth_values = [float(d) for d in depth_values]
    depth_indices = resolve_depth_indices(depth_values, depth_levels)
    depth_tensor = torch.as_tensor(depth_values, dtype=torch.float32, device=device)
    target_levels = []
    if args.spatial_panels:
        target_levels = validate_target_levels(
            parse_index_list(args.target_levels, "--target-levels"),
            depth_values,
        )
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
    loss_config = ckpt.get(
        "loss_config",
        {
            "primary": "masked_rmse_loss",
            "lambda_delta_z": 0.0,
            "delta_z": None,
            "source": "checkpoint_without_loss_config",
        },
    )
    model = HVCARefiner(**model_config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    train_config_path = checkpoint_train_config_path(args.checkpoint)
    training_strategy = load_training_strategy_reference(train_config_path)
    if isinstance(training_strategy.get("epoch"), dict):
        training_strategy["epoch"]["checkpoint_epoch"] = ckpt.get("epoch")
    training_strategy["checkpoint_best_val"] = ckpt.get("best_val")
    dataset = Dataset2Dto2D(
        args.data_dir,
        normalize=True,
        target_var=args.target_var,
        start_date=args.start_date,
        end_date=args.end_date,
        dataset_profile=args.dataset_profile,
        lazy=use_lazy_2dto2d_data(args),
        normalization_chunk_size=args.normalization_chunk_size,
        normalization_stats_path=args.normalization_stats_path,
        rebuild_normalization_stats=args.rebuild_normalization_stats,
    )
    args.normalization_stats_path = dataset.normalization_stats_path
    actual_norm_ref = dataset.get_normalization_reference()
    if (
        checkpoint_norm_ref
        and checkpoint_norm_ref.get("fingerprint")
        and actual_norm_ref.get("fingerprint") != checkpoint_norm_ref.get("fingerprint")
    ):
        raise ValueError(
            "当前归一化统计文件与 HVCA checkpoint 使用的统计量不一致；"
            "请传入 checkpoint 对应的 --normalization-stats-path。"
        )
    print_and_save_run_config(
        "HVCARefiner 测试超参数",
        {
            "stage": "test",
            "single": False,
            "model": "HVCARefiner",
            "paradigm": PARADIGM_2DTO2D,
            "target_var": args.target_var,
            "device": device,
            "data": {
                "data_dir": args.data_dir,
                "dataset_profile": args.dataset_profile,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "eval_start": args.eval_start,
                "eval_end": args.eval_end,
                "max_samples": args.max_samples,
                "normalization_stats_path": args.normalization_stats_path,
                "normalization_stats": actual_norm_ref,
                "rebuild_normalization_stats": args.rebuild_normalization_stats,
            },
            "hyperparameters": {
                "batch_size": args.batch_size,
                "cache_shard_size": args.cache_shard_size,
                "num_workers": args.num_workers,
                "prefetch_factor": args.prefetch_factor,
                "shard_cache_size": args.shard_cache_size,
                "depths": depth_values,
                "depth_indices": depth_indices,
                "thermocline_min": args.thermocline_min,
                "thermocline_max": args.thermocline_max,
                "spatial_panels": args.spatial_panels,
                "target_levels": target_levels,
                "panel_days": args.panel_days,
                "section_panels": args.section_panels,
                "multi_section_panels": args.multi_section_panels,
                "section_axes": (
                    ["lat", "lon"] if args.multi_section_panels else [args.section_axis]
                ),
                "section_axis": args.section_axis,
                "section_index": args.section_index,
                "section_percentiles": args.section_percentiles,
                "section_width": args.section_width,
            },
            "model_config": model_config,
            "loss_config": loss_config,
            "training_strategy": training_strategy,
            "parameter_count": count_model_parameters(model),
            "paths": {
                "save_dir": args.save_dir,
                "checkpoint": args.checkpoint,
                "t0_cache_dir": args.t0_cache_dir,
                "base_checkpoint_dir": args.base_checkpoint_dir,
                "train_config": train_config_path,
            },
        },
        run_config_path(args.save_dir, "test"),
    )

    ensure_eval_cache(
        args, dataset, depth_values, depth_indices, depth_levels, device
    )
    cache_norm_ref = load_cache_metadata(args.t0_cache_dir).get(
        "normalization_stats"
    )
    if (
        cache_norm_ref
        and cache_norm_ref.get("fingerprint")
        and cache_norm_ref.get("fingerprint") != actual_norm_ref.get("fingerprint")
    ):
        raise ValueError(
            "T0 cache 与当前归一化统计文件不一致；请使用匹配的 cache，"
            "或换一个空的 --t0-cache-dir 重新生成。"
        )
    target_stats = dataset.get_norm_stats()["target"]

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
        "depth_m": np.asarray(depth_values, dtype=np.float32),
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

    panel_day_records = []
    rendered_spatial_panels = []
    rendered_section_panels = []
    section_requested = args.section_panels or args.multi_section_panels
    if args.spatial_panels or section_requested:
        panel_day_records = resolve_panel_day_records(
            args,
            eval_dates,
            baseline_daily_records,
            refined_daily_records,
        )
    if args.spatial_panels:
        rendered_spatial_panels = render_spatial_panels(
            args,
            model,
            loader,
            depth_tensor,
            target_stats,
            depth_indices,
            depth_values,
            target_levels,
            panel_day_records,
            device,
            use_amp,
        )
    if section_requested:
        rendered_section_panels = render_section_panels(
            args,
            model,
            loader,
            depth_tensor,
            target_stats,
            depth_indices,
            depth_values,
            panel_day_records,
            device,
            use_amp,
        )

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
        "spatial_panels": {
            "enabled": bool(args.spatial_panels),
            "target_levels": target_levels,
            "panel_days": panel_day_records if args.spatial_panels else [],
            "selection_metric": (
                "refined_rmse - baseline_rmse"
                if args.spatial_panels and not args.panel_days
                else "manual panel-days" if args.spatial_panels else None
            ),
            "files": rendered_spatial_panels,
        },
        "section_panels": {
            "enabled": bool(section_requested),
            "multi_section_panels": bool(args.multi_section_panels),
            "section_axes": (
                ["lat", "lon"]
                if args.multi_section_panels
                else [args.section_axis] if args.section_panels else []
            ),
            "section_axis": (
                args.section_axis
                if args.section_panels and not args.multi_section_panels
                else None
            ),
            "section_index": args.section_index if args.section_panels else None,
            "section_percentiles": (
                args.section_percentiles if args.multi_section_panels else None
            ),
            "section_width": args.section_width if section_requested else None,
            "panel_days": panel_day_records if section_requested else [],
            "selection_metric": (
                "refined_rmse - baseline_rmse"
                if section_requested and not args.panel_days
                else "manual panel-days" if section_requested else None
            ),
            "files": rendered_section_panels,
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
    parser = argparse.ArgumentParser(description="Test HVCARefiner")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--t0-cache-dir", required=True)
    parser.add_argument("--base-checkpoint-dir", default=default_ckpt_dir)
    parser.add_argument("--target-var", choices=["temperature", "salinity"], default="temperature")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument(
        "--dataset-profile",
        choices=dataset_profile_names(),
        default=DEFAULT_DATASET_PROFILE,
        help="2dto2d 数据集配置；旧实验用 legacy_small，新大区域用 glorys12v1_all_size",
    )
    parser.add_argument("--start-date", default=DATA_START_DATE)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--eval-start", default=TEST_START_DATE)
    parser.add_argument("--eval-end", default=TEST_END_DATE)
    parser.add_argument("--depths", default=None, help="仅用于旧 checkpoint 无 depth_values 时")
    parser.add_argument(
        "--lazy-data",
        action="store_true",
        help="按样本延迟清洗/归一化；glorys12v1_all_size 会自动启用",
    )
    parser.add_argument(
        "--normalization-chunk-size",
        type=int,
        default=32,
        help="lazy 归一化统计量拟合时的时间分块大小",
    )
    parser.add_argument(
        "--normalization-stats-path",
        default=None,
        help="归一化统计量 .npz；默认读取 data-dir/normalization_stats/",
    )
    parser.add_argument(
        "--rebuild-normalization-stats",
        action="store_true",
        help="测试前显式重算并覆盖归一化统计量",
    )
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
    parser.add_argument("--save-dir", default=default_hvca_output_dir())
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--spatial-panels",
        action="store_true",
        help="输出 T0/HVCA/truth/error/delta 的二维解释面板",
    )
    parser.add_argument(
        "--target-levels",
        default=DEFAULT_PANEL_LEVELS,
        help="--spatial-panels 使用的逗号分隔深度层索引",
    )
    parser.add_argument(
        "--panel-days",
        default=None,
        help="可选逗号分隔日期 YYYY-MM-DD；不传则自动选择 best/median/worst",
    )
    parser.add_argument(
        "--section-panels",
        action="store_true",
        help="输出 T0/HVCA/truth/error/delta 的垂向剖面六子图",
    )
    parser.add_argument(
        "--multi-section-panels",
        action="store_true",
        help="按多个纬度和经度位置批量输出垂向剖面六子图",
    )
    parser.add_argument(
        "--section-axis",
        choices=["lat", "lon"],
        default="lat",
        help="垂向剖面固定的水平轴；lat 表示固定纬度画 depth-longitude",
    )
    parser.add_argument(
        "--section-index",
        default="middle",
        help="垂向剖面固定轴索引，整数或 middle",
    )
    parser.add_argument(
        "--section-percentiles",
        default=DEFAULT_SECTION_PERCENTILES,
        help="--multi-section-panels 使用的逗号分隔位置百分位，如 10,25,50,75,90",
    )
    parser.add_argument(
        "--section-width",
        type=int,
        default=1,
        help="剖面固定轴方向平均的格点宽度，默认 1 表示不做带平均",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.cache_shard_size <= 0:
        raise ValueError("--batch-size 和 --cache-shard-size 必须大于 0")
    if args.section_width <= 0:
        raise ValueError("--section-width 必须大于 0")
    if args.multi_section_panels:
        parse_percentile_list(args.section_percentiles, "--section-percentiles")
    evaluate(args)


if __name__ == "__main__":
    main()
