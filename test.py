"""统一预测评估入口（2dto2d / 2dto3d）。"""

import argparse
import json
import os
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from models.inversion_2dvar import inversion_2dvar
from models.inversion_modas import inversion_modas

from config import (
    CHECKPOINTS_ROOT,
    DATA_DIR,
    DATA_END_DATE,
    DATA_START_DATE,
    DEPTH_LEVELS_25M,
    DX,
    DY,
    DU_UNET_CKPT_NAME_TEMPLATE,
    INFER_DEFAULT_TARGET_LEVEL,
    INFER_PROFILE_METRIC,
    INFER_PROFILE_ZMAX,
    TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE,
    TEST_START_DATE, TEST_END_DATE,
    LON_RANGE, LAT_RANGE, DEPTH_MAX, OUTPUTS_ROOT,
    PARADIGM_2DTO2D, PARADIGM_2DTO2D_METHODS,
    PARADIGM_2DTO3D, PARADIGM_2DTO3D_METHODS,
    TWODTO3D_DATA_END_DATE,
    TWODTO3D_DATA_START_DATE,
    TWODTO3D_DEPTH_LEVELS,
    get_checkpoint_dir,
    get_output_dir,
)
from datasets.date_utils import date_to_index, generate_date_list
from datasets.climatology_normalizer import MonthlyClimatologyLayerStdNormalizer
from datasets.non_dl_preprocess import get_dataset_split, load_and_validate, load_sla_sss
from utils.metrics import (
    compute_grid_metrics, extract_level_map, save_grid_metrics,
    RegressionMetricAccumulator, scalar_metrics,
)
from utils.viz_layer_2d import plot_level_map
from utils.viz_layer_profile import plot_3d_metric_profile
from utils.physics import compute_eke, compute_grad_ssh

warnings.filterwarnings("ignore")

# ==================== 方法分类 ====================

NON_DL_METHODS = {"2dvar", "modas"}
DL_2DTO2D_METHODS = set(PARADIGM_2DTO2D_METHODS)
DL_2DTO3D_METHODS = set(PARADIGM_2DTO3D_METHODS) - NON_DL_METHODS
ALL_METHODS = sorted(DL_2DTO2D_METHODS | DL_2DTO3D_METHODS | NON_DL_METHODS)


# ==================== 命名工具 ====================

def _tag(method, date_str, target_var=None):
    d = date_str.replace("-", "")
    if target_var:
        return f"{method}_{target_var}", d
    return method, d


def pred_filename(method, date_str, target_var=None):
    m, d = _tag(method, date_str, target_var=target_var)
    return f"pred_{m}_{d}.npy"


def metrics_filename(method, date_str, target_var=None):
    m, d = _tag(method, date_str, target_var=target_var)
    return f"grid_metrics_{m}_{d}.npz"


def map_filename(var, level, method, date_str, target_var=None):
    m, d = _tag(method, date_str)
    if target_var and var != target_var:
        return f"map_{var}_{target_var}_lvl{level}_{m}_{d}.png"
    return f"map_{var}_lvl{level}_{m}_{d}.png"


def profile_filename(metric_name, method, date_str, target_var=None):
    m, d = _tag(method, date_str, target_var=target_var)
    return f"profile_{metric_name}_{m}_{d}.png"


def summary_filename(method, date_str, target_var=None):
    m, d = _tag(method, date_str, target_var=target_var)
    return f"summary_{m}_{d}.json"


def evaluation_metrics_filename(method, date_str, target_var=None):
    m, d = _tag(method, date_str, target_var=target_var)
    return f"evaluation_metrics_{m}_{d}.npz"


def depth_metrics_plot_filename(method, date_str, target_var=None):
    m, d = _tag(method, date_str, target_var=target_var)
    return f"metrics_by_depth_{m}_{d}.png"


def daily_metrics_plot_filename(method, date_str, target_var=None):
    m, d = _tag(method, date_str, target_var=target_var)
    return f"metrics_by_day_{m}_{d}.png"


# ==================== Non-DL 预测 (2DVar / MODAS) ====================


def predict_non_dl(args):
    """运行 2DVar 或 MODAS 反演，返回原始单位的 (y_pred, y_true_or_None)。"""
    sws_true_full = None

    if args.method == "2dvar":
        sla_sss = load_sla_sss(args.sla_sss_path)
        t = date_to_index(args.select_day, TWODTO3D_DATA_START_DATE, TWODTO3D_DATA_END_DATE)
        y_pred = inversion_2dvar(sla_sss, t, c_depth=args.c_depth)
        if args.sws_true_path:
            sws_true_full = np.load(args.sws_true_path).astype(np.float32)

    elif args.method == "modas":
        if not args.sws_true_path:
            raise ValueError("MODAS 需要 --sws-true-path")
        sla_sss, sws_true_full = load_and_validate(
            args.sla_sss_path, args.sws_true_path)
        t = date_to_index(args.select_day, TWODTO3D_DATA_START_DATE, TWODTO3D_DATA_END_DATE)
        ds = get_dataset_split(
            TWODTO3D_DATA_START_DATE, TWODTO3D_DATA_END_DATE,
            TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE,
            TEST_START_DATE, TEST_END_DATE)
        y_pred = inversion_modas(sla_sss, sws_true_full, t, ds["train_slice"])

    y_true = None
    if sws_true_full is not None:
        y_true = sws_true_full[t:t + 1]

    return y_pred, y_true


# ==================== DL 预测 (2dto3d / 2dto2d) ====================

def _get_device():
    """
    设备优先级：
    1) CUDA (NVIDIA)
    2) MPS   (Apple Silicon)
    3) CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_2dto3d_features(surface_raw):
    """
    将原始 2 通道海表输入扩展为 4 通道:
        [SLA(SSH), SSS, gradSSH, EKE]
    """
    ssh = surface_raw[:, 0:1]
    sss = surface_raw[:, 1:2]
    grad = compute_grad_ssh(ssh)
    eke = compute_eke(ssh, DX, DY)
    return torch.cat([ssh, sss, grad, eke], dim=1)


def predict_2dto3d(args):
    """2dto3d(ocean_transformer) 前向推理，返回原始单位的 (y_pred, y_true)。"""
    from train import build_model
    device = _get_device()

    model = build_model("ocean_transformer").to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device, weights_only=True))
        print(f"已加载权重: {args.checkpoint}")
    else:
        print("未提供/找到权重，使用随机初始化模型")

    if args.dummy:
        from datasets.dataset_2dto3d import DummyTwoDto3DDataset
        dataset = DummyTwoDto3DDataset(num_samples=200, H=32, W=32)
        t = min(date_to_index(args.select_day, TWODTO3D_DATA_START_DATE, TWODTO3D_DATA_END_DATE),
                len(dataset) - 1)
    else:
        from datasets.dataset_2dto3d import TwoDto3DDataset
        dataset = TwoDto3DDataset(args.data_dir, normalize=True)
        t = date_to_index(args.select_day, TWODTO3D_DATA_START_DATE, TWODTO3D_DATA_END_DATE)

    sample = dataset[t]
    model.eval()
    with torch.no_grad():
        surface_raw = sample["surface_raw"].unsqueeze(0).to(device)
        surface = build_2dto3d_features(surface_raw)
        pred = model(surface)

    y_pred = pred.cpu().numpy()                    # (1, D, H, W, 2)
    y_true = sample["target"].numpy()[np.newaxis]  # (1, D, H, W, 2)
    if not args.dummy:
        target_norm = MonthlyClimatologyLayerStdNormalizer.from_stats(
            dataset.get_norm_stats()["target"]
        )
        months = np.asarray([dataset.months[t]], dtype=np.int64)
        y_pred = target_norm.inverse_transform(y_pred, months)
        y_true = target_norm.inverse_transform(y_true, months)
    return y_pred, y_true


def predict_2dto2d(args):
    """Du_Unet 逐层推理并拼装 25 层结果。"""
    device = _get_device()

    from datasets.dataset_2dto2d import Dataset2Dto2D
    dataset = Dataset2Dto2D(
        args.data_dir,
        normalize=True,
        target_var=args.target_var,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    t = date_to_index(args.select_day, dataset.start_date, dataset.end_date)
    sample = dataset[t]
    norm_stats = dataset.get_norm_stats()
    sst = sample["sst"].unsqueeze(0).to(device)
    ssh_sss = sample["ssh_sss"].unsqueeze(0).to(device)

    models = _load_du_unet_models(args, device, require_all=False)
    pred_list = []
    for model in models:
        with torch.no_grad():
            pred_layer = model(sst, ssh_sss).cpu().numpy()
        pred_list.append(pred_layer)
    y_pred = np.concatenate(pred_list, axis=1)

    y_true = sample["target"].unsqueeze(0).numpy()      # (1, C, H, W)
    target_mask = sample["target_mask"].unsqueeze(0).numpy()

    if norm_stats.get("normalization") == "monthly_climatology_layer_std":
        target_norm = MonthlyClimatologyLayerStdNormalizer.from_stats(norm_stats["target"])
        months = np.asarray([dataset.months[t]], dtype=np.int64)
        y_pred = target_norm.inverse_transform(y_pred, months)
        y_true = target_norm.inverse_transform(y_true, months)

    return y_pred, y_true, target_mask


# ==================== 统一输出 ====================

def _metric_units(is_2dto3d):
    if is_2dto3d:
        return {
            "temperature": {
                "mse": "degC^2",
                "rmse": "degC",
                "mae": "degC",
                "r2": "dimensionless",
                "correlation": "dimensionless",
            },
            "salinity": {
                "mse": "psu^2",
                "rmse": "psu",
                "mae": "psu",
                "r2": "dimensionless",
                "correlation": "dimensionless",
            },
        }
    return {
        "mse": "physical_unit^2",
        "rmse": "physical_unit",
        "mae": "physical_unit",
        "r2": "dimensionless",
        "correlation": "dimensionless",
    }


def _target_var_for_outputs(args, is_2dto3d):
    if is_2dto3d:
        return None
    if getattr(args, "method", None) == "Du_Unet":
        return getattr(args, "target_var", None)
    return None


def _du_unet_checkpoint_dirs(args):
    base_dir = args.checkpoint_dir or get_checkpoint_dir(
        PARADIGM_2DTO2D, "Du_Unet", base_dir=CHECKPOINTS_ROOT
    )
    if os.path.basename(os.path.normpath(base_dir)) == args.target_var:
        candidates = [base_dir, os.path.dirname(os.path.normpath(base_dir))]
    else:
        candidates = [os.path.join(base_dir, args.target_var), base_dir]

    ordered = []
    for path in candidates:
        if path and path not in ordered:
            ordered.append(path)
    return ordered


def _load_du_unet_models(args, device, require_all):
    """一次加载 Du_Unet 的全部深度模型。"""
    from train import build_model

    ckpt_dirs = _du_unet_checkpoint_dirs(args)
    models = []
    for depth_m in DEPTH_LEVELS_25M:
        ckpt_name = DU_UNET_CKPT_NAME_TEMPLATE.format(
            target_var=args.target_var, depth_m=depth_m
        )
        searched_paths = [os.path.join(path, ckpt_name) for path in ckpt_dirs]
        ckpt_path = next((path for path in searched_paths if os.path.exists(path)), None)
        if ckpt_path is None and require_all:
            raise FileNotFoundError(
                f"缺少 {args.target_var} 深度 {depth_m}m checkpoint；"
                f"已查找：{', '.join(searched_paths)}"
            )

        model = build_model("du_unet", out_channels=1).to(device)
        if ckpt_path is not None:
            model.load_state_dict(
                torch.load(ckpt_path, map_location=device, weights_only=True)
            )
        else:
            print(
                f"[警告] 未找到 {args.target_var} 深度 {depth_m}m checkpoint；"
                f"已查找：{', '.join(searched_paths)}，将使用随机权重"
            )
        model.eval()
        models.append(model)
    return models


def _depth_label_2dto2d(level_idx):
    if 0 <= level_idx < len(DEPTH_LEVELS_25M):
        return f"Level-{level_idx} {DEPTH_LEVELS_25M[level_idx]}m"
    return f"Level-{level_idx}"


def _field_cmap_and_label(target_var):
    if target_var == "temperature":
        return "RdYlBu_r", "Temperature (°C)", "Absolute error (°C)"
    if target_var == "salinity":
        return "viridis", "Salinity (psu)", "Absolute error (psu)"
    return "viridis", "Value", "Absolute error"


def _metric_cbar_label(metric_name, target_var):
    unit = {
        "temperature": "°C",
        "salinity": "psu",
    }.get(target_var, "physical unit")
    metric = metric_name.upper()
    if metric_name.lower() == "mse":
        return f"{metric} ({unit}²)"
    return f"{metric} ({unit})"


def _finite_vmin_vmax(*arrays):
    vals = []
    for array in arrays:
        finite = np.asarray(array)[np.isfinite(array)]
        if finite.size:
            vals.append(finite)
    if not vals:
        return 0.0, 1.0
    merged = np.concatenate(vals)
    vmin = float(np.nanmin(merged))
    vmax = float(np.nanmax(merged))
    if vmin == vmax:
        pad = abs(vmin) * 0.05 if vmin else 1.0
        return vmin - pad, vmax + pad
    return vmin, vmax


def _apply_mask(array, mask):
    if mask is None:
        return array
    return np.where(mask.astype(bool), array, np.nan)


def plot_prediction_truth_error_panel(
    pred_2d,
    true_2d,
    error_2d,
    target_var,
    level,
    day,
    method,
    output_path,
):
    """绘制指定层的预测、真值和绝对误差三联图。"""
    cmap, value_label, error_label = _field_cmap_and_label(target_var)
    value_vmin, value_vmax = _finite_vmin_vmax(pred_2d, true_2d)
    err_vmin, err_vmax = _finite_vmin_vmax(error_2d)
    extent = [LON_RANGE[0], LON_RANGE[1], LAT_RANGE[0], LAT_RANGE[1]]
    level_label = _depth_label_2dto2d(level)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
    panels = [
        ("Prediction", pred_2d, cmap, value_label, value_vmin, value_vmax),
        ("Truth", true_2d, cmap, value_label, value_vmin, value_vmax),
        ("Absolute Error", error_2d, "magma", error_label, err_vmin, err_vmax),
    ]
    for ax, (title, data, panel_cmap, cbar_label, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(
            data,
            cmap=panel_cmap,
            aspect="auto",
            origin="lower",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude / °E")
        ax.set_ylabel("Latitude / °N")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)

    fig.suptitle(f"{method} {target_var} {day} {level_label}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"预测-真值-误差三联图已保存至：{output_path}")


PERIOD_METRIC_NAMES = ("mae", "rmse", "r2", "correlation")


def _period_metric_units(target_var):
    unit = "degC" if target_var == "temperature" else "psu"
    return {
        "mae": unit,
        "rmse": unit,
        "r2": "dimensionless",
        "correlation": "dimensionless",
    }


def _finite_json_value(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _period_metrics_to_json(metrics):
    result = {
        name: _finite_json_value(metrics[name])
        for name in PERIOD_METRIC_NAMES
    }
    result["valid_count"] = int(np.asarray(metrics["valid_count"]).item())
    return result


def _grouped_metrics_to_records(metrics, labels, label_name, extra_values=None):
    records = []
    for index, label in enumerate(labels):
        record = {label_name: label}
        if extra_values:
            record.update({name: values[index] for name, values in extra_values.items()})
        for metric_name in PERIOD_METRIC_NAMES:
            record[metric_name] = _finite_json_value(
                np.asarray(metrics[metric_name])[index]
            )
        record["valid_count"] = int(np.asarray(metrics["valid_count"])[index])
        records.append(record)
    return records


def _plot_period_metrics(
    x,
    metrics,
    xlabel,
    title,
    output_path,
    date_labels=None,
):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150)
    titles = {
        "mae": "MAE",
        "rmse": "RMSE",
        "r2": "R²",
        "correlation": "Correlation",
    }
    for ax, metric_name in zip(axes.flat, PERIOD_METRIC_NAMES):
        ax.plot(x, metrics[metric_name], linewidth=1.5)
        ax.set_title(titles[metric_name])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(titles[metric_name])
        ax.grid(True, alpha=0.3)
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
    print(f"时间段指标图已保存至：{output_path}")


def evaluate_2dto2d_period(args, out_dir):
    """在日期闭区间上评估 Du_Unet 的整体、逐深度和逐日指标。"""
    from datasets.dataset_2dto2d import Dataset2Dto2D

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
        raise ValueError(
            f"eval-end 不能早于 eval-start：{args.eval_end} < {args.eval_start}"
        )

    eval_indices = list(range(start_idx, end_idx + 1))
    eval_dates = generate_date_list(args.eval_start, args.eval_end)
    loader = DataLoader(
        Subset(dataset, eval_indices),
        batch_size=args.eval_batch_size,
        shuffle=False,
    )

    device = _get_device()
    models = _load_du_unet_models(args, device, require_all=True)
    norm_stats = dataset.get_norm_stats()
    target_norm = MonthlyClimatologyLayerStdNormalizer.from_stats(
        norm_stats["target"]
    )

    overall_acc = RegressionMetricAccumulator()
    depth_acc = RegressionMetricAccumulator()
    daily_records = []
    offset = 0

    with torch.no_grad():
        for batch in loader:
            sst = batch["sst"].to(device)
            ssh_sss = batch["ssh_sss"].to(device)
            pred_norm = torch.cat(
                [model(sst, ssh_sss) for model in models],
                dim=1,
            ).cpu().numpy()
            true_norm = batch["target"].numpy()
            target_mask = batch["target_mask"].numpy()

            batch_size = pred_norm.shape[0]
            batch_indices = eval_indices[offset:offset + batch_size]
            months = dataset.months[batch_indices]
            y_pred = target_norm.inverse_transform(pred_norm, months)
            y_true = target_norm.inverse_transform(true_norm, months)

            overall_acc.update(y_true, y_pred, mask=target_mask, axis=None)
            depth_acc.update(
                y_true,
                y_pred,
                mask=target_mask,
                axis=(0, 2, 3),
            )

            for local_index in range(batch_size):
                daily = scalar_metrics(
                    y_true[local_index],
                    y_pred[local_index],
                    mask=target_mask[local_index],
                )
                valid = (
                    target_mask[local_index].astype(bool)
                    & np.isfinite(y_true[local_index])
                    & np.isfinite(y_pred[local_index])
                )
                daily_records.append(
                    {
                        "date": eval_dates[offset + local_index],
                        **{
                            name: _finite_json_value(daily[name])
                            for name in PERIOD_METRIC_NAMES
                        },
                        "valid_count": int(np.count_nonzero(valid)),
                    }
                )

            offset += batch_size
            print(f"评估进度：{offset}/{len(eval_indices)} 天")

    overall_metrics = overall_acc.compute()
    depth_metrics = depth_acc.compute()
    daily_metrics = {
        name: np.asarray([record[name] for record in daily_records], dtype=np.float64)
        for name in PERIOD_METRIC_NAMES
    }
    daily_metrics["valid_count"] = np.asarray(
        [record["valid_count"] for record in daily_records],
        dtype=np.int64,
    )

    period_tag = f"{args.eval_start}_{args.eval_end}"
    evaluation_name = evaluation_metrics_filename(
        args.method,
        period_tag,
        target_var=args.target_var,
    )
    payload = {
        "dates": np.asarray(eval_dates),
        "depth_m": np.asarray(DEPTH_LEVELS_25M, dtype=np.int32),
        "overall_valid_count": overall_metrics["valid_count"],
        "by_depth_valid_count": depth_metrics["valid_count"],
        "by_day_valid_count": daily_metrics["valid_count"],
    }
    for metric_name in PERIOD_METRIC_NAMES:
        payload[f"overall_{metric_name}"] = overall_metrics[metric_name]
        payload[f"by_depth_{metric_name}"] = depth_metrics[metric_name]
        payload[f"by_day_{metric_name}"] = daily_metrics[metric_name]
    np.savez(os.path.join(out_dir, evaluation_name), **payload)

    depth_records = _grouped_metrics_to_records(
        depth_metrics,
        list(range(len(DEPTH_LEVELS_25M))),
        "depth_index",
        extra_values={"depth_m": DEPTH_LEVELS_25M},
    )
    summary = {
        "mode": "period",
        "method": args.method,
        "target_var": args.target_var,
        "eval_start": args.eval_start,
        "eval_end": args.eval_end,
        "num_days": len(eval_dates),
        "metric_units": _period_metric_units(args.target_var),
        "variables": {
            args.target_var: {
                "overall": _period_metrics_to_json(overall_metrics),
                "by_depth": depth_records,
                "by_day": daily_records,
            }
        },
    }
    summary_name = summary_filename(
        args.method,
        period_tag,
        target_var=args.target_var,
    )
    with open(os.path.join(out_dir, summary_name), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    depth_plot_name = depth_metrics_plot_filename(
        args.method,
        period_tag,
        target_var=args.target_var,
    )
    _plot_period_metrics(
        np.asarray(DEPTH_LEVELS_25M),
        depth_metrics,
        "Depth / m",
        f"{args.method} {args.target_var} metrics by depth",
        os.path.join(out_dir, depth_plot_name),
    )

    daily_plot_name = daily_metrics_plot_filename(
        args.method,
        period_tag,
        target_var=args.target_var,
    )
    _plot_period_metrics(
        np.arange(len(eval_dates)),
        daily_metrics,
        "Date",
        f"{args.method} {args.target_var} metrics by day",
        os.path.join(out_dir, daily_plot_name),
        date_labels=eval_dates,
    )

    print(f"时间段指标: {evaluation_name}")
    print(f"时间段汇总: {summary_name}")


def save_outputs(args, y_pred, y_true, out_dir, mask=None):
    """统一产物保存：npy / npz / 2D 图 / 3D 剖面 / summary.json"""
    method = args.method
    day = args.select_day
    level = args.target_level
    is_2dto3d = method == "ocean_transformer"
    output_target_var = _target_var_for_outputs(args, is_2dto3d)

    # ----- 1. 保存预测场 npy -----
    pred_name = pred_filename(method, day, target_var=output_target_var)
    np.save(os.path.join(out_dir, pred_name), y_pred)
    print(f"预测结果: {pred_name}  shape={y_pred.shape}")

    summary = {
        "method": method,
        "target_var": getattr(args, "target_var", None),
        "select_day": day,
        "target_level": level,
        "pred_shape": list(y_pred.shape),
        "metric_units": _metric_units(is_2dto3d),
        "variables": {},
    }

    # ----- 2. 二维层图 (预测值) -----
    if is_2dto3d:
        temp_map = y_pred[0, level, :, :, 0]
        salt_map = y_pred[0, level, :, :, 1]
        plot_level_map(
            temp_map,
            title=f"{method} {day} Temp Level-{level}",
            output_path=os.path.join(out_dir, map_filename("temp", level, method, day)),
            lon_range=LON_RANGE, lat_range=LAT_RANGE,
            cmap="RdYlBu_r", cbar_label="Temperature (°C)")
        plot_level_map(
            salt_map,
            title=f"{method} {day} Salt Level-{level}",
            output_path=os.path.join(out_dir, map_filename("salt", level, method, day)),
            lon_range=LON_RANGE, lat_range=LAT_RANGE,
            cmap="viridis", cbar_label="Salinity (psu)")
    else:
        if not output_target_var:
            target_var = getattr(args, "target_var", "target")
            if y_pred.ndim == 4 and y_pred.shape[1] > 1:
                vis_data = y_pred[0, level]
            elif y_pred.ndim == 4:
                vis_data = y_pred[0, 0]
            else:
                vis_data = y_pred[0]
            plot_level_map(
                vis_data,
                title=f"{method} {day} Level-{level}",
                output_path=os.path.join(
                    out_dir,
                    map_filename(target_var, level, method, day, target_var=output_target_var),
                ),
                lon_range=LON_RANGE, lat_range=LAT_RANGE)

    # ----- 3. 有真值时：网格指标 + 误差图 + 剖面图 -----
    if y_true is not None:
        if is_2dto3d:
            for vi, vname in enumerate(["temperature", "salinity"]):
                vp = y_pred[..., vi]
                vt = y_true[..., vi]
                summary["variables"][vname] = scalar_metrics(vt, vp)

            pred_all = y_pred[0]
            true_all = y_true[0]
            mae_grid = np.abs(pred_all - true_all).mean(axis=-1)
            rmse_grid = np.sqrt(np.square(pred_all - true_all).mean(axis=-1))
            npz_path = os.path.join(out_dir, metrics_filename(method, day))
            np.savez(npz_path, mae=mae_grid, rmse=rmse_grid)
            print(f"网格指标: {metrics_filename(method, day)}")

            mae_level = mae_grid[level]
            plot_level_map(
                mae_level,
                title=f"MAE {method} {day} Level-{level}",
                output_path=os.path.join(out_dir, map_filename("mae", level, method, day)),
                lon_range=LON_RANGE, lat_range=LAT_RANGE,
                cmap="RdYlBu_r", cbar_label="MAE")

            profile_path = os.path.join(out_dir, profile_filename("rmse", method, day))
            plot_3d_metric_profile(
                npz_path, metric_name=INFER_PROFILE_METRIC,
                output_img_path=profile_path,
                lon_range=LON_RANGE, lat_range=LAT_RANGE,
                z_max=INFER_PROFILE_ZMAX)
        else:
            grid = compute_grid_metrics(y_true, y_pred, mask=mask)
            metrics_name = metrics_filename(method, day, target_var=output_target_var)
            npz_path = os.path.join(out_dir, metrics_name)
            save_grid_metrics(grid, npz_path)

            summary["variables"][getattr(args, "target_var", "target")] = scalar_metrics(
                y_true, y_pred, mask=mask)

            if y_true.ndim >= 3 and (y_true.ndim < 4 or y_true.shape[1] > 1):
                mae_level = extract_level_map(grid["mae"], level)
                if output_target_var:
                    pred_level = y_pred[0, level] if y_pred.ndim == 4 else y_pred[level]
                    true_level = y_true[0, level] if y_true.ndim == 4 else y_true[level]
                    mask_level = None
                    if mask is not None:
                        mask_level = mask[0, level] if mask.ndim == 4 else mask[level]
                        pred_level = _apply_mask(pred_level, mask_level)
                        true_level = _apply_mask(true_level, mask_level)
                    panel_path = os.path.join(
                        out_dir,
                        map_filename("panel", level, method, day, target_var=output_target_var),
                    )
                    plot_prediction_truth_error_panel(
                        pred_level,
                        true_level,
                        mae_level,
                        output_target_var,
                        level,
                        day,
                        method,
                        panel_path,
                    )
                else:
                    plot_level_map(
                        mae_level,
                        title=f"MAE {method} {day} Level-{level}",
                        output_path=os.path.join(
                            out_dir,
                            map_filename("mae", level, method, day, target_var=output_target_var),
                        ),
                        lon_range=LON_RANGE, lat_range=LAT_RANGE,
                        cmap="RdYlBu_r", cbar_label="MAE")

                profile_path = os.path.join(
                    out_dir, profile_filename("rmse", method, day, target_var=output_target_var))
                plot_3d_metric_profile(
                    npz_path, metric_name=INFER_PROFILE_METRIC,
                    output_img_path=profile_path,
                    lon_range=LON_RANGE, lat_range=LAT_RANGE,
                    z_max=INFER_PROFILE_ZMAX,
                    cbar_label=_metric_cbar_label(INFER_PROFILE_METRIC, output_target_var))

    # ----- 4. summary.json -----
    summary_name = summary_filename(method, day, target_var=output_target_var)
    summary_path = os.path.join(out_dir, summary_name)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"汇总指标: {summary_name}")


# ==================== 主入口 ====================

def parse_args():
    p = argparse.ArgumentParser(description="统一预测评估入口")
    p.add_argument("--method", required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--select-day", help="单日预测日期 YYYY-MM-DD")
    mode.add_argument("--eval-start", help="时间段评估起始日期 YYYY-MM-DD")
    p.add_argument(
        "--eval-end",
        help="时间段评估结束日期 YYYY-MM-DD；使用 --eval-start 时必须提供",
    )
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=4,
        help="时间段评估推理 batch size，仅 Du_Unet 使用",
    )
    p.add_argument("--target-level", type=int, default=INFER_DEFAULT_TARGET_LEVEL,
                   help="可视化目标深度层索引")
    p.add_argument("--output-dir", default=OUTPUTS_ROOT)

    g_dl = p.add_argument_group("DL 方法参数")
    g_dl.add_argument("--checkpoint", default=None, help="模型权重路径")
    g_dl.add_argument("--checkpoint-dir", default=None, help="2dto2d 多模型权重目录")
    g_dl.add_argument("--data-dir", default=DATA_DIR)
    g_dl.add_argument("--target-var", choices=["temperature", "salinity"], default="temperature",
                      help="Du_Unet 预测目标变量")
    g_dl.add_argument("--start-date", default=DATA_START_DATE,
                      help="Du_Unet 使用的数据起始日期 YYYY-MM-DD")
    g_dl.add_argument("--end-date", default=None,
                      help="Du_Unet 使用的数据结束日期 YYYY-MM-DD，默认使用配置中的结束日期")
    g_dl.add_argument("--dummy", action="store_true",
                      help="使用合成数据 (2dto3d)")

    g_ndl = p.add_argument_group("Non-DL 方法参数")
    g_ndl.add_argument("--sla-sss-path", default=None)
    g_ndl.add_argument("--sws-true-path", default=None)
    g_ndl.add_argument("--c-depth", type=int, default=26,
                       help="2DVar 深度层数")
    return p.parse_args()


def main():
    args = parse_args()
    if args.eval_start and not args.eval_end:
        raise ValueError("使用 --eval-start 时必须同时提供 --eval-end")
    if args.eval_end and not args.eval_start:
        raise ValueError("使用 --eval-end 时必须同时提供 --eval-start")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size 必须大于 0")

    method = args.method.strip().lower()
    if method == "du-unet":
        method = "du_unet"
    if method not in ALL_METHODS:
        raise ValueError(f"未知方法: {args.method}")
    args.method = "Du_Unet" if method == "du_unet" else method
    if method in PARADIGM_2DTO2D_METHODS:
        paradigm = PARADIGM_2DTO2D
    else:
        paradigm = PARADIGM_2DTO3D
    if method == "du_unet":
        out_dir = get_output_dir(
            paradigm, args.method, base_dir=args.output_dir, target_var=args.target_var
        )
    else:
        out_dir = get_output_dir(paradigm, args.method, base_dir=args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.eval_start:
        if method != "du_unet":
            raise ValueError("时间段整体评估目前仅支持 Du_Unet")
        print("=" * 60)
        print(
            f"时间段评估：paradigm={paradigm}, method={args.method}, "
            f"period=[{args.eval_start}, {args.eval_end}]"
        )
        print("=" * 60)
        evaluate_2dto2d_period(args, out_dir)
        print("\n时间段评估完成。")
        return

    print("=" * 60)
    print(f"预测评估：paradigm={paradigm}, method={args.method}, day={args.select_day}, "
          f"level={args.target_level}")
    print("=" * 60)

    if method in NON_DL_METHODS:
        y_pred, y_true = predict_non_dl(args)
        mask = None
    elif method == "ocean_transformer":
        y_pred, y_true = predict_2dto3d(args)
        mask = None
    elif method in DL_2DTO2D_METHODS:
        y_pred, y_true, mask = predict_2dto2d(args)
    else:
        raise ValueError(f"未知方法: {method}")

    save_outputs(args, y_pred, y_true, out_dir, mask=mask)
    print("\n预测评估完成。")


if __name__ == "__main__":
    main()
