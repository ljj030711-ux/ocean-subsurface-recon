"""统一测试入口，支持 Du_Unet 单日预测、时间段评估和 2dto3d/Non-DL 方法。"""

import argparse
import json
import os
import warnings

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from models.inversion_2dvar import inversion_2dvar
from models.inversion_modas import inversion_modas

from config import (
    B3_DEPTHFORMER_CKPT_NAME_TEMPLATE,
    CHECKPOINTS_ROOT,
    DATA_DIR,
    DATA_END_DATE,
    DATA_START_DATE,
    DEFAULT_DATASET_PROFILE,
    DX,
    DY,
    DU_UNET_CKPT_NAME_TEMPLATE,
    INFER_DEFAULT_TARGET_LEVEL,
    JOINT_DU_UNET_CKPT_NAME_TEMPLATE,
    JOINT_DUVIT_CKPT_NAME_TEMPLATE,
    TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE,
    TEST_START_DATE, TEST_END_DATE,
    LON_RANGE, LAT_RANGE, DEPTH_MAX, OUTPUTS_ROOT,
    PARADIGM_2DTO2D, PARADIGM_2DTO2D_METHODS,
    PARADIGM_2DTO3D, PARADIGM_2DTO3D_METHODS,
    TWODTO3D_DATA_END_DATE,
    TWODTO3D_DATA_START_DATE,
    TWODTO3D_DEPTH_LEVELS,
    dataset_profile_names,
    get_2dto2d_depth_levels,
    get_2dto2d_lon_lat_range,
    get_checkpoint_dir,
    get_output_dir,
    joint_du_unet_depth_tag,
)
from datasets.date_utils import date_to_index, generate_date_list
from datasets.climatology_normalizer import MonthlyClimatologyLayerStdNormalizer
from datasets.non_dl_preprocess import get_dataset_split, load_and_validate, load_sla_sss
from utils.metrics import (
    DAY_DEPTH_METRIC_NAMES,
    RegressionMetricAccumulator,
    compute_day_depth_metrics,
    compute_grid_metrics,
    extract_level_map,
    save_grid_metrics,
    scalar_metrics,
    select_structure_quantile_days,
    summarize_horizontal_structure,
)
from utils.viz_layer_2d import (
    plot_level_map,
    plot_prediction_truth_error_panel,
)
from utils.viz_layer_profile import (
    build_section_specs,
    plot_joint_du_unet_section_panel,
)
from utils.physics import compute_eke, compute_grad_ssh
from utils.run_config import (
    count_model_parameters,
    load_run_config,
    model_config_path,
    print_and_save_run_config,
)

warnings.filterwarnings("ignore")

# ==================== 方法分类 ====================

NON_DL_METHODS = {"2dvar", "modas"}
DL_2DTO2D_METHODS = set(PARADIGM_2DTO2D_METHODS)
DL_2DTO3D_METHODS = set(PARADIGM_2DTO3D_METHODS) - NON_DL_METHODS
ALL_METHODS = sorted(DL_2DTO2D_METHODS | DL_2DTO3D_METHODS | NON_DL_METHODS)
JOINT_METHOD_NAMES = {"Joint_Du_Unet", "Joint_DUViT", "B3_DepthFormer"}


def _is_joint_method(args):
    return getattr(args, "method", None) in JOINT_METHOD_NAMES


def _joint_method_spec(args):
    if getattr(args, "method", None) == "B3_DepthFormer":
        return {
            "method_key": "b3_depthformer",
            "model_name": "B3_DepthFormer",
            "checkpoint_type": "B3_DepthFormer",
            "checkpoint_template": B3_DEPTHFORMER_CKPT_NAME_TEMPLATE,
            "experiment": "B3_BOS_depth_autoregressive",
        }
    if getattr(args, "method", None) == "Joint_DUViT":
        return {
            "method_key": "joint_duvit",
            "model_name": "Joint_DUViT",
            "checkpoint_type": "Joint_DUViT",
            "checkpoint_template": JOINT_DUVIT_CKPT_NAME_TEMPLATE,
            "experiment": "B2_spatial_vit_bottlenecks",
        }
    return {
        "method_key": "joint_du_unet",
        "model_name": "Joint_Du_Unet",
        "checkpoint_type": "Joint_Du_Unet",
        "checkpoint_template": JOINT_DU_UNET_CKPT_NAME_TEMPLATE,
        "experiment": "B1_joint_output_baseline",
    }


def _du_unet_depth_levels(args):
    return get_2dto2d_depth_levels(
        getattr(args, "dataset_profile", DEFAULT_DATASET_PROFILE)
    )


def _du_unet_depth_indices(args, depth_levels=None):
    depth_levels = depth_levels or _du_unet_depth_levels(args)
    depth_indices_arg = getattr(args, "depth_indices", None)
    if not depth_indices_arg:
        return list(range(len(depth_levels)))

    indices = []
    for item in depth_indices_arg.split(","):
        idx = int(item.strip())
        if idx < 0 or idx >= len(depth_levels):
            raise ValueError(
                f"depth index 越界: {idx}；当前 profile 共 {len(depth_levels)} 层"
            )
        if idx in indices:
            raise ValueError(f"depth index 重复: {idx}")
        indices.append(idx)
    if not indices:
        raise ValueError("--depth-indices 不能为空")
    return indices


def _du_unet_selected_depth_levels(args, depth_levels=None):
    resolved = getattr(args, "resolved_depth_levels", None)
    if resolved is not None:
        return list(resolved)
    depth_levels = depth_levels or _du_unet_depth_levels(args)
    return [depth_levels[idx] for idx in _du_unet_depth_indices(args, depth_levels)]


def _joint_evaluation_depths(args, depth_levels, checkpoint_depth_indices):
    """Map original profile indices to the joint model's output positions."""
    if getattr(args, "depth_indices", None):
        evaluation_indices = _du_unet_depth_indices(args, depth_levels)
    else:
        evaluation_indices = list(checkpoint_depth_indices)

    position_by_depth = {
        depth_idx: position
        for position, depth_idx in enumerate(checkpoint_depth_indices)
    }
    missing = [
        depth_idx
        for depth_idx in evaluation_indices
        if depth_idx not in position_by_depth
    ]
    if missing:
        raise ValueError(
            "请求评估的原始深度索引未包含在该联合模型checkpoint中: "
            f"{missing}；checkpoint 可用层为 {list(checkpoint_depth_indices)}"
        )
    output_positions = [position_by_depth[idx] for idx in evaluation_indices]
    return evaluation_indices, output_positions


def _target_normalizer_for_depth_indices(norm_stats, depth_indices):
    target_stats = norm_stats["target"]
    selected_stats = dict(target_stats)
    selected_stats["climatology"] = np.asarray(target_stats["climatology"])[
        :, depth_indices
    ]
    selected_stats["layer_std"] = np.asarray(target_stats["layer_std"])[depth_indices]
    return MonthlyClimatologyLayerStdNormalizer.from_stats(selected_stats)


def _du_unet_lon_lat_range(args):
    return get_2dto2d_lon_lat_range(
        getattr(args, "dataset_profile", DEFAULT_DATASET_PROFILE)
    )


def use_lazy_2dto2d_data(args):
    return args.lazy_data or args.dataset_profile == "glorys12v1_all_size"


def _parse_index_list(value, name):
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


def _validate_panel_levels(value, num_depths):
    levels = (
        list(range(num_depths))
        if value is None
        else _parse_index_list(value, "--target-levels")
    )
    if not levels:
        raise ValueError("--target-levels 不能为空")
    invalid = [level for level in levels if level < 0 or level >= num_depths]
    if invalid:
        raise ValueError(
            "--target-levels 是当前评估深度序列的局部位置；"
            f"有效范围 [0,{num_depths - 1}]，实际越界: {invalid}"
        )
    ordered = []
    for level in levels:
        if level not in ordered:
            ordered.append(level)
    return ordered


def _parse_panel_days(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_percentile_list(value, name):
    percentiles = []
    for item in str(value).split(","):
        text = item.strip()
        if not text:
            continue
        try:
            percentile = float(text)
        except ValueError as exc:
            raise ValueError(f"{name} 包含非法百分位: {text}") from exc
        if percentile < 0.0 or percentile > 100.0:
            raise ValueError(f"{name} 百分位需位于 [0,100]: {percentile}")
        percentiles.append(percentile)
    if not percentiles:
        raise ValueError(f"{name} 不能为空")
    return percentiles


def _resolve_joint_panel_days(args, eval_dates, horizontal_summary):
    explicit_days = _parse_panel_days(args.panel_days)
    if explicit_days:
        missing = [day for day in explicit_days if day not in set(eval_dates)]
        if missing:
            raise ValueError(f"--panel-days 中日期不在评估区间: {missing}")
        date_to_index_map = {day: index for index, day in enumerate(eval_dates)}
        records = []
        for day in explicit_days:
            index = date_to_index_map[day]
            records.append(
                {
                    "role": "manual",
                    "quantile": None,
                    "date": day,
                    "day_index": int(index),
                    "ssim": _finite_json_value(
                        horizontal_summary["by_day_ssim"][index]
                    ),
                    "acc": _finite_json_value(
                        horizontal_summary["by_day_acc"][index]
                    ),
                    "structure_score": _finite_json_value(
                        horizontal_summary["by_day_structure_score"][index]
                    ),
                }
            )
        return records

    records = select_structure_quantile_days(
        eval_dates,
        horizontal_summary["by_day_structure_score"],
        horizontal_summary["complete_day_mask"],
        quantiles=(10.0, 50.0, 90.0),
    )
    for record in records:
        index = record["day_index"]
        record["ssim"] = float(horizontal_summary["by_day_ssim"][index])
        record["acc"] = float(horizontal_summary["by_day_acc"][index])
    return records


# ==================== 命名工具 ====================

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


def horizontal_structure_depth_plot_filename(method, date_str, target_var=None):
    m, d = _tag(method, date_str, target_var=target_var)
    return f"horizontal_structure_by_depth_{m}_{d}.png"


def horizontal_structure_day_plot_filename(method, date_str, target_var=None):
    m, d = _tag(method, date_str, target_var=target_var)
    return f"horizontal_structure_by_day_{m}_{d}.png"


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
    """Run B0 layer-wise or a joint-depth B1/B2/B3 prediction."""
    device = _get_device()

    from datasets.dataset_2dto2d import Dataset2Dto2D
    from train import build_model
    dataset = Dataset2Dto2D(
        args.data_dir,
        normalize=True,
        target_var=args.target_var,
        start_date=args.start_date,
        end_date=args.end_date,
        quality_report=args.quality_report,
        dataset_profile=args.dataset_profile,
        lazy=use_lazy_2dto2d_data(args),
        normalization_chunk_size=args.normalization_chunk_size,
        normalization_stats_path=args.normalization_stats_path,
        rebuild_normalization_stats=args.rebuild_normalization_stats,
    )
    t = date_to_index(args.select_day, dataset.start_date, dataset.end_date)
    sample = dataset[t]
    norm_stats = dataset.get_norm_stats()
    all_depth_levels = _du_unet_depth_levels(args)
    if len(all_depth_levels) != dataset.num_depths:
        raise ValueError(
            f"dataset_profile={args.dataset_profile} 深度层数 {len(all_depth_levels)} "
            f"与 target 数据层数 {dataset.num_depths} 不一致"
        )
    is_joint = _is_joint_method(args)
    joint_spec = _joint_method_spec(args) if is_joint else None
    model_name = joint_spec["model_name"] if is_joint else "Du_Unet"
    joint_model = None
    joint_output_positions = None
    checkpoint_depth_indices = None
    checkpoint_metadata = None
    loaded_checkpoint = None
    if is_joint:
        (
            joint_model,
            loaded_checkpoint,
            checkpoint_depth_indices,
            checkpoint_metadata,
        ) = _load_joint_du_unet_model(args, device, dataset)
        depth_indices, joint_output_positions = _joint_evaluation_depths(
            args, all_depth_levels, checkpoint_depth_indices
        )
    else:
        depth_indices = _du_unet_depth_indices(args, all_depth_levels)
    depth_levels = [all_depth_levels[idx] for idx in depth_indices]
    args.resolved_depth_indices = list(depth_indices)
    args.resolved_depth_levels = list(depth_levels)
    joint_depth_tag = (
        joint_du_unet_depth_tag(
            checkpoint_depth_indices, dataset.num_depths
        )
        if is_joint
        else None
    )
    if args.target_level < 0 or args.target_level >= len(depth_indices):
        raise ValueError(
            f"--target-level 需位于 [0, {len(depth_indices) - 1}]，"
            f"实际为 {args.target_level}"
        )
    sst = sample["sst"].unsqueeze(0).to(device)
    ssh_sss = sample["ssh_sss"].unsqueeze(0).to(device)
    config_target_var = args.target_var if is_joint else None
    train_config_path = model_config_path(
        args.output_dir,
        model_name,
        "train",
        target_var=config_target_var,
        run_tag=joint_depth_tag,
    )

    if is_joint:
        param_stats = count_model_parameters(joint_model)
        parameter_payload = param_stats
        checkpoint_candidates = _joint_du_unet_checkpoint_candidates(
            args,
            dataset.num_depths,
        )
        model_config = dict((checkpoint_metadata or {}).get("model_config") or {})
        model_config.update(
            {
                "out_channels": len(checkpoint_depth_indices),
                "joint_depth_output": True,
                "checkpoint_depth_indices": checkpoint_depth_indices,
                "evaluation_output_positions": joint_output_positions,
            }
        )
    else:
        param_stats = count_model_parameters(build_model("du_unet", out_channels=1))
        parameter_payload = {
            "per_depth_model": param_stats,
            "num_depth_models": len(depth_levels),
            "total_if_all_depth_models_loaded": {
                name: value * len(depth_levels)
                for name, value in param_stats.items()
            },
        }
        checkpoint_candidates = _du_unet_checkpoint_dirs(args)
        model_config = {"out_channels": 1}

    print_and_save_run_config(
        f"{model_name} 单日测试超参数",
        {
            "stage": "test",
            "mode": "single_day",
            "single": True,
            "experiment": joint_spec["experiment"] if is_joint else "B0",
            "model": model_name,
            "method": args.method,
            "paradigm": PARADIGM_2DTO2D,
            "target_var": args.target_var,
            "device": device,
            "data": {
                "data_dir": args.data_dir,
                "dataset_profile": args.dataset_profile,
                "normalization_stats_path": dataset.normalization_stats_path,
                "normalization_stats": dataset.get_normalization_reference(),
                "rebuild_normalization_stats": args.rebuild_normalization_stats,
                "start_date": dataset.start_date,
                "end_date": dataset.end_date,
                "select_day": args.select_day,
                "sample_index": t,
            },
            "hyperparameters": {
                "target_level": args.target_level,
                "depth_indices": depth_indices,
                "depth_m": depth_levels,
            },
            "model_config": model_config,
            "training_strategy": load_training_strategy_reference(train_config_path),
            "parameter_count": parameter_payload,
            "paths": {
                "output_dir": get_output_dir(
                    PARADIGM_2DTO2D,
                    model_name,
                    base_dir=args.output_dir,
                    target_var=args.target_var,
                ),
                "checkpoint": args.checkpoint,
                "loaded_checkpoint": loaded_checkpoint,
                "checkpoint_metadata": checkpoint_metadata,
                "checkpoint_candidates": checkpoint_candidates,
                "train_config": train_config_path,
            },
        },
        model_config_path(
            args.output_dir,
            model_name,
            "test",
            target_var=config_target_var,
            run_tag=joint_depth_tag,
        ),
    )

    if is_joint:
        with torch.no_grad():
            pred_full = joint_model(
                sst, ssh_sss, output_size=sample["target"].shape[-2:]
            )
        expected_shape = (
            1,
            len(checkpoint_depth_indices),
            sample["target"].shape[-2],
            sample["target"].shape[-1],
        )
        if tuple(pred_full.shape) != expected_shape:
            raise ValueError(
                f"{model_name} 输出形状错误，"
                f"expected={expected_shape}, actual={tuple(pred_full.shape)}"
            )
        y_pred = pred_full[:, joint_output_positions].cpu().numpy()
    else:
        models = _load_du_unet_models(
            args, device, require_all=bool(args.depth_indices)
        )
        pred_list = []
        for model in models:
            with torch.no_grad():
                pred_layer = model(
                    sst, ssh_sss, output_size=sample["target"].shape[-2:]
                ).cpu().numpy()
            pred_list.append(pred_layer)
        y_pred = np.concatenate(pred_list, axis=1)

    y_true = sample["target"][depth_indices].unsqueeze(0).numpy()
    target_mask = sample["target_mask"][depth_indices].unsqueeze(0).numpy()

    if norm_stats.get("normalization") == "monthly_climatology_layer_std":
        target_norm = _target_normalizer_for_depth_indices(norm_stats, depth_indices)
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
    if getattr(args, "method", None) in {
        "Du_Unet",
        "Joint_Du_Unet",
        "Joint_DUViT",
        "B3_DepthFormer",
    }:
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
    depth_levels = _du_unet_selected_depth_levels(args)
    models = []
    for depth_m in depth_levels:
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


def _joint_du_unet_checkpoint_candidates(args, dataset_num_depths):
    spec = _joint_method_spec(args)
    if args.checkpoint:
        return [args.checkpoint]

    default_method_dir = get_checkpoint_dir(
        PARADIGM_2DTO2D,
        spec["model_name"],
        base_dir=CHECKPOINTS_ROOT,
    )
    base_dir = args.checkpoint_dir or default_method_dir
    method_dir = get_checkpoint_dir(
        PARADIGM_2DTO2D,
        spec["model_name"],
        base_dir=base_dir,
    )
    directories = [
        os.path.join(base_dir, args.target_var),
        base_dir,
        os.path.join(method_dir, args.target_var),
        method_dir,
    ]
    if os.path.basename(os.path.normpath(base_dir)) == args.target_var:
        directories.insert(0, base_dir)

    requested_depth_indices = _du_unet_depth_indices(
        args, list(range(dataset_num_depths))
    )
    depth_tag = joint_du_unet_depth_tag(
        requested_depth_indices, dataset_num_depths
    )
    ckpt_name = spec["checkpoint_template"].format(
        target_var=args.target_var,
        depth_tag=depth_tag,
    )
    candidates = []
    for directory in directories:
        path = os.path.join(directory, ckpt_name)
        if path not in candidates:
            candidates.append(path)
    return candidates


def _joint_output_channels_from_state_dict(state_dict, method_key):
    if method_key == "b3_depthformer":
        depth_values = state_dict.get("depth_embedding.depth_values")
        if depth_values is None or not hasattr(depth_values, "numel"):
            raise ValueError(
                "无法从 B3 checkpoint 推断深度数；缺少 depth_embedding.depth_values"
            )
        return int(depth_values.numel())
    candidates = [
        value
        for key, value in state_dict.items()
        if key.endswith("fusion.4.weight") and hasattr(value, "shape")
    ]
    if len(candidates) != 1 or len(candidates[0].shape) != 4:
        raise ValueError(
            "无法从 Joint_Du_Unet checkpoint 推断输出通道数；"
            "缺少唯一的 fusion.4.weight"
        )
    return int(candidates[0].shape[0])


def _load_joint_du_unet_model(args, device, dataset):
    """Load one B1/B2/B3 joint checkpoint and recover its depth mapping."""
    from train import build_model

    spec = _joint_method_spec(args)
    candidates = _joint_du_unet_checkpoint_candidates(args, dataset.num_depths)
    ckpt_path = next(
        (path for path in candidates if os.path.isfile(path)),
        None,
    )
    if ckpt_path is None:
        raise FileNotFoundError(
            f"缺少 {spec['model_name']} checkpoint；已查找："
            + ", ".join(candidates)
        )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    structured_checkpoint = isinstance(checkpoint, dict) and "state_dict" in checkpoint
    if structured_checkpoint:
        if checkpoint.get("format_version") != 1:
            raise ValueError(
                f"不支持的 {spec['model_name']} checkpoint format_version: "
                f"{checkpoint.get('format_version')!r}"
            )
        state_dict = checkpoint["state_dict"]
        metadata = dict(checkpoint.get("metadata") or {})
    else:
        state_dict = checkpoint
        metadata = {}
    if not isinstance(state_dict, dict):
        raise ValueError(f"checkpoint 中的 state_dict 格式错误: {ckpt_path}")

    if structured_checkpoint:
        required_metadata = {
            "checkpoint_type",
            "target_var",
            "dataset_profile",
            "dataset_num_depths",
            "normalize",
            "depth_indices",
            "depth_m",
            "num_outputs",
            "depth_tag",
            "model_config",
        }
        missing_metadata = sorted(required_metadata - set(metadata))
        if missing_metadata:
            raise ValueError(
                f"{spec['model_name']} checkpoint 缺少必要元数据: "
                + ", ".join(missing_metadata)
            )

    num_outputs = _joint_output_channels_from_state_dict(
        state_dict, spec["method_key"]
    )
    stored_indices = metadata.get("depth_indices")
    if stored_indices is None:
        if num_outputs != dataset.num_depths:
            raise ValueError(
                f"该旧版 {spec['model_name']} checkpoint 只有 "
                f"{num_outputs} 个输出且未记录原始 depth_indices，无法安全映射；"
                "请使用新版 train.py 重新训练"
            )
        checkpoint_depth_indices = list(range(dataset.num_depths))
    else:
        checkpoint_depth_indices = [int(idx) for idx in stored_indices]

    if len(checkpoint_depth_indices) != num_outputs:
        raise ValueError(
            "checkpoint 元数据与输出通道数不一致: "
            f"depth_indices={checkpoint_depth_indices}, out_channels={num_outputs}"
        )
    if len(set(checkpoint_depth_indices)) != len(checkpoint_depth_indices):
        raise ValueError(
            f"checkpoint depth_indices 存在重复: {checkpoint_depth_indices}"
        )
    invalid = [
        idx
        for idx in checkpoint_depth_indices
        if idx < 0 or idx >= dataset.num_depths
    ]
    if invalid:
        raise ValueError(
            f"checkpoint depth_indices 超出当前数据集范围: {invalid}"
        )

    metadata_num_outputs = metadata.get("num_outputs")
    if metadata_num_outputs is not None and int(metadata_num_outputs) != num_outputs:
        raise ValueError(
            "checkpoint num_outputs 与权重输出通道数不一致: "
            f"metadata={metadata_num_outputs}, weights={num_outputs}"
        )
    metadata_model_config = metadata.get("model_config") or {}
    metadata_out_channels = metadata_model_config.get("out_channels")
    if (
        metadata_out_channels is not None
        and int(metadata_out_channels) != num_outputs
    ):
        raise ValueError(
            "checkpoint model_config.out_channels 与权重不一致: "
            f"metadata={metadata_out_channels}, weights={num_outputs}"
        )
    metadata_num_depths = metadata_model_config.get("num_depths")
    if (
        metadata_num_depths is not None
        and int(metadata_num_depths) != num_outputs
    ):
        raise ValueError(
            "checkpoint model_config.num_depths 与权重不一致: "
            f"metadata={metadata_num_depths}, weights={num_outputs}"
        )

    expected_depth_values = [
        float(value)
        for value in get_2dto2d_depth_levels(args.dataset_profile)
    ]
    stored_depth_values = metadata.get("depth_m")
    if stored_depth_values is not None:
        wanted = [expected_depth_values[idx] for idx in checkpoint_depth_indices]
        actual = [float(value) for value in stored_depth_values]
        if len(actual) != len(wanted) or not np.allclose(
            actual, wanted, rtol=0.0, atol=1e-5
        ):
            raise ValueError(
                "checkpoint depth_m 与当前 profile 的 depth_indices 不一致: "
                f"文件={actual}, 当前={wanted}"
            )
    expected_depth_tag = joint_du_unet_depth_tag(
        checkpoint_depth_indices, dataset.num_depths
    )
    stored_depth_tag = metadata.get("depth_tag")
    if stored_depth_tag is not None and stored_depth_tag != expected_depth_tag:
        raise ValueError(
            "checkpoint depth_tag 与 depth_indices 不一致: "
            f"文件={stored_depth_tag!r}, 当前={expected_depth_tag!r}"
        )

    expected_metadata = {
        "checkpoint_type": spec["checkpoint_type"],
        "target_var": args.target_var,
        "dataset_profile": args.dataset_profile,
        "dataset_num_depths": dataset.num_depths,
        "normalize": True,
    }
    for key, expected in expected_metadata.items():
        actual = metadata.get(key)
        if actual is not None and actual != expected:
            raise ValueError(
                f"checkpoint {key} 不匹配: 文件={actual!r}, 当前={expected!r}"
            )

    checkpoint_normalization = metadata.get("normalization_stats")
    current_normalization = dataset.get_normalization_reference()
    if checkpoint_normalization and current_normalization:
        checkpoint_fingerprint = checkpoint_normalization.get("fingerprint")
        current_fingerprint = current_normalization.get("fingerprint")
        if (
            checkpoint_fingerprint
            and current_fingerprint
            and checkpoint_fingerprint != current_fingerprint
        ):
            raise ValueError(
                "checkpoint 与测试数据使用的归一化统计量指纹不一致"
            )

    if spec["method_key"] == "b3_depthformer":
        if not structured_checkpoint:
            raise ValueError("B3 仅支持包含结构元数据的新版 checkpoint")
        build_kwargs = {
            "depth_values": metadata_model_config.get(
                "depth_values", metadata.get("depth_m")
            ),
            "d_model": metadata_model_config.get("d_model"),
            "nhead": metadata_model_config.get("nhead"),
            "encoder_layers": metadata_model_config.get("encoder_layers"),
            "decoder_layers": metadata_model_config.get("decoder_layers"),
            "dim_feedforward": metadata_model_config.get("dim_feedforward"),
            "dropout": metadata_model_config.get("dropout"),
            "sst_patch_size": metadata_model_config.get("sst_patch_size"),
            "surface_patch_size": metadata_model_config.get(
                "surface_patch_size"
            ),
            "output_patch_size": metadata_model_config.get("output_patch_size"),
            "refinement_channels": metadata_model_config.get(
                "refinement_channels"
            ),
            "base_grid_size": metadata_model_config.get("base_grid_size"),
        }
        missing = sorted(
            key for key, value in build_kwargs.items() if value is None
        )
        if missing:
            raise ValueError(
                "B3 checkpoint model_config 缺少模型参数: " + ", ".join(missing)
            )
    else:
        build_kwargs = {"num_depths": num_outputs}
    if spec["method_key"] == "joint_duvit" and metadata_model_config:
        sst_vit = metadata_model_config.get("sst_vit") or {}
        surface_vit = metadata_model_config.get("surface_vit") or {}
        build_kwargs.update(
            {
                "sst_embed_dim": sst_vit.get("embed_dim"),
                "sst_patch_size": sst_vit.get("patch_size"),
                "sst_num_heads": sst_vit.get("num_heads"),
                "sst_num_layers": sst_vit.get("num_layers"),
                "sst_max_tokens": sst_vit.get("max_tokens"),
                "surface_embed_dim": surface_vit.get("embed_dim"),
                "surface_patch_size": surface_vit.get("patch_size"),
                "surface_num_heads": surface_vit.get("num_heads"),
                "surface_num_layers": surface_vit.get("num_layers"),
                "surface_max_tokens": surface_vit.get("max_tokens"),
                "mlp_ratio": sst_vit.get("mlp_ratio"),
                "dropout": sst_vit.get("dropout"),
            }
        )
        build_kwargs = {
            key: value for key, value in build_kwargs.items() if value is not None
        }
    model = build_model(spec["method_key"], **build_kwargs).to(device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{spec['model_name']} checkpoint 与当前 {num_outputs} 输出模型不兼容: "
            f"{ckpt_path}"
        ) from exc
    model.eval()
    return model, ckpt_path, checkpoint_depth_indices, metadata


def _apply_mask(array, mask):
    if mask is None:
        return array
    return np.where(mask.astype(bool), array, np.nan)


PERIOD_METRIC_NAMES = ("mae", "rmse", "r2", "correlation")
SSIM_WIN_SIZE = 11
SSIM_SIGMA = 1.5
SSIM_MIN_VALID_FRACTION = 0.8
SSIM_DATA_RANGE_STD_MULTIPLIER = 6.0


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


def _plot_horizontal_structure_metrics(
    x,
    ssim_values,
    acc_values,
    structure_scores,
    xlabel,
    title,
    output_path,
    date_labels=None,
    selected_days=None,
):
    """绘制SSIM、ACC和用于代表日期选择的结构得分。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150)
    series = [
        ("SSIM", ssim_values, "#2E74B5"),
        ("ACC", acc_values, "#3A7D44"),
        ("Structure score", structure_scores, "#C47F17"),
    ]
    for ax, (label, values, color) in zip(axes, series):
        ax.plot(x, values, linewidth=1.5, color=color)
        ax.set_title(label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(label)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)
        if selected_days:
            for record in selected_days:
                day_index = int(record["day_index"])
                ax.axvline(day_index, color="#8C2F39", alpha=0.55, linewidth=1.0)
                ax.text(
                    day_index,
                    1.0,
                    record["role"].upper(),
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=8,
                    color="#8C2F39",
                )
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
    print(f"水平结构指标图已保存至：{output_path}")


@torch.no_grad()
def _render_joint_horizontal_panels(
    args,
    model,
    loader,
    target_norm,
    checkpoint_depth_indices,
    output_positions,
    depth_indices,
    depth_levels,
    target_levels,
    eval_indices,
    eval_dates,
    panel_day_records,
    day_depth_metrics,
    dataset,
    device,
    out_dir,
):
    """为联合深度模型的代表日期生成指定深度水平三联图。"""
    if not panel_day_records:
        print("[spatial-panels] 没有可绘制的代表日期，跳过水平图")
        return []

    panel_days = {record["date"] for record in panel_day_records}
    panel_record_by_day = {record["date"]: record for record in panel_day_records}
    day_index_by_date = {day: index for index, day in enumerate(eval_dates)}
    rendered_spatial = []
    lon_range, lat_range = _du_unet_lon_lat_range(args)
    profile_depth_levels = _du_unet_depth_levels(args)
    evaluation_depth_tag = joint_du_unet_depth_tag(
        depth_indices, len(profile_depth_levels)
    )
    visualization_root = os.path.join(
        out_dir, "visualizations", evaluation_depth_tag
    )
    horizontal_dir = os.path.join(visualization_root, "horizontal")
    os.makedirs(horizontal_dir, exist_ok=True)
    model.eval()
    offset = 0

    for batch in loader:
        batch_size = batch["target"].shape[0]
        batch_dates = eval_dates[offset:offset + batch_size]
        batch_indices = eval_indices[offset:offset + batch_size]
        offset += batch_size
        if not any(day in panel_days for day in batch_dates):
            continue

        sst = batch["sst"].to(device)
        ssh_sss = batch["ssh_sss"].to(device)
        target_shape = batch["target"].shape[-2:]
        pred_full = model(sst, ssh_sss, output_size=target_shape)
        expected_shape = (
            batch_size,
            len(checkpoint_depth_indices),
            target_shape[0],
            target_shape[1],
        )
        if tuple(pred_full.shape) != expected_shape:
            raise ValueError(
                f"{args.method} 可视化推理输出形状错误，"
                f"expected={expected_shape}, actual={tuple(pred_full.shape)}"
            )

        pred_norm = pred_full[:, output_positions].cpu().numpy()
        true_norm = batch["target"][:, depth_indices].numpy()
        masks = batch["target_mask"][:, depth_indices].numpy().astype(bool)
        months = dataset.months[batch_indices]
        y_pred = target_norm.inverse_transform(pred_norm, months)
        y_true = target_norm.inverse_transform(true_norm, months)

        for local_index, day in enumerate(batch_dates):
            if day not in panel_days:
                continue
            date_tag = day.replace("-", "")
            valid = masks[local_index]
            pred_3d = np.where(valid, y_pred[local_index], np.nan)
            true_3d = np.where(valid, y_true[local_index], np.nan)
            record = panel_record_by_day[day]
            day_index = day_index_by_date[day]
            for level in target_levels:
                global_depth_index = int(depth_indices[level])
                ssim_value = float(day_depth_metrics["ssim"][day_index, level])
                acc_value = float(day_depth_metrics["acc"][day_index, level])
                role = record["role"].upper()
                method_label = (
                    f"{args.method} {role} | SSIM={ssim_value:.3f}, "
                    f"ACC={acc_value:.3f}"
                )
                filename = (
                    f"map_joint_panel_{args.target_var}_"
                    f"gidx{global_depth_index}_lvl{level}_{date_tag}.png"
                )
                output_path = os.path.join(horizontal_dir, filename)
                plot_prediction_truth_error_panel(
                    pred_3d[level],
                    true_3d[level],
                    np.abs(pred_3d[level] - true_3d[level]),
                    args.target_var,
                    global_depth_index,
                    day,
                    method_label,
                    output_path,
                    lon_range=lon_range,
                    lat_range=lat_range,
                    depth_values=profile_depth_levels,
                )
                rendered_spatial.append(
                    {
                        "date": day,
                        "role": record["role"],
                        "local_level": int(level),
                        "depth_index": global_depth_index,
                        "depth_m": float(depth_levels[level]),
                        "ssim": _finite_json_value(ssim_value),
                        "acc": _finite_json_value(acc_value),
                        "file": os.path.relpath(output_path, out_dir),
                    }
                )

    missing = sorted(panel_days - {record["date"] for record in rendered_spatial})
    if missing:
        raise RuntimeError(f"以下日期未成功绘制二维水平图: {missing}")
    return rendered_spatial


@torch.no_grad()
def _render_joint_section_panels(
    args,
    model,
    loader,
    target_norm,
    checkpoint_depth_indices,
    output_positions,
    depth_indices,
    depth_levels,
    eval_indices,
    eval_dates,
    panel_day_records,
    dataset,
    device,
    out_dir,
):
    """按需为联合深度模型绘制垂向剖面，不参与默认流程。"""
    if not panel_day_records:
        print("[section-panels] 没有可绘制的代表日期，跳过垂向剖面")
        return []

    panel_days = {record["date"] for record in panel_day_records}
    panel_record_by_day = {
        record["date"]: record for record in panel_day_records
    }
    lon_range, lat_range = _du_unet_lon_lat_range(args)
    profile_depth_levels = _du_unet_depth_levels(args)
    evaluation_depth_tag = joint_du_unet_depth_tag(
        depth_indices, len(profile_depth_levels)
    )
    vertical_dir = os.path.join(
        out_dir, "visualizations", evaluation_depth_tag, "vertical"
    )
    os.makedirs(vertical_dir, exist_ok=True)
    percentiles = _parse_percentile_list(
        args.section_percentiles, "--section-percentiles"
    )
    rendered = []
    model.eval()
    offset = 0

    for batch in loader:
        batch_size = batch["target"].shape[0]
        batch_dates = eval_dates[offset:offset + batch_size]
        batch_indices = eval_indices[offset:offset + batch_size]
        offset += batch_size
        if not any(day in panel_days for day in batch_dates):
            continue

        sst = batch["sst"].to(device)
        ssh_sss = batch["ssh_sss"].to(device)
        target_shape = batch["target"].shape[-2:]
        pred_full = model(sst, ssh_sss, output_size=target_shape)
        expected_shape = (
            batch_size,
            len(checkpoint_depth_indices),
            target_shape[0],
            target_shape[1],
        )
        if tuple(pred_full.shape) != expected_shape:
            raise ValueError(
                f"{args.method} 垂向可视化输出形状错误，"
                f"expected={expected_shape}, actual={tuple(pred_full.shape)}"
            )

        pred_norm = pred_full[:, output_positions].cpu().numpy()
        true_norm = batch["target"][:, depth_indices].numpy()
        masks = batch["target_mask"][:, depth_indices].numpy().astype(bool)
        months = dataset.months[batch_indices]
        y_pred = target_norm.inverse_transform(pred_norm, months)
        y_true = target_norm.inverse_transform(true_norm, months)
        section_specs = build_section_specs(
            target_shape[0],
            target_shape[1],
            multi=args.multi_section_panels,
            axis=args.section_axis,
            index=args.section_index,
            percentiles=percentiles,
        )

        for local_index, day in enumerate(batch_dates):
            if day not in panel_days:
                continue
            valid = masks[local_index]
            pred_3d = np.where(valid, y_pred[local_index], np.nan)
            true_3d = np.where(valid, y_true[local_index], np.nan)
            date_tag = day.replace("-", "")
            role = panel_record_by_day[day]["role"]
            for spec in section_specs:
                if spec["mode"] == "multi":
                    percentile_tag = f"p{spec['percentile']:g}".replace(".", "p")
                    section_tag = (
                        f"{spec['axis']}_{percentile_tag}_idx{spec['index']}"
                    )
                else:
                    section_tag = f"{spec['axis']}{spec['index']}"
                filename = (
                    f"section_joint_panel_{args.target_var}_"
                    f"{section_tag}_{date_tag}.png"
                )
                output_path = os.path.join(vertical_dir, filename)
                plot_joint_du_unet_section_panel(
                    pred_3d,
                    true_3d,
                    args.target_var,
                    spec["axis"],
                    spec["index"],
                    args.section_width,
                    day,
                    output_path,
                    depth_values=depth_levels,
                    lon_range=lon_range,
                    lat_range=lat_range,
                    title_prefix=f"{args.method} {role.upper()}",
                )
                rendered.append(
                    {
                        "date": day,
                        "role": role,
                        "section_axis": spec["axis"],
                        "section_index": int(spec["index"]),
                        "section_percentile": spec["percentile"],
                        "section_width": int(args.section_width),
                        "depth_indices": [int(index) for index in depth_indices],
                        "depth_m": [float(value) for value in depth_levels],
                        "file": os.path.relpath(output_path, out_dir),
                    }
                )

    missing = sorted(panel_days - {record["date"] for record in rendered})
    if missing:
        raise RuntimeError(f"以下日期未成功绘制垂向剖面: {missing}")
    return rendered


def evaluate_2dto2d_period(args, out_dir):
    """Evaluate B0 layer-wise or B1/B2/B3 joint-depth predictions."""
    from datasets.dataset_2dto2d import Dataset2Dto2D
    from train import build_model

    dataset = Dataset2Dto2D(
        args.data_dir,
        normalize=True,
        target_var=args.target_var,
        start_date=args.start_date,
        end_date=args.end_date,
        quality_report=args.quality_report,
        dataset_profile=args.dataset_profile,
        lazy=use_lazy_2dto2d_data(args),
        normalization_chunk_size=args.normalization_chunk_size,
        normalization_stats_path=args.normalization_stats_path,
        rebuild_normalization_stats=args.rebuild_normalization_stats,
    )
    all_depth_levels = _du_unet_depth_levels(args)
    if len(all_depth_levels) != dataset.num_depths:
        raise ValueError(
            f"dataset_profile={args.dataset_profile} 深度层数 {len(all_depth_levels)} "
            f"与 target 数据层数 {dataset.num_depths} 不一致"
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
    is_joint = _is_joint_method(args)
    joint_spec = _joint_method_spec(args) if is_joint else None
    model_name = joint_spec["model_name"] if is_joint else "Du_Unet"
    config_target_var = args.target_var if is_joint else None

    joint_model = None
    joint_output_positions = None
    checkpoint_depth_indices = None
    checkpoint_metadata = None
    loaded_checkpoint = None
    models = None
    if is_joint:
        (
            joint_model,
            loaded_checkpoint,
            checkpoint_depth_indices,
            checkpoint_metadata,
        ) = _load_joint_du_unet_model(args, device, dataset)
        depth_indices, joint_output_positions = _joint_evaluation_depths(
            args, all_depth_levels, checkpoint_depth_indices
        )
        parameter_payload = count_model_parameters(joint_model)
        checkpoint_candidates = _joint_du_unet_checkpoint_candidates(
            args,
            dataset.num_depths,
        )
        model_config = dict((checkpoint_metadata or {}).get("model_config") or {})
        model_config.update(
            {
                "out_channels": len(checkpoint_depth_indices),
                "joint_depth_output": True,
                "checkpoint_depth_indices": checkpoint_depth_indices,
                "evaluation_output_positions": joint_output_positions,
            }
        )
    else:
        depth_indices = _du_unet_depth_indices(args, all_depth_levels)
        param_stats = count_model_parameters(build_model("du_unet", out_channels=1))
        parameter_payload = {
            "per_depth_model": param_stats,
            "num_depth_models": len(depth_indices),
            "total_if_all_depth_models_loaded": {
                name: value * len(depth_indices)
                for name, value in param_stats.items()
            },
        }
        checkpoint_candidates = _du_unet_checkpoint_dirs(args)
        model_config = {"out_channels": 1}
        models = _load_du_unet_models(args, device, require_all=True)
    depth_levels = [all_depth_levels[idx] for idx in depth_indices]
    args.resolved_depth_indices = list(depth_indices)
    args.resolved_depth_levels = list(depth_levels)
    section_requested = args.section_panels or args.multi_section_panels
    if (args.spatial_panels or section_requested) and not is_joint:
        raise ValueError(
            "水平/垂向解释面板当前仅用于联合深度模型时间段评估"
        )
    target_levels = (
        _validate_panel_levels(args.target_levels, len(depth_indices))
        if args.spatial_panels
        else []
    )
    joint_depth_tag = (
        joint_du_unet_depth_tag(
            checkpoint_depth_indices, dataset.num_depths
        )
        if is_joint
        else None
    )
    train_config_path = model_config_path(
        args.output_dir,
        model_name,
        "train",
        target_var=config_target_var,
        run_tag=joint_depth_tag,
    )

    print_and_save_run_config(
        f"{model_name} 时间段测试超参数",
        {
            "stage": "test",
            "mode": "period",
            "single": False,
            "experiment": joint_spec["experiment"] if is_joint else "B0",
            "model": model_name,
            "method": args.method,
            "paradigm": PARADIGM_2DTO2D,
            "target_var": args.target_var,
            "device": device,
            "data": {
                "data_dir": args.data_dir,
                "dataset_profile": args.dataset_profile,
                "normalization_stats_path": dataset.normalization_stats_path,
                "normalization_stats": dataset.get_normalization_reference(),
                "rebuild_normalization_stats": args.rebuild_normalization_stats,
                "start_date": dataset.start_date,
                "end_date": dataset.end_date,
                "eval_start": args.eval_start,
                "eval_end": args.eval_end,
                "eval_samples": len(eval_indices),
            },
            "hyperparameters": {
                "eval_batch_size": args.eval_batch_size,
                "depth_indices": depth_indices,
                "depth_m": depth_levels,
                "spatial_panels": args.spatial_panels,
                "target_levels": target_levels,
                "panel_days": args.panel_days,
                "section_panels": args.section_panels,
                "multi_section_panels": args.multi_section_panels,
                "section_axis": args.section_axis,
                "section_index": args.section_index,
                "section_percentiles": args.section_percentiles,
                "section_width": args.section_width,
                "panel_day_strategy": (
                    "manual" if args.panel_days else "ssim_acc_quantiles_p10_p50_p90"
                ),
                "horizontal_structure_metrics": {
                    "ssim": {
                        "anomaly_reference": "training_monthly_climatology",
                        "win_size": SSIM_WIN_SIZE,
                        "sigma": SSIM_SIGMA,
                        "minimum_valid_fraction": SSIM_MIN_VALID_FRACTION,
                        "data_range": "6 * training_layer_std",
                    },
                    "acc": {
                        "anomaly_reference": "training_monthly_climatology",
                        "definition": "masked_spatial_pearson_correlation",
                    },
                },
            },
            "model_config": model_config,
            "training_strategy": load_training_strategy_reference(train_config_path),
            "parameter_count": parameter_payload,
            "paths": {
                "output_dir": out_dir,
                "checkpoint": args.checkpoint,
                "loaded_checkpoint": loaded_checkpoint,
                "checkpoint_metadata": checkpoint_metadata,
                "checkpoint_candidates": checkpoint_candidates,
                "train_config": train_config_path,
            },
        },
        model_config_path(
            args.output_dir,
            model_name,
            "test",
            target_var=config_target_var,
            run_tag=joint_depth_tag,
        ),
    )
    norm_stats = dataset.get_norm_stats()
    target_norm = _target_normalizer_for_depth_indices(norm_stats, depth_indices)
    target_stats = norm_stats["target"]
    selected_training_climatology = np.asarray(
        target_stats["climatology"], dtype=np.float32
    )[:, depth_indices]
    selected_layer_std = np.asarray(
        target_stats["layer_std"], dtype=np.float64
    )[depth_indices]
    ssim_data_range = SSIM_DATA_RANGE_STD_MULTIPLIER * selected_layer_std

    overall_acc = RegressionMetricAccumulator()
    depth_acc = RegressionMetricAccumulator()
    daily_records = []
    day_depth_chunks = []
    offset = 0

    with torch.no_grad():
        for batch in loader:
            sst = batch["sst"].to(device)
            ssh_sss = batch["ssh_sss"].to(device)
            target_shape = batch["target"].shape[-2:]
            if is_joint:
                pred_full = joint_model(sst, ssh_sss, output_size=target_shape)
                expected_shape = (
                    batch["target"].shape[0],
                    len(checkpoint_depth_indices),
                    target_shape[0],
                    target_shape[1],
                )
                if tuple(pred_full.shape) != expected_shape:
                    raise ValueError(
                        f"{model_name} 输出形状错误，"
                        f"expected={expected_shape}, actual={tuple(pred_full.shape)}"
                    )
                pred_norm = pred_full[:, joint_output_positions].cpu().numpy()
            else:
                pred_norm = torch.cat(
                    [
                        model(sst, ssh_sss, output_size=target_shape)
                        for model in models
                    ],
                    dim=1,
                ).cpu().numpy()
            true_norm = batch["target"][:, depth_indices].numpy()
            target_mask = batch["target_mask"][:, depth_indices].numpy()

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
            if is_joint:
                day_depth_chunks.append(
                    compute_day_depth_metrics(
                        y_true,
                        y_pred,
                        target_mask,
                        months,
                        selected_training_climatology,
                        ssim_data_range,
                        ssim_win_size=SSIM_WIN_SIZE,
                        ssim_sigma=SSIM_SIGMA,
                        ssim_min_valid_fraction=SSIM_MIN_VALID_FRACTION,
                    )
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
    day_depth_metrics = None
    horizontal_summary = None
    panel_day_records = []
    if is_joint:
        day_depth_metrics = {
            name: np.concatenate([chunk[name] for chunk in day_depth_chunks], axis=0)
            for name in (
                *DAY_DEPTH_METRIC_NAMES,
                "valid_count",
                "ssim_valid_windows",
                "acc_valid_count",
            )
        }
        if day_depth_metrics["ssim"].shape != (len(eval_dates), len(depth_indices)):
            raise RuntimeError(
                "逐日逐深度指标形状错误: "
                f"{day_depth_metrics['ssim'].shape}"
            )
        horizontal_summary = summarize_horizontal_structure(
            day_depth_metrics["ssim"], day_depth_metrics["acc"]
        )
        panel_day_records = _resolve_joint_panel_days(
            args, eval_dates, horizontal_summary
        ) if (args.spatial_panels or section_requested) else []

    period_tag = f"{args.eval_start}_{args.eval_end}"
    evaluation_name = evaluation_metrics_filename(
        args.method,
        period_tag,
        target_var=args.target_var,
    )
    payload = {
        "dates": np.asarray(eval_dates),
        "depth_indices": np.asarray(depth_indices, dtype=np.int64),
        "depth_m": np.asarray(depth_levels, dtype=np.float32),
        "overall_valid_count": overall_metrics["valid_count"],
        "by_depth_valid_count": depth_metrics["valid_count"],
        "by_day_valid_count": daily_metrics["valid_count"],
    }
    for metric_name in PERIOD_METRIC_NAMES:
        payload[f"overall_{metric_name}"] = overall_metrics[metric_name]
        payload[f"by_depth_{metric_name}"] = depth_metrics[metric_name]
        payload[f"by_day_{metric_name}"] = daily_metrics[metric_name]
    rendered_spatial_panels = []
    rendered_section_panels = []
    annual_equal_depth_mae = None
    annual_equal_depth_rmse = None
    if is_joint:
        for metric_name in ("mae", "rmse", "r2", "correlation"):
            payload[f"by_day_depth_{metric_name}"] = day_depth_metrics[metric_name]
        payload["by_day_depth_valid_count"] = day_depth_metrics["valid_count"]
        payload["by_day_depth_ssim"] = day_depth_metrics["ssim"]
        payload["by_day_depth_acc"] = day_depth_metrics["acc"]
        payload["by_day_depth_ssim_valid_windows"] = day_depth_metrics[
            "ssim_valid_windows"
        ]
        payload["by_day_depth_acc_valid_count"] = day_depth_metrics[
            "acc_valid_count"
        ]
        payload["by_depth_mean_ssim"] = horizontal_summary["by_depth_ssim"]
        payload["by_depth_mean_acc"] = horizontal_summary["by_depth_acc"]
        payload["by_depth_structure_score"] = horizontal_summary[
            "by_depth_structure_score"
        ]
        payload["by_depth_structure_valid_days"] = horizontal_summary[
            "by_depth_valid_days"
        ]
        payload["by_day_mean_ssim"] = horizontal_summary["by_day_ssim"]
        payload["by_day_mean_acc"] = horizontal_summary["by_day_acc"]
        payload["by_day_structure_score"] = horizontal_summary[
            "by_day_structure_score"
        ]
        payload["annual_mean_ssim"] = horizontal_summary["annual_mean_ssim"]
        payload["annual_mean_acc"] = horizontal_summary["annual_mean_acc"]
        payload["annual_mean_structure_score"] = horizontal_summary[
            "annual_mean_structure_score"
        ]
        annual_equal_depth_mae = float(np.nanmean(depth_metrics["mae"]))
        annual_equal_depth_rmse = float(
            np.sqrt(np.nanmean(np.square(depth_metrics["rmse"])))
        )
        payload["annual_equal_depth_mae"] = annual_equal_depth_mae
        payload["annual_equal_depth_rmse"] = annual_equal_depth_rmse
        payload["ssim_data_range_by_depth"] = ssim_data_range
        payload["selected_panel_dates"] = np.asarray(
            [record["date"] for record in panel_day_records]
        )
        payload["selected_panel_roles"] = np.asarray(
            [record["role"] for record in panel_day_records]
        )
        payload["selected_panel_quantiles"] = np.asarray(
            [
                np.nan if record.get("quantile") is None else record["quantile"]
                for record in panel_day_records
            ],
            dtype=np.float64,
        )

    np.savez(os.path.join(out_dir, evaluation_name), **payload)

    if args.spatial_panels:
        rendered_spatial_panels = _render_joint_horizontal_panels(
            args,
            joint_model,
            loader,
            target_norm,
            checkpoint_depth_indices,
            joint_output_positions,
            depth_indices,
            depth_levels,
            target_levels,
            eval_indices,
            eval_dates,
            panel_day_records,
            day_depth_metrics,
            dataset,
            device,
            out_dir,
        )
    if section_requested:
        rendered_section_panels = _render_joint_section_panels(
            args,
            joint_model,
            loader,
            target_norm,
            checkpoint_depth_indices,
            joint_output_positions,
            depth_indices,
            depth_levels,
            eval_indices,
            eval_dates,
            panel_day_records,
            dataset,
            device,
            out_dir,
        )

    depth_records = _grouped_metrics_to_records(
        depth_metrics,
        depth_indices,
        "depth_index",
        extra_values={"depth_m": depth_levels},
    )
    horizontal_structure = None
    if is_joint:
        by_depth_structure_records = []
        for position, (depth_index, depth_m) in enumerate(
            zip(depth_indices, depth_levels)
        ):
            by_depth_structure_records.append(
                {
                    "depth_index": int(depth_index),
                    "depth_m": float(depth_m),
                    "mean_ssim": _finite_json_value(
                        horizontal_summary["by_depth_ssim"][position]
                    ),
                    "std_ssim": _finite_json_value(
                        np.nanstd(day_depth_metrics["ssim"][:, position])
                    ),
                    "mean_acc": _finite_json_value(
                        horizontal_summary["by_depth_acc"][position]
                    ),
                    "std_acc": _finite_json_value(
                        np.nanstd(day_depth_metrics["acc"][:, position])
                    ),
                    "structure_score": _finite_json_value(
                        horizontal_summary["by_depth_structure_score"][position]
                    ),
                    "valid_days": int(
                        horizontal_summary["by_depth_valid_days"][position]
                    ),
                }
            )
        horizontal_structure = {
            "definitions": {
                "anomaly_reference": (
                    "2002-2021 training monthly climatology from normalization stats"
                ),
                "ssim": {
                    "window": "11x11 Gaussian",
                    "sigma": SSIM_SIGMA,
                    "minimum_valid_ocean_fraction": SSIM_MIN_VALID_FRACTION,
                    "data_range": "6 * training anomaly layer_std",
                },
                "acc": "masked spatial Pearson correlation of anomalies",
                "structure_score": "(daily_mean_ssim + daily_mean_acc) / 2",
            },
            "annual_equal_day_depth_mean": {
                "ssim": _finite_json_value(horizontal_summary["annual_mean_ssim"]),
                "acc": _finite_json_value(horizontal_summary["annual_mean_acc"]),
                "structure_score": _finite_json_value(
                    horizontal_summary["annual_mean_structure_score"]
                ),
            },
            "by_depth": by_depth_structure_records,
            "selected_days": panel_day_records,
        }
    summary = {
        "mode": "period",
        "method": args.method,
        "target_var": args.target_var,
        "eval_start": args.eval_start,
        "eval_end": args.eval_end,
        "num_days": len(eval_dates),
        "depth_indices": list(depth_indices),
        "depth_m": list(depth_levels),
        "metric_units": _period_metric_units(args.target_var),
        "variables": {
            args.target_var: {
                "overall": _period_metrics_to_json(overall_metrics),
                "by_depth": depth_records,
                "by_day": daily_records,
            }
        },
        "spatial_panels": {
            "enabled": bool(args.spatial_panels),
            "target_levels": target_levels,
            "target_depth_indices": [
                int(depth_indices[level]) for level in target_levels
            ],
            "panel_days": panel_day_records if args.spatial_panels else [],
            "selection_metric": (
                "daily_structure_score_p10_p50_p90"
                if args.spatial_panels and not args.panel_days
                else "manual panel-days" if args.spatial_panels else None
            ),
            "files": rendered_spatial_panels,
        },
        "section_panels": {
            "enabled": bool(section_requested),
            "single_section": bool(args.section_panels),
            "multi_section": bool(args.multi_section_panels),
            "section_axis": args.section_axis if args.section_panels else None,
            "section_index": args.section_index if args.section_panels else None,
            "section_percentiles": (
                _parse_percentile_list(
                    args.section_percentiles, "--section-percentiles"
                )
                if args.multi_section_panels
                else []
            ),
            "section_width": int(args.section_width),
            "panel_days": panel_day_records if section_requested else [],
            "files": rendered_section_panels,
        },
    }
    if is_joint:
        summary.update(
            {
                "aggregation_definition": {
                    "overall": (
                        "pooled over all valid days, depths and ocean grid cells"
                    ),
                    "by_depth": (
                        "pooled over all valid days and ocean grid cells per depth"
                    ),
                    "by_day": (
                        "pooled over all depths and ocean grid cells per day; pooled R2 "
                        "and correlation are auxiliary because vertical temperature "
                        "contrast can inflate them"
                    ),
                    "by_day_depth": (
                        "computed independently for each day and depth; stored in NPZ"
                    ),
                },
                "primary_numeric_metrics": {
                    "annual_equal_depth_mae": annual_equal_depth_mae,
                    "annual_equal_depth_rmse": annual_equal_depth_rmse,
                    "preferred_reporting": (
                        "by_depth MAE/RMSE plus horizontal SSIM/ACC"
                    ),
                },
                "horizontal_structure": horizontal_structure,
            }
        )
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
        np.asarray(depth_levels),
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

    if is_joint:
        structure_depth_plot_name = horizontal_structure_depth_plot_filename(
            args.method,
            period_tag,
            target_var=args.target_var,
        )
        _plot_horizontal_structure_metrics(
            np.asarray(depth_levels),
            horizontal_summary["by_depth_ssim"],
            horizontal_summary["by_depth_acc"],
            horizontal_summary["by_depth_structure_score"],
            "Depth / m",
            f"{args.method} {args.target_var} horizontal structure by depth",
            os.path.join(out_dir, structure_depth_plot_name),
        )
        structure_day_plot_name = horizontal_structure_day_plot_filename(
            args.method,
            period_tag,
            target_var=args.target_var,
        )
        _plot_horizontal_structure_metrics(
            np.arange(len(eval_dates)),
            horizontal_summary["by_day_ssim"],
            horizontal_summary["by_day_acc"],
            horizontal_summary["by_day_structure_score"],
            "Date",
            f"{args.method} {args.target_var} horizontal structure by day",
            os.path.join(out_dir, structure_day_plot_name),
            date_labels=eval_dates,
            selected_days=panel_day_records,
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
    if is_2dto3d:
        lon_range, lat_range = LON_RANGE, LAT_RANGE
    else:
        lon_range, lat_range = _du_unet_lon_lat_range(args)

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
    resolved_depth_indices = getattr(args, "resolved_depth_indices", None)
    resolved_depth_levels = getattr(args, "resolved_depth_levels", None)
    if resolved_depth_indices is not None and resolved_depth_levels is not None:
        summary.update(
            {
                "depth_indices": list(resolved_depth_indices),
                "depth_m": list(resolved_depth_levels),
                "target_depth_index": int(resolved_depth_indices[level]),
                "target_depth_m": float(resolved_depth_levels[level]),
            }
        )

    # ----- 2. 二维层图 (预测值) -----
    if is_2dto3d:
        temp_map = y_pred[0, level, :, :, 0]
        salt_map = y_pred[0, level, :, :, 1]
        plot_level_map(
            temp_map,
            title=f"{method} {day} Temp Level-{level}",
            output_path=os.path.join(out_dir, map_filename("temp", level, method, day)),
            lon_range=lon_range, lat_range=lat_range,
            cmap="RdYlBu_r", cbar_label="Temperature (°C)")
        plot_level_map(
            salt_map,
            title=f"{method} {day} Salt Level-{level}",
            output_path=os.path.join(out_dir, map_filename("salt", level, method, day)),
            lon_range=lon_range, lat_range=lat_range,
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
                lon_range=lon_range, lat_range=lat_range)

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
                lon_range=lon_range, lat_range=lat_range,
                cmap="RdYlBu_r", cbar_label="MAE")

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
                        lon_range=lon_range,
                        lat_range=lat_range,
                        depth_values=_du_unet_selected_depth_levels(args),
                    )
                else:
                    plot_level_map(
                        mae_level,
                        title=f"MAE {method} {day} Level-{level}",
                        output_path=os.path.join(
                            out_dir,
                            map_filename("mae", level, method, day, target_var=output_target_var),
                        ),
                        lon_range=lon_range, lat_range=lat_range,
                        cmap="RdYlBu_r", cbar_label="MAE")

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
        help="时间段评估推理 batch size，Du_Unet及联合深度模型使用",
    )
    p.add_argument(
        "--target-level",
        type=int,
        default=INFER_DEFAULT_TARGET_LEVEL,
        help="可视化深度在当前评估序列中的索引；使用--depth-indices时为切片后局部索引",
    )
    p.add_argument("--output-dir", default=OUTPUTS_ROOT)

    g_dl = p.add_argument_group("DL 方法参数")
    g_dl.add_argument("--checkpoint", default=None, help="模型权重路径")
    g_dl.add_argument(
        "--checkpoint-dir",
        default=None,
        help="2dto2d 权重目录；联合深度模型推荐直接传 --checkpoint",
    )
    g_dl.add_argument("--data-dir", default=DATA_DIR)
    g_dl.add_argument(
        "--dataset-profile",
        choices=dataset_profile_names(),
        default=DEFAULT_DATASET_PROFILE,
        help="2dto2d 数据集配置；旧实验用 legacy_small，新大区域用 glorys12v1_all_size",
    )
    g_dl.add_argument(
        "--lazy-data",
        action="store_true",
        help="2dto2d模型按样本延迟清洗/归一化；glorys12v1_all_size会自动启用",
    )
    g_dl.add_argument(
        "--normalization-chunk-size",
        type=int,
        default=32,
        help="lazy 归一化统计量拟合时的时间分块大小",
    )
    g_dl.add_argument(
        "--normalization-stats-path",
        default=None,
        help="归一化统计量 .npz；默认读取 data-dir/normalization_stats/",
    )
    g_dl.add_argument(
        "--rebuild-normalization-stats",
        action="store_true",
        help="忽略已有统计文件并用训练时段重新拟合后覆盖",
    )
    g_dl.add_argument("--target-var", choices=["temperature", "salinity"], default="temperature",
                      help="Du_Unet/联合深度模型预测目标变量")
    g_dl.add_argument("--start-date", default=DATA_START_DATE,
                      help="Du_Unet 使用的数据起始日期 YYYY-MM-DD")
    g_dl.add_argument("--end-date", default=None,
                      help="Du_Unet 使用的数据结束日期 YYYY-MM-DD，默认使用配置中的结束日期")
    g_dl.add_argument(
        "--depth-indices",
        default=None,
        help=(
            "按原始profile层号评估指定深度；联合模型必须是其训练层的子集；"
            "不传时自动评估checkpoint中记录的全部输出层"
        ),
    )
    g_dl.add_argument(
        "--spatial-panels",
        action="store_true",
        help="联合深度模型时间段评估输出预测/真值/绝对误差二维水平面板",
    )
    g_dl.add_argument(
        "--target-levels",
        default=None,
        help=(
            "--spatial-panels 绘制的局部通道位置，如 0,1,2,3,4,5；"
            "不传则绘制当前评估的全部深度"
        ),
    )
    g_dl.add_argument(
        "--panel-days",
        default=None,
        help=(
            "可选逗号分隔日期 YYYY-MM-DD；不传时按逐日 "
            "(SSIM+ACC)/2 自动选择 P10/P50/P90"
        ),
    )
    g_dl.add_argument(
        "--section-panels",
        action="store_true",
        help=(
            "可选：联合深度模型时间段评估输出一个垂向剖面三联图；"
            "不传则不调用垂向可视化"
        ),
    )
    g_dl.add_argument(
        "--multi-section-panels",
        action="store_true",
        help="可选：按多个经纬度百分位批量输出两个方向的垂向剖面",
    )
    g_dl.add_argument(
        "--section-axis",
        choices=["lat", "lon"],
        default="lat",
        help="--section-panels 固定轴；lat画depth-longitude，lon画depth-latitude",
    )
    g_dl.add_argument(
        "--section-index",
        default="middle",
        help="--section-panels 固定轴索引，整数或middle",
    )
    g_dl.add_argument(
        "--section-percentiles",
        default="10,25,50,75,90",
        help="--multi-section-panels 使用的位置百分位",
    )
    g_dl.add_argument(
        "--section-width",
        type=int,
        default=1,
        help="固定轴方向平均格点宽度，默认1表示不做带平均",
    )
    g_dl.add_argument("--dummy", action="store_true",
                      help="使用合成数据 (2dto3d)")
    g_dl.add_argument("--quality-report", action="store_true",
                      help="2dto2d模型加载数据时生成缺失率质检图")

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
    visualization_requested = (
        args.spatial_panels
        or args.section_panels
        or args.multi_section_panels
    )
    if visualization_requested and not args.eval_start:
        raise ValueError(
            "水平/垂向面板需要在时间段评估中使用 --eval-start/--eval-end"
        )
    if args.section_width <= 0:
        raise ValueError("--section-width 必须大于0")
    if args.section_panels and args.multi_section_panels:
        raise ValueError(
            "--section-panels 与 --multi-section-panels 二选一，不能同时传入"
        )
    if args.multi_section_panels:
        _parse_percentile_list(
            args.section_percentiles, "--section-percentiles"
        )

    method = args.method.strip().lower()
    if method == "du-unet":
        method = "du_unet"
    if method in {"joint-du-unet", "joint_unet", "joint-unet"}:
        method = "joint_du_unet"
    if method in {"joint-duvit", "joint_vit", "joint-vit"}:
        method = "joint_duvit"
    if method in {"b3-depthformer", "b3"}:
        method = "b3_depthformer"
    if method not in ALL_METHODS:
        raise ValueError(f"未知方法: {args.method}")
    args.method = {
        "du_unet": "Du_Unet",
        "joint_du_unet": "Joint_Du_Unet",
        "joint_duvit": "Joint_DUViT",
        "b3_depthformer": "B3_DepthFormer",
    }.get(method, method)
    if method in PARADIGM_2DTO2D_METHODS:
        paradigm = PARADIGM_2DTO2D
    else:
        paradigm = PARADIGM_2DTO3D
    if method in {
        "du_unet",
        "joint_du_unet",
        "joint_duvit",
        "b3_depthformer",
    }:
        out_dir = get_output_dir(
            paradigm, args.method, base_dir=args.output_dir, target_var=args.target_var
        )
    else:
        out_dir = get_output_dir(paradigm, args.method, base_dir=args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.eval_start:
        if method not in {
            "du_unet",
            "joint_du_unet",
            "joint_duvit",
            "b3_depthformer",
        }:
            raise ValueError(
                "时间段整体评估目前仅支持Du_Unet和联合深度模型"
            )
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
