"""统一预测评估入口（2dto2d / 2dto3d）。"""

import argparse
import json
import os
import warnings

import numpy as np
import torch
from models.inversion_2dvar import inversion_2dvar
from models.inversion_modas import inversion_modas

from config import (
    CHECKPOINTS_ROOT,
    DATA_DIR,
    DATA_END_DATE,
    DATA_START_DATE,
    DEPTH_LEVELS_26M,
    DX,
    DY,
    EDDY_UNET_CKPT_NAME_TEMPLATE,
    EDDY_UNET_USE_PHYSICS_FEATURES,
    INFER_DEFAULT_TARGET_LEVEL,
    INFER_PROFILE_METRIC,
    INFER_PROFILE_ZMAX,
    TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE,
    TEST_START_DATE, TEST_END_DATE,
    LON_RANGE, LAT_RANGE, DEPTH_MAX, OUTPUTS_ROOT,
    PARADIGM_2DTO2D, PARADIGM_2DTO2D_METHODS,
    PARADIGM_2DTO3D, PARADIGM_2DTO3D_METHODS,
    TWODTO3D_DEPTH_LEVELS,
    get_checkpoint_dir,
    get_output_dir,
)
from datasets.date_utils import date_to_index
from datasets.climatology_normalizer import MonthlyClimatologyLayerStdNormalizer
from datasets.non_dl_preprocess import get_dataset_split, load_and_validate, load_sla_sss
from utils.metrics import (
    compute_grid_metrics, extract_level_map, save_grid_metrics,
    mse, rmse, mae, r2,
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

def _tag(method, date_str):
    return method, date_str.replace("-", "")


def pred_filename(method, date_str):
    m, d = _tag(method, date_str)
    return f"pred_{m}_{d}.npy"


def metrics_filename(method, date_str):
    m, d = _tag(method, date_str)
    return f"grid_metrics_{m}_{d}.npz"


def map_filename(var, level, method, date_str):
    m, d = _tag(method, date_str)
    return f"map_{var}_lvl{level}_{m}_{d}.png"


def profile_filename(metric_name, method, date_str):
    m, d = _tag(method, date_str)
    return f"profile_{metric_name}_{m}_{d}.png"


def summary_filename(method, date_str):
    m, d = _tag(method, date_str)
    return f"summary_{m}_{d}.json"


# ==================== Non-DL 预测 (2DVar / MODAS) ====================


def predict_non_dl(args):
    """运行 2DVar 或 MODAS 反演，返回原始单位的 (y_pred, y_true_or_None)。"""
    sws_true_full = None

    if args.method == "2dvar":
        sla_sss = load_sla_sss(args.sla_sss_path)
        t = date_to_index(args.select_day, DATA_START_DATE, DATA_END_DATE)
        y_pred = inversion_2dvar(sla_sss, t, c_depth=args.c_depth)
        if args.sws_true_path:
            sws_true_full = np.load(args.sws_true_path).astype(np.float32)

    elif args.method == "modas":
        if not args.sws_true_path:
            raise ValueError("MODAS 需要 --sws-true-path")
        sla_sss, sws_true_full = load_and_validate(
            args.sla_sss_path, args.sws_true_path)
        t = date_to_index(args.select_day, DATA_START_DATE, DATA_END_DATE)
        ds = get_dataset_split(
            DATA_START_DATE, DATA_END_DATE,
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
        t = min(date_to_index(args.select_day, DATA_START_DATE, DATA_END_DATE),
                len(dataset) - 1)
    else:
        from datasets.dataset_2dto3d import TwoDto3DDataset
        dataset = TwoDto3DDataset(args.data_dir, normalize=True)
        t = date_to_index(args.select_day, DATA_START_DATE, DATA_END_DATE)

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
    """2dto2d / 2dto3d 的 2D 输入类模型推理。"""
    from train import build_model, build_2dto2d_features
    device = _get_device()

    use_phys = args.use_physics_features
    in_ch = 4 if use_phys else 2

    from datasets.eddy_dataset import EddyDataset
    dataset = EddyDataset(args.data_dir, normalize=True)
    t = date_to_index(args.select_day, DATA_START_DATE, DATA_END_DATE)
    sample = dataset[t]
    norm_stats = dataset.get_norm_stats()
    sss = sample["sss"].unsqueeze(0).to(device)
    ssh = sample["ssh"].unsqueeze(0).to(device)
    x = build_2dto2d_features(sss, ssh, use_phys)

    if args.method == "eddy_unet":
        # 2dto2d：26 层分别加载 checkpoint 推理并拼装
        ckpt_dir = args.checkpoint_dir or get_checkpoint_dir(
            PARADIGM_2DTO2D, "eddy_unet", base_dir=CHECKPOINTS_ROOT
        )
        pred_list = []
        for depth_m in DEPTH_LEVELS_26M:
            ckpt_name = EDDY_UNET_CKPT_NAME_TEMPLATE.format(depth_m=depth_m)
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            model = build_model("eddy_unet", in_channels=in_ch, out_channels=1).to(device)
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            else:
                print(f"[警告] 未找到深度 {depth_m}m checkpoint：{ckpt_path}，将使用随机权重")
            model.eval()
            with torch.no_grad():
                pred_layer = model(x).cpu().numpy()  # (1,1,H,W)
            pred_list.append(pred_layer)
        y_pred = np.concatenate(pred_list, axis=1)  # (1,26,H,W)
    else:
        # 2dto3d 内的 CNN 兼容分支
        num_out = int(sample["target"].shape[0])
        model = build_model(args.method, in_channels=in_ch, out_channels=num_out).to(device)
        if args.checkpoint and os.path.exists(args.checkpoint):
            model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
            print(f"已加载权重: {args.checkpoint}")
        else:
            print("未提供/找到权重，使用随机初始化模型")
        model.eval()
        with torch.no_grad():
            y_pred = model(x).cpu().numpy()

    y_true = sample["target"].unsqueeze(0).numpy()      # (1, C, H, W)

    if norm_stats.get("normalization") == "monthly_climatology_layer_std":
        target_norm = MonthlyClimatologyLayerStdNormalizer.from_stats(norm_stats["target"])
        months = np.asarray([dataset.months[t]], dtype=np.int64)
        y_pred = target_norm.inverse_transform(y_pred, months)
        y_true = target_norm.inverse_transform(y_true, months)

    return y_pred, y_true


# ==================== 统一输出 ====================

def _scalar_metrics(y_true_flat, y_pred_flat):
    mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
    yt, yp = y_true_flat[mask], y_pred_flat[mask]
    return {
        "mse": float(mse(yt, yp)),
        "rmse": float(rmse(yt, yp)),
        "mae": float(mae(yt, yp)),
        "r2": float(r2(yt, yp)),
    }


def _metric_units(is_2dto3d):
    if is_2dto3d:
        return {
            "temperature": {
                "mse": "degC^2",
                "rmse": "degC",
                "mae": "degC",
                "r2": "dimensionless",
            },
            "salinity": {
                "mse": "psu^2",
                "rmse": "psu",
                "mae": "psu",
                "r2": "dimensionless",
            },
        }
    return {
        "mse": "psu^2",
        "rmse": "psu",
        "mae": "psu",
        "r2": "dimensionless",
    }


def save_outputs(args, y_pred, y_true, out_dir):
    """统一产物保存：npy / npz / 2D 图 / 3D 剖面 / summary.json"""
    method = args.method
    day = args.select_day
    level = args.target_level
    is_2dto3d = method == "ocean_transformer"

    # ----- 1. 保存预测场 npy -----
    np.save(os.path.join(out_dir, pred_filename(method, day)), y_pred)
    print(f"预测结果: {pred_filename(method, day)}  shape={y_pred.shape}")

    summary = {
        "method": method,
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
        if y_pred.ndim == 4 and y_pred.shape[1] > 1:
            vis_data = y_pred[0, level]
        elif y_pred.ndim == 4:
            vis_data = y_pred[0, 0]
        else:
            vis_data = y_pred[0]
        plot_level_map(
            vis_data,
            title=f"{method} {day} Level-{level}",
            output_path=os.path.join(out_dir, map_filename("salt", level, method, day)),
            lon_range=LON_RANGE, lat_range=LAT_RANGE)

    # ----- 3. 有真值时：网格指标 + 误差图 + 剖面图 -----
    if y_true is not None:
        if is_2dto3d:
            for vi, vname in enumerate(["temperature", "salinity"]):
                vp = y_pred[..., vi]
                vt = y_true[..., vi]
                summary["variables"][vname] = _scalar_metrics(vt.flatten(), vp.flatten())

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
            grid = compute_grid_metrics(y_true, y_pred)
            npz_path = os.path.join(out_dir, metrics_filename(method, day))
            save_grid_metrics(grid, npz_path)

            summary["variables"]["salinity"] = _scalar_metrics(
                y_true.flatten(), y_pred.flatten())

            if y_true.ndim >= 3 and (y_true.ndim < 4 or y_true.shape[1] > 1):
                mae_level = extract_level_map(grid["mae"], level)
                plot_level_map(
                    mae_level,
                    title=f"MAE {method} {day} Level-{level}",
                    output_path=os.path.join(
                        out_dir, map_filename("mae", level, method, day)),
                    lon_range=LON_RANGE, lat_range=LAT_RANGE,
                    cmap="RdYlBu_r", cbar_label="MAE (psu)")

                profile_path = os.path.join(
                    out_dir, profile_filename("rmse", method, day))
                plot_3d_metric_profile(
                    npz_path, metric_name=INFER_PROFILE_METRIC,
                    output_img_path=profile_path,
                    lon_range=LON_RANGE, lat_range=LAT_RANGE,
                    z_max=INFER_PROFILE_ZMAX)

    # ----- 4. summary.json -----
    summary_path = os.path.join(out_dir, summary_filename(method, day))
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"汇总指标: {summary_filename(method, day)}")


# ==================== 主入口 ====================

def parse_args():
    p = argparse.ArgumentParser(description="统一预测评估入口")
    p.add_argument("--method", required=True, choices=ALL_METHODS)
    p.add_argument("--select-day", required=True, help="预测日期 YYYY-MM-DD")
    p.add_argument("--target-level", type=int, default=INFER_DEFAULT_TARGET_LEVEL,
                   help="可视化目标深度层索引")
    p.add_argument("--output-dir", default=OUTPUTS_ROOT)

    g_dl = p.add_argument_group("DL 方法参数")
    g_dl.add_argument("--checkpoint", default=None, help="模型权重路径")
    g_dl.add_argument("--checkpoint-dir", default=None, help="2dto2d 多模型权重目录")
    g_dl.add_argument("--data-dir", default=DATA_DIR)
    g_dl.add_argument("--use-physics-features", action="store_true",
                      default=EDDY_UNET_USE_PHYSICS_FEATURES)
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
    method = args.method.strip().lower()
    if method in PARADIGM_2DTO2D_METHODS:
        paradigm = PARADIGM_2DTO2D
    else:
        paradigm = PARADIGM_2DTO3D
    out_dir = get_output_dir(paradigm, method, base_dir=args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"预测评估：paradigm={paradigm}, method={method}, day={args.select_day}, "
          f"level={args.target_level}")
    print("=" * 60)

    if method in NON_DL_METHODS:
        y_pred, y_true = predict_non_dl(args)
    elif method == "ocean_transformer":
        y_pred, y_true = predict_2dto3d(args)
    elif method in (DL_2DTO2D_METHODS | DL_2DTO3D_METHODS):
        y_pred, y_true = predict_2dto2d(args)
    else:
        raise ValueError(f"未知方法: {method}")

    save_outputs(args, y_pred, y_true, out_dir)
    print("\n预测评估完成。")


if __name__ == "__main__":
    main()
