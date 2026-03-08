"""
测试脚本（按天序列）
按目标日期范围执行：数据加载 + 推理 + 保存 + 指标评估
"""

import json
import os
from datetime import datetime, timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config import (
    DATA_DIR,
    DATA_START_DATE,
    DX,
    DY,
    MODEL_NAME,
    MODEL_SAVE_PATH,
    PRED_SAVE_PATH,
    PREDICT_HORIZON_DAYS,
    USE_PHYSICS_FEATURES,
    WINDOW_DAYS,
)
from datasets.eddy_dataset import DailySequenceEddyDataset
from models.eddy_cnn import EddyAwareCNN, EddyResNet, EddyUNet
from utils.metrics import evaluate_prediction, print_metrics
from utils.physics import compute_eke, compute_grad_ssh


def build_model(model_name, in_channels):
    name = model_name.strip().lower()
    if name in {"eddyawarecnn", "eddycnn", "cnn"}:
        return EddyAwareCNN(in_channels=in_channels, out_channels=1)
    if name in {"eddyunet", "unet"}:
        return EddyUNet(in_channels=in_channels, out_channels=1)
    if name in {"eddyresnet", "resnet"}:
        return EddyResNet(in_channels=in_channels, out_channels=1)
    raise ValueError(f"未识别的模型名称: {model_name}")


def build_input_features(sst, sss, ssh, use_physics_features):
    if use_physics_features:
        eke = compute_eke(ssh, dx=DX, dy=DY)
        grad = compute_grad_ssh(ssh)
        return torch.cat([sst, sss, ssh, eke, grad], dim=1)
    return torch.cat([sst, sss, ssh], dim=1)


def _load_model(model_name, in_channels, device):
    model = build_model(model_name, in_channels=in_channels).to(device)
    lower_name = model_name.strip().lower()
    candidate_paths = [
        f"./checkpoints/{lower_name}_best_model.pth",
        "./checkpoints/best_model.pth",
        MODEL_SAVE_PATH,
    ]

    last_error = None
    for weight_path in candidate_paths:
        if not os.path.exists(weight_path):
            continue
        try:
            state_dict = torch.load(weight_path, map_location=device)
            model.load_state_dict(state_dict, strict=True)
            return model, weight_path
        except RuntimeError as e:
            last_error = e

    print(f"⚠️ {model_name} 未找到可用匹配权重，将使用随机初始化模型。")
    if last_error is not None:
        print(f"   最近一次加载错误: {last_error}")
    return model, None


def _date_to_day_idx(date_str):
    series_start = datetime.strptime(DATA_START_DATE, "%Y-%m-%d")
    cur = datetime.strptime(date_str, "%Y-%m-%d")
    return (cur - series_start).days


def _resolve_sample_indices(dataset, start_date, end_date):
    start_day = _date_to_day_idx(start_date)
    end_day = _date_to_day_idx(end_date)
    if end_day < start_day:
        raise ValueError("end_date 不能早于 start_date")

    available_start = int(dataset.target_day_indices[0])
    available_end = int(dataset.target_day_indices[-1])
    if start_day < available_start or end_day > available_end:
        series_start = datetime.strptime(DATA_START_DATE, "%Y-%m-%d")
        avail_start_date = (series_start + timedelta(days=available_start)).strftime("%Y-%m-%d")
        avail_end_date = (series_start + timedelta(days=available_end)).strftime("%Y-%m-%d")
        raise ValueError(
            f"时间范围超出可预测目标日期: 请求 [{start_date}, {end_date}], "
            f"可用 [{avail_start_date}, {avail_end_date}]"
        )

    sample_indices = np.where(
        (dataset.target_day_indices >= start_day) & (dataset.target_day_indices <= end_day)
    )[0]
    if len(sample_indices) == 0:
        raise ValueError("给定日期范围内没有可用样本")
    return sample_indices.tolist()


def perform_inversion(
    start_date,
    end_date,
    model_names,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    按日期范围（天）进行反演评估。日期表示目标 100m 温场对应日期。
    """
    os.makedirs("./outputs", exist_ok=True)

    print("=" * 60)
    print("模型测试与推理（按天序列）")
    print("=" * 60)
    print(f"时间范围(目标日期): {start_date} ~ {end_date}")
    print(f"模型列表: {model_names}")
    print(
        f"序列配置: window_days={WINDOW_DAYS}, horizon_days={PREDICT_HORIZON_DAYS}, "
        f"use_physics_features={USE_PHYSICS_FEATURES}"
    )

    dataset = DailySequenceEddyDataset(
        DATA_DIR,
        window_days=WINDOW_DAYS,
        horizon_days=PREDICT_HORIZON_DAYS,
        normalize=True,
    )
    indices = _resolve_sample_indices(dataset, start_date, end_date)
    infer_dataset = Subset(dataset, indices)
    loader = DataLoader(infer_dataset, batch_size=1, shuffle=False)
    print(f"可用样本总数: {len(dataset)}，本次评估样本: {len(infer_dataset)}")

    per_day_channels = 5 if USE_PHYSICS_FEATURES else 3
    in_channels = per_day_channels * WINDOW_DAYS

    metrics_json = {}
    predictions_map = {}
    all_targets = []

    for model_idx, model_name in enumerate(model_names):
        print("\n" + "-" * 60)
        print(f"加载模型: {model_name}")
        model, weight_path = _load_model(model_name, in_channels=in_channels, device=device)
        model.eval()

        pred_list = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                sst = batch["sst"].to(device)       # [B,D,H,W]
                sss = batch["sss"].to(device)       # [B,D,H,W]
                ssh = batch["ssh"].to(device)       # [B,D,H,W]
                target = batch["target"]            # [B,1,H,W]

                x = build_input_features(sst, sss, ssh, USE_PHYSICS_FEATURES)
                pred = model(x)

                pred_list.append(pred.cpu().numpy())
                if model_idx == 0:
                    all_targets.append(target.numpy())

                if (i + 1) % max(1, len(loader) // 10) == 0:
                    print(f"  进度: {i + 1}/{len(loader)}")

        predictions = np.concatenate(pred_list, axis=0)  # [N,1,H,W]
        targets = np.concatenate(all_targets, axis=0)    # [N,1,H,W]
        predictions_map[model_name] = predictions

        lower_name = model_name.strip().lower()
        npz_path = f"./outputs/inversion_{start_date}_{end_date}_{lower_name}.npz"
        np.savez_compressed(npz_path, targets=targets, predictions=predictions)
        print(f"✓ 保存完成: {npz_path}")
        print(f"权重信息: {weight_path if weight_path else 'random_init'}")
        print(f"  - targets: shape={targets.shape}")
        print(f"  - predictions: shape={predictions.shape}")

        metrics = evaluate_prediction(targets, predictions)
        metrics_json[model_name] = {k: float(v) for k, v in metrics.items()}
        print_metrics(metrics, prefix=f"{model_name} 全局")

        if len(model_names) == 1:
            np.save(PRED_SAVE_PATH, predictions)
            np.save("./outputs/targets.npy", targets)
            print(f"✓ 预测结果已保存到 {PRED_SAVE_PATH}")
            print("✓ 目标值已保存到 ./outputs/targets.npy")

    metrics_path = f"./outputs/inversion_metrics_{start_date}_{end_date}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 指标已保存到 {metrics_path}")

    print("\n" + "=" * 60)
    print("✓ 测试完成（按天序列）")
    print("=" * 60)

    if len(model_names) == 1 and model_names[0] in predictions_map:
        return predictions_map[model_names[0]]
    return predictions_map


if __name__ == "__main__":
    # window_days=1 时，30 天数据可预测目标日期范围是 2025-01-01 到 2025-01-30
    perform_inversion("2025-01-01", "2025-01-30", [MODEL_NAME])
