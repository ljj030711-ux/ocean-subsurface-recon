"""
原始场可视化

功能：
    - 绘制指定日期的原始 SLA / SSS 二维图
    - 可选绘制原始场、全局 Z-score、月气候态 Z-score 三栏对比图
    - 若提供 sws 真值文件，绘制指定深度层真值盐度图和可选标准化对比图

用法示例：
    python utils/plot_raw_fields.py \
        --select-day 2023-03-12 \
        --sla-sss-path ./data/raw/sla_sss_2019-01-01_2023-12-31_10_18_110_118.npy \
        --sws-true-path ./data/raw/sws_2019-01-01_2023-12-31_10_18_110_118_0-300.npy \
        --level 10 \
        --compare-normalized \
        --output-dir ./outputs/raw_fields
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# 允许在 scripts/ 目录下直接运行本脚本
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    DATA_START_DATE,
    DATA_END_DATE,
    DEPTH_LEVELS_26M,
    TRAIN_END_DATE,
    LON_RANGE,
    LAT_RANGE,
)
from datasets.climatology_normalizer import MonthlyClimatologyLayerStdNormalizer
from datasets.date_utils import date_to_index, generate_month_numbers, indices_until_date
from utils.viz_layer_2d import plot_level_map


def parse_args():
    parser = argparse.ArgumentParser(description="原始场可视化")
    parser.add_argument("--select-day", required=True, help="日期 YYYY-MM-DD")
    parser.add_argument("--sla-sss-path", required=True,
                        help="海表原始输入 npy 路径，形状 (T,2,H,W)")
    parser.add_argument("--sws-true-path", default=None,
                        help="水下真值 npy 路径，形状 (T,D,H,W)")
    parser.add_argument("--level", type=int, default=10,
                        help="真值盐度图深度层索引（默认 10）")
    parser.add_argument("--output-dir", default="./outputs/raw_fields",
                        help="输出目录")
    parser.add_argument("--compare-normalized", action="store_true",
                        help="输出原始输入场、全局 Z-score、月气候态 Z-score 三栏对比图")
    parser.add_argument("--fit-end-date", default=TRAIN_END_DATE,
                        help="拟合 climatology/std 的结束日期，默认 TRAIN_END_DATE")
    return parser.parse_args()


def _finite_symmetric_limit(data_2d):
    finite = np.asarray(data_2d)[np.isfinite(data_2d)]
    if finite.size == 0:
        return 1.0
    vmax = float(np.nanmax(np.abs(finite)))
    return vmax if vmax > 0 else 1.0


def _plot_zscore_panel(ax, data_2d, title, extent):
    limit = _finite_symmetric_limit(data_2d)
    im = ax.imshow(
        data_2d,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
        origin="lower",
        extent=extent,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude / °E")
    ax.set_ylabel("Latitude / °N")
    return im


def plot_input_normalization_comparison(
    raw_2d,
    global_z_2d,
    monthly_z_2d,
    title,
    output_path,
    raw_cmap,
    raw_cbar_label,
):
    """绘制原始物理场、全局 Z-score、月气候态 Z-score 三栏对比图。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
    extent = [LON_RANGE[0], LON_RANGE[1], LAT_RANGE[0], LAT_RANGE[1]]

    raw_im = axes[0].imshow(
        raw_2d,
        cmap=raw_cmap,
        aspect="auto",
        origin="lower",
        extent=extent,
    )
    axes[0].set_title("Raw field")
    axes[0].set_xlabel("Longitude / °E")
    axes[0].set_ylabel("Latitude / °N")
    raw_cbar = fig.colorbar(raw_im, ax=axes[0])
    raw_cbar.set_label(raw_cbar_label)

    global_im = _plot_zscore_panel(axes[1], global_z_2d, "Global Z-score", extent)
    global_cbar = fig.colorbar(global_im, ax=axes[1])
    global_cbar.set_label("z-score")

    monthly_im = _plot_zscore_panel(
        axes[2], monthly_z_2d, "Monthly climatology Z-score", extent
    )
    monthly_cbar = fig.colorbar(monthly_im, ax=axes[2])
    monthly_cbar.set_label("z-score")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"标准化对比图已保存至：{output_path}")


def normalize_surface_channel(channel_data, months, fit_indices):
    """按训练期统计量对单个海表通道做月气候态标准化。"""
    normalizer = MonthlyClimatologyLayerStdNormalizer().fit(
        channel_data, months, fit_indices=fit_indices
    )
    return normalizer.transform(channel_data, months)


def global_zscore_surface_channel(channel_data, fit_indices, eps=1e-6):
    """按训练期全局均值和标准差对单个海表通道做 Z-score 标准化。"""
    arr = np.asarray(channel_data, dtype=np.float32)
    arr = arr.copy()
    arr[~np.isfinite(arr)] = np.nan

    fit_data = arr[np.asarray(fit_indices, dtype=np.int64)]
    mean = float(np.nanmean(fit_data))
    std = float(np.nanstd(fit_data))
    if not np.isfinite(mean):
        mean = 0.0
    if not np.isfinite(std) or std < eps:
        std = 1.0

    normalized = (arr - mean) / std
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def depth_label(level_idx):
    """返回包含索引和实际深度的层标签。"""
    if 0 <= level_idx < len(DEPTH_LEVELS_26M):
        return f"Level-{level_idx} {DEPTH_LEVELS_26M[level_idx]}m"
    return f"Level-{level_idx}"


def build_months_and_fit_indices(total_len, fit_end_date):
    months = np.asarray(generate_month_numbers(DATA_START_DATE, total_len), dtype=np.int64)
    fit_indices = indices_until_date(total_len, DATA_START_DATE, fit_end_date)
    if not fit_indices:
        raise ValueError(
            f"拟合时间范围为空：DATA_START_DATE={DATA_START_DATE}, "
            f"fit_end_date={fit_end_date}"
        )
    return months, fit_indices


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sla_sss = np.load(args.sla_sss_path).astype(np.float32)
    if sla_sss.ndim != 4 or sla_sss.shape[1] != 2:
        raise ValueError(f"sla_sss 需要 (T,2,H,W)，实际: {sla_sss.shape}")

    t = date_to_index(args.select_day, DATA_START_DATE, DATA_END_DATE)
    date_tag = args.select_day.replace("-", "")

    sla_2d = sla_sss[t, 0]
    sss_2d = sla_sss[t, 1]

    sla_img = os.path.join(args.output_dir, f"map_raw_sla_{date_tag}.png")
    sss_img = os.path.join(args.output_dir, f"map_raw_sss_{date_tag}.png")

    plot_level_map(
        sla_2d,
        title=f"Raw SLA {args.select_day}",
        output_path=sla_img,
        lon_range=LON_RANGE,
        lat_range=LAT_RANGE,
        cmap="RdBu_r",
        cbar_label="m",
    )
    plot_level_map(
        sss_2d,
        title=f"Raw SSS {args.select_day}",
        output_path=sss_img,
        lon_range=LON_RANGE,
        lat_range=LAT_RANGE,
        cmap="viridis",
        cbar_label="psu",
    )

    print(f"已输出：{sla_img}")
    print(f"已输出：{sss_img}")

    if args.compare_normalized:
        months, fit_indices = build_months_and_fit_indices(
            sla_sss.shape[0], args.fit_end_date
        )

        sla_global = global_zscore_surface_channel(sla_sss[:, 0], fit_indices)
        sss_global = global_zscore_surface_channel(sla_sss[:, 1], fit_indices)
        sla_monthly = normalize_surface_channel(sla_sss[:, 0], months, fit_indices)
        sss_monthly = normalize_surface_channel(sla_sss[:, 1], months, fit_indices)

        sla_compare_img = os.path.join(
            args.output_dir, f"compare_sla_raw_vs_norm_{date_tag}.png"
        )
        sss_compare_img = os.path.join(
            args.output_dir, f"compare_sss_raw_vs_norm_{date_tag}.png"
        )
        plot_input_normalization_comparison(
            sla_2d,
            sla_global[t],
            sla_monthly[t],
            title=f"SLA Input Normalization Comparison {args.select_day}",
            output_path=sla_compare_img,
            raw_cmap="RdBu_r",
            raw_cbar_label="m",
        )
        plot_input_normalization_comparison(
            sss_2d,
            sss_global[t],
            sss_monthly[t],
            title=f"SSS Input Normalization Comparison {args.select_day}",
            output_path=sss_compare_img,
            raw_cmap="viridis",
            raw_cbar_label="psu",
        )

    if args.sws_true_path:
        sws = np.load(args.sws_true_path).astype(np.float32)
        if sws.ndim != 4:
            raise ValueError(f"sws_true 需要 (T,D,H,W)，实际: {sws.shape}")
        if t >= sws.shape[0]:
            raise IndexError(f"t={t} 超出 sws 时间维范围 [0, {sws.shape[0]-1}]")
        if args.level < 0 or args.level >= sws.shape[1]:
            raise IndexError(f"level={args.level} 超出范围 [0, {sws.shape[1]-1}]")

        truth_2d = sws[t, args.level]
        level_label = depth_label(args.level)
        truth_img = os.path.join(
            args.output_dir, f"map_truth_salt_lvl{args.level}_{date_tag}.png"
        )
        plot_level_map(
            truth_2d,
            title=f"Truth Salt {args.select_day} {level_label}",
            output_path=truth_img,
            lon_range=LON_RANGE,
            lat_range=LAT_RANGE,
            cmap="viridis",
            cbar_label="psu",
        )
        print(f"已输出：{truth_img}")

        if args.compare_normalized:
            months, fit_indices = build_months_and_fit_indices(
                sws.shape[0], args.fit_end_date
            )
            target_level_series = sws[:, args.level]
            target_global = global_zscore_surface_channel(
                target_level_series, fit_indices
            )
            target_monthly = normalize_surface_channel(
                target_level_series, months, fit_indices
            )
            target_compare_img = os.path.join(
                args.output_dir,
                f"compare_sws_target_lvl{args.level}_raw_vs_norm_{date_tag}.png",
            )
            plot_input_normalization_comparison(
                truth_2d,
                target_global[t],
                target_monthly[t],
                title=f"SWS Target Normalization Comparison {args.select_day} {level_label}",
                output_path=target_compare_img,
                raw_cmap="viridis",
                raw_cbar_label="psu",
            )


if __name__ == "__main__":
    main()
