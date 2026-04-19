"""
原始场可视化脚本

功能：
    - 绘制指定日期的原始 SLA / SSS 二维图
    - 若提供 sws 真值文件，绘制指定深度层真值盐度图

用法示例：
    python utils/plot_raw_fields.py \
        --select-day 2023-03-12 \
        --sla-sss-path ./data/raw/sla_sss_2019-01-01_2023-12-31_10_18_110_118.npy \
        --sws-true-path ./data/raw/sws_2019-01-01_2023-12-31_10_18_110_118_0-300.npy \
        --level 10 \
        --output-dir ./outputs/raw_fields
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# 允许在 scripts/ 目录下直接运行本脚本
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DATA_START_DATE, DATA_END_DATE, LON_RANGE, LAT_RANGE
from datasets.date_utils import date_to_index
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
    return parser.parse_args()


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

    if args.sws_true_path:
        sws = np.load(args.sws_true_path).astype(np.float32)
        if sws.ndim != 4:
            raise ValueError(f"sws_true 需要 (T,D,H,W)，实际: {sws.shape}")
        if t >= sws.shape[0]:
            raise IndexError(f"t={t} 超出 sws 时间维范围 [0, {sws.shape[0]-1}]")
        if args.level < 0 or args.level >= sws.shape[1]:
            raise IndexError(f"level={args.level} 超出范围 [0, {sws.shape[1]-1}]")

        truth_2d = sws[t, args.level]
        truth_img = os.path.join(
            args.output_dir, f"map_truth_salt_lvl{args.level}_{date_tag}.png"
        )
        plot_level_map(
            truth_2d,
            title=f"Truth Salt {args.select_day} Level-{args.level}",
            output_path=truth_img,
            lon_range=LON_RANGE,
            lat_range=LAT_RANGE,
            cmap="viridis",
            cbar_label="psu",
        )
        print(f"已输出：{truth_img}")


if __name__ == "__main__":
    main()
