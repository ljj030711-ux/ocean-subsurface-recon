"""输入数据质量检查工具。"""

from datetime import datetime, timedelta
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import PercentFormatter

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import DEPTH_LEVELS_25M


def _yearly_missing_stats(arr, start_date):
    """按时间维第 0 维统计每年 NaN/Inf 分布。"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    years = np.asarray(
        [(start + timedelta(days=i)).year for i in range(arr.shape[0])],
        dtype=np.int64,
    )
    out = {}
    for year in np.unique(years):
        year_arr = arr[years == year]
        nan_count = int(np.isnan(year_arr).sum())
        inf_count = int(np.isinf(year_arr).sum())
        total = int(year_arr.size)
        invalid = nan_count + inf_count
        out[int(year)] = {
            "nan_count": nan_count,
            "inf_count": inf_count,
            "invalid_count": invalid,
            "total_count": total,
            "ratio": 100.0 * invalid / max(total, 1),
        }
    return out


def _depth_missing_stats(arr, depth_dim):
    """按指定深度维统计每层 NaN/Inf 分布。"""
    if depth_dim < 0:
        depth_dim += arr.ndim
    if depth_dim < 0 or depth_dim >= arr.ndim:
        raise ValueError(f"depth_dim={depth_dim} 超出数组维度范围：ndim={arr.ndim}")

    depth_count = int(arr.shape[depth_dim])
    if depth_count > len(DEPTH_LEVELS_25M):
        raise ValueError(
            f"深度层数 D={depth_count} 超过 DEPTH_LEVELS_25M 长度 "
            f"{len(DEPTH_LEVELS_25M)}"
        )

    depth_first = np.moveaxis(arr, depth_dim, 0)
    out = {}
    for idx in range(depth_count):
        layer_arr = depth_first[idx]
        nan_count = int(np.isnan(layer_arr).sum())
        inf_count = int(np.isinf(layer_arr).sum())
        total = int(layer_arr.size)
        invalid = nan_count + inf_count
        out[int(idx)] = {
            "depth_m": DEPTH_LEVELS_25M[idx],
            "nan_count": nan_count,
            "inf_count": inf_count,
            "invalid_count": invalid,
            "total_count": total,
            "ratio": 100.0 * invalid / max(total, 1),
        }
    return out


def _safe_filename(name):
    """把数据名转换为安全文件名前缀。"""
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(name)).strip("._")
    return safe or "data"


def _ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def _plot_yearly_missing(name, yearly, output_path):
    years = np.asarray(sorted(yearly), dtype=np.int64)
    ratios = np.asarray([yearly[int(year)]["ratio"] for year in years], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(years, ratios, marker="o", linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Missing rate")
    ax.set_title(f"{name} yearly missing rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", labelrotation=35)
    upper = min(100.0, max(1.0, float(np.nanmax(ratios)) * 1.1 if ratios.size else 1.0))
    ax.set_ylim(0.0, upper)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_depth_missing(name, depth_stats, output_path):
    items = [depth_stats[idx] for idx in sorted(depth_stats)]
    depths = np.asarray([item["depth_m"] for item in items], dtype=np.float64)
    ratios = np.asarray([item["ratio"] for item in items], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot(depths, ratios, color="#1f77b4", linewidth=2)
    ax.set_xlabel("Depth(m)")
    ax.set_ylabel("Missing rate")
    ax.set_title(f"{name} depth missing rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(False)

    if depths.size:
        max_idx = int(np.nanargmax(ratios))
        max_depth = float(depths[max_idx])
        max_ratio = float(ratios[max_idx])
        ax.axvline(max_depth, color="#ff4d5a", linestyle="--", linewidth=1.2)
        ax.axhline(max_ratio, color="#ff4d5a", linestyle="--", linewidth=1.2)
        ax.annotate(
            f"({max_depth:g}m,{max_ratio:.0f}%)",
            xy=(max_depth, max_ratio),
            xytext=(24, -32),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "#ff4d5a", "lw": 1.1},
            fontsize=10,
        )
        upper = min(100.0, max(1.0, float(np.nanmax(ratios)) * 1.08))
        ax.set_ylim(0.0, upper)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def report_missing_values(
    name,
    array,
    start_date=None,
    depth_dim=None,
    output_dir="outputs/data_quality",
    plot=True,
):
    """
    检查数组中的 NaN/Inf，并可生成缺失率统计图。

    Args:
        name: 数据名（用于日志）
        array: 待检查数组
        start_date: 可选，时间维第 0 维对应的起始日期 YYYY-MM-DD
        depth_dim: 可选，深度维索引；传入时按 DEPTH_LEVELS_25M[:D] 绘制深度缺失率
        output_dir: 缺失率统计图输出目录
        plot: 是否保存统计图

    Returns:
        dict: 缺失值统计
    """
    arr = np.asarray(array)
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    total = int(arr.size)
    invalid = nan_count + inf_count
    ratio = 100.0 * invalid / max(total, 1)

    stats = {
        "name": name,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "invalid_count": invalid,
        "total_count": total,
        "ratio": ratio,
        "plot_paths": {},
    }

    safe_name = _safe_filename(name)
    if plot:
        _ensure_output_dir(output_dir)

    if start_date is not None:
        yearly = _yearly_missing_stats(arr, start_date)
        stats["yearly_missing"] = yearly
        if plot:
            yearly_path = os.path.join(output_dir, f"{safe_name}_yearly_missing.png")
            _plot_yearly_missing(name, yearly, yearly_path)
            stats["plot_paths"]["yearly_missing"] = yearly_path

    if depth_dim is not None:
        depth_stats = _depth_missing_stats(arr, depth_dim)
        stats["depth_missing"] = depth_stats
        if plot:
            depth_path = os.path.join(output_dir, f"{safe_name}_depth_missing.png")
            _plot_depth_missing(name, depth_stats, depth_path)
            stats["plot_paths"]["depth_missing"] = depth_path

    return stats


def sanitize_with_value(array, fill_value=0.0):
    """
    将 NaN/Inf 统一替换为给定值。
    """
    return np.nan_to_num(
        array,
        nan=fill_value,
        posinf=fill_value,
        neginf=fill_value,
    )

def main():
    """
    主函数
    """
    # report_missing_values("sla_sss", np.load("data/raw/SSH_2002-01-01_2023-12-31_10_18_110_118.npy"))
    # report_missing_values("T_true", np.load("data/raw/T-FIELD_2002-01-01_2023-12-31_10_18_110_118.npy"), start_date="2002-01-01")
    # report_missing_values("S_true", np.load("data/raw/S-FIELD_2002-01-01_2023-12-31_10_18_110_118.npy"), start_date="2002-01-01")
    report_missing_values("SWS_true", array=np.load("data/raw/S-FIELD_2002-01-01_2023-12-31_10_18_110_118.npy"), start_date="2002-01-01", depth_dim=1, output_dir="outputs/data_quality")

if __name__ == "__main__":
    main()
