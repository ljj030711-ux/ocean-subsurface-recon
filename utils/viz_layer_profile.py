"""HVCARefiner 垂向剖面六子图可视化。"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from utils.viz_layer_2d import (
    field_cmap_and_label,
    finite_symmetric_limit,
    finite_vmin_vmax,
)

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _section_bounds(index, width, size):
    width = max(int(width), 1)
    half = width // 2
    start = max(index - half, 0)
    end = min(start + width, size)
    start = max(end - width, 0)
    return start, end


def _axis_coords(axis, lat_range, lon_range, h, w):
    if axis == "lat":
        return np.linspace(lon_range[0], lon_range[1], w), "Longitude / °E"
    if axis == "lon":
        return np.linspace(lat_range[0], lat_range[1], h), "Latitude / °N"
    raise ValueError(f"section_axis 必须是 lat 或 lon，实际: {axis}")


def _extract_section(field_3d, axis, index, width):
    arr = np.asarray(field_3d, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"field_3d 需要 (D,H,W)，实际: {arr.shape}")
    _, h, w = arr.shape
    if axis == "lat":
        start, end = _section_bounds(index, width, h)
        return np.nanmean(arr[:, start:end, :], axis=1), start, end
    if axis == "lon":
        start, end = _section_bounds(index, width, w)
        return np.nanmean(arr[:, :, start:end], axis=2), start, end
    raise ValueError(f"section_axis 必须是 lat 或 lon，实际: {axis}")


def plot_hvca_refiner_section_panel(
    baseline_3d,
    refined_3d,
    true_3d,
    target_var,
    section_axis,
    section_index,
    section_width,
    day,
    output_path,
    depth_values,
    lon_range=(110, 118),
    lat_range=(10, 18),
):
    """绘制 HVCARefiner 的 T0/refined/truth/error/delta 垂向剖面面板。"""
    baseline_3d = np.asarray(baseline_3d, dtype=np.float32)
    refined_3d = np.asarray(refined_3d, dtype=np.float32)
    true_3d = np.asarray(true_3d, dtype=np.float32)
    if baseline_3d.shape != refined_3d.shape or baseline_3d.shape != true_3d.shape:
        raise ValueError(
            f"baseline/refined/true 形状不一致: "
            f"{baseline_3d.shape}, {refined_3d.shape}, {true_3d.shape}"
        )
    if baseline_3d.ndim != 3:
        raise ValueError(f"输入场需要 (D,H,W)，实际: {baseline_3d.shape}")

    d, h, w = baseline_3d.shape
    depth = np.asarray(depth_values, dtype=np.float32)
    if depth.size != d:
        raise ValueError(f"depth_values 长度 {depth.size} 与 D={d} 不一致")

    baseline_sec, start, end = _extract_section(
        baseline_3d, section_axis, section_index, section_width
    )
    refined_sec, _, _ = _extract_section(
        refined_3d, section_axis, section_index, section_width
    )
    true_sec, _, _ = _extract_section(
        true_3d, section_axis, section_index, section_width
    )
    baseline_err = np.abs(baseline_sec - true_sec)
    refined_err = np.abs(refined_sec - true_sec)
    delta = refined_sec - baseline_sec

    x_coords, x_label = _axis_coords(section_axis, lat_range, lon_range, h, w)
    section_center = (start + end - 1) / 2
    if section_axis == "lat":
        section_coord = np.interp(section_center, np.arange(h), np.linspace(lat_range[0], lat_range[1], h))
        section_label = f"lat idx {start}:{end} (~{section_coord:.2f}°N)"
    else:
        section_coord = np.interp(section_center, np.arange(w), np.linspace(lon_range[0], lon_range[1], w))
        section_label = f"lon idx {start}:{end} (~{section_coord:.2f}°E)"

    cmap, value_label, error_label = field_cmap_and_label(target_var)
    value_vmin, value_vmax = finite_vmin_vmax(baseline_sec, refined_sec, true_sec)
    err_vmin, err_vmax = finite_vmin_vmax(
        baseline_err, refined_err, default=(0.0, 1.0)
    )
    err_vmin = max(err_vmin, 0.0)
    delta_limit = finite_symmetric_limit(delta)
    delta_norm = TwoSlopeNorm(vmin=-delta_limit, vcenter=0.0, vmax=delta_limit)

    extent = [float(x_coords[0]), float(x_coords[-1]), float(depth[-1]), float(depth[0])]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150)
    panels = [
        ("Baseline T0", baseline_sec, cmap, value_label, value_vmin, value_vmax, None),
        ("HVCA Refined", refined_sec, cmap, value_label, value_vmin, value_vmax, None),
        ("Truth", true_sec, cmap, value_label, value_vmin, value_vmax, None),
        ("|T0 - Truth|", baseline_err, "magma", error_label, err_vmin, err_vmax, None),
        ("|HVCA - Truth|", refined_err, "magma", error_label, err_vmin, err_vmax, None),
        ("Delta = HVCA - T0", delta, "RdBu_r", value_label, None, None, delta_norm),
    ]
    for ax, (title, data, panel_cmap, cbar_label, vmin, vmax, norm) in zip(
        axes.flat, panels
    ):
        im = ax.imshow(
            data,
            cmap=panel_cmap,
            aspect="auto",
            origin="upper",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
        )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Depth / m")
        ax.set_ylim(float(depth[-1]), float(depth[0]))
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)

    fig.suptitle(f"HVCARefiner {target_var} section {day} | {section_label}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"HVCA 垂向剖面解释图已保存至：{output_path}")
