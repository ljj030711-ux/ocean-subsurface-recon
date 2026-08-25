"""2dto2d 模型的垂向剖面可视化。"""

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


def resolve_section_index(value, size):
    """把 ``middle`` 或整数文本解析为合法的剖面固定轴索引。"""
    text = str(value).strip().lower()
    if text in {"middle", "mid", "center", "centre"}:
        return size // 2
    try:
        index = int(text)
    except ValueError as exc:
        raise ValueError(
            f"section_index 必须是整数或 middle，实际: {value}"
        ) from exc
    if index < 0 or index >= size:
        raise ValueError(f"section_index 超出范围 [0,{size - 1}]: {index}")
    return index


def build_section_specs(
    height,
    width,
    multi=False,
    axis="lat",
    index="middle",
    percentiles=(10.0, 25.0, 50.0, 75.0, 90.0),
):
    """生成单剖面或经纬两个方向的多位置剖面配置。"""
    if multi:
        specs = []
        for current_axis, size in (("lat", height), ("lon", width)):
            seen = set()
            for percentile in percentiles:
                percentile = float(percentile)
                if percentile < 0.0 or percentile > 100.0:
                    raise ValueError(
                        f"section percentile 需位于 [0,100]，实际: {percentile}"
                    )
                current_index = int(round(percentile / 100.0 * (size - 1)))
                if current_index in seen:
                    continue
                seen.add(current_index)
                specs.append(
                    {
                        "axis": current_axis,
                        "index": current_index,
                        "percentile": percentile,
                        "mode": "multi",
                    }
                )
        return specs

    if axis not in {"lat", "lon"}:
        raise ValueError(f"section_axis 必须是 lat 或 lon，实际: {axis}")
    size = height if axis == "lat" else width
    return [
        {
            "axis": axis,
            "index": resolve_section_index(index, size),
            "percentile": None,
            "mode": "single",
        }
    ]


def plot_joint_du_unet_section_panel(
    prediction_3d,
    truth_3d,
    target_var,
    section_axis,
    section_index,
    section_width,
    day,
    output_path,
    depth_values,
    lon_range=(110, 118),
    lat_range=(10, 18),
    title_prefix="Joint_Du_Unet",
):
    """绘制 Joint_Du_Unet 的预测、真值和绝对误差垂向剖面。"""
    prediction_3d = np.asarray(prediction_3d, dtype=np.float32)
    truth_3d = np.asarray(truth_3d, dtype=np.float32)
    if prediction_3d.shape != truth_3d.shape or prediction_3d.ndim != 3:
        raise ValueError(
            "prediction/truth 需要同形状 (D,H,W)，实际: "
            f"{prediction_3d.shape}, {truth_3d.shape}"
        )

    depth = np.asarray(depth_values, dtype=np.float32)
    d, h, w = prediction_3d.shape
    if depth.shape != (d,) or not np.all(np.diff(depth) > 0):
        raise ValueError(
            f"depth_values 需要长度D且严格递增，实际: {depth.tolist()}"
        )

    prediction_section, start, end = _extract_section(
        prediction_3d, section_axis, section_index, section_width
    )
    truth_section, _, _ = _extract_section(
        truth_3d, section_axis, section_index, section_width
    )
    error_section = np.abs(prediction_section - truth_section)
    x_coords, x_label = _axis_coords(
        section_axis, lat_range, lon_range, h, w
    )
    section_center = (start + end - 1) / 2
    if section_axis == "lat":
        section_coord = np.interp(
            section_center,
            np.arange(h),
            np.linspace(lat_range[0], lat_range[1], h),
        )
        section_label = f"lat idx {start}:{end} (~{section_coord:.2f}°N)"
    else:
        section_coord = np.interp(
            section_center,
            np.arange(w),
            np.linspace(lon_range[0], lon_range[1], w),
        )
        section_label = f"lon idx {start}:{end} (~{section_coord:.2f}°E)"

    cmap, value_label, error_label = field_cmap_and_label(target_var)
    value_vmin, value_vmax = finite_vmin_vmax(
        prediction_section, truth_section
    )
    error_vmin, error_vmax = finite_vmin_vmax(
        error_section, default=(0.0, 1.0)
    )
    error_vmin = max(error_vmin, 0.0)
    panels = [
        ("Prediction", prediction_section, cmap, value_label, value_vmin, value_vmax),
        ("Truth", truth_section, cmap, value_label, value_vmin, value_vmax),
        ("Absolute error", error_section, "magma", error_label, error_vmin, error_vmax),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
    for ax, (title, data, panel_cmap, label, vmin, vmax) in zip(axes, panels):
        image = ax.pcolormesh(
            x_coords,
            depth,
            data,
            cmap=panel_cmap,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Depth / m")
        ax.invert_yaxis()
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label(label)
    fig.suptitle(f"{title_prefix} {target_var} section {day} | {section_label}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Joint_Du_Unet 垂向剖面图已保存至：{output_path}")


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
