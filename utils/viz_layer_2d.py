"""指定层二维平面图和对比面板可视化。"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def field_cmap_and_label(target_var):
    if target_var == "temperature":
        return "RdYlBu_r", "Temperature (°C)", "Absolute error (°C)"
    if target_var == "salinity":
        return "viridis", "Salinity (psu)", "Absolute error (psu)"
    return "viridis", "Value", "Absolute error"


def metric_cbar_label(metric_name, target_var):
    unit = {
        "temperature": "°C",
        "salinity": "psu",
    }.get(target_var, "physical unit")
    metric = metric_name.upper()
    if metric_name.lower() == "mse":
        return f"{metric} ({unit}²)"
    return f"{metric} ({unit})"


def finite_vmin_vmax(*arrays, default=(0.0, 1.0)):
    vals = []
    for array in arrays:
        finite = np.asarray(array)[np.isfinite(array)]
        if finite.size:
            vals.append(finite)
    if not vals:
        return default
    merged = np.concatenate(vals)
    vmin = float(np.nanmin(merged))
    vmax = float(np.nanmax(merged))
    if vmin == vmax:
        pad = abs(vmin) * 0.05 if vmin else 1.0
        return vmin - pad, vmax + pad
    return vmin, vmax


def finite_symmetric_limit(*arrays):
    vals = []
    for array in arrays:
        finite = np.asarray(array)[np.isfinite(array)]
        if finite.size:
            vals.append(np.abs(finite))
    if not vals:
        return 1.0
    limit = float(np.nanmax(np.concatenate(vals)))
    return limit if limit > 0 else 1.0


def depth_label(level_idx, depth_values=None):
    if depth_values is not None and 0 <= level_idx < len(depth_values):
        return f"Level-{level_idx} {float(depth_values[level_idx]):g}m"
    return f"Level-{level_idx}"


def plot_level_map(data_2d, title="", output_path=None,
                   lon_range=(110, 118), lat_range=(10, 18),
                   cmap='viridis', cbar_label='psu'):
    """
    绘制指定层的二维平面图（imshow）

    Args:
        data_2d: (H, W) 二维数组
        title: 图标题
        output_path: 保存路径，None 则 show
        lon_range: 经度范围 (min, max)
        lat_range: 纬度范围 (min, max)
        cmap: colormap 名称
        cbar_label: 色标标签
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    extent = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]
    im = ax.imshow(data_2d, cmap=cmap, aspect='auto', origin='lower',
                   extent=extent)

    ax.set_xlabel('Longitude / °E')
    ax.set_ylabel('Latitude / °N')
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"二维平面图已保存至：{output_path}")
    else:
        plt.show()
    plt.close()


def plot_prediction_truth_error_panel(
    pred_2d,
    true_2d,
    error_2d,
    target_var,
    level,
    day,
    method,
    output_path,
    lon_range=(110, 118),
    lat_range=(10, 18),
    depth_values=None,
):
    """绘制指定层的预测、真值和绝对误差三联图。"""
    cmap, value_label, error_label = field_cmap_and_label(target_var)
    value_vmin, value_vmax = finite_vmin_vmax(pred_2d, true_2d)
    err_vmin, err_vmax = finite_vmin_vmax(error_2d)
    extent = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]
    level_text = depth_label(level, depth_values=depth_values)

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

    fig.suptitle(f"{method} {target_var} {day} {level_text}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"预测-真值-误差三联图已保存至：{output_path}")


def plot_hvca_refiner_panel(
    baseline_2d,
    refined_2d,
    true_2d,
    target_var,
    level,
    day,
    output_path,
    lon_range=(110, 118),
    lat_range=(10, 18),
    depth_values=None,
):
    """绘制 HVCARefiner 的 T0/refined/truth/error/delta 二维解释面板。"""
    baseline_error = np.abs(baseline_2d - true_2d)
    refined_error = np.abs(refined_2d - true_2d)
    delta = refined_2d - baseline_2d

    cmap, value_label, error_label = field_cmap_and_label(target_var)
    value_vmin, value_vmax = finite_vmin_vmax(baseline_2d, refined_2d, true_2d)
    err_vmin, err_vmax = finite_vmin_vmax(
        baseline_error, refined_error, default=(0.0, 1.0)
    )
    err_vmin = max(err_vmin, 0.0)
    delta_limit = finite_symmetric_limit(delta)
    delta_norm = TwoSlopeNorm(vmin=-delta_limit, vcenter=0.0, vmax=delta_limit)
    extent = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]
    level_text = depth_label(level, depth_values=depth_values)
    delta_label = value_label

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=150)
    panels = [
        ("Baseline T0", baseline_2d, cmap, value_label, value_vmin, value_vmax, None),
        ("HVCA Refined", refined_2d, cmap, value_label, value_vmin, value_vmax, None),
        ("Truth", true_2d, cmap, value_label, value_vmin, value_vmax, None),
        ("|T0 - Truth|", baseline_error, "magma", error_label, err_vmin, err_vmax, None),
        ("|HVCA - Truth|", refined_error, "magma", error_label, err_vmin, err_vmax, None),
        ("Delta = HVCA - T0", delta, "RdBu_r", delta_label, None, None, delta_norm),
    ]
    for ax, (title, data, panel_cmap, cbar_label, vmin, vmax, norm) in zip(
        axes.flat, panels
    ):
        im = ax.imshow(
            data,
            cmap=panel_cmap,
            aspect="auto",
            origin="lower",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude / °E")
        ax.set_ylabel("Latitude / °N")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)

    fig.suptitle(f"HVCARefiner {target_var} {day} {level_text}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"HVCA 二维解释图已保存至：{output_path}")
