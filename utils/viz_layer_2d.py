"""本地水平对比图：真值 / 模型A / 模型B / 两个绝对误差，共享掩膜与色标。

在 B4 仓库根目录运行；本地仅需 numpy 和 matplotlib，无需 Torch 或 checkpoint。
服务器导出示例（两个模型必须指定相同日期；此处只保存原始层 17、27）：
    python test.py --method Joint_DUViT --target-var temperature \
      --dataset-profile glorys12v1_all_size --data-dir ./data/raw \
      --checkpoint /server/B2/best.pth --output-dir ./B2 \
      --eval-start 2023-01-01 --eval-end 2023-12-31 --eval-batch-size 1 \
      --depth-indices 0,4,9,17,24,27 --target-levels 3,5 --panel-days 2023-05-26
    python test.py --method B3_DepthFormer --target-var temperature \
      --dataset-profile glorys12v1_all_size --data-dir ./data/raw \
      --checkpoint /server/B3/best.pth --output-dir ./B3 \
      --eval-start 2023-01-01 --eval-end 2023-12-31 --eval-batch-size 1 \
      --depth-indices 0,4,9,17,24,27 --target-levels 3,5 --panel-days 2023-05-26
下载 B2、B3 文件夹后绘图（输出到 ./comparison_figures）：
    python -m utils.viz_layer_2d --run-a /local/results/B2 --run-b /local/results/B3 \
      --dates 2023-05-26 --depth-indices 17,27
    python -m utils.viz_layer_2d --dates 2023-05-26 --depth-indices 17,27
第二条默认读取当前工作目录的 ./B2、./B3。也可直接传入两个 case_fields_*.npz。
服务端 --target-levels 是评估序列的局部位置；本地 --depth-indices 是原始层号。
没有保存请求日期或层会报错，必须重新推理；不会自动用别的日期或层代替。
旧版本1 NPZ需同时下载匹配的 *_test_config*.json 或日志，必要时指定：
    --metadata-a /local/B2/test.log --metadata-b /local/B3/test_config.json
不指定 --panel-days 时仍按各模型的 P10/P50/P90 保存，两个模型可能日期不同。
垂向单剖面与多剖面示例见 viz_layer_profile.py 文件头；两者使用同一案例NPZ。
"""

from pathlib import Path
import sys
import textwrap

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

from utils.case_fields import comparison_parser, comparison_from_args

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


def create_comparison_figure(num_rows, title):
    """Five panels and two dedicated colorbar columns per row."""
    fig = plt.figure(figsize=(22, 4.4 * num_rows), dpi=150, layout="constrained")
    grid = fig.add_gridspec(num_rows, 7, width_ratios=[1, 1, 1, 0.045, 1, 1, 0.045])
    rows = []
    for row in range(num_rows):
        axes = [fig.add_subplot(grid[row, column]) for column in (0, 1, 2, 4, 5)]
        color_axes = [fig.add_subplot(grid[row, column]) for column in (3, 6)]
        rows.append((axes, color_axes))
    fig.suptitle(title, fontsize=17)
    return fig, rows


def draw_comparison_row(fig, axes, color_axes, truth, prediction_a, prediction_b,
                        x, y, target_var, truth_label, label_a, label_b,
                        x_label, y_label, invert_depth=False):
    """Input fields already have the shared mask, including before band averaging."""
    if not np.isfinite(truth).any():
        raise ValueError("该剖面没有共同有效海洋区域，请选择其他剖面位置")
    _, value_label, error_label = field_cmap_and_label(target_var)
    value_limits = finite_vmin_vmax(truth, prediction_a, prediction_b)
    error_a, error_b = np.abs(prediction_a - truth), np.abs(prediction_b - truth)
    error_limit = max(float(np.nanmax(error_a)), float(np.nanmax(error_b))) or 1.0
    value_norm, error_norm = Normalize(*value_limits), Normalize(0, error_limit)
    titles = [truth_label, label_a, label_b,
              f"|{label_a.replace(chr(10), ' ')} - truth|",
              f"|{label_b.replace(chr(10), ' ')} - truth|"]
    fields = [truth, prediction_a, prediction_b, error_a, error_b]
    images = []
    for position, (ax, field, title) in enumerate(zip(axes, fields, titles)):
        cmap = plt.get_cmap("viridis" if position < 3 else "magma").copy()
        cmap.set_bad("white")
        images.append(ax.pcolormesh(x, y, field, shading="auto", cmap=cmap,
                                   norm=value_norm if position < 3 else error_norm,
                                   rasterized=True))
        ax.set_title("\n".join(textwrap.fill(line, width=34) for line in title.splitlines()), fontsize=10)
        ax.set_xlabel(x_label)
        if position == 0:
            ax.set_ylabel(y_label)
        ax.grid(alpha=0.2)
        if len(x) > 1:
            ax.set_xlim(float(np.min(x)), float(np.max(x)))
        if invert_depth:
            ax.set_ylim(float(np.max(y)), float(np.min(y)))
        elif len(y) > 1:
            ax.set_ylim(float(np.min(y)), float(np.max(y)))
    fig.colorbar(images[0], cax=color_axes[0]).set_label(value_label)
    fig.colorbar(images[3], cax=color_axes[1]).set_label(error_label)


def plot_horizontal_comparison(comparison, day, output_path):
    """One date per figure; each requested physical depth gets its own color scales."""
    t = comparison.dates.index(day)
    fig, rows = create_comparison_figure(
        len(comparison.depth_indices), f"Case study: {day}, shared ocean mask and color scale"
    )
    try:
        for d, (axes, color_axes) in enumerate(rows):
            draw_comparison_row(
                fig, axes, color_axes, comparison.truth[t, d], comparison.prediction_a[t, d],
                comparison.prediction_b[t, d], comparison.longitude, comparison.latitude,
                comparison.target_var, comparison.truth_label, comparison.label_a, comparison.label_b,
                "Longitude / °E", f"{comparison.depth_m[d]:.2f} m\nLatitude / °N",
            )
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"水平对比图: {output_path}")
    finally:
        plt.close(fig)
    return fig


def main(argv=None):
    parser = comparison_parser("读取两组案例 NPZ，绘制共享色标的水平对比图")
    args = parser.parse_args(argv)
    try:
        comparison = comparison_from_args(args)
        depths_tag = "-".join(map(str, comparison.depth_indices))
        for day in comparison.dates:
            filename = f"comparison_horizontal_{comparison.target_var}_{day.replace('-', '')}_gidx{depths_tag}.png"
            plot_horizontal_comparison(comparison, day, Path(args.output_dir) / filename)
    except (ValueError, OSError) as error:
        parser.error(str(error))


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


if __name__ == "__main__":
    main()
