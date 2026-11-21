"""本地垂向对比图：真值 / 模型A / 模型B / 两个绝对误差，共享掩膜与色标。

在 B4 仓库根目录运行；本地仅需 numpy 和 matplotlib，无需 Torch 或 checkpoint。
服务器导出两个模型的相同日期、全部希望画剖面的目标层：
    python test.py --method B3_DepthFormer --target-var temperature \
      --dataset-profile glorys12v1_all_size --data-dir ./data/raw \
      --checkpoint /server/B3/best.pth --output-dir ./B3 \
      --eval-start 2023-01-01 --eval-end 2023-12-31 --eval-batch-size 1 \
      --depth-indices 0,4,9,17,24,27 --target-levels 0,1,2,3,4,5 --panel-days 2023-05-26
B2 将 --method 改为 Joint_DUViT，并设置对应 --checkpoint 与 --output-dir ./B2。
下载 B2、B3 后，固定中间纬度画 depth-longitude：
    python -m utils.viz_layer_profile --run-a /local/results/B2 --run-b /local/results/B3 \
      --dates 2023-05-26 --depth-indices 0,4,9,17,24,27 \
      --section-axis lat --section-index middle --section-width 1
两个方向的多个位置（每个日期、每个剖面单独输出一张对比图）：
    python -m utils.viz_layer_profile --dates 2023-05-26 --depth-indices 0,4,9,17,24,27 \
      --multi-section-panels --section-percentiles 10,25,50,75,90 --section-width 1
默认输入当前工作目录的 ./B2、./B3，输出 ./comparison_figures；可直接传案例NPZ。
--section-axis lon 固定经度画 depth-latitude；宽度 >1 时先应用共同掩膜再带平均。
本地 --depth-indices 是数据集原始层号，与服务端 --target-levels 的局部位置不同。
只使用 NPZ 中已保存的实际深度（至少两层），不补出其他层；缺少日期或层必须重新推理。
旧 NPZ 如无法确定版本，可加 --metadata-a /local/B2/test.log --metadata-b /local/B3/test.log。
水平图的完整 B2/B3 导出与绘图示例见 viz_layer_2d.py 文件头。
"""

from pathlib import Path
import sys
import warnings

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from utils.viz_layer_2d import (
    create_comparison_figure,
    draw_comparison_row,
    field_cmap_and_label,
    finite_symmetric_limit,
    finite_vmin_vmax,
)

from utils.case_fields import comparison_parser, comparison_from_args

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


def plot_section_comparison(comparison, day, axis, index, width, output_path):
    """Apply the common mask before extracting/averaging physical depth sections."""
    if len(comparison.depth_m) < 2:
        raise ValueError("垂向图至少需要两个已保存的深度；缺少目标层时需要重新推理")
    if width <= 0:
        raise ValueError("section-width 必须大于0")
    size = len(comparison.latitude) if axis == "lat" else len(comparison.longitude)
    index = resolve_section_index(index, size)
    t, order = comparison.dates.index(day), np.argsort(comparison.depth_m)
    sections = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for fields in (comparison.truth, comparison.prediction_a, comparison.prediction_b):
            section, start, end = _extract_section(fields[t, order], axis, index, width)
            sections.append(section)
    if axis == "lat":
        x, x_label = comparison.longitude, "Longitude / °E"
        coordinate = float(np.mean(comparison.latitude[start:end]))
        section_label = f"Latitude {coordinate:.3f}°N"
    else:
        x, x_label = comparison.latitude, "Latitude / °N"
        coordinate = float(np.mean(comparison.longitude[start:end]))
        section_label = f"Longitude {coordinate:.3f}°E"
    fig, rows = create_comparison_figure(
        1, f"Section study: {day}, {section_label}, width {end - start}, shared ocean mask and color scale"
    )
    try:
        axes, color_axes = rows[0]
        draw_comparison_row(
            fig, axes, color_axes, *sections, x, comparison.depth_m[order],
            comparison.target_var, comparison.truth_label, comparison.label_a, comparison.label_b,
            x_label, "Depth / m", invert_depth=True,
        )
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"垂向对比图: {output_path}")
    finally:
        plt.close(fig)
    return fig


def main(argv=None):
    parser = comparison_parser("读取两组案例 NPZ，绘制共享色标的垂向剖面对比图")
    parser.add_argument("--section-axis", choices=["lat", "lon"], default="lat")
    parser.add_argument("--section-index", default="middle")
    parser.add_argument("--section-width", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--section-panels", action="store_true", help="单剖面，默认模式")
    mode.add_argument("--multi-section-panels", action="store_true")
    parser.add_argument("--section-percentiles", default="10,25,50,75,90")
    args = parser.parse_args(argv)
    try:
        if args.section_width <= 0:
            raise ValueError("section-width 必须大于0")
        comparison = comparison_from_args(args)
        if len(comparison.depth_m) < 2:
            raise ValueError("垂向图至少需要两个已保存的深度，需要重新推理保存所需目标层")
        specs = build_section_specs(
            len(comparison.latitude), len(comparison.longitude), multi=args.multi_section_panels,
            axis=args.section_axis, index=args.section_index,
            percentiles=[float(value) for value in args.section_percentiles.split(",")],
        )
        depths_tag = "-".join(map(str, comparison.depth_indices))
        for day in comparison.dates:
            for spec in specs:
                filename = (f"comparison_section_{comparison.target_var}_{day.replace('-', '')}_"
                            f"{spec['axis']}_idx{spec['index']}_width{args.section_width}_gidx{depths_tag}.png")
                plot_section_comparison(
                    comparison, day, spec["axis"], spec["index"], args.section_width,
                    Path(args.output_dir) / filename,
                )
    except (ValueError, OSError) as error:
        parser.error(str(error))


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


if __name__ == "__main__":
    main()
