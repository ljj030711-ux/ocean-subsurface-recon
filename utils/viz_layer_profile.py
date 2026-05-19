"""三维经纬-深度剖面图可视化（迁移自 utils/draw.py plot_3d_metric_no_overlap）"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _nan_gaussian_filter(data, sigma):
    """Smooth finite values without letting NaNs contaminate the full field."""
    finite = np.isfinite(data)
    if not np.any(finite):
        return np.full_like(data, np.nan, dtype=np.float32)
    filled = np.where(finite, data, 0.0)
    weights = finite.astype(np.float32)
    smoothed = gaussian_filter(filled, sigma=sigma)
    smoothed_weights = gaussian_filter(weights, sigma=sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = smoothed / smoothed_weights
    out[smoothed_weights <= 1e-6] = np.nan
    return out


def plot_3d_metric_profile(npz_path, metric_name="rmse", output_img_path=None,
                           lon_range=(110, 118), lat_range=(10, 18),
                           z_max=300, smooth_sigma=1.2, cbar_label=None):
    """
    三维经纬-深度剖面图（3 条经度剖面）。

    Args:
        npz_path: grid_metrics npz 文件路径（键名需含 metric_name）
        metric_name: 指标名（rmse / mae / mse）
        output_img_path: 保存路径，None 则 show
        lon_range: 经度范围 (min, max)
        lat_range: 纬度范围 (min, max)
        z_max: 最大深度（m）
        smooth_sigma: 高斯平滑 sigma
    """
    npz_data = np.load(npz_path)
    available_keys = sorted(npz_data.files)

    candidates = [
        metric_name, metric_name.lower(), metric_name.upper(),
        metric_name.replace("2", "^2"), metric_name.replace("^2", "2"),
    ]
    metric_key = next((c for c in candidates if c in available_keys), None)
    if metric_key is None:
        raise ValueError(f"未找到指标 '{metric_name}'，可用: {available_keys}")

    metric_data = npz_data[metric_key]
    if metric_data.ndim == 4:
        metric_data = np.squeeze(metric_data, axis=0)
    D, H, W = metric_data.shape

    metric_smoothed = _nan_gaussian_filter(
        metric_data, sigma=(smooth_sigma, smooth_sigma, smooth_sigma)
    )

    lon = np.linspace(lon_range[0], lon_range[1], W)
    lat = np.linspace(lat_range[0], lat_range[1], H)
    z = np.linspace(0, z_max, D)

    x_idx_list = [0, W // 2, W - 1]
    lon_pos_list = lon[x_idx_list]

    finite_metric = metric_smoothed[np.isfinite(metric_smoothed)]
    if finite_metric.size:
        vmin = float(np.nanmin(finite_metric))
        vmax = float(np.nanmax(finite_metric))
        if vmin == vmax:
            pad = abs(vmin) * 0.05 if vmin else 1.0
            vmin -= pad
            vmax += pad
    else:
        vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap("jet").copy()
    cmap_obj.set_bad((0, 0, 0, 0))

    fig = plt.figure(figsize=(16, 9), dpi=150)
    ax = fig.add_subplot(projection='3d')

    edge_colors = ['white', 'lightgray', 'dimgray']
    for idx, x_idx in enumerate(x_idx_list):
        profile = metric_smoothed[:, :, x_idx]
        Y, Z = np.meshgrid(lat, z)
        X = np.full_like(Y, lon_pos_list[idx])
        face_colors = cmap_obj(norm(np.ma.masked_invalid(profile)))
        ax.plot_surface(
            X, Y, Z, rstride=2, cstride=2,
            facecolors=face_colors, shade=False,
            alpha=0.98, linewidth=0.3,
            edgecolor=edge_colors[idx], antialiased=True,
        )

    ax.set_box_aspect([4, 4, 1])
    ax.set_xlabel('Longitude / °E', fontsize=9, labelpad=10)
    ax.set_ylabel('Latitude / °N', fontsize=9, labelpad=10)
    ax.set_zlabel('Depth / m', fontsize=9, labelpad=8)

    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_zlim(z_max, 0)

    ax.set_xticks(np.linspace(lon_range[0], lon_range[1], 5))
    ax.set_yticks(np.linspace(lat_range[0], lat_range[1], 5))
    ax.set_zticks(np.linspace(0, z_max, 7))
    ax.view_init(elev=15, azim=-60)

    if cbar_label is None:
        metric_label_map = {
            "rmse": "RMSE (psu)", "mae": "MAE (psu)", "mse": "MSE (psu²)",
        }
        cbar_label = metric_label_map.get(
            metric_key.lower(), metric_key.upper().replace("^2", "²")
        )
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    mappable.set_array(np.ma.masked_invalid(metric_smoothed))
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.grid(False)
    ax.set_facecolor('white')

    plt.tight_layout()
    if output_img_path is not None:
        plt.savefig(output_img_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"三维剖面图已保存至：{output_img_path}")
    else:
        plt.show()
    plt.close()
