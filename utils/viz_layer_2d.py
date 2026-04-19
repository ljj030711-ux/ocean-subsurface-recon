"""指定层二维平面图可视化"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


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
