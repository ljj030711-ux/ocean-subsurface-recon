"""输入数据质量检查工具。"""

import numpy as np


def report_missing_values(name, array):
    """
    检查数组中的 NaN/Inf，并打印统计信息。

    Args:
        name: 数据名（用于日志）
        array: 待检查数组

    Returns:
        dict: 缺失值统计
    """
    arr = np.asarray(array)
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    total = int(arr.size)
    invalid = nan_count + inf_count

    if invalid > 0:
        ratio = 100.0 * invalid / max(total, 1)
        print(
            f"[数据质检] {name}: 检测到缺失/异常值 "
            f"(NaN={nan_count}, Inf={inf_count}, 占比={ratio:.6f}%)"
        )
    else:
        print(f"[数据质检] {name}: 未检测到缺失/异常值")

    return {
        "name": name,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "invalid_count": invalid,
        "total_count": total,
    }


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
    print(report_missing_values("sla_sss", np.load("data/raw/sla_sss_2019-01-01_2023-12-31_10_18_110_118.npy")))
    print(report_missing_values("sws_true", np.load("data/raw/sws_2019-01-01_2023-12-31_10_18_110_118_0-300.npy")))

if __name__ == "__main__":
    main()