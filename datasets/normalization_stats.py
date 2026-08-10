"""2dto2d 归一化统计量的持久化与一致性校验。"""

import hashlib
import json
import os
import tempfile

import numpy as np


FORMAT_VERSION = 1
NORMALIZATION_METHOD = "monthly_climatology_layer_std"
VARIABLE_NAMES = ("sst", "ssh", "sss", "target")


def default_normalization_stats_path(
    data_dir,
    dataset_profile,
    target_var,
    fit_start_date,
    fit_end_date,
):
    """返回与数据、变量和训练统计时段绑定的默认统计文件路径。"""
    filename = (
        f"{dataset_profile}_{target_var}_"
        f"{fit_start_date}_{fit_end_date}.npz"
    )
    return os.path.join(os.path.abspath(data_dir), "normalization_stats", filename)


def build_normalization_metadata(
    *,
    dataset_profile,
    target_var,
    fit_start_date,
    fit_end_date,
    fit_sample_count,
    arrays,
    sst_unit_conversion,
):
    """构造用于拒绝误用统计文件的最小数据身份信息。"""
    return {
        "format_version": FORMAT_VERSION,
        "normalization": NORMALIZATION_METHOD,
        "dataset_profile": str(dataset_profile),
        "target_var": str(target_var),
        "fit_start_date": str(fit_start_date),
        "fit_end_date": str(fit_end_date),
        "fit_sample_count": int(fit_sample_count),
        "spatial_shapes": {
            name: list(map(int, arrays[name].shape[1:]))
            for name in VARIABLE_NAMES
        },
        "dtypes": {
            name: str(arrays[name].dtype)
            for name in VARIABLE_NAMES
        },
        "sst_unit_conversion": str(sst_unit_conversion),
    }


def _normalizer_payload(stats):
    payload = {}
    normalizer_config = {}
    for name in VARIABLE_NAMES:
        variable_stats = stats[name]
        payload[f"{name}_climatology"] = np.asarray(
            variable_stats["climatology"], dtype=np.float32
        )
        payload[f"{name}_layer_std"] = np.asarray(
            variable_stats["layer_std"], dtype=np.float32
        )
        normalizer_config[name] = {
            "eps": float(variable_stats.get("eps", 1e-6)),
            "fill_value": float(variable_stats.get("fill_value", 0.0)),
        }
    return payload, normalizer_config


def _fingerprint(metadata, payload):
    """对元数据和数值数组计算稳定指纹，不额外复制大数组。"""
    metadata_without_hash = dict(metadata)
    metadata_without_hash.pop("fingerprint", None)
    digest = hashlib.sha256(
        json.dumps(
            metadata_without_hash,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for key in sorted(payload):
        array = np.ascontiguousarray(payload[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def save_normalization_stats(path, stats, metadata):
    """原子保存统计量；返回包含 fingerprint 的元数据。"""
    path = os.path.abspath(path)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    payload, normalizer_config = _normalizer_payload(stats)
    metadata = dict(metadata)
    metadata["normalizer_config"] = normalizer_config
    metadata["fingerprint"] = _fingerprint(metadata, payload)
    payload["metadata_json"] = np.asarray(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    )

    fd, temp_path = tempfile.mkstemp(
        prefix=".normalization_", suffix=".tmp", dir=directory
    )
    os.close(fd)
    try:
        with open(temp_path, "wb") as file_obj:
            np.savez(file_obj, **payload)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return metadata


def load_normalization_stats(path):
    """加载统计量并校验文件内部指纹。"""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到归一化统计文件: {path}")

    with np.load(path, allow_pickle=False) as bundle:
        metadata = json.loads(str(bundle["metadata_json"].item()))
        config = metadata.get("normalizer_config", {})
        stats = {
            "normalization": metadata.get("normalization"),
            "sst_unit": "degree_C",
            "sst_unit_conversion": metadata.get("sst_unit_conversion"),
        }
        payload = {}
        for name in VARIABLE_NAMES:
            climatology = np.asarray(
                bundle[f"{name}_climatology"], dtype=np.float32
            )
            layer_std = np.asarray(
                bundle[f"{name}_layer_std"], dtype=np.float32
            )
            payload[f"{name}_climatology"] = climatology
            payload[f"{name}_layer_std"] = layer_std
            variable_config = config.get(name, {})
            stats[name] = {
                "climatology": climatology,
                "layer_std": layer_std,
                "eps": float(variable_config.get("eps", 1e-6)),
                "fill_value": float(variable_config.get("fill_value", 0.0)),
            }

    expected_fingerprint = metadata.get("fingerprint")
    actual_fingerprint = _fingerprint(metadata, payload)
    if not expected_fingerprint or actual_fingerprint != expected_fingerprint:
        raise ValueError(f"归一化统计文件指纹校验失败，文件可能不完整或已损坏: {path}")
    return stats, metadata


def validate_normalization_stats(stats, metadata, expected_metadata, path):
    """校验统计文件是否属于当前数据集和训练统计时段。"""
    errors = []
    scalar_keys = (
        "format_version",
        "normalization",
        "dataset_profile",
        "target_var",
        "fit_start_date",
        "fit_end_date",
        "fit_sample_count",
        "sst_unit_conversion",
    )
    for key in scalar_keys:
        if metadata.get(key) != expected_metadata.get(key):
            errors.append(
                f"{key}: 文件={metadata.get(key)!r}, 当前={expected_metadata.get(key)!r}"
            )

    for key in ("spatial_shapes", "dtypes"):
        if metadata.get(key) != expected_metadata.get(key):
            errors.append(
                f"{key}: 文件={metadata.get(key)!r}, 当前={expected_metadata.get(key)!r}"
            )

    expected_shapes = expected_metadata["spatial_shapes"]
    for name in VARIABLE_NAMES:
        climatology_shape = tuple(stats[name]["climatology"].shape)
        wanted_climatology = (12,) + tuple(expected_shapes[name])
        if climatology_shape != wanted_climatology:
            errors.append(
                f"{name}.climatology shape: 文件={climatology_shape}, "
                f"当前应为={wanted_climatology}"
            )
        expected_std_shape = (1,) if name != "target" else (expected_shapes[name][0],)
        std_shape = tuple(stats[name]["layer_std"].shape)
        if std_shape != expected_std_shape:
            errors.append(
                f"{name}.layer_std shape: 文件={std_shape}, 当前应为={expected_std_shape}"
            )

    if errors:
        details = "\n  - ".join(errors)
        raise ValueError(
            f"归一化统计文件与当前数据不匹配: {os.path.abspath(path)}\n"
            f"  - {details}\n"
            "确认数据配置后使用 --rebuild-normalization-stats 显式重算。"
        )


def normalization_reference(path, metadata):
    """返回可写入 checkpoint/cache metadata 的轻量引用。"""
    return {
        "path": os.path.abspath(path),
        "fingerprint": metadata.get("fingerprint"),
        "format_version": metadata.get("format_version"),
        "dataset_profile": metadata.get("dataset_profile"),
        "target_var": metadata.get("target_var"),
        "fit_start_date": metadata.get("fit_start_date"),
        "fit_end_date": metadata.get("fit_end_date"),
    }
