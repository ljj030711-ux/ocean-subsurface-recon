"""Portable case-field archives and strict matching, without Torch or plotting."""

import argparse
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import tempfile
import zipfile
import zlib

import numpy as np


REQUIRED_FIELDS = (
    "format_version", "method", "target_var", "dates", "depth_indices",
    "depth_m", "longitude", "latitude", "prediction", "truth", "valid_mask",
)
MODEL_NAMES = {
    "Du_Unet": "B0 Du_Unet", "Joint_Du_Unet": "B1 Joint_Du_Unet",
    "Joint_DUViT": "B2 Joint_DUViT", "B3_DepthFormer": "B3",
    "B4_CPlus": "B4 C+",
}


def csv_dates(value):
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or len(set(values)) != len(values):
        raise ValueError("日期不能为空或重复")
    for item in values:
        if date.fromisoformat(item).isoformat() != item:
            raise ValueError(f"日期需要 YYYY-MM-DD 格式: {item}")
    return values


def csv_depth_indices(value):
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or min(values) < 0 or len(set(values)) != len(values):
        raise ValueError("原始深度索引不能为空、为负数或重复")
    return values


def scalar_text(data, name, default=None):
    if name not in data:
        if default is not None:
            return default
        raise ValueError(f"缺少字段 {name}")
    value = np.asarray(data[name])
    if value.shape != () or value.dtype.kind not in "US":
        raise ValueError(f"{name} 必须是字符串标量")
    return str(value.item())


def validate_archive(data, path):
    missing = [key for key in REQUIRED_FIELDS if key not in data]
    if missing:
        raise ValueError(f"{path}: 缺少案例字段 {missing}；请重新导出案例 NPZ")
    version = np.asarray(data["format_version"])
    if version.shape != () or version.dtype.kind not in "iu" or int(version) not in (1, 2):
        raise ValueError(f"{path}: 不支持的案例 format_version: {version}")
    for key in ("method", "target_var"):
        if not scalar_text(data, key):
            raise ValueError(f"{path}: {key} 不能为空")
    if scalar_text(data, "target_var") not in ("temperature", "salinity"):
        raise ValueError(f"{path}: 目标变量必须是 temperature 或 salinity")
    dates = np.asarray(data["dates"])
    if dates.ndim != 1 or dates.dtype.kind not in "US":
        raise ValueError(f"{path}: dates 需要一维日期字符串")
    csv_dates(",".join(dates.tolist()))
    indices, depths = np.asarray(data["depth_indices"]), np.asarray(data["depth_m"])
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError(f"{path}: depth_indices 需要一维整数")
    csv_depth_indices(",".join(str(int(index)) for index in indices))
    if (depths.shape != indices.shape or depths.dtype.kind not in "fiu"
            or not np.isfinite(depths).all() or np.any(depths < 0)):
        raise ValueError(f"{path}: depth_m 与原始层号不对应或含非法深度")
    if len(np.unique(depths)) != len(depths):
        raise ValueError(f"{path}: 不同层号对应了重复的实际深度")
    prediction = np.asarray(data["prediction"])
    if prediction.ndim != 4 or prediction.shape[:2] != (len(dates), len(indices)):
        raise ValueError(f"{path}: prediction 需要 (日期, 深度, 纬度, 经度)，实际 {prediction.shape}")
    for key in ("prediction", "truth", "valid_mask"):
        array = np.asarray(data[key])
        if array.shape != prediction.shape or array.dtype.kind not in "fibu":
            raise ValueError(f"{path}: {key} 的形状或数值类型与预测不对应")
    mask = np.asarray(data["valid_mask"])
    if not np.isin(mask, [0, 1]).all():
        raise ValueError(f"{path}: valid_mask 只能包含 0/1 或布尔值")
    for key, size in (("latitude", prediction.shape[2]), ("longitude", prediction.shape[3])):
        coordinates = np.asarray(data[key])
        if (coordinates.shape != (size,) or coordinates.dtype.kind not in "fiu"
                or not size or not np.isfinite(coordinates).all()):
            raise ValueError(f"{path}: {key} 与网格尺寸不对应")
        delta = np.diff(coordinates)
        if not (np.all(delta > 0) or np.all(delta < 0)):
            raise ValueError(f"{path}: {key} 必须严格单调")
    if "target_levels" in data:
        levels = np.asarray(data["target_levels"])
        if levels.shape != indices.shape or levels.dtype.kind not in "iu" or np.any(levels < 0):
            raise ValueError(f"{path}: target_levels 与保存层不对应")
        if len(np.unique(levels)) != len(levels):
            raise ValueError(f"{path}: target_levels 重复")
    for key, size in (("roles", len(dates)), ("months", len(dates))):
        if key in data and np.asarray(data[key]).shape != (size,):
            raise ValueError(f"{path}: {key} 与日期不对应")
    if "months" in data and not np.array_equal(
        data["months"], [date.fromisoformat(day).month for day in dates]
    ):
        raise ValueError(f"{path}: months 与 dates 不对应")
    if int(version) == 2:
        metadata = json.loads(scalar_text(data, "metadata_json"))
        if not isinstance(metadata, dict) or metadata.get("method") != scalar_text(data, "method"):
            raise ValueError(f"{path}: 内嵌元数据的方法与 NPZ 不对应")


def write_case_archive(path, payload, metadata):
    """Preserve numeric arrays and add pickle-free JSON provenance atomically."""
    path = Path(path)
    data = dict(payload)
    data["format_version"] = np.asarray(2, dtype=np.int64)
    data["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False, allow_nan=False))
    validate_archive(data, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_case_archive(path):
    try:
        with Path(path).open("rb") as stream:
            with np.load(stream, allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
        validate_archive(data, path)
        return data
    except (zipfile.BadZipFile, EOFError, zlib.error) as error:
        raise ValueError(
            f"无法读取案例 {path}: NPZ 文件损坏或下载不完整，请从服务器重新下载该文件；"
            "若服务器原文件也无法读取，需要重新推理并导出"
        ) from error
    except (ValueError, TypeError, KeyError, OSError) as error:
        raise ValueError(f"无法读取案例 {path}: {error}") from error


def _model_config(metadata):
    config = metadata.get("model_config")
    if not isinstance(config, dict) or not config:
        raise ValueError("缺少实际模型配置，请提供对应 --metadata-a/--metadata-b 测试配置或日志")
    return config


def model_label(metadata, method):
    config = _model_config(metadata)
    label = MODEL_NAMES.get(method, method)
    details = []
    if "architecture_version" in config:
        details.append(f"v{config['architecture_version']}")
    if method in ("B3_DepthFormer", "B4_CPlus"):
        if config.get("depth_attention") not in ("causal", "independent"):
            raise ValueError("缺少可靠的深度注意力版本，请提供对应测试配置或日志")
        details.append(config["depth_attention"])
    if "output_patch_size" in config:
        details.append(f"ps{config['output_patch_size']}")
    if method == "Joint_DUViT":
        for name, caption in (("sst_vit", "SST"), ("surface_vit", "surface")):
            patch = config.get(name, {}).get("patch_size")
            if patch is not None:
                details.append(f"{caption} ps{patch}")
    return label + ("\n" + ", ".join(details) if details else "")


def _identity(metadata):
    config = dict(_model_config(metadata))
    for key in ("evaluation_output_positions", "checkpoint_depth_indices", "source_depth_indices"):
        config.pop(key, None)
    paths = metadata.get("paths") or {}
    checkpoint = paths.get("loaded_checkpoint") or paths.get("checkpoint")
    return json.dumps(config, sort_keys=True), checkpoint


def _metadata_records(path):
    text = Path(path).read_text(encoding="utf-8")
    decoder, offset, records = json.JSONDecoder(), 0, []
    # Run logs print complete JSON configs. Parse these blocks as data only.
    while True:
        start = text.find("{", offset)
        if start < 0:
            break
        try:
            record, length = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            offset = start + 1
            continue
        offset = start + length
        if isinstance(record, dict) and record.get("stage") == "test":
            records.append(record)
    return records


def _matches_metadata(record, data):
    if record.get("method", record.get("model")) != scalar_text(data, "method"):
        return False
    if record.get("target_var") != scalar_text(data, "target_var"):
        return False
    source = record.get("data") or {}
    if source.get("dataset_profile") != scalar_text(data, "dataset_profile", ""):
        return False
    dates = data["dates"].tolist()
    if source.get("select_day"):
        if dates != [source["select_day"]]:
            return False
    elif source.get("eval_start") and source.get("eval_end"):
        if any(day < source["eval_start"] or day > source["eval_end"] for day in dates):
            return False
    else:
        return False
    hyperparameters = record.get("hyperparameters") or {}
    indices, depths = hyperparameters.get("depth_indices", []), hyperparameters.get("depth_m", [])
    mapping = dict(zip(indices, depths))
    if "target_levels" in data and any(
        int(level) >= len(indices) or indices[int(level)] != int(index)
        for level, index in zip(data["target_levels"], data["depth_indices"])
    ):
        return False
    return all(
        int(index) in mapping and np.isclose(mapping[int(index)], depth, rtol=0, atol=1e-3)
        for index, depth in zip(data["depth_indices"], data["depth_m"])
    )


def resolve_metadata(data, archive_path, search_root, explicit=None):
    embedded = json.loads(scalar_text(data, "metadata_json")) if "metadata_json" in data else None
    if embedded is not None:
        model_label(embedded, scalar_text(data, "method"))
        if embedded.get("target_var") != scalar_text(data, "target_var"):
            raise ValueError(f"{archive_path}: 元数据目标变量冲突")
        if not _matches_metadata(embedded, data):
            raise ValueError(f"{archive_path}: 内嵌元数据日期、深度或数据集不对应")
        if explicit is None:
            return embedded
    if explicit is not None:
        paths = [Path(explicit).expanduser().resolve()]
        if not paths[0].is_file():
            raise ValueError(f"元数据路径不存在: {paths[0]}")
    else:
        paths = sorted(search_root.rglob("*_test_config*.json"))
        paths += sorted(search_root.rglob("*.log"))
    candidates = {}
    for path in paths:
        for record in _metadata_records(path):
            if _matches_metadata(record, data):
                model_label(record, scalar_text(data, "method"))
                candidates[_identity(record)] = record
    if len(candidates) != 1:
        raise ValueError(
            f"{archive_path}: 无法唯一确定模型版本，匹配到 {len(candidates)} 种配置；"
            "请通过 --metadata-a/--metadata-b 提供对应测试配置或日志"
        )
    metadata = next(iter(candidates.values()))
    if embedded is not None and _identity(metadata) != _identity(embedded):
        raise ValueError(f"{archive_path}: NPZ 与指定配置/日志的模型版本或权重来源冲突")
    return metadata if embedded is None else embedded


def _load_run(path, dates, indices, explicit_metadata=None):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"结果路径不存在: {path}")
    if path.is_file():
        if path.suffix.lower() != ".npz":
            raise ValueError(f"结果文件需要是 NPZ: {path}")
        files, search_root = [path], path.parent
        # Include adjacent configs when the NPZ is inside case_fields/horizontal.
        while search_root.name in ("case_fields", "horizontal", "visualizations") or search_root.name.startswith("idx"):
            search_root = search_root.parent
    else:
        files, search_root = sorted(path.rglob("case_fields_*.npz")), path
    if not files:
        raise ValueError(f"{path}: 没有预测场案例 NPZ（指标文件不能重画案例），需要重新推理并导出")
    cases, metadata_by_file = {}, {}
    for file in files:
        data = read_case_archive(file)
        for day_position, day in enumerate(data["dates"].tolist()):
            if day not in dates:
                continue
            for level_position, index in enumerate(data["depth_indices"].tolist()):
                if index not in indices:
                    continue
                key = day, index
                if key in cases:
                    raise ValueError(f"{path}: 日期 {day} 原始层 {index} 有重复案例来源: {cases[key][0]} 和 {file}")
                cases[key] = file, data, day_position, level_position
    for day in dates:
        for index in indices:
            if (day, index) not in cases:
                if not any(key[0] == day for key in cases):
                    # Distinguish a missing layer on an existing date from a missing date.
                    available = any(day in read_case_archive(file)["dates"] for file in files)
                    if not available:
                        raise ValueError(f"{path}: 没有储存 {day} 日期的推理结果，需要重新推理")
                raise ValueError(f"{path}: {day} 没有储存原始索引 {index} 层的推理结果，需要重新推理")
    for file, data, _, _ in cases.values():
        if file not in metadata_by_file:
            metadata_by_file[file] = resolve_metadata(data, file, search_root, explicit_metadata)
    identities = {_identity(metadata) for metadata in metadata_by_file.values()}
    if len(identities) != 1:
        raise ValueError(f"{path}: 所请求案例来自不同模型版本或权重，不能合并")
    first = cases[dates[0], indices[0]][1]
    method, target_var = scalar_text(first, "method"), scalar_text(first, "target_var")
    depths = np.asarray([cases[dates[0], index][1]["depth_m"][cases[dates[0], index][3]] for index in indices])
    arrays = {key: [] for key in ("prediction", "truth", "valid_mask")}
    for day in dates:
        rows = {key: [] for key in arrays}
        for position, index in enumerate(indices):
            file, data, t, d = cases[day, index]
            for key in ("method", "target_var", "dataset_profile"):
                if scalar_text(data, key, "") != scalar_text(first, key, ""):
                    raise ValueError(f"{file}: 案例 {key} 不一致")
            for key in ("longitude", "latitude"):
                if data[key].shape != first[key].shape or not np.allclose(data[key], first[key], rtol=0, atol=1e-5):
                    raise ValueError(f"{file}: 案例 {key} 网格不一致，不自动重网格")
            if not np.isclose(data["depth_m"][d], depths[position], rtol=0, atol=1e-3):
                raise ValueError(f"{file}: 原始层 {index} 对应的实际深度不一致")
            for key in arrays:
                rows[key].append(data[key][t, d])
        for key in arrays:
            arrays[key].append(np.stack(rows[key]))
    return {
        **{key: np.stack(value) for key, value in arrays.items()},
        "longitude": first["longitude"], "latitude": first["latitude"], "depth_m": depths,
        "target_var": target_var, "dataset_profile": scalar_text(first, "dataset_profile", ""),
        "label": model_label(next(iter(metadata_by_file.values())), method),
    }


@dataclass
class CaseComparison:
    dates: list
    depth_indices: list
    depth_m: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray
    truth: np.ndarray
    prediction_a: np.ndarray
    prediction_b: np.ndarray
    valid_mask: np.ndarray
    target_var: str
    label_a: str
    label_b: str
    truth_label: str


def load_comparison(run_a, run_b, dates, depth_indices, metadata_a=None, metadata_b=None):
    csv_dates(",".join(dates))
    csv_depth_indices(",".join(map(str, depth_indices)))
    a, b = _load_run(run_a, dates, depth_indices, metadata_a), _load_run(run_b, dates, depth_indices, metadata_b)
    for key in ("target_var", "dataset_profile"):
        if a[key] != b[key]:
            raise ValueError(f"两组结果的 {key} 不一致: {a[key]} / {b[key]}")
    for key, tolerance in (("longitude", 1e-5), ("latitude", 1e-5), ("depth_m", 1e-3)):
        if a[key].shape != b[key].shape or not np.allclose(a[key], b[key], rtol=0, atol=tolerance):
            raise ValueError(f"两组结果的 {key} 不对应，不自动重网格或改选深度")
    if a["prediction"].shape != b["prediction"].shape:
        raise ValueError("两组结果的预测网格形状不一致")
    mask = a["valid_mask"].astype(bool) & b["valid_mask"].astype(bool)
    for run in (a, b):
        mask &= np.isfinite(run["truth"]) & np.isfinite(run["prediction"])
    for t, day in enumerate(dates):
        for d, index in enumerate(depth_indices):
            valid = mask[t, d]
            if not valid.any():
                raise ValueError(f"{day} 原始层 {index}: 两组结果没有共同有效海洋区域")
            if not np.allclose(a["truth"][t, d][valid], b["truth"][t, d][valid], rtol=1e-6, atol=1e-5):
                raise ValueError(f"{day} 原始层 {index}: 两组结果的共同区域真值不一致")
    return CaseComparison(
        dates, depth_indices, a["depth_m"], a["longitude"], a["latitude"],
        np.where(mask, a["truth"], np.nan), np.where(mask, a["prediction"], np.nan),
        np.where(mask, b["prediction"], np.nan), mask, a["target_var"], a["label"], b["label"],
        "GLORYS truth" if a["dataset_profile"].startswith("glorys") else "Truth",
    )


def comparison_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run-a", default="B2", help="结果文件夹或案例 NPZ，默认 ./B2")
    parser.add_argument("--run-b", default="B3", help="结果文件夹或案例 NPZ，默认 ./B3")
    parser.add_argument("--dates", required=True, help="逗号分隔 YYYY-MM-DD，必须两组均已保存")
    parser.add_argument("--depth-indices", required=True, help="逗号分隔的数据集原始层号，不是局部通道")
    parser.add_argument("--output-dir", default="comparison_figures")
    parser.add_argument("--metadata-a", help="旧 NPZ 对应的测试配置 JSON 或完整测试日志")
    parser.add_argument("--metadata-b", help="旧 NPZ 对应的测试配置 JSON 或完整测试日志")
    return parser


def comparison_from_args(args):
    return load_comparison(
        args.run_a, args.run_b, csv_dates(args.dates), csv_depth_indices(args.depth_indices),
        args.metadata_a, args.metadata_b,
    )
