#!/usr/bin/env python3
"""Download and validate Global Ocean Physics Reanalysis T/S NetCDF files."""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
VARIABLES = {
    "temperature": "thetao",
    "salinity": "so",
}
START_DATE = date(2002, 1, 1)
END_DATE = date(2023, 12, 31)
MIN_LONGITUDE = 105.0
MAX_LONGITUDE = 121.0
MIN_LATITUDE = 0.0
MAX_LATITUDE = 24.0
MIN_DEPTH = 0.0
MAX_DEPTH = 300.0


@dataclass(frozen=True)
class TimeChunk:
    start: date
    end: date

    @property
    def label(self) -> str:
        if self.start.year == self.end.year and self.start.month == self.end.month:
            return f"{self.start.year:04d}_{self.start.month:02d}"
        return f"{self.start:%Y%m%d}_{self.end:%Y%m%d}"


@dataclass(frozen=True)
class ValidationResult:
    path: Path
    valid: bool
    dates: pd.DatetimeIndex
    errors: tuple[str, ...]


def expected_dates(start: date, end: date) -> pd.DatetimeIndex:
    return pd.date_range(start=start.isoformat(), end=end.isoformat(), freq="D")


def normalize_dates(values: np.ndarray) -> pd.DatetimeIndex:
    timestamps = pd.DatetimeIndex(pd.to_datetime(values))
    if timestamps.tz is not None:
        timestamps = timestamps.tz_convert(None)
    return timestamps.normalize()


def add_one_month(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def add_one_year(value: date) -> date:
    return date(value.year + 1, 1, 1)


def make_chunks(start: date, end: date, chunk: str) -> list[TimeChunk]:
    chunks: list[TimeChunk] = []
    current = start
    while current <= end:
        next_start = add_one_month(current) if chunk == "month" else add_one_year(current)
        chunk_end = min(end, next_start - timedelta(days=1))
        chunks.append(TimeChunk(current, chunk_end))
        current = next_start
    return chunks


def output_paths(output_dir: Path, variable: str, chunk: TimeChunk) -> tuple[Path, Path]:
    var_dir = output_dir / variable
    target = var_dir / f"{variable}_{chunk.label}.nc"
    temporary = var_dir / f"{variable}_{chunk.label}.download.nc"
    return target, temporary


def validate_file(
    path: Path,
    variable: str,
    expected_start: date,
    expected_end: date,
    min_depth: float,
    max_depth: float,
) -> ValidationResult:
    errors: list[str] = []
    dates = pd.DatetimeIndex([])

    if not path.is_file():
        return ValidationResult(path, False, dates, ("文件不存在",))

    try:
        import xarray as xr

        with xr.open_dataset(path) as ds:
            if variable not in ds.data_vars:
                errors.append(f"缺少变量 {variable}")
            if "time" not in ds.coords:
                errors.append("缺少 time 坐标")
            else:
                dates = normalize_dates(ds["time"].values)

            for coordinate in ("depth", "latitude", "longitude"):
                if coordinate not in ds.coords:
                    errors.append(f"缺少 {coordinate} 坐标")

            if "depth" in ds.coords:
                depths = np.asarray(ds["depth"].values, dtype=np.float64)
                if depths.size == 0:
                    errors.append("depth 坐标为空")
                else:
                    if np.any(np.diff(depths) < 0):
                        errors.append("depth 坐标未按升序排列")
                    if float(np.nanmin(depths)) < min_depth - 1e-6:
                        errors.append(f"depth 最小值小于 {min_depth:g} m")
                    if float(np.nanmax(depths)) > max_depth + 1e-6:
                        errors.append(f"depth 最大值大于 {max_depth:g} m")

            if variable in ds.data_vars:
                required_dims = {"time", "depth", "latitude", "longitude"}
                actual_dims = set(ds[variable].dims)
                missing_dims = required_dims - actual_dims
                if missing_dims:
                    errors.append(
                        f"{variable} 缺少维度: {', '.join(sorted(missing_dims))}"
                    )
    except Exception as exc:
        return ValidationResult(
            path,
            False,
            dates,
            (f"无法读取 NetCDF: {type(exc).__name__}: {exc}",),
        )

    if len(dates):
        expected = expected_dates(expected_start, expected_end)
        duplicated = dates[dates.duplicated()].unique()
        if len(duplicated):
            errors.append(f"存在 {len(duplicated)} 个重复日期")
        if not dates.is_monotonic_increasing:
            errors.append("时间坐标未按升序排列")

        unique_dates = pd.DatetimeIndex(dates.unique()).sort_values()
        missing = expected.difference(unique_dates)
        extra = unique_dates.difference(expected)
        if len(missing):
            errors.append(f"缺少 {len(missing)} 天")
        if len(extra):
            errors.append(f"包含范围外的 {len(extra)} 天")
        if len(dates) != len(expected):
            errors.append(f"时间步数为 {len(dates)}，预期 {len(expected)}")

    return ValidationResult(path, not errors, dates, tuple(errors))


def print_result(result: ValidationResult) -> None:
    if result.valid:
        print(f"[OK] {result.path.name}: {len(result.dates)} 天")
        return
    print(f"[FAIL] {result.path.name}: {'; '.join(result.errors)}")


def download_chunk(
    output_dir: Path,
    variable: str,
    chunk: TimeChunk,
    args: argparse.Namespace,
) -> Path:
    copernicusmarine = importlib.import_module("copernicusmarine")
    target, temporary = output_paths(output_dir, variable, chunk)
    target.parent.mkdir(parents=True, exist_ok=True)

    copernicusmarine.subset(
        dataset_id=DATASET_ID,
        variables=[variable],
        minimum_longitude=args.min_lon,
        maximum_longitude=args.max_lon,
        minimum_latitude=args.min_lat,
        maximum_latitude=args.max_lat,
        minimum_depth=args.min_depth,
        maximum_depth=args.max_depth,
        start_datetime=f"{chunk.start.isoformat()}T00:00:00",
        end_datetime=f"{chunk.end.isoformat()}T23:59:59",
        coordinates_selection_method="inside",
        output_filename=temporary.name,
        output_directory=target.parent,
        file_format="netcdf",
        overwrite=True,
        netcdf_compression_level=1,
    )

    result = validate_file(
        temporary,
        variable,
        chunk.start,
        chunk.end,
        args.min_depth,
        args.max_depth,
    )
    if not result.valid:
        raise RuntimeError(f"下载文件校验失败: {'; '.join(result.errors)}")
    temporary.replace(target)
    return target


def validate_all(
    output_dir: Path,
    variable: str,
    chunks: list[TimeChunk],
    min_depth: float,
    max_depth: float,
) -> bool:
    all_dates: list[pd.Timestamp] = []
    all_valid = True

    for chunk in chunks:
        target, _ = output_paths(output_dir, variable, chunk)
        result = validate_file(target, variable, chunk.start, chunk.end, min_depth, max_depth)
        print_result(result)
        all_valid = all_valid and result.valid
        all_dates.extend(result.dates)

    combined = pd.DatetimeIndex(all_dates)
    expected = expected_dates(chunks[0].start, chunks[-1].end) if chunks else pd.DatetimeIndex([])
    duplicates = combined[combined.duplicated()].unique()
    unique_dates = pd.DatetimeIndex(combined.unique()).sort_values()
    missing = expected.difference(unique_dates)
    extra = unique_dates.difference(expected)

    print(
        "\n时间轴汇总: "
        f"实际 {len(combined)}，预期 {len(expected)}，"
        f"重复 {len(duplicates)}，缺失 {len(missing)}，范围外 {len(extra)}"
    )
    if len(expected) == 8035 and chunks[0].start == START_DATE and chunks[-1].end == END_DATE:
        print("完整范围预期时间步数: 8035")

    timeline_valid = (
        len(combined) == len(expected)
        and not len(duplicates)
        and not len(missing)
        and not len(extra)
        and combined.is_monotonic_increasing
    )
    if not timeline_valid:
        print("[FAIL] 汇总时间轴不完整或顺序异常")
    else:
        print("[OK] 汇总时间轴连续、无重复且顺序正确")
    return all_valid and timeline_valid


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"日期必须使用 YYYY-MM-DD 格式: {value}"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按月/年下载并校验 Copernicus Global Ocean Physics Reanalysis 温盐数据。"
    )
    parser.add_argument(
        "--var",
        choices=sorted(VARIABLES),
        default="temperature",
        help="下载变量：temperature=thetao，salinity=so。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="服务器大容量数据盘输出目录；脚本会自动创建变量子目录。",
    )
    parser.add_argument("--start-date", type=parse_date, default=START_DATE)
    parser.add_argument("--end-date", type=parse_date, default=END_DATE)
    parser.add_argument("--min-lon", type=float, default=MIN_LONGITUDE)
    parser.add_argument("--max-lon", type=float, default=MAX_LONGITUDE)
    parser.add_argument("--min-lat", type=float, default=MIN_LATITUDE)
    parser.add_argument("--max-lat", type=float, default=MAX_LATITUDE)
    parser.add_argument("--min-depth", type=float, default=MIN_DEPTH)
    parser.add_argument("--max-depth", type=float, default=MAX_DEPTH)
    parser.add_argument(
        "--chunk",
        choices=("month", "year"),
        default="month",
        help="时间分块粒度，默认按月，适合大范围 3D 温盐数据。",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="仅校验现有分块文件，不执行下载。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印下载计划，不联网、不创建目录、不下载。",
    )
    return parser


def print_plan(output_dir: Path, variable: str, chunks: list[TimeChunk], args: argparse.Namespace) -> None:
    print(f"dataset_id: {DATASET_ID}")
    print(f"variable: {variable}")
    print(
        "region: "
        f"lon {args.min_lon:g}..{args.max_lon:g}, "
        f"lat {args.min_lat:g}..{args.max_lat:g}, "
        f"depth {args.min_depth:g}..{args.max_depth:g} m"
    )
    print(f"time: {chunks[0].start}..{chunks[-1].end} ({len(chunks)} chunks)")
    print(f"output_dir: {output_dir}")
    preview = chunks[:3]
    if len(chunks) > 6:
        preview = chunks[:3] + chunks[-3:]
    for chunk in preview:
        target, _ = output_paths(output_dir, variable, chunk)
        print(f"  {chunk.start}..{chunk.end} -> {target}")
    if len(chunks) > len(preview):
        print(f"  ... {len(chunks) - len(preview)} more chunks")


def main() -> int:
    args = build_parser().parse_args()
    if args.end_date < args.start_date:
        print("错误: --end-date 不能早于 --start-date", file=sys.stderr)
        return 2
    if args.max_lon < args.min_lon or args.max_lat < args.min_lat:
        print("错误: 经纬度最大值不能小于最小值", file=sys.stderr)
        return 2
    if args.max_depth < args.min_depth:
        print("错误: --max-depth 不能小于 --min-depth", file=sys.stderr)
        return 2

    output_dir = args.output_dir.expanduser()
    variable = VARIABLES[args.var]
    chunks = make_chunks(args.start_date, args.end_date, args.chunk)

    if args.dry_run:
        print_plan(output_dir, variable, chunks, args)
        return 0

    if args.validate_only:
        return 0 if validate_all(
            output_dir, variable, chunks, args.min_depth, args.max_depth
        ) else 1

    try:
        importlib.import_module("copernicusmarine")
    except ImportError:
        print(
            "错误: 当前 Python 环境未安装 copernicusmarine。\n"
            f"当前解释器: {sys.executable}\n"
            "请执行:\n"
            f'  "{sys.executable}" -m pip install "copernicusmarine>=2.4.1,<3"',
            file=sys.stderr,
        )
        return 1

    failed_chunks: list[TimeChunk] = []
    for chunk in chunks:
        target, _ = output_paths(output_dir, variable, chunk)
        existing = validate_file(
            target,
            variable,
            chunk.start,
            chunk.end,
            args.min_depth,
            args.max_depth,
        )
        if existing.valid:
            print(f"[SKIP] {target.name} 已存在且校验通过")
            continue
        if target.exists():
            print(f"[REDOWNLOAD] {target.name}: {'; '.join(existing.errors)}")

        print(f"[DOWNLOAD] {variable} {chunk.start} 至 {chunk.end}")
        try:
            downloaded = download_chunk(output_dir, variable, chunk, args)
            print(f"[OK] 已保存 {downloaded}")
        except Exception as exc:
            failed_chunks.append(chunk)
            print(
                f"[FAIL] {chunk.label}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    if failed_chunks:
        print(
            "下载失败分块: " + ", ".join(chunk.label for chunk in failed_chunks),
            file=sys.stderr,
        )
        return 1
    return 0 if validate_all(
        output_dir, variable, chunks, args.min_depth, args.max_depth
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
