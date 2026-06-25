#!/usr/bin/env python3
"""Download and validate annual OSTIA foundation SST NetCDF files."""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


DATASET_ID = "METOFFICE-GLO-SST-L4-REP-OBS-SST"
VARIABLE = "analysed_sst"
START_DATE = date(2002, 1, 1)
END_DATE = date(2023, 12, 31)
MIN_LONGITUDE = 105.0
MAX_LONGITUDE = 121.0
MIN_LATITUDE = 0.0
MAX_LATITUDE = 24.0
DEFAULT_OUTPUT_DIR = Path(
    "/Users/lijunjie/Documents/上大硕士/ocean-subsurface-recon/"
    "data/inversion_data_depth_all_size/SST"
)


@dataclass(frozen=True)
class ValidationResult:
    path: Path
    valid: bool
    dates: pd.DatetimeIndex
    errors: tuple[str, ...]


def expected_dates(start: date, end: date) -> pd.DatetimeIndex:
    return pd.date_range(start=start.isoformat(), end=end.isoformat(), freq="D")


def annual_bounds(year: int, start: date, end: date) -> tuple[date, date]:
    return max(start, date(year, 1, 1)), min(end, date(year, 12, 31))


def normalize_dates(values: np.ndarray) -> pd.DatetimeIndex:
    timestamps = pd.DatetimeIndex(pd.to_datetime(values))
    if timestamps.tz is not None:
        timestamps = timestamps.tz_convert(None)
    return timestamps.normalize()


def validate_file(
    path: Path,
    expected_start: date,
    expected_end: date,
) -> ValidationResult:
    errors: list[str] = []
    dates = pd.DatetimeIndex([])

    if not path.is_file():
        return ValidationResult(path, False, dates, ("文件不存在",))

    try:
        with xr.open_dataset(path) as ds:
            if VARIABLE not in ds.data_vars:
                errors.append(f"缺少变量 {VARIABLE}")
            if "time" not in ds.coords:
                errors.append("缺少 time 坐标")
            else:
                dates = normalize_dates(ds["time"].values)

            for coordinate in ("latitude", "longitude"):
                if coordinate not in ds.coords:
                    errors.append(f"缺少 {coordinate} 坐标")

            if VARIABLE in ds.data_vars:
                required_dims = {"time", "latitude", "longitude"}
                actual_dims = set(ds[VARIABLE].dims)
                missing_dims = required_dims - actual_dims
                if missing_dims:
                    errors.append(
                        f"{VARIABLE} 缺少维度: {', '.join(sorted(missing_dims))}"
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


def download_year(
    output_dir: Path,
    year: int,
    start: date,
    end: date,
) -> Path:
    copernicusmarine = importlib.import_module("copernicusmarine")

    target = output_dir / f"ostia_sst_{year}.nc"
    temporary = output_dir / f"ostia_sst_{year}.download.nc"

    copernicusmarine.subset(
        dataset_id=DATASET_ID,
        variables=[VARIABLE],
        minimum_longitude=MIN_LONGITUDE,
        maximum_longitude=MAX_LONGITUDE,
        minimum_latitude=MIN_LATITUDE,
        maximum_latitude=MAX_LATITUDE,
        start_datetime=f"{start.isoformat()}T00:00:00",
        end_datetime=f"{end.isoformat()}T23:59:59",
        coordinates_selection_method="inside",
        output_filename=temporary.name,
        output_directory=output_dir,
        file_format="netcdf",
        overwrite=True,
        netcdf_compression_level=1,
    )

    result = validate_file(temporary, start, end)
    if not result.valid:
        raise RuntimeError(
            f"下载文件校验失败: {'; '.join(result.errors)}"
        )
    temporary.replace(target)
    return target


def validate_all(
    output_dir: Path,
    start: date,
    end: date,
) -> bool:
    all_dates: list[pd.Timestamp] = []
    all_valid = True

    for year in range(start.year, end.year + 1):
        year_start, year_end = annual_bounds(year, start, end)
        result = validate_file(
            output_dir / f"ostia_sst_{year}.nc",
            year_start,
            year_end,
        )
        print_result(result)
        all_valid = all_valid and result.valid
        all_dates.extend(result.dates)

    combined = pd.DatetimeIndex(all_dates)
    expected = expected_dates(start, end)
    duplicates = combined[combined.duplicated()].unique()
    unique_dates = pd.DatetimeIndex(combined.unique()).sort_values()
    missing = expected.difference(unique_dates)
    extra = unique_dates.difference(expected)

    print(
        "\n时间轴汇总: "
        f"实际 {len(combined)}，预期 {len(expected)}，"
        f"重复 {len(duplicates)}，缺失 {len(missing)}，范围外 {len(extra)}"
    )
    if len(expected) == 8035 and start == START_DATE and end == END_DATE:
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
        description="按年下载并校验 Copernicus Marine OSTIA foundation SST。"
    )
    parser.add_argument("--start-date", type=parse_date, default=START_DATE)
    parser.add_argument("--end-date", type=parse_date, default=END_DATE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"NetCDF 输出目录（默认: {DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="仅校验现有年度文件，不执行下载。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.end_date < args.start_date:
        print("错误: --end-date 不能早于 --start-date", file=sys.stderr)
        return 2

    output_dir = args.output_dir.expanduser()
    if args.validate_only:
        if not output_dir.is_dir():
            print(f"错误: 下载目录不存在: {output_dir}", file=sys.stderr)
            return 1
        return 0 if validate_all(output_dir, args.start_date, args.end_date) else 1

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

    output_dir.mkdir(parents=True, exist_ok=True)
    failed_years: list[int] = []
    for year in range(args.start_date.year, args.end_date.year + 1):
        year_start, year_end = annual_bounds(
            year, args.start_date, args.end_date
        )
        target = output_dir / f"ostia_sst_{year}.nc"
        existing = validate_file(target, year_start, year_end)
        if existing.valid:
            print(f"[SKIP] {target.name} 已存在且校验通过")
            continue
        if target.exists():
            print(f"[REDOWNLOAD] {target.name}: {'; '.join(existing.errors)}")

        print(f"[DOWNLOAD] {year_start} 至 {year_end}")
        try:
            downloaded = download_year(
                output_dir, year, year_start, year_end
            )
            print(f"[OK] 已保存 {downloaded}")
        except Exception as exc:
            failed_years.append(year)
            print(
                f"[FAIL] {year}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    if failed_years:
        print(
            "下载失败年份: " + ", ".join(map(str, failed_years)),
            file=sys.stderr,
        )
        return 1
    return 0 if validate_all(output_dir, args.start_date, args.end_date) else 1


if __name__ == "__main__":
    raise SystemExit(main())
