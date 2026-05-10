#!/usr/bin/env python3
"""
TODO：Download a South China Sea subset of GHRSST MUR L4 SST v4.1 from NASA Harmony.

"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import shlex
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlencode


COLLECTION_ID = "C1996881146-POCLOUD"
DATASET_SHORT_NAME = "MUR-JPL-L4-GLOB-v4.1"
HARMONY_BASE_URL = (
    f"https://harmony.earthdata.nasa.gov/{COLLECTION_ID}"
    "/ogc-api-coverages/1.0.0/collections/{variable}/coverage/rangeset"
)


def build_harmony_url(args: argparse.Namespace) -> str:
    min_lon, max_lon = args.lon
    min_lat, max_lat = args.lat
    if args.unquoted_time:
        time_subset = f"time({args.start}T00:00:00Z:{args.end}T23:59:59Z)"
    else:
        time_subset = f'time("{args.start}T00:00:00Z":"{args.end}T23:59:59Z")'
    params: list[tuple[str, str]] = [
        ("subset", f"lat({min_lat}:{max_lat})"),
        ("subset", f"lon({min_lon}:{max_lon})"),
        ("subset", time_subset),
        ("format", "application/x-netcdf4"),
        ("skipPreview", "true"),
    ]
    if args.concatenate:
        params.append(("concatenate", "true"))
    if args.max_results:
        params.append(("maxResults", str(args.max_results)))

    query = urlencode(params)
    return HARMONY_BASE_URL.format(variable=args.variable) + "?" + query


def output_path(args: argparse.Namespace) -> Path:
    min_lon, max_lon = args.lon
    min_lat, max_lat = args.lat
    filename = (
        f"mur_sst_{args.start}_{args.end}_"
        f"lat{min_lat:g}-{max_lat:g}_lon{min_lon:g}-{max_lon:g}.nc"
    )
    return Path(args.output_dir) / filename


def curl_command(url: str, target: Path, error_file: Path | None = None) -> list[str]:
    cmd = [
        "curl",
        "--http1.1",
        "-L",
        "-n",
        "-b",
        str(Path.home() / ".urs_cookies"),
        "-c",
        str(Path.home() / ".urs_cookies"),
        "--fail-with-body",
        "--retry",
        "10",
        "--retry-max-time",
        "900",
        "--retry-delay",
        "10",
        "--create-dirs",
        "-o",
        str(target),
        url,
    ]
    if error_file is not None:
        cmd[1:1] = ["--show-error", "--silent", "--dump-header", str(error_file)]
    return cmd


def display_command(url: str, target: Path) -> list[str]:
    return [
        "curl",
        "--http1.1",
        "-L",
        "-n",
        "-b",
        str(Path.home() / ".urs_cookies"),
        "-c",
        str(Path.home() / ".urs_cookies"),
        "--fail-with-body",
        "--retry",
        "10",
        "--retry-max-time",
        "900",
        "--retry-delay",
        "10",
        "--create-dirs",
        "-o",
        str(target),
        url,
    ]


def date_range(start: str, end: str):
    current = date.fromisoformat(start)
    stop = date.fromisoformat(end)
    while current <= stop:
        yield current.isoformat()
        current += timedelta(days=1)


def download_once(args: argparse.Namespace, quiet: bool = False) -> Path:
    target = output_path(args)
    url = build_harmony_url(args)
    cmd_for_display = display_command(url, target)

    if not quiet:
        print(f"Dataset: {DATASET_SHORT_NAME}")
        print(f"Collection: {COLLECTION_ID}")
        print(f"Target: {target}")
        print(f"Harmony URL: {url}")
        print("Command:", " ".join(shlex.quote(part) for part in cmd_for_display))

    if args.print_url:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w+", delete=False) as headers:
        header_path = Path(headers.name)
    cmd = curl_command(url, target, header_path)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        if target.exists():
            error_text = target.read_text(errors="replace").strip()
            if error_text:
                print("\nHarmony error response:")
                print(error_text)
        if header_path.exists():
            header_text = header_path.read_text(errors="replace").strip()
            if header_text:
                print("\nHTTP response headers:")
                print(header_text)
        raise
    if not quiet:
        print(f"Downloaded: {target}")
    return target


def download_daily(args: argparse.Namespace) -> int:
    days = list(date_range(args.start, args.end))
    failures: list[str] = []
    output_dir = Path(args.output_dir)
    failure_log = output_dir / f"failed_mur_sst_{args.start}_{args.end}.txt"

    print(f"Dataset: {DATASET_SHORT_NAME}")
    print(f"Collection: {COLLECTION_ID}")
    print(f"Daily downloads: {len(days)}")
    print(f"Output dir: {output_dir}")

    for index, day in enumerate(days, start=1):
        day_args = argparse.Namespace(**vars(args))
        day_args.start = day
        day_args.end = day
        target = output_path(day_args)
        if args.print_url:
            url = build_harmony_url(day_args)
            print(f"[{index}/{len(days)}] {day}: {url}")
            continue
        if target.exists() and target.stat().st_size > 1024:
            print(f"[{index}/{len(days)}] Skip existing: {target}")
            continue

        print(f"[{index}/{len(days)}] Downloading {day} -> {target}")
        try:
            download_once(day_args, quiet=True)
        except subprocess.CalledProcessError:
            failures.append(day)
            print(f"[{index}/{len(days)}] Failed: {day}")
            continue
        print(f"[{index}/{len(days)}] Done: {target}")

    if failures:
        output_dir.mkdir(parents=True, exist_ok=True)
        failure_log.write_text("\n".join(failures) + "\n")
        print(f"Failed dates: {len(failures)}")
        print(f"Failure log: {failure_log}")
        return 1

    if failure_log.exists():
        failure_log.unlink()
    print("All daily downloads completed.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download GHRSST MUR L4 SST v4.1 subset through NASA Harmony. "
            "Default bbox follows this project's current South China Sea domain."
        )
    )
    parser.add_argument("--start", required=True, help="Start date, e.g. 2019-01-01")
    parser.add_argument("--end", required=True, help="End date, e.g. 2019-01-31")
    parser.add_argument(
        "--lon",
        nargs=2,
        type=float,
        default=(110.0, 118.0),
        metavar=("MIN_LON", "MAX_LON"),
        help="Longitude bounds in degrees east. Default: 110 118",
    )
    parser.add_argument(
        "--lat",
        nargs=2,
        type=float,
        default=(10.0, 18.0),
        metavar=("MIN_LAT", "MAX_LAT"),
        help="Latitude bounds in degrees north. Default: 10 18",
    )
    parser.add_argument(
        "--variable",
        default="all",
        help=(
            "Harmony variable path. The official PO.DAAC MUR v4.1 example uses "
            "'all'. Default: all"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/mur_sst",
        help="Directory for downloaded NetCDF files.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Optional cap on processed granules, useful for testing.",
    )
    parser.add_argument(
        "--concatenate",
        dest="concatenate",
        action="store_true",
        help="Ask Harmony to concatenate matching granules into one NetCDF file.",
    )
    parser.add_argument(
        "--no-concatenate",
        dest="concatenate",
        action="store_false",
        help="Return per-granule Harmony output instead of one concatenated file. This is the default.",
    )
    parser.add_argument(
        "--print-url",
        action="store_true",
        help="Print the Harmony URL and curl command without downloading.",
    )
    parser.add_argument(
        "--unquoted-time",
        action="store_true",
        help="Use time(start:stop) instead of the default time(\"start\":\"stop\") syntax.",
    )
    parser.add_argument(
        "--daily",
        action="store_true",
        help="Download one NetCDF file per day for the requested date range.",
    )
    parser.set_defaults(concatenate=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.daily:
        return download_daily(args)
    target = download_once(args)
    if not args.print_url:
        print(f"Downloaded: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
