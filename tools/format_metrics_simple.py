#!/usr/bin/env python3
"""Convert the verbose 3D-GS metrics CSV into a six-column summary."""

import argparse
import csv
from pathlib import Path
from typing import Iterable, Optional


METRIC_KEYS = ("PSNR", "SSIM", "LPIPS", "FPS", "GPU_PEAK", "POWER_AVG")


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def summarise_csv(path: Path) -> Iterable[str]:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise ValueError(f"No data rows found in {path}")

    row = rows[0]
    summarised = []
    for key in METRIC_KEYS:
        summarised.append(parse_float(row.get(key)))
    return summarised


def write_summary(path: Path, values: Iterable[Optional[float]]) -> None:
    values = list(values)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(METRIC_KEYS)
        writer.writerow([
            f"{value:.5f}" if value is not None else ""
            for value in values
        ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to CSV produced by export_metrics_csv.py")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=" Optional output path (defaults to in-place overwrite)",
    )
    args = parser.parse_args()

    target_path = args.output or args.input
    values = summarise_csv(args.input)
    write_summary(target_path, values)

    print(f"Wrote simplified metrics to {target_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
