#!/usr/bin/env python3
"""Aggregate key quality/performance metrics from one or more student runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

FIELDS = [
    "run",
    "psnr",
    "ssim",
    "lpips",
    "fps",
    "gpu_mem_peak_mib",
    "power_avg_w",
]


def load_metrics(run_dir: Path) -> Dict[str, Optional[float]]:
    eval_dir = run_dir / "eval"
    metrics_path = eval_dir / "metrics.json"
    render_stats_path = eval_dir / "render_stats.json"

    def read_json(path: Path) -> Dict:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    metrics = read_json(metrics_path)
    stats = read_json(render_stats_path)

    result: Dict[str, Optional[float]] = {
        "run": run_dir.name,
        "psnr": metrics.get("psnr"),
        "ssim": metrics.get("ssim"),
        "lpips": metrics.get("lpips"),
        "fps": stats.get("avg_fps"),
        "gpu_mem_peak_mib": stats.get("gpu_mem_peak_mib"),
        "power_avg_w": stats.get("power_avg_watts"),
    }
    return result


def format_markdown(rows: Iterable[Dict[str, Optional[float]]]) -> str:
    rows = list(rows)
    header = "| Run | PSNR (dB) | SSIM | LPIPS | FPS | GPU peak (MiB) | Power (W) |"
    divider = "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"

    def fmt(value: Optional[float], digits: int = 3) -> str:
        if value is None:
            return "-"
        return f"{value:.{digits}f}"

    body_lines = []
    for row in rows:
        body_lines.append(
            "| {run} | {psnr} | {ssim} | {lpips} | {fps} | {mem} | {power} |".format(
                run=row["run"],
                psnr=fmt(row["psnr"], 3),
                ssim=fmt(row["ssim"], 3),
                lpips=fmt(row["lpips"], 3),
                fps=fmt(row["fps"], 3),
                mem=fmt(row["gpu_mem_peak_mib"], 1),
                power=fmt(row["power_avg_w"], 1),
            )
        )
    return "\n".join([header, divider, *body_lines])


def write_csv(rows: Iterable[Dict[str, Optional[float]]], output_path: Path) -> None:
    rows = list(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories (e.g. results/lego/stage1b_kilo_uniform_50k)")
    parser.add_argument("--output-md", type=Path, default=None, help="Optional Markdown output path")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV output path")
    args = parser.parse_args()

    summaries: List[Dict[str, Optional[float]]] = []
    for run_dir in args.runs:
        summaries.append(load_metrics(run_dir))

    md_table = format_markdown(summaries)
    print(md_table)

    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md_table + "\n", encoding="utf-8")

    if args.output_csv is not None:
        write_csv(summaries, args.output_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
