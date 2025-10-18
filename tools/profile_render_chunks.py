"""Profile student rendering performance across chunk sizes.

This helper runs ``distill.render_student`` for a list of chunk sizes and
collects the resulting FPS and VRAM statistics into JSON/CSV summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from distill import render_student


def _parse_float3(value: Iterable[float]) -> List[float]:
    values = list(float(v) for v in value)
    if len(values) != 3:
        raise ValueError("Expected exactly three values")
    return values


def run_profile(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    chunk_sizes = [int(c) for c in args.chunks]
    results = []

    for chunk in chunk_sizes:
        destination = output_root / f"chunk_{chunk}"
        destination.mkdir(parents=True, exist_ok=True)
        namespace = SimpleNamespace(
            config=str(config_path),
            checkpoint=str(checkpoint_path),
            output_dir=str(destination),
            num_samples=int(args.num_samples),
            chunk=int(chunk),
            near=float(args.near),
            far=float(args.far),
            background_color=tuple(args.background_color),
            bbox_min=tuple(args.bbox_min),
            bbox_max=tuple(args.bbox_max),
            device=args.device,
            enable_nvml=bool(args.enable_nvml),
            allow_mismatched_weights=bool(args.allow_mismatched_weights),
            max_frames=args.max_frames,
        )
        print(f"[profile] Rendering chunk={chunk} → {destination}")
        render_student.render_student_scene(namespace)
        stats_path = destination / "render_stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"Missing render stats for chunk {chunk}: {stats_path}")
        with stats_path.open("r", encoding="utf-8") as fp:
            stats = json.load(fp)
        stats["chunk"] = chunk
        results.append(stats)

    summary_json = output_root / "chunk_profile_summary.json"
    with summary_json.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"[profile] Summary written to {summary_json}")

    # Prepare CSV with a stable column order.
    fieldnames = [
        "chunk",
        "avg_fps",
        "total_render_time_s",
        "num_frames",
        "gpu_mem_peak_mib",
        "gpu_mem_reserved_peak_mib",
        "power_avg_watts",
        "nvml_enabled",
    ]
    csv_path = output_root / "chunk_profile_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for entry in results:
            row = {name: entry.get(name, "") for name in fieldnames}
            writer.writerow(row)
    print(f"[profile] CSV summary written to {csv_path}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Render student checkpoints across multiple chunk sizes")
    parser.add_argument("--config", required=True, help="Path to lego_response YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to student checkpoint (.pth)")
    parser.add_argument("--output-dir", required=True, help="Directory to store render outputs and summaries")
    parser.add_argument(
        "--chunks",
        required=True,
        nargs="+",
        help="List of chunk sizes to profile (e.g. 4096 8192 16384)",
    )
    parser.add_argument("--num-samples", type=int, default=128, help="Samples per ray")
    parser.add_argument("--near", type=float, default=2.0, help="Near plane distance")
    parser.add_argument("--far", type=float, default=6.0, help="Far plane distance")
    parser.add_argument(
        "--background-color",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        help="Background color (RGB) for alpha compositing",
    )
    parser.add_argument(
        "--bbox-min",
        type=float,
        nargs=3,
        default=(-1.5, -1.5, -1.5),
        help="Minimum corner of the bounding box",
    )
    parser.add_argument(
        "--bbox-max",
        type=float,
        nargs=3,
        default=(1.5, 1.5, 1.5),
        help="Maximum corner of the bounding box",
    )
    parser.add_argument("--device", default=None, help="Override rendering device (cuda|cpu)")
    parser.add_argument("--enable-nvml", action="store_true", help="Enable NVML power sampling")
    parser.add_argument(
        "--allow-mismatched-weights",
        action="store_true",
        help="Allow partial checkpoint loading for shape-mismatched weights",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames rendered per chunk")

    parsed = parser.parse_args(argv)
    parsed.background_color = _parse_float3(parsed.background_color)
    parsed.bbox_min = _parse_float3(parsed.bbox_min)
    parsed.bbox_max = _parse_float3(parsed.bbox_max)

    run_profile(parsed)


if __name__ == "__main__":
    main()
