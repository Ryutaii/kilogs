#!/usr/bin/env python3
"""One-stop helper to recompose RGBA renders onto a background and evaluate metrics."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch

from recompose_background import (  # type: ignore
    gather_npz_files,
    parse_background,
    process_files,
)
from evaluate_student_metrics import (  # type: ignore
    compute_metrics,
    list_matching_files,
    _load_render_stats,
    _update_summary_csv,
)


def _background_tag(color: Tuple[float, float, float]) -> str:
    return "bg_{:03d}_{:03d}_{:03d}".format(
        int(round(color[0] * 255.0)),
        int(round(color[1] * 255.0)),
        int(round(color[2] * 255.0)),
    )


def _recompose(run_dir: Path, background: Tuple[float, float, float], clean: bool) -> Path:
    rgba_root = run_dir / "rgba_npz"
    if not rgba_root.exists():
        raise FileNotFoundError(f"RGBA directory not found: {rgba_root}")

    output_root = run_dir / "renders_recomposed"
    background_dir = output_root / _background_tag(background)
    if clean and background_dir.exists():
        shutil.rmtree(background_dir)

    npz_files = gather_npz_files(rgba_root)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under {rgba_root}")

    process_files(npz_files, output_root, [background])
    student_dir = background_dir / "renders"
    if not student_dir.exists():
        raise FileNotFoundError(f"Expected recomposed renders at {student_dir}")
    return student_dir


def _run_evaluation(
    student_dir: Path,
    teacher_dir: Path,
    device: torch.device,
    progress_interval: int,
) -> Dict[str, float]:
    pairs = list_matching_files(student_dir, teacher_dir)
    if not pairs:
        raise FileNotFoundError(
            f"Could not align renders between {student_dir} and {teacher_dir}"
        )
    return compute_metrics(pairs, device, progress_interval)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Render evaluation directory (contains rgba_npz / renders / render_stats.json)",
    )
    parser.add_argument(
        "--teacher-renders",
        type=Path,
        required=True,
        help="Directory with teacher reference renders",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="white",
        help="Background color (preset or r,g,b in [0,1])",
    )
    parser.add_argument(
        "--method-name",
        type=str,
        default=None,
        help="Method label for metrics_summary.csv updates",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="metrics_summary.csv to update (optional)",
    )
    parser.add_argument(
        "--render-stats",
        type=Path,
        default=None,
        help="Path to render_stats.json (default: run_dir/render_stats.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for metric computation (default: cuda if available)",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Progress logging interval when computing metrics",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to dump metrics as JSON",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing recomposed renders for the background before processing",
    )
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    teacher_dir = args.teacher_renders.resolve()
    if not teacher_dir.exists():
        raise FileNotFoundError(f"Teacher renders directory not found: {teacher_dir}")

    background_color = parse_background(args.background)
    recomposed_student_dir = _recompose(run_dir, background_color, clean=args.clean)

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    metrics = _run_evaluation(
        recomposed_student_dir,
        teacher_dir,
        device=device,
        progress_interval=args.progress_interval,
    )

    render_stats_path = args.render_stats or (run_dir / "render_stats.json")
    render_stats: Dict[str, Optional[float]] = {}
    if render_stats_path.exists():
        render_stats = _load_render_stats(render_stats_path)
        if render_stats.get("avg_fps") is not None:
            metrics["avg_fps"] = float(render_stats["avg_fps"])  # type: ignore[arg-type]
        if render_stats.get("gpu_peak_gib") is not None:
            metrics["gpu_peak_gib"] = float(render_stats["gpu_peak_gib"])  # type: ignore[arg-type]
        if render_stats.get("power_avg_w") is not None:
            metrics["power_avg_w"] = float(render_stats["power_avg_w"])  # type: ignore[arg-type]

    metrics["background"] = args.background

    if args.summary is not None:
        if args.method_name is None:
            raise ValueError("--summary requires --method-name to be specified")
        _update_summary_csv(
            args.summary,
            args.method_name,
            args.background,
            metrics,
            render_stats,
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
