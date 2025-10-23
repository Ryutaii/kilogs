#!/usr/bin/env python3
"""Utility to validate LEGO evaluation assets and optionally recompute metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import shutil

import torch

from check_teacher_assets import check_teacher_assets
from evaluate_student_metrics import (  # type: ignore
    _load_render_stats,
    _update_summary_csv,
    compute_metrics,
    list_matching_files,
)
from recompose_background import gather_npz_files, parse_background, process_files  # type: ignore


def _count_matches(directory: Path, pattern: str) -> int:
    return sum(1 for _ in directory.glob(pattern))


def _check_student_assets(
    render_root: Path,
    expected_frames: Optional[int],
    require_rgba: bool,
    verbose: bool,
) -> tuple[list[str], Optional[Path], Optional[Path]]:
    issues: list[str] = []
    student_renders: Optional[Path] = None
    rgba_dir: Optional[Path] = None

    if not render_root.exists():
        issues.append(f"student render root missing: {render_root}")
        return issues, student_renders, rgba_dir
    if not render_root.is_dir():
        issues.append(f"student render root is not a directory: {render_root}")
        return issues, student_renders, rgba_dir

    student_renders = render_root / "renders"
    if not student_renders.exists():
        issues.append(f"missing student renders directory: {student_renders}")
    elif not student_renders.is_dir():
        issues.append(f"student renders path is not a directory: {student_renders}")
    else:
        png_count = _count_matches(student_renders, "*.png")
        if png_count == 0:
            nested = student_renders / "renders"
            if nested.exists() and nested.is_dir():
                student_renders = nested
                png_count = _count_matches(student_renders, "*.png")
        if png_count == 0:
            issues.append(f"student renders contain 0 png files under {student_renders}")
        elif expected_frames is not None:
            if png_count != expected_frames:
                issues.append(
                    f"student renders contain {png_count} png files, expected {expected_frames}"
                )
            elif verbose:
                print(f"[ok] student renders: {png_count} frames")

    rgba_dir = render_root / "rgba_npz"
    if require_rgba:
        if not rgba_dir.exists():
            issues.append(f"missing rgba_npz directory (required for recomposition): {rgba_dir}")
        elif not rgba_dir.is_dir():
            issues.append(f"rgba_npz path is not a directory: {rgba_dir}")
        elif expected_frames is not None:
            count = _count_matches(rgba_dir, "*.npz")
            if count != expected_frames:
                issues.append(
                    f"rgba_npz contains {count} npz files, expected {expected_frames}"
                )
            elif verbose:
                print(f"[ok] rgba_npz: {count} frames")

    stats_path = render_root / "render_stats.json"
    if not stats_path.exists():
        issues.append(f"missing render_stats.json under {render_root}")
    elif verbose:
        print(f"[ok] found render stats: {stats_path}")

    return issues, student_renders, rgba_dir


def _recompute_metrics(
    render_root: Path,
    rgba_dir: Path,
    student_renders: Path,
    teacher_renders: Path,
    background: str,
    device: torch.device,
    progress_interval: int,
    clean: bool,
    summary_csv: Optional[Path],
    method_name: Optional[str],
    output_json: Optional[Path],
) -> dict[str, float]:
    background_color = parse_background(background)
    output_root = render_root / "renders_recomposed"
    background_dir = output_root / "bg_{:03d}_{:03d}_{:03d}".format(
        int(round(background_color[0] * 255.0)),
        int(round(background_color[1] * 255.0)),
        int(round(background_color[2] * 255.0)),
    )

    npz_files = gather_npz_files(rgba_dir)
    if not npz_files:
        raise FileNotFoundError(f"no npz files found under {rgba_dir}")

    if clean and background_dir.exists():
        for child in background_dir.glob("*"):
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)
    process_files(npz_files, output_root, [background_color])

    recomposed_dir = background_dir / "renders"
    if not recomposed_dir.exists():
        alternate = background_dir / "rgba_npz"
        if alternate.exists():
            recomposed_dir = alternate
        else:
            raise FileNotFoundError(f"expected recomposed renders at {recomposed_dir}")

    pairs = list_matching_files(recomposed_dir, teacher_renders)
    if not pairs:
        raise FileNotFoundError(
            f"failed to align renders: {recomposed_dir} vs {teacher_renders}"
        )

    metrics = compute_metrics(pairs, device, progress_interval)

    stats_path = render_root / "render_stats.json"
    render_stats = {}
    if stats_path.exists():
        render_stats = _load_render_stats(stats_path)
        if render_stats.get("avg_fps") is not None:
            metrics["avg_fps"] = float(render_stats["avg_fps"])  # type: ignore[arg-type]
        if render_stats.get("gpu_peak_gib") is not None:
            metrics["gpu_peak_gib"] = float(render_stats["gpu_peak_gib"])  # type: ignore[arg-type]
        if render_stats.get("power_avg_w") is not None:
            metrics["power_avg_w"] = float(render_stats["power_avg_w"])  # type: ignore[arg-type]

    metrics["background"] = background

    if summary_csv is not None:
        if method_name is None:
            raise ValueError("--summary requires --method-name")
        _update_summary_csv(summary_csv, method_name, background, metrics, render_stats)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)

    return metrics


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-outputs",
        type=Path,
        default=Path("teacher/outputs/lego/test_white/ours_30000"),
        help="Teacher render root (contains renders/ gt/ depth/).",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=Path("teacher/checkpoints/lego/point_cloud/iteration_30000/point_cloud.ply"),
        help="Teacher point-cloud checkpoint path.",
    )
    parser.add_argument(
        "--student-render-root",
        type=Path,
        required=True,
        help="Directory produced by render_student.py (renders/, rgba_npz/, render_stats.json).",
    )
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        default=None,
        help="Optional student checkpoint to verify exists.",
    )
    parser.add_argument(
        "--expected-frames",
        type=int,
        default=200,
        help="Expected frame count for teacher/student renders (set to 0 to skip).",
    )
    parser.add_argument(
        "--skip-rgba",
        action="store_true",
        help="Skip verifying rgba_npz directory (useful if only png renders are available).",
    )
    parser.add_argument(
        "--recompute-metrics",
        action="store_true",
        help="Re-run recomposition + metric evaluation using evaluate_student_metrics.",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="white",
        help="Background preset or r,g,b tuple for recomposition.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for metric computation.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Logging interval for metric recomputation.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="metrics_summary.csv to update with recomputed metrics.",
    )
    parser.add_argument(
        "--method-name",
        type=str,
        default=None,
        help="Method label when updating metrics_summary.csv.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to dump metrics JSON report.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing recomposed renders for the requested background before running.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional diagnostics while checking assets.",
    )

    args = parser.parse_args(argv)

    expected_frames: Optional[int] = args.expected_frames if args.expected_frames > 0 else None

    print("[step] checking teacher assets …")
    teacher_status = check_teacher_assets(
        outputs_root=args.teacher_outputs,
        checkpoint_path=args.teacher_checkpoint,
        expected_frames=expected_frames or 0,
        verbose=args.verbose,
    )
    if teacher_status != 0:
        print("[error] teacher asset check failed; aborting.")
        return 1

    if args.student_checkpoint is not None:
        checkpoint_path = Path(args.student_checkpoint)
        if not checkpoint_path.exists():
            print(f"[error] student checkpoint missing: {checkpoint_path}")
            return 1
        if checkpoint_path.stat().st_size == 0:
            print(f"[error] student checkpoint is empty: {checkpoint_path}")
            return 1
        if args.verbose:
            size_mb = checkpoint_path.stat().st_size / (1024 ** 2)
            print(f"[ok] student checkpoint present ({size_mb:.2f} MiB)")

    print("[step] checking student renders …")
    issues, student_renders, rgba_dir = _check_student_assets(
        render_root=args.student_render_root,
        expected_frames=expected_frames,
        require_rgba=not args.skip_rgba,
        verbose=args.verbose,
    )
    if issues:
        print("[error] student render validation failed:")
        for item in issues:
            print(f"  - {item}")
        return 1

    if args.recompute_metrics:
        if student_renders is None or rgba_dir is None:
            print("[error] recompute requested but student renders or rgba_npz missing.")
            return 1
        teacher_renders = args.teacher_outputs / "renders"
        if not teacher_renders.exists():
            print(f"[error] teacher renders directory missing: {teacher_renders}")
            return 1

        device_str = args.device
        if device_str == "cuda" and not torch.cuda.is_available():
            device_str = "cpu"
        device = torch.device(device_str)

        print("[step] recomposing + evaluating metrics …")
        metrics = _recompute_metrics(
            render_root=args.student_render_root,
            rgba_dir=rgba_dir,
            student_renders=student_renders,
            teacher_renders=teacher_renders,
            background=args.background,
            device=device,
            progress_interval=args.progress_interval,
            clean=args.clean,
            summary_csv=args.summary_csv,
            method_name=args.method_name,
            output_json=args.output_json,
        )
        print(json.dumps(metrics, indent=2))
    else:
        print("[ok] student render assets look consistent. Skipping metric recomputation.")

    print("[done] evaluation consistency checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
