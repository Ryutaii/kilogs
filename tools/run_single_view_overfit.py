#!/usr/bin/env python3
"""Utility to run a single-view overfit sanity test end-to-end."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "generated" / "lego_single_view_overfit_v1.yaml"
DEFAULT_SUMMARY = REPO_ROOT / "metrics_summary.csv"


def _load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _resolve_teacher_renders(config: Dict) -> Path:
    base = Path(config["data"]["teacher_outputs"])
    renders_dir = base / "renders"
    if not renders_dir.exists():
        raise FileNotFoundError(f"Teacher renders directory not found: {renders_dir}")
    return renders_dir


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(checkpoint_dir.glob("step_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return candidates[-1]


def run_training(
    config_path: Path,
    max_steps: Optional[int],
    overfit_mode: Optional[str],
    overfit_steps: Optional[int],
    overfit_lr: Optional[float],
) -> None:
    cmd = [sys.executable, "-m", "distill.lego_response_distill", "--config", str(config_path)]
    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]
    if overfit_mode:
        cmd += ["--overfit-mode", overfit_mode]
    if overfit_steps is not None:
        cmd += ["--overfit-steps", str(overfit_steps)]
    if overfit_lr is not None:
        cmd += ["--overfit-lr", f"{overfit_lr:.6g}"]
    subprocess.run(cmd, check=True)


def run_render(config_path: Path, checkpoint: Path, output_dir: Path, frame_index: int) -> Path:
    _ensure_directory(output_dir)
    cmd = [
        sys.executable,
        "-m",
        "distill.render_student",
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--start-frame",
        str(frame_index),
        "--max-frames",
        "1",
        "--store-rgba",
    ]
    subprocess.run(cmd, check=True)
    return output_dir / "render_stats.json"


def run_evaluation(
    student_dir: Path,
    teacher_renders: Path,
    render_stats: Path,
    summary_csv: Path,
    method_name: str,
    output_json: Path,
) -> Dict:
    renders_dir = student_dir / "renders" / "renders"
    if not renders_dir.exists():
        raise FileNotFoundError(f"Student renders directory not found: {renders_dir}")

    cmd = [
        sys.executable,
        "tools/evaluate_student_metrics.py",
        str(renders_dir),
        str(teacher_renders),
        "--background",
        "white",
        "--render-stats",
        str(render_stats),
        "--summary",
        str(summary_csv),
        "--method-name",
        method_name,
        "--output-json",
        str(output_json),
        "--force-update",
        "--progress-interval",
        "0",
    ]
    subprocess.run(cmd, check=True)
    with output_json.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Single-view overfit config")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index to overfit and evaluate")
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="metrics_summary.csv to update with the overfit evaluation",
    )
    parser.add_argument(
        "--overfit-mode",
        type=str,
        default="student",
        choices=["none", "projector", "student", "all"],
        help="Optional overfit diagnostic mode to use during training",
    )
    parser.add_argument("--overfit-steps", type=int, default=None, help="Override overfit step count if desired")
    parser.add_argument("--overfit-lr", type=float, default=None, help="Override overfit learning rate")
    args = parser.parse_args()

    config_path = (args.config if args.config.is_absolute() else (REPO_ROOT / args.config)).resolve()
    config = _load_config(config_path)

    experiment_output = Path(config["experiment"]["output_dir"])
    checkpoints_dir = experiment_output / "checkpoints"
    run_training(
        config_path,
        args.max_steps,
        None if args.overfit_mode == "none" else args.overfit_mode,
        args.overfit_steps,
        args.overfit_lr,
    )

    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory missing after training: {checkpoints_dir}")
    checkpoint_path = _find_latest_checkpoint(checkpoints_dir)
    step_token = checkpoint_path.stem.split("_")[-1]
    eval_dir = experiment_output / f"eval_single_view_step{step_token}_view{args.frame_index:03d}"

    render_stats = run_render(config_path, checkpoint_path, eval_dir, args.frame_index)

    teacher_renders = _resolve_teacher_renders(config)
    metrics_path = eval_dir / "metrics_single_view.json"
    method_name = f"single_view_overfit_v1_step{step_token}_view{args.frame_index:03d}"

    metrics = run_evaluation(eval_dir, teacher_renders, render_stats, args.summary, method_name, metrics_path)

    print("Single-view overfit metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
