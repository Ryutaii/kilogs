#!/usr/bin/env python3
"""Automate LEGO feature-distillation capacity sweeps (grid/hidden variants).

This helper orchestrates three phases for each config:
1. Launch the training run if the final checkpoint is missing.
2. Render the trained checkpoint with the standard full-eval settings.
3. Recompose RGBA renders, compute metrics, and optionally update metrics_summary.csv.

It defaults to the 50k white-background configs covering grid8/grid10 and
hidden-dim 160/192 variants.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIGS = [
    Path("configs/lego_feature_teacher_full_rehab_masked_white_grid8.yaml"),
    Path("configs/lego_feature_teacher_full_rehab_masked_white_grid10.yaml"),
    Path("configs/lego_feature_teacher_full_rehab_masked_white_hidden160_layers5.yaml"),
    Path("configs/lego_feature_teacher_full_rehab_masked_white_hidden192_layers5.yaml"),
]


@dataclass
class RunTarget:
    config_path: Path
    config: Dict[str, Any]
    output_dir: Path
    method_name: str
    max_steps: int

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / "checkpoints" / f"step_{self.max_steps:06d}.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        type=Path,
        nargs="*",
        default=DEFAULT_CONFIGS,
        help="List of YAML configs to process (defaults to the planned 50k capacity sweep).",
    )

    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without executing them.")

    parser.add_argument("--force-train", action="store_true", help="Always rerun training even if the final checkpoint exists.")
    parser.add_argument("--force-render", action="store_true", help="Always rerun rendering even if render_stats.json exists.")
    parser.add_argument("--force-eval", action="store_true", help="Always recompute metrics even if the JSON cache exists.")

    parser.add_argument("--no-train", action="store_false", dest="run_train", help="Skip the training phase entirely.")
    parser.add_argument("--no-render", action="store_false", dest="run_render", help="Skip the rendering phase.")
    parser.add_argument("--no-evaluate", action="store_false", dest="run_eval", help="Skip the evaluation phase.")
    parser.set_defaults(run_train=True, run_render=True, run_eval=True)

    parser.add_argument("--render-chunk", type=int, default=8192, help="Chunk size to use for rendering (default: 8192).")
    parser.add_argument(
        "--render-max-frames",
        type=int,
        default=200,
        help="Maximum number of frames to render (default: 200 for full test set).",
    )
    parser.add_argument("--render-tag", type=str, default="full", help="Tag component for the render directory name.")
    parser.add_argument(
        "--render-dir-name",
        type=str,
        default=None,
        help="Override the name of the render directory inside the output dir.",
    )
    parser.add_argument(
        "--render-device",
        type=str,
        default=None,
        help="Device override for rendering (defaults to auto).",
    )
    parser.add_argument(
        "--render-start-frame",
        type=int,
        default=0,
        help="Starting frame index for rendering (default: 0).",
    )
    parser.add_argument(
        "--render-num-samples",
        type=int,
        default=None,
        help="Override samples per ray for rendering (defaults to config value).",
    )
    parser.add_argument(
        "--render-store-rgba",
        dest="render_store_rgba",
        action="store_true",
        help="Store RGBA .npz outputs alongside renders (default).",
    )
    parser.add_argument(
        "--no-render-store-rgba",
        dest="render_store_rgba",
        action="store_false",
        help="Skip writing RGBA .npz outputs during rendering.",
    )
    parser.set_defaults(render_store_rgba=True)
    parser.add_argument(
        "--render-enable-nvml",
        action="store_true",
        help="Enable NVML power logging during rendering.",
    )
    parser.add_argument(
        "--render-allow-mismatch",
        action="store_true",
        help="Allow loading checkpoints with minor shape mismatches.",
    )

    parser.add_argument(
        "--teacher-renders",
        type=Path,
        default=None,
        help="Directory containing teacher reference renders (required when evaluating).",
    )
    parser.add_argument(
        "--metrics-summary",
        type=Path,
        default=Path("metrics_summary.csv"),
        help="Path to metrics_summary.csv for summary updates.",
    )
    parser.add_argument(
        "--method-prefix",
        type=str,
        default="",
        help="Optional prefix to prepend to each method name when logging metrics.",
    )
    parser.add_argument(
        "--eval-background",
        type=str,
        default="white",
        help="Background specification to pass to run_eval_pipeline (default: white).",
    )
    parser.add_argument(
        "--eval-device",
        type=str,
        default=None,
        help="Device override for metric computation.",
    )
    parser.add_argument(
        "--eval-clean",
        action="store_true",
        help="Force recomposition by cleaning existing background directories before evaluation.",
    )
    parser.add_argument(
        "--eval-progress-interval",
        type=int,
        default=20,
        help="Progress interval when computing metrics (default: 20).",
    )
    parser.add_argument(
        "--eval-json-name",
        type=str,
        default=None,
        help="Optional filename for cached metrics JSON (default derives from background).",
    )
    parser.add_argument(
        "--no-summary",
        action="store_false",
        dest="update_summary",
        help="Do not update metrics_summary.csv after evaluation.",
    )
    parser.set_defaults(update_summary=True)

    args = parser.parse_args()

    if args.run_eval and args.teacher_renders is None:
        parser.error("--teacher-renders is required when evaluation is enabled")

    return args


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def prepare_target(config_path: Path) -> RunTarget:
    cfg = load_config(config_path)
    experiment = cfg.get("experiment", {})
    output_dir = resolve_path(Path(experiment.get("output_dir", "results")))
    max_steps = int(cfg.get("train", {}).get("max_steps", 50000))
    method_name = experiment.get("name", config_path.stem)
    return RunTarget(config_path=config_path, config=cfg, output_dir=output_dir, method_name=method_name, max_steps=max_steps)


def run_subprocess(cmd: Sequence[str], dry_run: bool, cwd: Optional[Path] = None) -> None:
    cmd_display = " ".join(cmd)
    if cwd is not None:
        print(f"[cwd] {cwd}")
    print(f"[cmd] {cmd_display}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd is not None else None)


def run_training(target: RunTarget, force: bool, dry_run: bool) -> bool:
    checkpoint = target.checkpoint_path
    if checkpoint.exists() and not force:
        print(f"[skip][train] {target.method_name}: checkpoint already exists at {checkpoint}")
        return False
    cmd = [
        sys.executable,
        "-m",
        "distill.lego_response_distill",
        "--config",
        str(target.config_path),
    ]
    run_subprocess(cmd, dry_run, cwd=PROJECT_ROOT)
    return True


def default_render_dir(target: RunTarget, args: argparse.Namespace) -> Path:
    if args.render_dir_name:
        return target.output_dir / args.render_dir_name
    frames_suffix = "all" if args.render_max_frames is None else f"all{args.render_max_frames}"
    return target.output_dir / f"render_eval_{args.render_tag}_chunk{args.render_chunk}_{frames_suffix}"


def extract_data_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("data", {})


def ensure_list(value: Any, length: int, fallback: float) -> List[float]:
    if isinstance(value, (list, tuple)) and len(value) == length:
        return [float(x) for x in value]
    return [fallback for _ in range(length)]


def run_render(target: RunTarget, args: argparse.Namespace, force: bool, dry_run: bool) -> Path:
    render_dir = default_render_dir(target, args)
    render_stats = render_dir / "render_stats.json"
    if render_stats.exists() and not force:
        print(f"[skip][render] {target.method_name}: existing render stats at {render_stats}")
        return render_dir

    data_cfg = extract_data_config(target.config)
    background = ensure_list(data_cfg.get("background_color", [1.0, 1.0, 1.0]), 3, 1.0)
    bbox_min = ensure_list(data_cfg.get("bbox_min", [-1.5, -1.5, -1.5]), 3, -1.5)
    bbox_max = ensure_list(data_cfg.get("bbox_max", [1.5, 1.5, 1.5]), 3, 1.5)
    near = float(data_cfg.get("near", 2.0))
    far = float(data_cfg.get("far", 6.0))
    samples_per_ray = args.render_num_samples or int(data_cfg.get("samples_per_ray", 128))

    checkpoint = target.checkpoint_path
    if not checkpoint.exists() and not dry_run:
        raise FileNotFoundError(f"Checkpoint missing for render: {checkpoint}")

    cmd: List[str] = [
        sys.executable,
        "-m",
        "distill.render_student",
        "--config",
        str(target.config_path),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(render_dir),
        "--num-samples",
        str(samples_per_ray),
        "--chunk",
        str(args.render_chunk),
        "--near",
        str(near),
        "--far",
        str(far),
        "--background-color",
        *[f"{c:.6f}" for c in background],
        "--bbox-min",
        *[f"{c:.6f}" for c in bbox_min],
        "--bbox-max",
        *[f"{c:.6f}" for c in bbox_max],
        "--start-frame",
        str(args.render_start_frame),
    ]
    if args.render_max_frames is not None:
        cmd.extend(["--max-frames", str(args.render_max_frames)])
    if args.render_device is not None:
        cmd.extend(["--device", args.render_device])
    if args.render_store_rgba:
        cmd.append("--store-rgba")
    if args.render_enable_nvml:
        cmd.append("--enable-nvml")
    if args.render_allow_mismatch:
        cmd.append("--allow-mismatched-weights")

    run_subprocess(cmd, dry_run, cwd=PROJECT_ROOT)
    return render_dir


def sanitise_label(value: str) -> str:
    value = value.strip()
    if not value:
        return "run"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def run_evaluation(
    target: RunTarget,
    render_dir: Path,
    args: argparse.Namespace,
    force: bool,
    dry_run: bool,
) -> Path:
    if not render_dir.exists() and not dry_run:
        raise FileNotFoundError(f"Render directory missing for evaluation: {render_dir}")

    label = sanitise_label(args.eval_background if args.eval_json_name is None else args.eval_json_name)
    metrics_json = render_dir / f"metrics_{label}.json"
    if metrics_json.exists() and not force:
        print(f"[skip][eval] {target.method_name}: cached metrics at {metrics_json}")
        return metrics_json

    teacher_dir = resolve_path(args.teacher_renders)
    summary_path = resolve_path(args.metrics_summary)
    method_label = f"{args.method_prefix}{target.method_name}" if args.method_prefix else target.method_name

    cmd: List[str] = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "run_eval_pipeline.py"),
        str(render_dir),
        "--teacher-renders",
        str(teacher_dir),
        "--background",
        args.eval_background,
        "--method-name",
        method_label,
        "--output-json",
        str(metrics_json),
        "--progress-interval",
        str(args.eval_progress_interval),
    ]
    render_stats = render_dir / "render_stats.json"
    if render_stats.exists() or not dry_run:
        cmd.extend(["--render-stats", str(render_stats)])
    if args.eval_device is not None:
        cmd.extend(["--device", args.eval_device])
    if args.eval_clean:
        cmd.append("--clean")
    if args.update_summary:
        cmd.extend(["--summary", str(summary_path)])

    run_subprocess(cmd, dry_run, cwd=PROJECT_ROOT)
    return metrics_json


def main() -> None:
    args = parse_args()

    config_paths = [resolve_path(path) for path in args.configs]
    for cfg_path in config_paths:
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")

    targets = [prepare_target(path) for path in config_paths]

    for target in targets:
        print("=" * 80)
        print(f"[target] {target.method_name}")
        print(f"         config: {target.config_path}")
        print(f"         output: {target.output_dir}")
        print(f"         steps : {target.max_steps}")

        render_dir: Optional[Path] = None
        if args.run_train:
            try:
                ran = run_training(target, force=args.force_train, dry_run=args.dry_run)
                if ran:
                    print(f"[done][train] {target.method_name}")
            except subprocess.CalledProcessError as exc:
                print(f"[error][train] {target.method_name}: {exc}")
                if not args.dry_run:
                    raise

        if args.run_render:
            try:
                render_dir = run_render(target, args, force=args.force_render, dry_run=args.dry_run)
                print(f"[done][render] {target.method_name} → {render_dir}")
            except subprocess.CalledProcessError as exc:
                print(f"[error][render] {target.method_name}: {exc}")
                if not args.dry_run:
                    raise
        else:
            render_dir = default_render_dir(target, args)

        if args.run_eval:
            try:
                metrics_path = run_evaluation(target, render_dir, args, force=args.force_eval, dry_run=args.dry_run)
                print(f"[done][eval] {target.method_name} → {metrics_path}")
            except subprocess.CalledProcessError as exc:
                print(f"[error][eval] {target.method_name}: {exc}")
                if not args.dry_run:
                    raise

    print("=" * 80)
    print("All targets processed.")


if __name__ == "__main__":
    main()
