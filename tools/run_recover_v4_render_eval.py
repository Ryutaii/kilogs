#!/usr/bin/env python3
"""Render and evaluate recover_v4 hidden_dim sweep checkpoints sequentially.

For each hidden_dim variant, this script generates the corresponding config,
invokes ``distill/render_student.py`` to render RGBA outputs, and then runs
``tools/run_eval_pipeline.py`` to compute white-background metrics. The results
are appended (or updated) in ``hyper_params_Lego.csv`` and recorded in the
standard summary CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import yaml

from run_recover_v4_hyperparam_sweep import (  # type: ignore
    default_values,
    format_suffix,
    load_config,
    override_config,
    save_config,
)

BASE_CONFIG = Path("configs/lego_feature_teacher_full_quickwin_relaxed_alpha045_recover_v4.yaml")
BASE_RUN_NAME = "teacher_full_quickwin_relaxed_alpha045_recover_v4"
RUNS_ROOT = Path("results/lego/feat_t_full/runs")
RENDERS_ROOT = Path("results/lego/feat_t_full/renders")
SUMMARY_CSV = Path("logs/lego/feat_t_full/hparam_sweeps/recover_v4/summary.csv")
TEACHER_RENDER_ROOT = Path("teacher/outputs/lego/test_white/ours_30000/renders")
HYPER_PARAM_CSV = Path("../hyper_params_Lego.csv").resolve()
RENDER_SCRIPT = Path("distill/render_student.py")
EVAL_SCRIPT = Path("tools/run_eval_pipeline.py")


@dataclass
class VariantInfo:
    value: int
    suffix: str
    method_name: str
    run_dir: Path
    checkpoint: Path
    render_dir: Path
    config_path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--values",
        type=int,
        nargs="*",
        help="Explicit hidden_dim values to process (default: sweep defaults)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Training steps the checkpoints correspond to (default: 10000)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=8192,
        help="Chunk size for render_student (default: 8192)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="Samples per ray for render_student (default: 128)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Rendering device (default: cuda)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip evaluation if hyper_params already has a row for the variant",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without executing render/eval",
    )
    return parser.parse_args(argv)


def resolve_variant_info(base_cfg: Dict, value: int, max_steps: int) -> VariantInfo:
    suffix = format_suffix("hidden_dim", value)
    tmp_dir = Path(tempfile.gettempdir()) / f"recover_v4_hidden_dim_eval_{suffix}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    variant_cfg = override_config(base_cfg, "hidden_dim", value, max_steps, suffix)
    config_path = tmp_dir / f"{suffix}.yaml"
    save_config(variant_cfg, config_path)

    run_dir_name = f"{BASE_RUN_NAME}_{suffix}"
    run_dir = RUNS_ROOT / run_dir_name
    checkpoint = run_dir / "checkpoints" / f"step_{max_steps:06d}.pth"
    render_dir = RENDERS_ROOT / run_dir_name
    method_name = f"recover_v4_{suffix}"

    return VariantInfo(
        value=value,
        suffix=suffix,
        method_name=method_name,
        run_dir=run_dir,
        checkpoint=checkpoint,
        render_dir=render_dir,
        config_path=config_path,
    )


def run_command(cmd: Sequence[str]) -> None:
    subprocess.run(cmd, check=True)


def run_render(info: VariantInfo, args: argparse.Namespace) -> None:
    render_dir = info.render_dir
    render_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(RENDER_SCRIPT),
        "--config",
        str(info.config_path),
        "--checkpoint",
        str(info.checkpoint),
        "--output-dir",
        str(render_dir),
        "--chunk",
        str(args.chunk),
        "--num-samples",
        str(args.num_samples),
        "--device",
        args.device,
        "--store-rgba",
        "--enable-nvml",
    ]
    print(f"[render] {info.method_name} → {render_dir}")
    run_command(cmd)


def run_eval(info: VariantInfo, args: argparse.Namespace) -> Path:
    render_dir = info.render_dir
    output_json = render_dir / "metrics_white.json"
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        str(render_dir),
        "--teacher-renders",
        str(TEACHER_RENDER_ROOT),
        "--background",
        "white",
        "--method-name",
        info.method_name,
        "--summary",
        str(SUMMARY_CSV),
        "--output-json",
        str(output_json),
        "--clean",
    ]
    print(f"[eval] {info.method_name} → {output_json}")
    run_command(cmd)
    return output_json


def load_metrics(json_path: Path) -> Dict[str, float]:
    with json_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def ensure_hyperparam_header(path: Path, headers: Sequence[str]) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(headers)


def update_hyperparams(path: Path, headers: Sequence[str], row: Dict[str, str], key_fields: Sequence[str]) -> None:
    rows: List[Dict[str, str]] = []
    found = False
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for existing in reader:
                if all(existing.get(field, "") == row.get(field, "") for field in key_fields):
                    rows.append(row)
                    found = True
                else:
                    rows.append(existing)
    if not found:
        rows.append(row)

    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(metrics: Dict[str, float], key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return ""
    try:
        return f"{float(value):.5f}"
    except (TypeError, ValueError):
        return ""


def build_hyperparam_row(metrics: Dict[str, float], base_cfg: Dict, variant_value: int, steps: int) -> Dict[str, str]:
    student_cfg = base_cfg["student"]
    resolution = int(student_cfg.get("grid_resolution", [6, 6, 6])[0])
    num_layers = int(student_cfg.get("num_layers", 4))
    hidden_dim = int(variant_value)
    ray_chunk = int(base_cfg["data"].get("ray_chunk", base_cfg["data"].get("batch_size", 0)))

    headers = [
        "scene",
        "psnr",
        "ssim",
        "lpips",
        "fps",
        "gpu_peak",
        "power_avg",
        "steps",
        "resolution",
        "hidden_dim",
        "num_layers",
        "proj_dimK",
        "alpha_T",
        "alpha_lambda",
        "blend_tau",
        "blend_eps",
        "ray_chunk",
    ]

    row = {key: "" for key in headers}
    row.update(
        {
            "scene": "lego",
            "psnr": _format_metric(metrics, "psnr"),
            "ssim": _format_metric(metrics, "ssim"),
            "lpips": _format_metric(metrics, "lpips"),
            "fps": _format_metric(metrics, "avg_fps"),
            "gpu_peak": _format_metric(metrics, "gpu_peak_gib"),
            "power_avg": _format_metric(metrics, "power_avg_w"),
            "steps": str(steps),
            "resolution": str(resolution),
            "hidden_dim": str(hidden_dim),
            "num_layers": str(num_layers),
            "proj_dimK": str(hidden_dim),
            "ray_chunk": str(ray_chunk),
        }
    )
    return row


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    base_cfg = load_config(BASE_CONFIG)
    values = args.values or tuple(default_values("hidden_dim"))
    headers = [
        "scene",
        "psnr",
        "ssim",
        "lpips",
        "fps",
        "gpu_peak",
        "power_avg",
        "steps",
        "resolution",
        "hidden_dim",
        "num_layers",
        "proj_dimK",
        "alpha_T",
        "alpha_lambda",
        "blend_tau",
        "blend_eps",
        "ray_chunk",
    ]
    ensure_hyperparam_header(HYPER_PARAM_CSV, headers)
    key_fields = ("scene", "steps", "resolution", "hidden_dim", "num_layers")

    for value in values:
        info = resolve_variant_info(base_cfg, value, args.max_steps)
        print(f"[variant] hidden_dim={value} checkpoint={info.checkpoint}")

        if not info.checkpoint.exists():
            print(f"  -> checkpoint missing, skipping", file=sys.stderr)
            continue

        if args.skip_existing:
            with HYPER_PARAM_CSV.open("r", newline="", encoding="utf-8") as fp:
                reader = csv.DictReader(fp)
                if any(
                    row.get("scene") == "lego"
                    and row.get("steps") == str(args.max_steps)
                    and row.get("hidden_dim") == str(value)
                    for row in reader
                ):
                    print("  -> metrics already recorded, skipping")
                    continue

        if args.dry_run:
            print("  -> dry-run: would render and evaluate")
            continue

        run_render(info, args)
        metrics_path = run_eval(info, args)
        metrics = load_metrics(metrics_path)
        hyper_row = build_hyperparam_row(metrics, base_cfg, value, args.max_steps)
        update_hyperparams(HYPER_PARAM_CSV, headers, hyper_row, key_fields)
        print(f"  -> metrics recorded for hidden_dim={value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
