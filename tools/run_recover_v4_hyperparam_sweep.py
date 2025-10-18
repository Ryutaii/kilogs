"""Sweep key student hyperparameters for the recover_v4 teacher (SH + alpha + log-scale).

The tool clones the base recover_v4 YAML config, overrides one hyperparameter
axis at a time, launches 10k smoke-test training runs (by default), and records
where the metrics CSV for each variant will be written.  It mirrors the staged
workflow described in docs/research_notes.md: first sweep hidden_dim, then
num_layers, then resolution r.
"""
from __future__ import annotations

import argparse
import copy
import csv
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import yaml

BASE_CONFIG = Path("configs/lego_feature_teacher_full_quickwin_relaxed_alpha045_recover_v4.yaml")
DEFAULT_OUTPUT = Path("logs/lego/feat_t_full/hparam_sweeps/recover_v4")

_HIDDEN_DIM_VALUES = (96, 128, 160, 192)
_NUM_LAYER_VALUES = (3, 4, 5)
_RESOLUTION_VALUES = (4, 6, 8, 10)

_AXIS_PREFIX = {
    "hidden_dim": "h",
    "num_layers": "l",
    "resolution": "r",
}


@dataclass
class SweepResult:
    axis: str
    value: int
    metrics_csv: Path
    last_row: dict[str, str]

    def to_row(self) -> List[str]:
        keys = (
            "step",
            "total",
            "color",
            "opacity",
            "depth",
            "feature",
            "feature_cosine",
            "feature_mask_fraction",
        )
        row = [self.axis, str(self.value), str(self.metrics_csv)]
        for key in keys:
            row.append(self.last_row.get(key, ""))
        return row

    @staticmethod
    def header() -> List[str]:
        keys = (
            "step",
            "total",
            "color",
            "opacity",
            "depth",
            "feature",
            "feature_cosine",
            "feature_mask_fraction",
        )
        return ["axis", "value", "metrics_csv", *keys]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--axis",
        choices=("hidden_dim", "num_layers", "resolution"),
        required=True,
        help="Hyperparameter axis to sweep.",
    )
    parser.add_argument(
        "--values",
        type=int,
        nargs="*",
        help=(
            "Explicit values to sweep. Default depends on axis: hidden_dim="
            f"{_HIDDEN_DIM_VALUES}, num_layers={_NUM_LAYER_VALUES}, resolution={_RESOLUTION_VALUES}."
        ),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=BASE_CONFIG,
        help="Base recover_v4 YAML config.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=DEFAULT_OUTPUT / "summary.csv",
        help="CSV file to write aggregated metrics.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Override train.max_steps for the generated configs (default: 10000).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip variants whose metrics CSV already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without launching training runs.",
    )
    return parser.parse_args()


def default_values(axis: str) -> Sequence[int]:
    if axis == "hidden_dim":
        return _HIDDEN_DIM_VALUES
    if axis == "num_layers":
        return _NUM_LAYER_VALUES
    if axis == "resolution":
        return _RESOLUTION_VALUES
    raise ValueError(axis)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def save_config(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(config, fp, sort_keys=False)


def format_suffix(axis: str, value: int) -> str:
    label = {
        "hidden_dim": "hd",
        "num_layers": "nl",
        "resolution": "r",
    }[axis]
    return f"{label}{value:03d}"


def format_step_label(steps: int) -> str:
    if steps <= 0:
        return str(steps)
    if steps % 1000 == 0:
        return f"{steps // 1000}k"
    if steps >= 1000:
        return f"{steps / 1000:.1f}k"
    return str(steps)


def format_axis_token(axis: str, value: int) -> str:
    prefix = _AXIS_PREFIX.get(axis, axis[:1])
    return f"{prefix}{value}"


def override_config(config: dict, axis: str, value: int, max_steps: int, suffix: str) -> dict:
    cfg = copy.deepcopy(config)

    # Experiment metadata
    experiment = cfg.setdefault("experiment", {})
    base_name = experiment.get("name", "experiment")
    experiment["name"] = f"{base_name}_{suffix}"
    output_dir = Path(experiment.get("output_dir", "results/experiment"))
    experiment["output_dir"] = str(output_dir.parent / f"{output_dir.name}_{suffix}")

    # Logging paths
    logging_cfg = cfg.setdefault("logging", {})
    for key in ("tensorboard", "csv"):
        if key in logging_cfg:
            path = Path(logging_cfg[key])
            logging_cfg[key] = str(path.parent / suffix / path.name)

    # Training schedule override
    train_cfg = cfg.setdefault("train", {})
    train_cfg["max_steps"] = int(max_steps)
    # Keep eval/checkpoint intervals <= max_steps
    for key in ("eval_interval", "checkpoint_interval"):
        interval = int(train_cfg.get(key, max_steps))
        train_cfg[key] = min(interval, int(max_steps))

    student = cfg.setdefault("student", {})
    feature_cfg = cfg.setdefault("feature_pipeline", {})
    stud_proj = feature_cfg.setdefault("student_projector", {})

    if axis == "hidden_dim":
        student["hidden_dim"] = int(value)
        # Keep projector aligned with new penultimate dimension
        feature_cfg["projector_input_dim"] = int(value)
        feature_cfg["projector_hidden_dim"] = int(value)
        feature_cfg["projector_output_dim"] = int(value)
        stud_proj["input_dim"] = int(value)
        stud_proj["hidden_dim"] = int(value)
        stud_proj["output_dim"] = int(value)
        # Allow the student projector to keep its expanded width and adapt the teacher instead.
        feature_cfg["allow_dim_mismatch"] = True
        teacher_adapter_cfg = feature_cfg.setdefault("teacher_adapter", {})
        teacher_adapter_cfg.setdefault("type", "linear")
        teacher_adapter_cfg["output_dim"] = int(value)
    elif axis == "num_layers":
        student["num_layers"] = int(value)
    elif axis == "resolution":
        student["grid_resolution"] = [int(value)] * 3
    else:
        raise ValueError(axis)

    student_hidden_dim = int(student.get("hidden_dim", config.get("student", {}).get("hidden_dim", value)))
    teacher_feature_dim = 52
    step_label = format_step_label(max_steps)
    axis_token = format_axis_token(axis, int(value))
    experiment["progress_desc"] = f"特徴蒸留のみ_教師{teacher_feature_dim}_{axis_token}_{step_label}"

    return cfg


def run_training(config_path: Path) -> None:
    cmd = [sys.executable, "-m", "distill.lego_response_distill", "--config", str(config_path)]
    subprocess.run(cmd, check=True)


def read_last_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
        if not rows:
            raise RuntimeError(f"No rows found in metrics CSV: {csv_path}")
        return rows[-1]


def summarise(results: Iterable[SweepResult], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = list(results)
    if not rows:
        return
    with destination.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(SweepResult.header())
        for result in rows:
            writer.writerow(result.to_row())


def main() -> None:
    args = parse_args()
    values = args.values or tuple(default_values(args.axis))
    base_cfg = load_config(args.base_config)
    value_list = ", ".join(str(v) for v in values)
    print(f"[sweep] axis={args.axis}, values=[{value_list}]")

    results: List[SweepResult] = []
    with tempfile.TemporaryDirectory(prefix=f"recover_v4_{args.axis}_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        for value in values:
            suffix = format_suffix(args.axis, value)
            variant_cfg = override_config(base_cfg, args.axis, value, args.max_steps, suffix)
            cfg_path = tmpdir / f"{suffix}.yaml"
            save_config(variant_cfg, cfg_path)

            metrics_csv = Path(variant_cfg["logging"]["csv"]).resolve()
            if args.skip_existing and metrics_csv.exists():
                print(f"[skip] {suffix} metrics already at {metrics_csv}")
                last_row = read_last_row(metrics_csv)
                results.append(SweepResult(args.axis, value, metrics_csv, last_row))
                continue

            print(f"[run] {suffix} ({args.axis}={value}) → config {cfg_path}")
            print(f"      metrics → {metrics_csv}")
            if args.dry_run:
                continue

            run_training(cfg_path)

            if not metrics_csv.exists():
                raise FileNotFoundError(f"Expected metrics CSV missing: {metrics_csv}")
            last_row = read_last_row(metrics_csv)
            results.append(SweepResult(args.axis, value, metrics_csv, last_row))

    summarise(results, args.output_summary)
    if results:
        print(f"[summary] Aggregated sweep metrics → {args.output_summary}")
    else:
        print("No runs executed; nothing to summarise.")


if __name__ == "__main__":
    main()
