"""Utility to sweep Stage 2 Gaussian SH feature distillation hyperparameters.

This helper clones a base YAML config, overrides the feature mask threshold and
projector output dimensionality, launches the smoketest training run, and
summarises the resulting metrics CSV.
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
from typing import Iterable, List

import yaml


@dataclass
class SweepResult:
    mask_threshold: float
    projector_dim: int
    metrics_csv: Path
    final_row: dict[str, str]

    def to_csv_row(self) -> List[str]:
        keys = (
            "step",
            "total",
            "color",
            "opacity",
            "depth",
            "feature_recon",
            "feature_cosine",
            "feature_mask_fraction",
        )
        row = [
            f"{self.mask_threshold:.2f}",
            str(self.projector_dim),
            str(self.metrics_csv),
        ]
        for key in keys:
            row.append(self.final_row.get(key, ""))
        return row

    @staticmethod
    def csv_header() -> List[str]:
        keys = (
            "step",
            "total",
            "color",
            "opacity",
            "depth",
            "feature_recon",
            "feature_cosine",
            "feature_mask_fraction",
        )
        return ["mask_threshold", "projector_dim", "metrics_csv", *keys]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/lego_response_stage2_kilo_feature_gaussian_sh_smoketest.yaml"),
        help="Base YAML config to clone for each sweep variant.",
    )
    parser.add_argument(
        "--mask-thresholds",
        type=float,
        nargs="+",
        default=[0.60, 0.75, 0.90],
        help="Boundary mask thresholds to evaluate.",
    )
    parser.add_argument(
        "--projector-dims",
        type=int,
        nargs="+",
        default=[48],
        help="Projector output dimensionalities to try.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("logs/lego/stage2_kilo_feature_gaussian_sh_smoketest/sweep_results.csv"),
        help="Destination CSV for aggregated sweep results.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip variants whose metrics CSV already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing training runs.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def dump_yaml(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(config, fp, sort_keys=False)


def format_threshold(threshold: float) -> str:
    return f"{int(round(threshold * 100)):03d}"


def apply_suffix_to_path(path_str: str, suffix: str) -> str:
    path = Path(path_str)
    return str(path.parent / suffix / path.name)


def prepare_variant_config(
    base_cfg: dict,
    mask_threshold: float,
    projector_dim: int,
    suffix: str,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("feature_pipeline", {})
    cfg["feature_pipeline"]["boundary_mask_threshold"] = float(mask_threshold)
    cfg["feature_pipeline"]["projector_output_dim"] = int(projector_dim)

    experiment = cfg.setdefault("experiment", {})
    base_name = experiment.get("name", "experiment")
    experiment["name"] = f"{base_name}_{suffix}"
    output_dir = Path(experiment.get("output_dir", "results"))
    experiment["output_dir"] = str(output_dir.parent / f"{output_dir.name}_{suffix}")

    logging_cfg = cfg.setdefault("logging", {})
    tensorboard = logging_cfg.get("tensorboard", "logs/tensorboard")
    csv_path = logging_cfg.get("csv", "logs/metrics.csv")
    logging_cfg["tensorboard"] = apply_suffix_to_path(tensorboard, suffix)
    logging_cfg["csv"] = apply_suffix_to_path(csv_path, suffix)

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


def summarise_runs(results: Iterable[SweepResult], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = list(results)
    with output_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(SweepResult.csv_header())
        for result in rows:
            writer.writerow(result.to_csv_row())


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(args.base_config)

    mask_thresholds = sorted(set(args.mask_thresholds))
    projector_dims = sorted(set(args.projector_dims))

    results: List[SweepResult] = []

    with tempfile.TemporaryDirectory(prefix="gaussian_sh_sweep_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        for mask_threshold in mask_thresholds:
            for projector_dim in projector_dims:
                suffix = f"mask{format_threshold(mask_threshold)}_proj{projector_dim:03d}"
                cfg = prepare_variant_config(base_cfg, mask_threshold, projector_dim, suffix)
                cfg_path = tmpdir / f"config_{suffix}.yaml"
                dump_yaml(cfg, cfg_path)
                metrics_csv = Path(cfg["logging"]["csv"]).resolve()

                if args.skip_existing and metrics_csv.exists():
                    print(f"[skip] {suffix} already has metrics at {metrics_csv}")
                    last_row = read_last_row(metrics_csv)
                    results.append(SweepResult(mask_threshold, projector_dim, metrics_csv, last_row))
                    continue

                print(f"[run] {suffix} → config {cfg_path}")
                print(f"      metrics → {metrics_csv}")
                if args.dry_run:
                    continue

                run_training(cfg_path)

                if not metrics_csv.exists():
                    raise FileNotFoundError(f"Expected metrics CSV missing: {metrics_csv}")
                last_row = read_last_row(metrics_csv)
                results.append(SweepResult(mask_threshold, projector_dim, metrics_csv, last_row))

    if results:
        summarise_runs(results, args.output_summary)
        print(f"[summary] Wrote aggregated results to {args.output_summary}")
    else:
        print("No runs executed; nothing to summarise.")


if __name__ == "__main__":
    main()
