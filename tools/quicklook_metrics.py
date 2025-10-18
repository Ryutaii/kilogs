"""Generate compact diagnostic plots from training metrics CSV (and optional PSNR logs)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _read_metrics(csv_path: Path) -> Dict[str, List[float]]:
    columns: Dict[str, List[float]] = {}
    with csv_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            for key, value in row.items():
                if key in {"timestamp", "_eor_checksum"}:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                columns.setdefault(key, []).append(numeric)
    return columns


def _normalise(series: List[float]) -> List[float]:
    if not series:
        return series
    minimum = min(series)
    maximum = max(series)
    if maximum - minimum < 1e-8:
        return [0.5 for _ in series]
    span = maximum - minimum
    return [(value - minimum) / span for value in series]


def plot_quicklook(metrics: Dict[str, List[float]], output_path: Path) -> None:
    steps = metrics.get("step")
    if not steps:
        raise RuntimeError("metrics CSV did not contain a 'step' column")

    series_map = {
        "loss_total": metrics.get("total"),
        "loss_color": metrics.get("color"),
        "feature_recon": metrics.get("feature_recon"),
        "feature_mask_fraction": metrics.get("feature_mask_fraction"),
        "opacity_weight": metrics.get("opacity_target_weight_effective"),
    }

    plt.figure(figsize=(8, 4.5))
    for label, values in series_map.items():
        if not values:
            continue
        normalised = _normalise(values)
        plt.plot(steps[: len(normalised)], normalised, label=label)
    plt.title("Quicklook (normalised)")
    plt.xlabel("step")
    plt.ylabel("normalised value")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _read_psnr(csv_path: Path) -> Dict[int, float]:
    psnr_map: Dict[int, float] = {}
    with csv_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                step = int(float(row.get("step", "0")))
                psnr = float(row.get("psnr", row.get("PSNR", "nan")))
            except (TypeError, ValueError):
                continue
            psnr_map[step] = psnr
    return psnr_map


def plot_mask_vs_psnr(
    metrics: Dict[str, List[float]],
    psnr_map: Dict[int, float],
    output_path: Path,
    *,
    step_stride: int = 1000,
) -> None:
    steps = metrics.get("step") or []
    mask_fraction = metrics.get("feature_mask_fraction") or []
    if not steps or not mask_fraction:
        raise RuntimeError("metrics CSV did not contain required columns for scatter plot")

    points: List[Tuple[float, float]] = []
    for idx, step in enumerate(steps):
        if int(step) % step_stride != 0:
            continue
        psnr = psnr_map.get(int(step))
        if psnr is None:
            continue
        if idx < len(mask_fraction):
            points.append((mask_fraction[idx], psnr))

    if not points:
        raise RuntimeError("No overlapping step entries found between mask metrics and PSNR data")

    xs, ys = zip(*points)
    plt.figure(figsize=(5.5, 4.5))
    plt.scatter(xs, ys, c=ys, cmap="viridis", s=24, edgecolor="none")
    plt.title("Mask fraction vs PSNR (stride {})".format(step_stride))
    plt.xlabel("feature_mask_fraction")
    plt.ylabel("PSNR (dB)")
    plt.grid(alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_csv", type=Path, help="Training metrics CSV path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to store output plots (default: 'plots').",
    )
    parser.add_argument("--psnr-csv", type=Path, help="Optional CSV containing columns 'step' and 'psnr'")
    parser.add_argument("--stride", type=int, default=1000, help="Step stride for the scatter plot")
    args = parser.parse_args()

    metrics = _read_metrics(args.metrics_csv)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    quicklook_path = output_dir / "quicklook_step.png"
    plot_quicklook(metrics, quicklook_path)

    if args.psnr_csv:
        psnr_map = _read_psnr(args.psnr_csv)
        scatter_path = output_dir / "mask_psnr_scatter.png"
        try:
            plot_mask_vs_psnr(metrics, psnr_map, scatter_path, step_stride=args.stride)
        except RuntimeError as err:
            print(f"[quicklook] Skipping mask/PSNR scatter: {err}")
    else:
        print("[quicklook] PSNR CSV not provided; skipping mask/PSNR scatter plot.")


if __name__ == "__main__":
    main()
