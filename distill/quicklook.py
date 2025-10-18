"""Utilities to generate training quicklook plots."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class QuicklookGenerationError(RuntimeError):
    """Raised when quicklook generation fails."""


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        stripped = value.strip()
    except AttributeError:
        return None
    if stripped == "":
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _rolling_average(values: Sequence[float], window: int) -> List[float]:
    window = max(int(window), 1)
    if window <= 1:
        return list(values)
    buffer: List[float] = []
    averaged: List[float] = []
    for value in values:
        buffer.append(value)
        if len(buffer) > window:
            buffer.pop(0)
        finite = [item for item in buffer if math.isfinite(item)]
        if not finite:
            averaged.append(float("nan"))
        else:
            averaged.append(sum(finite) / len(finite))
    return averaged


def _slice_recent(series: Sequence[Tuple[int, float]], recent_steps: Optional[int]) -> List[Tuple[int, float]]:
    if recent_steps is None or recent_steps <= 0:
        return list(series)
    if not series:
        return []
    cutoff = series[-1][0] - int(recent_steps)
    idx = 0
    for idx, (step, _) in enumerate(series):
        if step >= cutoff:
            break
    return list(series[idx:])


def _extract_series(rows: Iterable[Dict[str, str]], keys: Sequence[str]) -> Dict[str, List[Tuple[int, float]]]:
    storage: Dict[str, List[Tuple[int, float]]] = {key: [] for key in keys}
    for row in rows:
        step_raw = row.get("step")
        if step_raw is None:
            continue
        try:
            step_val = int(float(step_raw))
        except ValueError:
            continue
        for key in keys:
            raw_value = row.get(key)
            parsed = _parse_float(raw_value)
            if parsed is None:
                continue
            storage.setdefault(key, []).append((step_val, parsed))
    # Filter empty entries to avoid plotting legends for absent data.
    return {key: values for key, values in storage.items() if values}


def generate_quicklook(
    metrics_csv: Path,
    output_path: Path,
    *,
    recent_steps: Optional[int] = None,
    rolling: Optional[int] = None,
    dpi: int = 160,
) -> Path:
    """Create a compact PNG summarising key training metrics."""

    metrics_csv = Path(metrics_csv)
    if not metrics_csv.exists():
        raise QuicklookGenerationError(f"Metrics CSV not found: {metrics_csv}")

    try:
        with metrics_csv.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            rows = list(reader)
    except Exception as err:  # pragma: no cover - unexpected I/O errors
        raise QuicklookGenerationError(f"Failed to read metrics CSV: {err}") from err

    if not rows:
        raise QuicklookGenerationError("Metrics CSV is empty; quicklook has nothing to plot.")

    series = _extract_series(
        rows,
        (
            "total",
            "alpha_mean",
            "alpha_fraction_ge95",
            "alpha_fraction_le05",
            "feature_mask_fraction",
            "feature_mask_threshold",
            "opacity_target_weight_effective",
            "opacity_target_weight_base",
            "alpha_guard_penalty",
            "alpha_guard_avg_penalty",
            "alpha_penalty_weight",
        ),
    )

    if not series:
        raise QuicklookGenerationError(f"Selected metrics absent from CSV: {metrics_csv}")

    recent = {key: _slice_recent(values, recent_steps) for key, values in series.items()}

    steps_reference: Optional[List[int]] = None
    for values in recent.values():
        if values:
            steps_reference = [step for step, _ in values]
            break
    if steps_reference is None:
        raise QuicklookGenerationError("No valid numeric entries to plot.")

    def _prepare(key: str) -> Tuple[List[int], List[float]]:
        items = recent.get(key, [])
        steps = [item[0] for item in items]
        vals = [item[1] for item in items]
        if rolling and len(vals) > 1:
            vals = _rolling_average(vals, rolling)
        return steps, vals

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=dpi)

    # Panel 1: Total loss
    steps_total, values_total = _prepare("total")
    if steps_total:
        axes[0, 0].plot(steps_total, values_total, label="total", color="#2962ff")
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_ylabel("Total")
        axes[0, 0].grid(alpha=0.2)
        axes[0, 0].legend()

    # Panel 2: Alpha coverage
    steps_alpha, alpha_mean_vals = _prepare("alpha_mean")
    _, alpha_hi_vals = _prepare("alpha_fraction_ge95")
    _, alpha_lo_vals = _prepare("alpha_fraction_le05")
    if steps_alpha:
        axes[0, 1].plot(steps_alpha, alpha_mean_vals, label="alpha_mean", color="#43a047")
        if alpha_hi_vals:
            axes[0, 1].plot(steps_alpha, alpha_hi_vals, label="alpha>=0.95", color="#f57f17")
        if alpha_lo_vals:
            axes[0, 1].plot(steps_alpha, alpha_lo_vals, label="alpha<=0.05", color="#6a1b9a")
        axes[0, 1].set_title("Alpha Coverage")
        axes[0, 1].set_ylabel("ratio")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.2)

    # Panel 3: Mask behaviour
    steps_mask, mask_values = _prepare("feature_mask_fraction")
    _, mask_threshold_vals = _prepare("feature_mask_threshold")
    if steps_mask:
        axes[1, 0].plot(steps_mask, mask_values, label="mask fraction", color="#00838f")
        if mask_threshold_vals:
            axes[1, 0].plot(steps_mask, mask_threshold_vals, label="mask threshold", color="#d81b60")
        axes[1, 0].set_title("Mask Monitor")
        axes[1, 0].set_ylabel("value")
        axes[1, 0].legend()
    axes[1, 0].set_xlabel("step")
    axes[1, 0].grid(alpha=0.2)

    # Panel 4: Opacity and guard
    steps_weight, weight_vals = _prepare("opacity_target_weight_effective")
    base_steps, weight_base_vals = _prepare("opacity_target_weight_base")
    penalty_steps, penalty_vals = _prepare("alpha_guard_penalty")
    penalty_weight_steps, penalty_weight_vals = _prepare("alpha_penalty_weight")
    guard_avg_steps, guard_avg_vals = _prepare("alpha_guard_avg_penalty")
    if steps_weight:
        axes[1, 1].plot(steps_weight, weight_vals, label="opacity target (eff)", color="#1e88e5")
        if weight_base_vals:
            axes[1, 1].plot(base_steps, weight_base_vals, label="opacity target (base)", color="#4caf50")
        if penalty_vals:
            axes[1, 1].plot(penalty_steps, penalty_vals, label="alpha penalty", color="#ff7043")
        if penalty_weight_vals:
            axes[1, 1].plot(
                penalty_weight_steps,
                penalty_weight_vals,
                label="penalty weight",
                color="#5e35b1",
            )
        if guard_avg_vals:
            axes[1, 1].plot(guard_avg_steps, guard_avg_vals, label="penalty avg", color="#00897b")
        axes[1, 1].set_title("Opacity Guard")
        axes[1, 1].set_ylabel("value")
        axes[1, 1].set_xlabel("step")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.2)

    for ax in axes.flat:
        if steps_reference:
            ax.set_xlim(steps_reference[0], steps_reference[-1])

    fig.tight_layout()
    try:
        fig.savefig(output_path, dpi=dpi)
    except Exception as err:  # pragma: no cover - filesystem errors
        raise QuicklookGenerationError(f"Failed to save quicklook PNG: {err}") from err
    finally:
        plt.close(fig)

    return output_path


__all__ = ["generate_quicklook", "QuicklookGenerationError"]
