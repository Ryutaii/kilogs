#!/usr/bin/env python3
"""Quick inspection utility for teacher depth buffers.

This helper loads the LEGO response distillation config and summarises the
teacher-rendered depth maps. It reports valid-pixel coverage, min/max ranges,
and simple quantiles so we can reason about scaling and masking strategies.

Example usage:
    python tools/inspect_teacher_depth.py --config configs/lego_response_stage1a_50k_depth.yaml

Use --max-frames to limit the inspection to the first N frames when the depth
folder is large, and --per-frame N to print the N frames with the lowest depth
coverage for quick debugging.
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from distill.lego_response_distill import LegoRayDataset, parse_config


_TORCH = None


def _require_torch():
    global _TORCH
    if _TORCH is None:
        try:
            _TORCH = importlib.import_module("torch")
        except ImportError as exc:  # pragma: no cover - convenience guard
            raise ImportError(
                "PyTorch is required to inspect teacher depth buffers. Install torch in the kilogs environment."
            ) from exc
    return _TORCH


def _gather_depth_values(
    dataset: LegoRayDataset, max_frames: int | None
) -> Tuple[List[Any], List[Dict[str, float]]]:
    torch = _require_torch()

    depth_batches: List[Any] = []
    per_frame: List[Dict[str, float]] = []

    frames_with_depth = 0
    total_frames = len(dataset.teacher_depth)

    for frame_idx, depth_tensor in enumerate(dataset.teacher_depth):
        if depth_tensor is None:
            per_frame.append(
                {
                    "frame": float(frame_idx),
                    "coverage": 0.0,
                    "min": math.nan,
                    "max": math.nan,
                    "mean": math.nan,
                }
            )
            continue

        if max_frames is not None and frames_with_depth >= max_frames:
            break

        depth_values = depth_tensor.flatten().float()
        valid_mask = torch.isfinite(depth_values) & (depth_values > 0.0)
        coverage = valid_mask.float().mean().item()

        if valid_mask.any():
            valid_depth = depth_values[valid_mask]
            depth_batches.append(valid_depth)
            per_frame.append(
                {
                    "frame": float(frame_idx),
                    "coverage": coverage,
                    "min": valid_depth.min().item(),
                    "max": valid_depth.max().item(),
                    "mean": valid_depth.mean().item(),
                }
            )
        else:
            per_frame.append(
                {
                    "frame": float(frame_idx),
                    "coverage": coverage,
                    "min": math.nan,
                    "max": math.nan,
                    "mean": math.nan,
                }
            )
        frames_with_depth += 1

    return depth_batches, per_frame


def _summary_line(label: str, value: float, unit: str = "") -> str:
    if math.isnan(value):
        numeric = "nan"
    else:
        numeric = f"{value:.6f}"
    return f"{label:<20}: {numeric}{unit}"


def main() -> None:
    torch = _require_torch()

    parser = argparse.ArgumentParser(description="Inspect teacher depth buffers")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a response-distillation YAML config",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Inspect at most this many frames with valid depth (0 = all)",
    )
    parser.add_argument(
        "--per-frame",
        type=int,
        default=5,
        help="Show the N frames with the lowest coverage (0 to disable)",
    )
    args = parser.parse_args()

    config_path = args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    (
        experiment_cfg,
        data_cfg,
        _teacher_cfg,
        _student_cfg,
        _train_cfg,
        _loss_cfg,
        _logging_cfg,
        _feature_cfg,
        _feature_aux_cfg,
    ) = parse_config(config_path)

    dataset = LegoRayDataset(data_cfg)
    max_frames = args.max_frames if args.max_frames > 0 else None

    depth_batches, per_frame_stats = _gather_depth_values(dataset, max_frames)

    total_frames = len(dataset.teacher_depth)
    frames_with_depth = sum(1 for stats in per_frame_stats if not math.isnan(stats["coverage"]))

    print("\nTeacher depth overview")
    print("----------------------")
    print(f"Total frames          : {total_frames}")
    print(f"Frames with depth     : {frames_with_depth}")

    if not depth_batches:
        print("No valid depth values were found. Ensure teacher depth exports are available.")
        return

    concatenated = torch.cat(depth_batches)
    near = getattr(dataset, "near", float("nan"))
    far = getattr(dataset, "far", float("nan"))

    stats = {
        "min": concatenated.min().item(),
        "max": concatenated.max().item(),
        "mean": concatenated.mean().item(),
        "median": concatenated.median().item(),
        "q05": torch.quantile(concatenated, torch.tensor(0.05)).item(),
        "q95": torch.quantile(concatenated, torch.tensor(0.95)).item(),
    }

    for key, value in stats.items():
        print(_summary_line(f"Depth {key}", value, " m"))

    if not math.isnan(near) and not math.isnan(far):
        scale = far - near
        if scale > 0:
            normalized_min = (stats["min"] - near) / scale
            normalized_max = (stats["max"] - near) / scale
            print(
                _summary_line(
                    "Normalized range", normalized_min, " to {:.6f}".format(normalized_max)
                )
            )

    if args.per_frame > 0:
        sorted_frames = sorted(per_frame_stats, key=lambda item: item["coverage"])
        print("\nFrames with lowest depth coverage")
        print("---------------------------------")
        for entry in sorted_frames[: args.per_frame]:
            coverage_pct = entry["coverage"] * 100.0 if not math.isnan(entry["coverage"]) else float("nan")
            min_val = entry["min"]
            max_val = entry["max"]
            print(
                f"Frame {int(entry['frame']):03d}: coverage {coverage_pct:5.2f}% | min {min_val:.6f} m | max {max_val:.6f} m"
            )

    print("\nDone.")


if __name__ == "__main__":
    torch = _require_torch()
    torch.set_grad_enabled(False)
    main()
