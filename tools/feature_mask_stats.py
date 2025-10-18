#!/usr/bin/env python3
"""Summarise feature mask statistics from a Stage 2 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from tools.inspect_feature_alignment import inspect_alignment


def _summarise(stats: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    summary["mask_fraction"] = stats.get("mask_fraction")
    summary["feature_loss"] = stats.get("feature_loss")
    summary["primary_weight_sum_stats"] = stats.get("primary_weight_sum_stats")
    summary["teacher_stats"] = stats.get("teacher_stats")
    summary["student_stats"] = stats.get("student_stats")
    summary["delta_stats"] = stats.get("delta_stats")
    if warning := stats.get("warning"):
        summary["warning"] = warning
    if info := stats.get("info"):
        summary["info"] = info
    return summary


def main() -> None:  # pragma: no cover - CLI utility
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Stage 2 training config used for the checkpoint")
    parser.add_argument("checkpoint", type=Path, help="Checkpoint path to analyse")
    parser.add_argument("--num-rays", type=int, default=8192, help="Number of random rays to sample")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override (default: same heuristic as inspect_alignment)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional destination for saving the summary JSON",
    )
    args = parser.parse_args()

    stats = inspect_alignment(
        args.config,
        args.checkpoint,
        num_rays=args.num_rays,
        device=args.device,
    )

    summary = _summarise(stats)
    print(json.dumps(summary, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)


if __name__ == "__main__":  # pragma: no cover
    torch.set_grad_enabled(False)
    main()
