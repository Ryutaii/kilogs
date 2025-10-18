"""Sanity-check opacity target scheduler monotonicity and resume behaviour."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distill.lego_response_distill import (
    OpacityTargetScheduler,
    parse_config,
)


def check_monotonic(weights: Iterable[float], *, decreasing: bool = False, eps: float = 1e-6) -> bool:
    iterator = iter(weights)
    try:
        prev = next(iterator)
    except StopIteration:
        return True
    ok = True
    for current in iterator:
        if decreasing:
            if current > prev + eps:
                ok = False
                break
        else:
            if current + eps < prev:
                ok = False
                break
        prev = current
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=Path, help="Optional checkpoint to resume state from", default=None)
    parser.add_argument("--start-step", type=int, default=0, help="Starting global step (exclusive). Use checkpoint step if resuming.")
    parser.add_argument("--end-step", type=int, default=60000, help="Final global step to evaluate (inclusive)")
    parser.add_argument("--print", dest="print_values", action="store_true", help="Print per-step weights")
    parser.add_argument("--eps", type=float, default=1e-6, help="Tolerance for monotonic comparison")
    args = parser.parse_args()

    (
        _experiment,
        _data_cfg,
        _teacher_cfg,
        _student_cfg,
        _train_cfg,
        loss_cfg,
        _logging_cfg,
        _feature_cfg,
        _feature_aux_cfg,
    ) = parse_config(args.config)

    scheduler = OpacityTargetScheduler(loss_cfg)

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("opacity_scheduler_state")
        scheduler.load_state_dict(state)
        if "step" in ckpt and args.start_step == 0:
            args.start_step = int(ckpt["step"])

    start_step = max(int(args.start_step), 0)
    end_step = max(int(args.end_step), start_step)

    weights: List[float] = []
    for step in range(start_step + 1, end_step + 1):
        weight = scheduler.compute(step)
        weights.append(weight)
        if args.print_values:
            print(f"step={step:06d} weight={weight:.8f}")

    target_weight = float(loss_cfg.opacity_target_weight or 0.0)
    start_weight_cfg = getattr(loss_cfg, "opacity_target_start_weight", None)
    start_weight = float(start_weight_cfg) if start_weight_cfg is not None else target_weight
    increasing = target_weight >= start_weight
    monotonic_ok = check_monotonic(weights, decreasing=not increasing, eps=args.eps)

    if monotonic_ok:
        print(
            f"Monotonicity OK for steps {start_step + 1}..{end_step} (increasing={increasing}, hysteresis={loss_cfg.opacity_target_hysteresis})."
        )
    else:
        raise SystemExit(
            f"Detected monotonicity violation between steps {start_step + 1}..{end_step}. "
            "Inspect with --print to debug scheduler settings."
        )


if __name__ == "__main__":
    main()
