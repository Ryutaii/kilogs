"""Inspect distilled training checkpoint metadata."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch


def describe_optimizer_state(state: Dict[str, Any]) -> Dict[str, Any]:
    keys = [key for key in state if key != "param_groups"]
    return {
        "keys": keys,
        "param_groups": len(state.get("param_groups", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to saved checkpoint (.pth)")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")

    step = ckpt.get("step")
    print(f"step: {step}")

    scheduler_state = ckpt.get("opacity_scheduler_state")
    if scheduler_state is not None:
        last_weight = scheduler_state.get("last_weight")
        max_weight = scheduler_state.get("max_weight")
        print("opacity_scheduler_state:")
        print(f"  last_weight: {last_weight}")
        print(f"  max_weight: {max_weight}")
    else:
        print("opacity_scheduler_state: <missing>")

    optimizer_state = ckpt.get("optimizer_state")
    if optimizer_state is not None:
        summary = describe_optimizer_state(optimizer_state)
        print("optimizer_state:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    else:
        print("optimizer_state: <missing>")

    extra_keys = sorted(k for k in ckpt.keys() if k not in {"step", "model_state", "optimizer_state", "feature_projector_state", "student_feature_adapter_state", "teacher_feature_adapter_state", "opacity_scheduler_state"})
    if extra_keys:
        print("other keys:")
        for key in extra_keys:
            value = ckpt[key]
            if isinstance(value, (int, float, str)):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: <{type(value).__name__}>")


if __name__ == "__main__":
    main()
