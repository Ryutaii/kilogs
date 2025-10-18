#!/usr/bin/env python3
"""Run a short training burst to verify early-run health metrics."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distill.lego_response_distill import main as distill_main  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a short burst (default 5k steps) and emit health snapshots before full training."
    )
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Number of steps to run before evaluating the fail gate (default: 5000)",
    )
    parser.add_argument(
        "--warn-steps",
        type=int,
        default=2000,
        help="Step at which to issue health warnings (default: 2000)",
    )
    parser.add_argument(
        "--median-window",
        type=int,
        default=400,
        help="Number of recent records used when computing rolling medians (default: 400)",
    )
    parser.add_argument(
        "--no-fail-fast",
        action="store_true",
        help="Do not exit immediately when the health snapshot flags an abnormality",
    )
    return parser


def run(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    os.environ["KILOGS_HEALTHCHECK"] = "1"
    os.environ["KILOGS_HEALTHCHECK_MAX_STEPS"] = str(args.max_steps)
    os.environ["KILOGS_HEALTHCHECK_WARN_STEPS"] = str(args.warn_steps)
    os.environ["KILOGS_HEALTHCHECK_MEDIAN_WINDOW"] = str(args.median_window)
    fail_fast = not args.no_fail_fast
    os.environ["KILOGS_HEALTHCHECK_FAILFAST"] = "1" if fail_fast else "0"

    distill_main(["--config", str(args.config)])


if __name__ == "__main__":
    run()
