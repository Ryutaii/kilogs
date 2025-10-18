#!/usr/bin/env python3
"""Command-line wrapper for generating quicklook PNGs from training metrics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from distill.quicklook import QuicklookGenerationError, generate_quicklook


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a quicklook PNG from training metrics CSV data.")
    parser.add_argument(
        "metrics",
        type=Path,
        help="Path to the metrics CSV file (typically training_metrics.csv).",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Destination path for the generated PNG (e.g. quicklook.png).",
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=None,
        metavar="STEPS",
        help="Only render the most recent STEPS worth of data (defaults to all).",
    )
    parser.add_argument(
        "--rolling",
        type=int,
        default=64,
        metavar="WINDOW",
        help="Apply a rolling average with WINDOW samples (default: 64).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output DPI for the PNG (default: 160).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        generate_quicklook(
            metrics_csv=args.metrics,
            output_path=args.output,
            recent_steps=args.recent,
            rolling=args.rolling,
            dpi=args.dpi,
        )
    except QuicklookGenerationError as err:
        print(f"[quicklook] generation failed: {err}", file=sys.stderr)
        return 1
    except Exception as err:  # pragma: no cover - unexpected failures
        print(f"[quicklook] unexpected failure: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
