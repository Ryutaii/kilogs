#!/usr/bin/env python3
"""Display training progress bars from log files in a table view."""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROGRESS_LINE = re.compile(
    r"^(?P<label>[^:]+):\s+(?P<pct>\d+)%\|(?P<bar>[^|]+)\|\s+(?P<current>\d+)/(?:\s*)?(?P<total>\d+)"
)

# Default bar width keeps output concise while staying readable in terminals.
BAR_WIDTH = 24

def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show tqdm-style progress bars from one or more log files.",
    )
    parser.add_argument(
        "log",
        nargs="*",
        type=Path,
        help="One or more log files to monitor (defaults to newest train.log files).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory to search for logs when none are provided (default: current working directory).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum number of auto-detected logs to display.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Refresh interval in seconds (ignored when --once is set).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Render the table once and exit instead of refreshing.",
    )
    return parser.parse_args(list(argv))


def discover_logs(root: Path, limit: int) -> List[Path]:
    if not root.exists():
        return []
    matches: List[Tuple[float, Path]] = []
    for candidate in root.glob("**/train.log"):
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            continue
        matches.append((mtime, candidate))
    matches.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in matches[:limit]]


def read_progress(log_path: Path) -> Dict[str, Tuple[int, int, int]]:
    entries: Dict[str, Tuple[int, int, int]] = {}
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.rstrip()
                match = PROGRESS_LINE.match(line)
                if not match:
                    continue
                label = match.group("label").strip()
                pct = int(match.group("pct"))
                current = int(match.group("current"))
                total = int(match.group("total"))
                entries[label] = (pct, current, total)
    except FileNotFoundError:
        return {}
    return entries


def make_bar(pct: int, width: int = BAR_WIDTH) -> str:
    filled = int(round(pct * width / 100))
    filled = max(0, min(width, filled))
    empty = width - filled
    return "[" + "#" * filled + "-" * empty + "]"


def render_table(entries: Dict[str, Tuple[int, int, int]]) -> List[str]:
    if not entries:
        return ["(no progress lines detected yet)"]
    header = f"{'Task':<28} {'Progress':<34} {'Count':<12}"
    border = "-" * len(header)
    lines = [header, border]
    for label, (pct, current, total) in entries.items():
        bar = make_bar(pct)
        progress = f"{bar} {pct:3d}%"
        count = f"{current}/{total}"
        lines.append(f"{label:<28} {progress:<34} {count:<12}")
    return lines


def describe_log(log_path: Path) -> str:
    try:
        mtime = datetime.fromtimestamp(log_path.stat().st_mtime)
        timestamp = mtime.strftime("%Y-%m-%d %H:%M:%S")
    except FileNotFoundError:
        timestamp = "unavailable"
    run_name = log_path.parent.name
    return f"Run: {run_name} (updated: {timestamp})\nPath: {log_path}"


def clear_screen() -> None:
    sys.stdout.write("\033[H\033[J")
    sys.stdout.flush()


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logs = args.log or discover_logs(args.root, args.limit)
    logs = [p for p in logs if p.exists()]

    if not logs:
        print("No log files found. Provide a path or adjust --root.")
        return 1

    try:
        while True:
            clear_screen()
            for index, log_path in enumerate(logs, start=1):
                print(describe_log(log_path))
                table_lines = render_table(read_progress(log_path))
                for line in table_lines:
                    print(line)
                if index < len(logs):
                    print()
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
