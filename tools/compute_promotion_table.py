"""Utility to aggregate 10k smoke metrics and compute ΔPSNR vs baseline."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class MetricsRow:
    method: str
    psnr_white: Optional[float]
    lpips_white: Optional[float]
    step_token: Optional[str]

    @property
    def steps_display(self) -> str:
        return self.step_token or "?"


def parse_float(value: str) -> Optional[float]:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def infer_step_token(name: str, default: Optional[str] = None) -> Optional[str]:
    lowered = name.lower()
    if "step" in lowered:
        parts = lowered.split("step", 1)[1]
        token = parts.split("_")[0]
        token = token.split("-")[0]
        token = token.strip()
        if token:
            return token
    components = lowered.replace("-", "_").split("_")
    for comp in reversed(components):
        if comp.endswith("k") and comp[:-1].isdigit():
            return comp
    for comp in components:
        if comp.isdigit():
            return comp
    return default


def read_metrics(path: Path, *, step_filter: Optional[str] = None) -> list[MetricsRow]:
    rows: list[MetricsRow] = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for record in reader:
            method = (record.get("method") or "").strip()
            if not method:
                continue
            step_token = infer_step_token(method)
            if step_filter and step_token != step_filter:
                continue
            psnr_white = parse_float(record.get("psnr_white", ""))
            lpips_white = parse_float(record.get("lpips_white", ""))
            rows.append(MetricsRow(method, psnr_white, lpips_white, step_token))
    return rows


def compute_deltas(rows: Iterable[MetricsRow], baseline: str) -> tuple[Optional[MetricsRow], list[tuple[MetricsRow, Optional[float]]]]:
    baseline_row: Optional[MetricsRow] = None
    for row in rows:
        if row.method == baseline:
            baseline_row = row
            break
    deltas: list[tuple[MetricsRow, Optional[float]]] = []
    for row in rows:
        if baseline_row and baseline_row.psnr_white is not None and row.psnr_white is not None:
            delta = row.psnr_white - baseline_row.psnr_white
        else:
            delta = None
        deltas.append((row, delta))
    return baseline_row, deltas


def format_markdown_table(rows_with_delta: Iterable[tuple[MetricsRow, Optional[float]]], baseline: Optional[MetricsRow]) -> str:
    header = "| Config | Steps | PSNR (white) | ΔPSNR vs baseline | LPIPS (white) |"
    separator = "| --- | --- | --- | --- | --- |"
    lines = [header, separator]

    def fmt(value: Optional[float], precision: int = 2) -> str:
        if value is None:
            return "—"
        return f"{value:.{precision}f}"

    baseline_method = baseline.method if baseline else None

    for row, delta in rows_with_delta:
        mark = "**" if row.method == baseline_method else ""
        config = f"{mark}{row.method}{mark}"
        steps = row.steps_display
        psnr = fmt(row.psnr_white, precision=2)
        delta_fmt = fmt(delta, precision=2)
        lpips = fmt(row.lpips_white, precision=3)
        lines.append(f"| {config} | {steps} | {psnr} | {delta_fmt} | {lpips} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_csv", type=Path, help="Path to metrics_summary.csv")
    parser.add_argument("--baseline", type=str, default="FeatureDistill_recover_v2_10k", help="Method name to treat as baseline")
    parser.add_argument("--steps", type=str, default="10k", help="Step token to filter (e.g. 10k)")
    args = parser.parse_args()

    rows = read_metrics(args.metrics_csv, step_filter=args.steps)
    if not rows:
        raise SystemExit(f"No rows found for step token '{args.steps}'.")

    baseline_row, deltas = compute_deltas(rows, args.baseline)
    if baseline_row is None:
        print(f"Warning: baseline '{args.baseline}' not found; ΔPSNR will be reported as —.")

    table = format_markdown_table(deltas, baseline_row)
    print(table)


if __name__ == "__main__":
    main()
