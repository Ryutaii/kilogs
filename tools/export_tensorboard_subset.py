import argparse
import csv
import math
import shutil
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

# TensorBoard tags -> CSV column names
DEFAULT_TAG_MAP = {
    "loss/total": "total",
    "loss/color": "color",
    "loss/opacity": "opacity",
    "loss/depth": "depth",
    "loss/feature_recon": "feature_recon",
    "loss/feature_cosine": "feature_cosine",
    "feature_mask/fraction": "feature_mask_fraction",
    "feature_mask/threshold": "feature_mask_threshold",
    "feature_mask/weight_min": "feature_mask_weight_min",
    "opacity/target_weight_effective": "opacity_target_weight_effective",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a subset of metrics to a TensorBoard log.")
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to training_metrics.csv",
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        required=True,
        help="Destination directory for the filtered TensorBoard event files.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=tuple(DEFAULT_TAG_MAP.keys()),
        help="Optional list of TensorBoard tags to export (subset of defaults).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing log directory instead of deleting it first.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep monitoring the CSV file and stream new rows as they appear.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds when --watch is enabled.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as fp:
        reader = [row for row in csv.reader(fp) if row]
    if len(reader) <= 1:
        raise RuntimeError("CSV contains no data rows")
    header = reader[0]
    rows = reader[1:]
    column_index = {name: idx for idx, name in enumerate(header)}
    return header, rows, column_index


def to_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except ValueError:
        value_lower = value.lower()
        if value_lower == "nan":
            return None
        if value_lower in {"inf", "+inf", "infinity"}:
            return None
        if value_lower in {"-inf", "-infinity"}:
            return None
        return None
    if math.isfinite(numeric):
        return numeric
    return None


def main() -> None:
    args = parse_args()
    header, data_rows, column_index = load_rows(args.csv)

    requested_tags = []
    tag_map: dict[str, str] = {}
    for tag in args.tags:
        if tag not in DEFAULT_TAG_MAP:
            raise ValueError(f"Unsupported tag '{tag}'. Valid options: {sorted(DEFAULT_TAG_MAP)}")
        column = DEFAULT_TAG_MAP[tag]
        if column not in column_index:
            print(f"[warn] Column '{column}' missing in CSV; skipping tag '{tag}'")
            continue
        requested_tags.append(tag)
        tag_map[tag] = column

    if not requested_tags:
        raise RuntimeError("No valid tags requested; nothing to export.")

    if args.logdir.exists() and not args.append:
        shutil.rmtree(args.logdir)
    args.logdir.mkdir(parents=True, exist_ok=True)

    def export_rows(rows: list[list[str]], indices: dict[str, int], processed_steps: set[int], writer: SummaryWriter) -> int:
        total = 0
        step_column = indices.get("step")
        if step_column is None:
            raise RuntimeError("CSV is missing required 'step' column")

        for row in rows:
            step_value = row[step_column]
            try:
                step = int(step_value)
            except (TypeError, ValueError):
                continue
            if step in processed_steps:
                continue

            wrote_scalar = False
            for tag in requested_tags:
                column = tag_map[tag]
                column_idx = indices[column]
                value = to_float(row[column_idx])
                if value is None:
                    continue
                writer.add_scalar(tag, value, step)
                total += 1
                wrote_scalar = True
            if wrote_scalar:
                processed_steps.add(step)
        return total

    writer = SummaryWriter(log_dir=str(args.logdir))
    processed_steps: set[int] = set()
    total_written = export_rows(data_rows, column_index, processed_steps, writer)
    writer.flush()

    if args.watch:
        interval = max(float(args.poll_interval), 0.5)
        print(f"[watch] Monitoring {args.csv} every {interval:.1f} s — press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(interval)
                try:
                    _, new_rows, new_index = load_rows(args.csv)
                except FileNotFoundError:
                    continue
                total_written += export_rows(new_rows, new_index, processed_steps, writer)
                writer.flush()
        except KeyboardInterrupt:
            print("[watch] Stopped monitoring.")

    writer.close()
    print(f"[export] Wrote {total_written} scalar points to {args.logdir}")


if __name__ == "__main__":
    main()
