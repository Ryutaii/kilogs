#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

DEFAULT_OUTDIR = Path(
    "results/lego/feat_t_full/runs/teacher_full_rehab_masked_white/renders/step_001000"
)
DEFAULT_CSV = DEFAULT_OUTDIR / "rgba_white" / "metrics_per_frame_psnr.csv"
DEFAULT_STUDENT = DEFAULT_OUTDIR / "rgba_white" / "renders"
DEFAULT_TEACHER = Path("teacher/outputs/lego/test_white/ours_30000/renders")
DEFAULT_DEST = DEFAULT_OUTDIR / "rgba_white" / "debug_diffs"
TOP_N = 16


def read_psnr_table(path: Path) -> List[Tuple[str, float]]:
    class _FallbackDialect(csv.Dialect):
        delimiter = "\t"
        quotechar = '"'
        doublequote = True
        skipinitialspace = False
        lineterminator = "\n"
        quoting = csv.QUOTE_MINIMAL

    rows: List[Tuple[str, float]] = []
    with path.open("r", newline="") as fp:
        sample = fp.read(4096)
        fp.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t,; ")
        except csv.Error:
            dialect = _FallbackDialect()
        reader = csv.reader(fp, dialect)
        for record in reader:
            if not record:
                continue
            if record[0].strip().lower() in {"name", "file", "filename"}:
                continue
            if len(record) < 2:
                continue
            name = record[0].strip()
            raw_value = record[1].strip()
            try:
                psnr = float(raw_value)
            except ValueError:
                try:
                    psnr = float(raw_value.split("=")[-1])
                except ValueError:
                    continue
            rows.append((name, psnr))
    return rows


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
    return array / 255.0


def to_u8_image(array: np.ndarray) -> Image.Image:
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def ensure_exists(path: Path, kind: str) -> None:
    if not path.exists():
        raise SystemExit(f"[ERR] missing {kind}: {path}")


def main(
    csv_path: Path,
    student_dir: Path,
    teacher_dir: Path,
    dest_dir: Path,
    top_n: int = TOP_N,
) -> None:
    ensure_exists(csv_path, "metrics table")
    ensure_exists(student_dir, "student render directory")
    ensure_exists(teacher_dir, "teacher render directory")

    rows = read_psnr_table(csv_path)
    if not rows:
        raise SystemExit(f"[ERR] no (name, psnr) rows in {csv_path}. Check file format.")

    rows.sort(key=lambda item: item[1])
    selected = rows[:top_n]

    dest_dir.mkdir(parents=True, exist_ok=True)

    for name, _ in selected:
        student_path = student_dir / name
        teacher_path = teacher_dir / name
        ensure_exists(student_path, "student frame")
        ensure_exists(teacher_path, "teacher frame")

        student = load_rgb(student_path)
        teacher = load_rgb(teacher_path)
        diff = np.abs(student - teacher).mean(axis=2, keepdims=True)
        diff = np.repeat(diff, 3, axis=2)
        diff = np.power(diff, 0.5)

        triptych = np.concatenate([teacher, student, diff], axis=1)
        output_name = f"{name.replace('.png', '')}_triptych.png"
        to_u8_image(triptych).save(dest_dir / output_name)

    print(f"[done] saved {len(selected)} triptychs -> {dest_dir}")


if __name__ == "__main__":
    main(DEFAULT_CSV, DEFAULT_STUDENT, DEFAULT_TEACHER, DEFAULT_DEST)
