#!/usr/bin/env python3
"""Analyse opacity map distributions and highlight saturation issues."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass
class ImageStats:
    name: str
    mean: float
    std: float
    min: float
    max: float
    quantiles: Dict[float, float]
    saturation: Dict[float, float]
    teacher_diff_mean: Optional[float] = None
    teacher_diff_max: Optional[float] = None
    background_rmse: Optional[Dict[str, float]] = None


FloatArray = np.ndarray


def _natural_key(name: str) -> List[object]:
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def list_png_pairs(student_dir: Path, teacher_dir: Optional[Path]) -> List[Tuple[Path, Optional[Path]]]:
    student_paths = sorted(student_dir.glob("**/*.png"), key=lambda p: _natural_key(str(p.relative_to(student_dir))))
    if not student_paths:
        raise FileNotFoundError(f"No PNG files found in {student_dir}")

    if teacher_dir is None:
        return [(p, None) for p in student_paths]

    student_basenames = {p.name for p in student_paths}
    teacher_candidates = [p for p in teacher_dir.glob("**/*.png") if p.name in student_basenames]
    if teacher_candidates:
        teacher_paths = sorted(teacher_candidates, key=lambda p: _natural_key(str(p.relative_to(teacher_dir))))
    else:
        teacher_paths = sorted(teacher_dir.glob("**/*.png"), key=lambda p: _natural_key(str(p.relative_to(teacher_dir))))

    teacher_lookup: Dict[str, Path] = {
        str(p.relative_to(teacher_dir)).replace("\\", "/"): p for p in teacher_paths
    }
    pairs: List[Tuple[Path, Optional[Path]]] = []
    missing_teacher: List[str] = []
    for student_path in student_paths:
        rel = str(student_path.relative_to(student_dir)).replace("\\", "/")
        teacher_path = teacher_lookup.get(rel)
        if teacher_path is None:
            missing_teacher.append(rel)
            pairs.append((student_path, None))
        else:
            pairs.append((student_path, teacher_path))

    if missing_teacher and len(student_paths) == len(teacher_paths):
        print(
            "[warn] student/teacher filenames differ; falling back to natural sort pairing"
        )
        pairs = list(zip(student_paths, teacher_paths))
        missing_teacher = []

    if missing_teacher:
        print(
            f"[warn] {len(missing_teacher)} student images lack teacher counterparts (e.g. {missing_teacher[:3]})"
        )
    return pairs


def load_opacity(path: Path) -> FloatArray:
    image = Image.open(path)
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        channels = array.shape[-1]
        if channels == 4:
            array = array[..., 3]
        elif channels >= 1:
            array = array[..., 0]
    if array.max() > 1.0:
        array = array / 255.0
    return array


def compute_statistics(
    values: FloatArray,
    *,
    thresholds: Sequence[float],
    quantiles: Sequence[float],
) -> Tuple[float, float, float, float, Dict[float, float], Dict[float, float]]:
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    quantile_dict = {q: float(np.quantile(values, q)) for q in quantiles}
    saturation = {thr: float(np.mean(values >= thr)) for thr in thresholds}
    return mean_val, std_val, min_val, max_val, quantile_dict, saturation


def aggregate_image_stats(
    pairs: Iterable[Tuple[Path, Optional[Path]]],
    *,
    thresholds: Sequence[float],
    quantiles: Sequence[float],
    student_root: Path,
    student_rgba_dir: Optional[Path],
    backgrounds: Dict[str, np.ndarray],
    background_alpha_threshold: float,
) -> Tuple[List[ImageStats], Dict[str, float]]:
    per_image: List[ImageStats] = []
    all_values: List[FloatArray] = []
    all_teacher: List[FloatArray] = []
    diff_values: List[FloatArray] = []
    background_rmse_values: Dict[str, List[float]] = {name: [] for name in backgrounds}

    for student_path, teacher_path in pairs:
        student_vals = load_opacity(student_path)
        mean_val, std_val, min_val, max_val, quantile_dict, saturation = compute_statistics(
            student_vals,
            thresholds=thresholds,
            quantiles=quantiles,
        )

        diff_mean = None
        diff_max = None
        background_metrics: Dict[str, float] = {}
        if teacher_path is not None and teacher_path.exists():
            teacher_vals = load_opacity(teacher_path)
            diff = np.abs(student_vals - teacher_vals)
            diff_mean = float(np.mean(diff))
            diff_max = float(np.max(diff))
            all_teacher.append(teacher_vals)
            diff_values.append(diff)

        if backgrounds and student_rgba_dir is not None:
            rel_npz = student_path.relative_to(student_root).with_suffix(".npz")
            npz_path = student_rgba_dir / rel_npz
            if npz_path.exists():
                data = np.load(npz_path)
                rgb = data["rgb"].astype(np.float32)
                alpha_vals = data["alpha"].astype(np.float32)
                alpha_mask = alpha_vals <= background_alpha_threshold
                for name, color in backgrounds.items():
                    if not np.any(alpha_mask):
                        continue
                    composed = rgb + (1.0 - alpha_vals[..., None]) * color
                    bg_target = np.reshape(color, (1, 1, 3))
                    diff = composed[alpha_mask] - bg_target.reshape(-1, 3)
                    rmse = float(np.sqrt(np.mean(np.square(diff))))
                    background_metrics[name] = rmse
                    background_rmse_values[name].append(rmse)
            else:
                missing_msg = getattr(aggregate_image_stats, "_missing_npz_warn", False)
                if not missing_msg:
                    print(f"[warn] Missing RGBA npz counterpart for {student_path}; skipping background RMSE")
                    setattr(aggregate_image_stats, "_missing_npz_warn", True)

        per_image.append(
            ImageStats(
                name=str(student_path.name),
                mean=mean_val,
                std=std_val,
                min=min_val,
                max=max_val,
                quantiles=quantile_dict,
                saturation=saturation,
                teacher_diff_mean=diff_mean,
                teacher_diff_max=diff_max,
                background_rmse=background_metrics if background_metrics else None,
            )
        )
        all_values.append(student_vals)

    merged = np.concatenate([v.reshape(-1) for v in all_values])
    overall_mean, overall_std, overall_min, overall_max, overall_quantiles, overall_saturation = compute_statistics(
        merged,
        thresholds=thresholds,
        quantiles=quantiles,
    )

    summary: Dict[str, float] = {
        "num_images": float(len(per_image)),
        "mean": overall_mean,
        "std": overall_std,
        "min": overall_min,
        "max": overall_max,
    }
    for q, value in overall_quantiles.items():
        summary[f"quantile_{q:.2f}"] = value
    for thr, frac in overall_saturation.items():
        summary[f"fraction_ge_{thr:.2f}"] = frac

    if all_teacher:
        teacher_merged = np.concatenate([v.reshape(-1) for v in all_teacher])
        diff_merged = np.concatenate([v.reshape(-1) for v in diff_values]) if diff_values else np.zeros(1)
        summary["teacher_mean"] = float(np.mean(teacher_merged))
        summary["teacher_std"] = float(np.std(teacher_merged))
        summary["teacher_fraction_ge_0.95"] = float(np.mean(teacher_merged >= 0.95))
        summary["student_teacher_absdiff_mean"] = float(np.mean(diff_merged))
        summary["student_teacher_absdiff_max"] = float(np.max(diff_merged))

    for name, values in background_rmse_values.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float32)
        summary[f"background_{name}_rmse_mean"] = float(np.mean(arr))
        summary[f"background_{name}_rmse_max"] = float(np.max(arr))
        summary[f"background_{name}_rmse_std"] = float(np.std(arr))

    return per_image, summary


def dump_json(path: Path, summary: Dict[str, float], per_image: List[ImageStats]) -> None:
    payload = {
        "summary": summary,
        "per_image": [
            {
                "name": stats.name,
                "mean": stats.mean,
                "std": stats.std,
                "min": stats.min,
                "max": stats.max,
                "quantiles": stats.quantiles,
                "saturation": stats.saturation,
                "teacher_diff_mean": stats.teacher_diff_mean,
                "teacher_diff_max": stats.teacher_diff_max,
                "background_rmse": stats.background_rmse,
            }
            for stats in per_image
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def dump_csv(path: Path, summary: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write("metric,value\n")
        for key, value in sorted(summary.items()):
            fp.write(f"{key},{value:.6f}\n")


def dump_per_image_csv(
    path: Path,
    per_image: List[ImageStats],
    thresholds: Sequence[float],
    quantiles: Sequence[float],
    background_names: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "name",
        "mean",
        "std",
        "min",
        "max",
    ]
    header += [f"quantile_{q:.2f}" for q in quantiles]
    header += [f"fraction_ge_{thr:.2f}" for thr in thresholds]
    header += ["teacher_diff_mean", "teacher_diff_max"]
    header += [f"background_{name}_rmse" for name in background_names]

    with path.open("w", encoding="utf-8") as fp:
        fp.write(",".join(header) + "\n")
        for stats in per_image:
            row = [
                stats.name,
                f"{stats.mean:.6f}",
                f"{stats.std:.6f}",
                f"{stats.min:.6f}",
                f"{stats.max:.6f}",
            ]
            row += [f"{stats.quantiles[q]:.6f}" for q in quantiles]
            row += [f"{stats.saturation[thr]:.6f}" for thr in thresholds]
            if stats.teacher_diff_mean is not None:
                row.append(f"{stats.teacher_diff_mean:.6f}")
            else:
                row.append("")
            if stats.teacher_diff_max is not None:
                row.append(f"{stats.teacher_diff_max:.6f}")
            else:
                row.append("")
            for name in background_names:
                if stats.background_rmse and name in stats.background_rmse:
                    row.append(f"{stats.background_rmse[name]:.6f}")
                else:
                    row.append("")
            fp.write(",".join(row) + "\n")


BACKGROUND_PRESETS: Dict[str, np.ndarray] = {
    "black": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    "white": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    "gray": np.array([0.5, 0.5, 0.5], dtype=np.float32),
    "grey": np.array([0.5, 0.5, 0.5], dtype=np.float32),
}


def parse_background_spec(spec: str) -> Tuple[str, np.ndarray]:
    if ":" in spec:
        name_part, color_part = spec.split(":", 1)
    else:
        name_part, color_part = spec, spec

    key = color_part.lower()
    if key in BACKGROUND_PRESETS:
        color = BACKGROUND_PRESETS[key].copy()
    else:
        parts = color_part.split(",")
        if len(parts) != 3:
            raise ValueError(
                f"Background specification '{spec}' must be preset name or name:r,g,b"
            )
        try:
            color = np.array([float(p) for p in parts], dtype=np.float32)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value in background specification '{spec}'") from exc
    if np.any(color < 0.0) or np.any(color > 1.0):
        raise ValueError(f"Background color must be within [0,1], got {color}")
    name = name_part if name_part else key
    return name, color


def parse_thresholds(raw: Sequence[str]) -> Sequence[float]:
    values = [float(x) for x in raw]
    for v in values:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Thresholds must be within [0,1]")
    return values


def parse_quantiles(raw: Sequence[str]) -> Sequence[float]:
    values = [float(x) for x in raw]
    for v in values:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Quantiles must be within [0,1]")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("student_opacity", type=Path, help="Directory containing student opacity PNGs")
    parser.add_argument("--teacher-alpha", type=Path, default=None, help="Optional directory with teacher alpha PNGs")
    parser.add_argument("--thresholds", nargs="*", default=(0.9, 0.95, 0.99), help="Saturation thresholds in [0,1]")
    parser.add_argument("--quantiles", nargs="*", default=(0.5, 0.9, 0.99), help="Quantiles to record")
    parser.add_argument("--output-json", type=Path, default=None, help="Path to write JSON summary")
    parser.add_argument("--output-csv", type=Path, default=None, help="Path to write CSV summary")
    parser.add_argument("--per-image-csv", type=Path, default=None, help="Path to write per-image CSV")
    parser.add_argument("--student-rgba", type=Path, default=None, help="Directory containing student RGBA npz files")
    parser.add_argument(
        "--background",
        action="append",
        default=[],
        help=(
            "Background specification (name:r,g,b in [0,1] or preset name such as white/black); "
            "repeat to enable multiple backgrounds."
        ),
    )
    parser.add_argument(
        "--background-alpha-threshold",
        type=float,
        default=0.05,
        help="Alpha threshold to treat pixel as background when computing background RMSE",
    )
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    quantiles = parse_quantiles(args.quantiles)
    if not 0.0 <= args.background_alpha_threshold <= 1.0:
        raise ValueError("background-alpha-threshold must be within [0,1]")

    backgrounds: Dict[str, np.ndarray] = {}
    for spec in args.background:
        name, color = parse_background_spec(spec)
        backgrounds[name] = color

    student_rgba_dir = args.student_rgba
    if backgrounds and student_rgba_dir is None:
        candidate = args.student_opacity.parent / "rgba_npz"
        if candidate.exists():
            student_rgba_dir = candidate
        else:
            print(
                f"[warn] No RGBA directory provided and default {candidate} not found; background RMSE will be skipped"
            )
            student_rgba_dir = None

    pairs = list_png_pairs(args.student_opacity, args.teacher_alpha)
    per_image, summary = aggregate_image_stats(
        pairs,
        thresholds=thresholds,
        quantiles=quantiles,
        student_root=args.student_opacity,
        student_rgba_dir=student_rgba_dir,
        backgrounds=backgrounds,
        background_alpha_threshold=args.background_alpha_threshold,
    )

    print("Opacity summary:")
    for key, value in sorted(summary.items()):
        print(f"  {key}: {value:.6f}")

    if args.output_json is not None:
        dump_json(args.output_json, summary, per_image)
    if args.output_csv is not None:
        dump_csv(args.output_csv, summary)
    if args.per_image_csv is not None:
        dump_per_image_csv(args.per_image_csv, per_image, thresholds, quantiles, list(backgrounds.keys()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
