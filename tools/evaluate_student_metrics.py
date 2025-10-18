#!/usr/bin/env python3
"""Compute PSNR / SSIM / LPIPS between student renders and teacher reference images."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Reuse the 3D-GS utility functions for PSNR / SSIM.
REPO_ROOT = Path(__file__).resolve().parents[1]
THREEDGS_ROOT = REPO_ROOT.parent / "3dgs"
if str(THREEDGS_ROOT) not in sys.path:
    sys.path.append(str(THREEDGS_ROOT))


def _import_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


image_utils = _import_from_path("threedgs_image_utils", THREEDGS_ROOT / "utils" / "image_utils.py")
loss_utils = _import_from_path("threedgs_loss_utils", THREEDGS_ROOT / "utils" / "loss_utils.py")

psnr_fn = image_utils.psnr  # type: ignore[attr-defined]
ssim_fn = loss_utils.ssim  # type: ignore[attr-defined]
from lpipsPyTorch.modules.lpips import LPIPS  # type: ignore  # noqa: E402


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor.to(device)


def _natural_key(name: str) -> List[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def list_matching_files(student_dir: Path, teacher_dir: Path) -> List[Tuple[Path, Path]]:
    student_paths = {p.name: p for p in student_dir.glob("*.png")}
    teacher_paths = {p.name: p for p in teacher_dir.glob("*.png")}

    shared_keys = sorted(student_paths.keys() & teacher_paths.keys(), key=_natural_key)
    if shared_keys:
        return [(student_paths[key], teacher_paths[key]) for key in shared_keys]

    student_sorted = sorted(student_dir.glob("*.png"), key=lambda p: _natural_key(p.name))
    teacher_sorted = sorted(teacher_dir.glob("*.png"), key=lambda p: _natural_key(p.name))

    if not student_sorted or not teacher_sorted:
        raise FileNotFoundError(
            f"No PNG files found in {student_dir} or {teacher_dir}."
        )

    if len(student_sorted) != len(teacher_sorted):
        raise FileNotFoundError(
            "PNG counts differ and no overlapping names were found; cannot align renders."
        )

    return list(zip(student_sorted, teacher_sorted))


def compute_metrics(
    pairs: Iterable[Tuple[Path, Path]],
    device: torch.device,
    progress_interval: int,
) -> Dict[str, float]:
    psnr_values: List[float] = []
    ssim_values: List[float] = []
    lpips_values: List[float] = []

    pair_list = list(pairs)
    total = len(pair_list)
    if total == 0:
        raise ValueError("No image pairs provided for metric computation.")

    start_time = time.perf_counter()
    print(f"[info] evaluating {total} image pairs...", flush=True)

    lpips_model = LPIPS(net_type="vgg").to(device)
    lpips_model.eval()

    progress_bar = tqdm(
        pair_list,
        total=total,
        unit="img",
        dynamic_ncols=True,
        leave=True,
        disable=total == 0,
    )

    for idx, (student_path, teacher_path) in enumerate(progress_bar, start=1):
        student_img = load_image(student_path, device)
        teacher_img = load_image(teacher_path, device)

        with torch.no_grad():
            psnr_val = psnr_fn(student_img, teacher_img).mean().item()
            ssim_val = ssim_fn(student_img, teacher_img).item()
            lpips_val = lpips_model(student_img, teacher_img).item()

        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        lpips_values.append(lpips_val)

        elapsed = time.perf_counter() - start_time
        views_per_s = idx / max(elapsed, 1e-6)
        remaining = (total - idx) / max(views_per_s, 1e-6)
        progress_bar.set_postfix(
            ordered_dict={
                "last": student_path.name,
                "views/s": f"{views_per_s:.2f}",
                "eta": _format_duration(remaining),
            }
        )

        if progress_interval > 0 and idx % progress_interval == 0:
            progress_bar.write(
                f"[progress] {idx}/{total} ({idx / total * 100.0:.1f}%) - {student_path.name}"
                f" | {views_per_s:.2f} views/s | elapsed {_format_duration(elapsed)}"
                f" | eta {_format_duration(remaining)}"
            )

    def avg(values: List[float]) -> float:
        return float(sum(values) / len(values))

    return {
        "num_images": len(psnr_values),
        "psnr": avg(psnr_values),
        "ssim": avg(ssim_values),
        "lpips": avg(lpips_values),
    }


def dump_results(metrics: Dict[str, float], output_json: Path | None, output_csv: Path | None) -> None:
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8") as fp:
            fp.write("metric,value\n")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    fp.write(f"{key},{value:.6f}\n")
                else:
                    fp.write(f"{key},{value}\n")


SUMMARY_COLUMNS = [
    "method",
    "psnr_white",
    "ssim_white",
    "lpips_white",
    "psnr_black",
    "ssim_black",
    "lpips_black",
    "fps",
    "gpu_peak_gib",
    "power_avg_w",
]


def _format_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.5f}"


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _load_render_stats(path: Path) -> Dict[str, Optional[float]]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    avg_fps = data.get("avg_fps")
    gpu_peak_mib = data.get("gpu_mem_peak_mib") or data.get("gpu_mem_peak")
    gpu_peak_gib = None
    if isinstance(gpu_peak_mib, (int, float)):
        gpu_peak_gib = float(gpu_peak_mib) / 1024.0
    power_avg = data.get("power_avg_watts") or data.get("power_avg")

    return {
        "avg_fps": float(avg_fps) if isinstance(avg_fps, (int, float)) else None,
        "gpu_peak_gib": gpu_peak_gib,
        "power_avg_w": float(power_avg) if isinstance(power_avg, (int, float)) else None,
    }


def _ensure_row_columns(row: Dict[str, str]) -> Dict[str, str]:
    return {column: row.get(column, "") for column in SUMMARY_COLUMNS}


def _update_summary_csv(
    summary_path: Path,
    method: str,
    background: str,
    metrics: Dict[str, float],
    render_stats: Dict[str, Optional[float]],
) -> None:
    background = background.lower()
    if background not in {"white", "black"}:
        raise ValueError("background must be 'white' or 'black' when updating summary")

    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                rows.append(_ensure_row_columns(row))

    target_row = None
    for row in rows:
        if row.get("method") == method:
            target_row = row
            break

    if target_row is None:
        target_row = {column: "" for column in SUMMARY_COLUMNS}
        target_row["method"] = method
        rows.append(target_row)

    should_update = True
    tolerance = 1e-6
    if background == "white":
        existing_psnr = _parse_float(target_row.get("psnr_white"))
        new_psnr = float(metrics.get("psnr")) if metrics.get("psnr") is not None else None
        if existing_psnr is not None and new_psnr is not None:
            if new_psnr < existing_psnr - tolerance:
                should_update = False
            elif abs(new_psnr - existing_psnr) <= tolerance:
                existing_lpips = _parse_float(target_row.get("lpips_white"))
                new_lpips = float(metrics.get("lpips")) if metrics.get("lpips") is not None else None
                if existing_lpips is not None and new_lpips is not None and new_lpips > existing_lpips + tolerance:
                    should_update = False
            if not should_update:
                print(
                    f"[info] Skipping summary update for '{method}' (white) because an existing run has better metrics."
                )

    prefix_map = {
        "psnr": f"psnr_{background}",
        "ssim": f"ssim_{background}",
        "lpips": f"lpips_{background}",
    }

    if should_update:
        for metric_key, column_name in prefix_map.items():
            value = metrics.get(metric_key)
            target_row[column_name] = _format_float(float(value)) if value is not None else ""

        if render_stats:
            if render_stats.get("avg_fps") is not None:
                target_row["fps"] = _format_float(render_stats["avg_fps"])
            if render_stats.get("gpu_peak_gib") is not None:
                target_row["gpu_peak_gib"] = _format_float(render_stats["gpu_peak_gib"])  # type: ignore[arg-type]
            if render_stats.get("power_avg_w") is not None:
                target_row["power_avg_w"] = _format_float(render_stats["power_avg_w"])

    with summary_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_path(path_str: str, description: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path

    pattern = None
    if "..." in path_str:
        pattern = path_str.replace("...", "**")
    elif any(token in path_str for token in ("*", "?", "[", "]")):
        pattern = path_str

    if pattern is not None:
        matches = sorted(Path().glob(pattern))
        matches = [p for p in matches if p.exists()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 0:
            raise FileNotFoundError(
                f"Could not resolve {description} pattern '{path_str}' (expanded to '{pattern}')"
            )
        raise FileNotFoundError(
            f"Pattern '{path_str}' (expanded to '{pattern}') matched multiple candidates for {description}:\n"
            + "\n".join(str(p) for p in matches)
        )

    raise FileNotFoundError(f"{description} path '{path_str}' does not exist.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("student_renders", type=str, help="Directory (or pattern) with student PNG renders")
    parser.add_argument("teacher_renders", type=str, help="Directory (or pattern) with teacher PNG renders")
    parser.add_argument(
        "--device", default="cuda", help="Torch device to use (default: cuda if available)"
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Print a progress line every N images (default: 10; 0 disables periodic updates)",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument(
        "--background",
        choices=["white", "black"],
        default=None,
        help="Background label to record with the metrics (required when updating summary)",
    )
    parser.add_argument(
        "--render-stats",
        type=Path,
        default=None,
        help="Optional render_stats.json to extract fps / gpu memory / power",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="metrics_summary.csv to update with the computed results",
    )
    parser.add_argument(
        "--method-name",
        type=str,
        default=None,
        help="Method label to use when updating the summary CSV",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    student_dir = _resolve_path(args.student_renders, "student renders")
    teacher_dir = _resolve_path(args.teacher_renders, "teacher renders")

    pairs = list_matching_files(student_dir, teacher_dir)
    metrics = compute_metrics(pairs, device, args.progress_interval)

    render_stats: Dict[str, Optional[float]] = {}
    if args.render_stats is not None:
        render_stats = _load_render_stats(args.render_stats)
        if render_stats.get("avg_fps") is not None:
            metrics["avg_fps"] = float(render_stats["avg_fps"])
        if render_stats.get("gpu_peak_gib") is not None:
            metrics["gpu_peak_gib"] = float(render_stats["gpu_peak_gib"])  # type: ignore[arg-type]
        if render_stats.get("power_avg_w") is not None:
            metrics["power_avg_w"] = float(render_stats["power_avg_w"])

    if args.background is not None:
        metrics["background"] = args.background

    dump_results(metrics, args.output_json, args.output_csv)

    if args.summary is not None:
        if args.method_name is None:
            parser.error("--summary requires --method-name to be provided")
        if args.background is None:
            parser.error("--summary requires --background to be specified")
        _update_summary_csv(args.summary, args.method_name, args.background, metrics, render_stats)

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# --- PATCH: override load_image (safe, RGBA->white compose) ---
def load_image(path, device):
    import imageio.v3 as iio
    import numpy as np
    import torch
    arr = iio.imread(path)  # HxWx(C)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[2] == 4:
        a = arr[..., 3:4] / 255.0
        rgb = arr[..., :3] / 255.0
        arr = rgb * a + (1.0 - a)   # white compose → 3ch
        arr = (arr * 255.0).astype(np.float32)
    arr = np.ascontiguousarray(arr / 255.0)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
