#!/usr/bin/env python3
"""Quick visual/metric inspection for student vs teacher render pairs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image


def _load_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    arr = np.array(Image.open(path)).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def _psnr(student: np.ndarray, teacher: np.ndarray) -> float:
    mse = np.mean((student - teacher) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return -10.0 * np.log10(mse)


def _mae(student: np.ndarray, teacher: np.ndarray) -> float:
    return float(np.mean(np.abs(student - teacher)))


def _resolve_student_path(run_dir: Path, frame: str, use_recomposed: bool, background: str) -> Path:
    if use_recomposed:
        bg_tag = background.lower()
        if "," in bg_tag:
            parts = [float(p) for p in bg_tag.split(",")]
            if len(parts) != 3:
                raise ValueError("--background must have 3 comma-separated values")
            bg_tag = "bg_{:03d}_{:03d}_{:03d}".format(
                int(round(parts[0] * 255.0)),
                int(round(parts[1] * 255.0)),
                int(round(parts[2] * 255.0)),
            )
        else:
            presets = {
                "white": "bg_255_255_255",
                "black": "bg_000_000_000",
                "grey": "bg_128_128_128",
                "gray": "bg_128_128_128",
            }
            bg_tag = presets.get(bg_tag, bg_tag)
        base = run_dir / "renders_recomposed" / bg_tag
        candidates = [base / "renders" / f"{frame}.png", base / "rgba_npz" / f"{frame}.png"]
    else:
        base = run_dir / "renders"
        candidates = [base / f"{frame}.png", base / "renders" / f"{frame}.png"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not locate student frame '{frame}' under {run_dir}")


def _resolve_teacher_path(teacher_dir: Path, frame: str) -> Path:
    candidates = [
        teacher_dir / f"{frame}.png",
        teacher_dir / "renders" / f"{frame}.png",
        teacher_dir / "gt" / f"{frame}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not locate teacher frame '{frame}' under {teacher_dir}")


def _ensure_frames(frames: Iterable[str]) -> List[str]:
    out: List[str] = []
    for frame in frames:
        token = frame.strip()
        if not token:
            continue
        if token.endswith(".png"):
            token = token[:-4]
        if len(token) != 5:
            token = token.zfill(5)
        out.append(token)
    if not out:
        raise ValueError("No frames specified")
    return out


def _save_image(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((data * 255.0).clip(0, 255).astype(np.uint8)).save(path)


def inspect(run_dir: Path, teacher_dir: Path, frames: List[str], use_recomposed: bool, background: str, output_dir: Path | None, diff_gamma: float, save_stack: bool) -> None:
    for frame in frames:
        student_path = _resolve_student_path(run_dir, frame, use_recomposed, background)
        teacher_path = _resolve_teacher_path(teacher_dir, frame)

        student = _load_image(student_path)
        teacher = _load_image(teacher_path)
        diff = np.abs(student - teacher)

        psnr_val = _psnr(student, teacher)
        mae_val = _mae(student, teacher)
        max_diff = float(diff.max())
        print(f"[{frame}] psnr={psnr_val:.4f} dB, mae={mae_val:.4f}, max_diff={max_diff:.4f}")

        if output_dir is not None:
            if max_diff > 0:
                norm = diff / max_diff
            else:
                norm = diff
            diff_img = norm ** (1.0 / max(diff_gamma, 1e-6))
            _save_image(output_dir / f"diff_{frame}.png", diff_img)
            if save_stack:
                stack = np.concatenate([teacher, student, diff_img], axis=1)
                _save_image(output_dir / f"stack_{frame}.png", stack)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-run", type=Path, required=True, help="Path to render_student output directory")
    parser.add_argument("--teacher-dir", type=Path, required=True, help="Path containing teacher renders (renders/ or gt/)")
    parser.add_argument("--frames", nargs="+", default=["00000"], help="Frame indices to inspect (e.g. 00000 00050)")
    parser.add_argument("--use-recomposed", action="store_true", help="Read images from renders_recomposed/<background>/ instead of raw renders/")
    parser.add_argument("--background", type=str, default="white", help="Background tag when --use-recomposed is set")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional directory to save diff/stack PNGs")
    parser.add_argument("--diff-gamma", type=float, default=2.2, help="Gamma to apply when visualising absolute difference")
    parser.add_argument("--save-stack", action="store_true", help="When set, save teacher|student|diff horizontal stacks")
    args = parser.parse_args()

    frames = _ensure_frames(args.frames)
    inspect(
        run_dir=args.student_run,
        teacher_dir=args.teacher_dir,
        frames=frames,
        use_recomposed=args.use_recomposed,
        background=args.background,
        output_dir=args.output_dir,
        diff_gamma=args.diff_gamma,
        save_stack=args.save_stack,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
