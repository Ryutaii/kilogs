"""Recompose white-background renders into black-background images using opacity maps.

This utility expects renders that were produced with a white background plus the
corresponding opacity maps. It outputs images that match an alpha compositing
done over a black background, which is required for fair comparisons against
black-background teacher renders.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def _load_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")
    img = imageio.imread(path)
    if img.dtype == np.uint16:
        img = (img / 257.0).astype(np.uint8)
    if img.dtype != np.uint8:
        raise ValueError(f"Unsupported dtype {img.dtype} for {path}; expected uint8")
    return img


def recompose_directory(white_dir: Path, opacity_dir: Path, output_dir: Path, subdir: str) -> None:
    white_root = white_dir / subdir
    opacity_root = opacity_dir / subdir
    output_root = output_dir / subdir

    if not white_root.exists():
        raise FileNotFoundError(f"White render directory not found: {white_root}")
    if not opacity_root.exists():
        raise FileNotFoundError(f"Opacity directory not found: {opacity_root}")

    for white_path in sorted(white_root.rglob("*.png")):
        rel_path = white_path.relative_to(white_root)
        opacity_path = opacity_root / rel_path
        output_path = output_root / rel_path

        opacity_img = _load_image(opacity_path).astype(np.float32) / 255.0
        white_img = _load_image(white_path).astype(np.float32) / 255.0

        if opacity_img.ndim == 2:
            alpha = np.expand_dims(opacity_img, axis=-1)
        elif opacity_img.ndim == 3 and opacity_img.shape[2] == 1:
            alpha = opacity_img
        elif opacity_img.ndim == 3 and opacity_img.shape[2] == 3:
            alpha = np.expand_dims(opacity_img[..., 0], axis=-1)
        else:
            raise ValueError(
                f"Unexpected opacity image shape {opacity_img.shape} for {opacity_path}"
            )

        alpha = np.clip(alpha, 0.0, 1.0)
        if white_img.ndim != 3 or white_img.shape[2] != 3:
            raise ValueError(f"Expected RGB image for {white_path}, got shape {white_img.shape}")

        black_img = white_img - (1.0 - alpha)
        black_img = np.clip(black_img, 0.0, 1.0)
        black_uint8 = (black_img * 255.0 + 0.5).astype(np.uint8)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(output_path, black_uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompose renders over a black background")
    parser.add_argument("--white-dir", type=Path, required=True, help="Directory containing white-background renders")
    parser.add_argument("--opacity-dir", type=Path, required=True, help="Directory containing opacity PNGs")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save recomposed renders")
    parser.add_argument(
        "--subdir",
        type=str,
        default="test",
        help="Sub-directory (relative) to process, defaults to 'test'",
    )
    parser.add_argument(
        "--copy-stats-from",
        type=Path,
        default=None,
        help="Optional directory containing render_stats.json to copy alongside outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    white_dir = args.white_dir.resolve()
    opacity_dir = args.opacity_dir.resolve()
    output_dir = args.output_dir.resolve()

    recompose_directory(white_dir, opacity_dir, output_dir, args.subdir)

    if args.copy_stats_from is not None:
        stats_path = args.copy_stats_from.resolve() / "render_stats.json"
        if stats_path.exists():
            output_stats = output_dir.parent / "render_stats.json"
            output_stats.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stats_path, output_stats)


if __name__ == "__main__":
    main()