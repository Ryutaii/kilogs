"""Recompose student RGBA outputs onto arbitrary background colors."""

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np


def parse_background(value: str) -> Tuple[float, float, float]:
    presets = {
        "black": (0.0, 0.0, 0.0),
        "white": (1.0, 1.0, 1.0),
        "grey": (0.5, 0.5, 0.5),
        "gray": (0.5, 0.5, 0.5),
    }
    lower = value.lower()
    if lower in presets:
        return presets[lower]

    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Background '{value}' must be preset name or comma-separated RGB"
        )
    try:
        rgb = tuple(float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid numeric value in background '{value}'"
        ) from exc

    if any(c < 0.0 or c > 1.0 for c in rgb):
        raise argparse.ArgumentTypeError(
            f"RGB components must be within [0, 1], got {rgb}"
        )
    return rgb  # type: ignore[return-value]


def gather_npz_files(root: Path) -> List[Path]:
    return sorted(root.glob("**/*.npz"))


def compose_rgba(rgb: np.ndarray, alpha: np.ndarray, background: Sequence[float]) -> np.ndarray:
    if rgb.shape[:2] != alpha.shape[:2]:
        raise ValueError("RGB and alpha dimensions do not match")
    bg = np.asarray(background, dtype=np.float32).reshape(1, 1, 3)
    alpha_expanded = alpha[..., None]
    composed = rgb + (1.0 - alpha_expanded) * bg
    return np.clip(composed, 0.0, 1.0)


def save_composed_image(output_path: Path, composed: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(output_path, (composed * 255.0).astype(np.uint8))


def process_files(npz_files: Iterable[Path], output_root: Path, backgrounds: Sequence[Tuple[float, float, float]]) -> None:
    for npz_path in npz_files:
        data = np.load(npz_path)
        if "rgb" not in data or "alpha" not in data:
            raise KeyError(f"Expected 'rgb' and 'alpha' arrays in {npz_path}")
        rgb = data["rgb"].astype(np.float32)
        alpha = data["alpha"].astype(np.float32)

        relative = npz_path.relative_to(npz_path.parents[1])  # parents[1] corresponds to 'rgba_npz'
        stem = relative.with_suffix("")

        for color in backgrounds:
            name = f"bg_{int(color[0]*255):03d}_{int(color[1]*255):03d}_{int(color[2]*255):03d}"
            composed = compose_rgba(rgb, alpha, color)
            output_path = output_root / name / stem.with_suffix(".png")
            save_composed_image(output_path, composed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompose RGBA npz outputs onto background colors")
    parser.add_argument("--input", type=Path, required=True, help="Directory containing rgba_npz outputs")
    parser.add_argument("--output", type=Path, required=True, help="Directory to store recomposed renders")
    parser.add_argument(
        "--background",
        type=parse_background,
        nargs="+",
        default=[(1.0, 1.0, 1.0)],
        help="Background colors (preset name or r,g,b in [0,1]); can be specified multiple times",
    )
    args = parser.parse_args()

    npz_dir = args.input
    if not npz_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {npz_dir}")

    npz_files = gather_npz_files(npz_dir)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under {npz_dir}")

    process_files(npz_files, args.output, args.background)


if __name__ == "__main__":
    main()
