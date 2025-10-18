#!/usr/bin/env python
"""Export aggregated Gaussian teacher features for downstream models (e.g. VAE).

This script loads a 3D Gaussian Splatting checkpoint (PLY), aggregates the per-Gaussian
attributes onto a KiloNeRF cell grid, and stores the resulting feature matrix alongside
basic statistics in a ``.npz`` file.

Example
-------
python tools/export_gaussian_features.py \
    --ply teacher/checkpoints/lego/point_cloud/iteration_30000/point_cloud.ply \
    --output data/vae/lego_gaussians_cells.npz \
    --grid-resolution 6 6 6 \
    --bbox-min -1.5 -1.5 -1.5 \
    --bbox-max 1.5 1.5 1.5 \
    --teacher-mode gaussian_all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from distill.lego_response_distill import _build_gaussian_cell_features
from distill.teacher_features import GaussianTeacherFeatures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export aggregated Gaussian teacher features")
    parser.add_argument("--ply", type=Path, required=True, help="Path to Gaussian point cloud .ply")
    parser.add_argument("--output", type=Path, required=True, help="Destination .npz file")
    parser.add_argument(
        "--grid-resolution",
        type=int,
        nargs=3,
        metavar=("GX", "GY", "GZ"),
        default=(6, 6, 6),
        help="Cell grid resolution (default: 6 6 6)",
    )
    parser.add_argument(
        "--bbox-min",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(-1.5, -1.5, -1.5),
        help="Scene bounding box minimum (default: -1.5 -1.5 -1.5)",
    )
    parser.add_argument(
        "--bbox-max",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(1.5, 1.5, 1.5),
        help="Scene bounding box maximum (default: 1.5 1.5 1.5)",
    )
    parser.add_argument(
        "--teacher-mode",
        type=str,
        default="gaussian_all",
        help="Teacher feature mode (matches FeaturePipelineConfig.teacher_mode)",
    )
    parser.add_argument(
        "--teacher-components",
        type=str,
        nargs="*",
        default=None,
        help="Explicit Gaussian components to aggregate (overrides teacher-mode preset)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=("float32", "float64"),
        help="Floating point precision for tensors",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to load tensors onto (default: cpu)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file",
    )
    return parser.parse_args()


def _components_argument(raw: Iterable[str] | None) -> Sequence[str]:
    if raw is None:
        return ()
    return tuple(str(comp) for comp in raw)


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output file already exists: {args.output} (use --overwrite to replace)")

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    teacher = GaussianTeacherFeatures.from_ply(Path(args.ply), device=device, dtype=dtype)
    features = _build_gaussian_cell_features(
        teacher,
        tuple(int(v) for v in args.grid_resolution),
        torch.tensor(args.bbox_min, dtype=dtype, device=device),
        torch.tensor(args.bbox_max, dtype=dtype, device=device),
        mode=args.teacher_mode,
        components=_components_argument(args.teacher_components),
    )

    features_cpu = features.cpu()
    mask = torch.any(features_cpu != 0.0, dim=-1)
    valid = features_cpu[mask]
    if valid.numel() == 0:
        raise RuntimeError("No valid cells found – check bounding box and grid resolution")

    mean = valid.mean(dim=0)
    std = valid.std(dim=0).clamp_min(1e-6)

    payload = {
        "features": features_cpu.numpy(),
        "mask": mask.numpy().astype(np.bool_),
        "mean": mean.numpy(),
        "std": std.numpy(),
        "metadata": json.dumps(
            {
                "ply": str(args.ply),
                "grid_resolution": [int(v) for v in args.grid_resolution],
                "bbox_min": list(map(float, args.bbox_min)),
                "bbox_max": list(map(float, args.bbox_max)),
                "teacher_mode": args.teacher_mode,
                "teacher_components": list(_components_argument(args.teacher_components)) or None,
                "dtype": args.dtype,
            }
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(
        f"Exported features to {args.output} (cells={features.shape[0]}, dim={features.shape[1]}, "
        f"valid={int(mask.sum())})"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
