#!/usr/bin/env python
"""Sanity-check that teacher assets required for LEGO white baseline exist and are self-consistent."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Tuple

REQUIRED_RENDER_FILES = (
    ("renders", "*.png"),
    ("gt", "*.png"),
    ("depth", "*.npy"),
)


def _count_matches(directory: Path, pattern: str) -> int:
    return sum(1 for _ in directory.glob(pattern))


def _check_directory(root: Path, relative: str, pattern: str, expected: int) -> Tuple[bool, str]:
    path = root / relative
    if not path.exists():
        return False, f"missing directory: {path}"
    if not path.is_dir():
        return False, f"expected directory but found file: {path}"
    count = _count_matches(path, pattern)
    if count != expected:
        return False, f"{path} contains {count} files matching {pattern}, expected {expected}"
    return True, ""


def _load_json(path: Path) -> Tuple[bool, str, dict | None]:
    if not path.exists():
        return False, f"missing file: {path}", None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:  # pylint: disable=broad-except
        return False, f"failed to parse JSON {path}: {exc}", None
    return True, "", data


def _verify_render_stats(stats: dict, expected_frames: int) -> Tuple[bool, str]:
    required_keys = {"num_frames", "avg_fps", "total_render_time_s", "gpu_mem_peak_mib"}
    missing = required_keys - stats.keys()
    if missing:
        return False, f"render_stats.json missing keys: {sorted(missing)}"
    if int(stats["num_frames"]) != expected_frames:
        return False, (
            f"render_stats.json reports {stats['num_frames']} frames but expected {expected_frames}"
        )
    return True, ""


def check_teacher_assets(
    outputs_root: Path,
    checkpoint_path: Path,
    expected_frames: int,
    verbose: bool = False,
) -> int:
    issues: list[str] = []

    if verbose:
        print(f"[check] outputs_root = {outputs_root}")
        print(f"[check] checkpoint   = {checkpoint_path}")
        print(f"[check] expected frames = {expected_frames}")

    if not outputs_root.exists():
        issues.append(f"teacher outputs directory missing: {outputs_root}")
    elif not outputs_root.is_dir():
        issues.append(f"teacher outputs path is not a directory: {outputs_root}")
    else:
        for rel_dir, pattern in REQUIRED_RENDER_FILES:
            ok, message = _check_directory(outputs_root, rel_dir, pattern, expected_frames)
            if not ok:
                issues.append(message)
            elif verbose:
                print(f"[ok] {rel_dir} matches {pattern} ×{expected_frames}")

        stats_ok, stats_msg, stats = _load_json(outputs_root / "render_stats.json")
        if not stats_ok:
            issues.append(stats_msg)
        else:
            ok, message = _verify_render_stats(stats, expected_frames)
            if not ok:
                issues.append(message)
            elif verbose:
                print(
                    "[ok] render_stats.json: "
                    f"{stats['num_frames']} frames, avg_fps={stats['avg_fps']:.3f}, "
                    f"gpu_peak={stats['gpu_mem_peak_mib']:.1f} MiB"
                )

        transforms_path = outputs_root / "transforms_test_white.json"
        if not transforms_path.exists():
            issues.append(f"missing transforms file: {transforms_path}")
        elif verbose:
            print(f"[ok] found transforms JSON: {transforms_path}")

    if not checkpoint_path.exists():
        issues.append(f"teacher checkpoint missing: {checkpoint_path}")
    elif not checkpoint_path.is_file():
        issues.append(f"teacher checkpoint is not a file: {checkpoint_path}")
    elif checkpoint_path.stat().st_size == 0:
        issues.append(f"teacher checkpoint file is empty: {checkpoint_path}")
    elif verbose:
        print(
            f"[ok] checkpoint size = {checkpoint_path.stat().st_size / (1024**2):.2f} MiB"
        )

    if issues:
        print("[FAIL] Teacher asset check failed:")
        for item in issues:
            print(f"  - {item}")
        return 1

    print("[PASS] Teacher assets are complete and consistent.")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("teacher/outputs/lego/test_white/ours_30000"),
        help="Path to the teacher RGBA/depth outputs for evaluation.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("teacher/checkpoints/lego/point_cloud/iteration_30000/point_cloud.ply"),
        help="Path to the teacher Gaussian checkpoint (.ply).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=200,
        help="Expected number of frames in the teacher outputs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed diagnostics while checking assets.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    return check_teacher_assets(
        outputs_root=args.outputs_root,
        checkpoint_path=args.checkpoint,
        expected_frames=args.frames,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
