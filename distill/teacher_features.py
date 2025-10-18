"""Utilities for extracting structured features from 3D Gaussian Splatting teachers.

This module provides minimal tooling to ingest a trained 3D-GS point cloud (e.g. the
``point_cloud/iteration_*.ply`` checkpoints emitted by the official GraphDECO codebase)
and expose spherical-harmonic coefficients, opacities, scaling parameters, and rotations
as PyTorch tensors.  These tensors form the basis for Stage 2 feature distillation, where
student representations will align against the teacher's intermediate attributes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch

_PLY_ERROR: Optional[ImportError]
try:  # optional dependency installed with 3D-GS tooling
    from plyfile import PlyData  # type: ignore
    _PLY_ERROR = None
except ImportError as err:  # pragma: no cover - informative error only
    PlyData = None  # type: ignore
    _PLY_ERROR = err


@dataclass(frozen=True)
class GaussianAttributeSet:
    """Container holding per-Gaussian attributes on a common device/dtype."""

    positions: torch.Tensor  # (N, 3)
    sh_coeffs: torch.Tensor  # (N, 3, (degree+1)^2)
    opacity: torch.Tensor  # (N, 1) in logit space; sigmoid yields opacity
    scaling: torch.Tensor  # (N, 3) log-scale parameters (exp -> actual scale)
    rotation: torch.Tensor  # (N, 4) quaternion in (w, x, y, z) order

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> "GaussianAttributeSet":
        if device is None and dtype is None:
            return self
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        return GaussianAttributeSet(
            positions=self.positions.to(**kwargs),
            sh_coeffs=self.sh_coeffs.to(**kwargs),
            opacity=self.opacity.to(**kwargs),
            scaling=self.scaling.to(**kwargs),
            rotation=self.rotation.to(**kwargs),
        )

    @property
    def num_gaussians(self) -> int:
        return int(self.positions.shape[0])

    @property
    def sh_degree(self) -> int:
        coeffs_per_channel = int(self.sh_coeffs.shape[-1])
        return int(round(math.sqrt(coeffs_per_channel) - 1))

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "positions": self.positions,
            "sh_coeffs": self.sh_coeffs,
            "opacity": self.opacity,
            "scaling": self.scaling,
            "rotation": self.rotation,
        }


def _sorted_property_names(names: Iterable[str], prefix: str) -> Sequence[str]:
    filtered = [name for name in names if name.startswith(prefix)]
    return tuple(sorted(filtered, key=lambda n: int(n.split("_")[-1])))


def _quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    q = torch.nn.functional.normalize(quaternion, dim=-1)
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    m00 = 1 - 2 * (yy + zz)
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)
    m10 = 2 * (xy + wz)
    m11 = 1 - 2 * (xx + zz)
    m12 = 2 * (yz - wx)
    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = 1 - 2 * (xx + yy)

    return torch.stack(
        [
            torch.stack([m00, m01, m02], dim=-1),
            torch.stack([m10, m11, m12], dim=-1),
            torch.stack([m20, m21, m22], dim=-1),
        ],
        dim=-2,
    )


class GaussianTeacherFeatures:
    """Loads and serves Gaussian attributes for feature distillation."""

    def __init__(self, attributes: GaussianAttributeSet):
        self._attributes = attributes

    @property
    def attributes(self) -> GaussianAttributeSet:
        return self._attributes

    @property
    def num_gaussians(self) -> int:
        return self._attributes.num_gaussians

    @property
    def sh_degree(self) -> int:
        return self._attributes.sh_degree

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> "GaussianTeacherFeatures":
        return GaussianTeacherFeatures(self._attributes.to(device=device, dtype=dtype))

    def covariance_matrices(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        scale = torch.exp(self._attributes.scaling) * float(scaling_modifier)
        rotation = self._attributes.rotation
        rot_mats = _quaternion_to_matrix(rotation)
        lower = torch.matmul(rot_mats, torch.diag_embed(scale))
        cov = lower @ lower.transpose(-1, -2)
        return cov

    def select(self, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("indices must be an integer tensor")
        return {
            name: tensor.index_select(0, indices.to(tensor.device))
            for name, tensor in self._attributes.as_dict().items()
        }

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return self._attributes.as_dict()

    @classmethod
    def from_ply(
        cls,
        ply_path: Path,
        *,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "GaussianTeacherFeatures":
        if PlyData is None:
            raise ImportError(
                "plyfile is required to load Gaussian teacher checkpoints. "
                "Install it via `pip install plyfile` inside the kilogs environment"
            ) from _PLY_ERROR
        path = Path(ply_path)
        if not path.exists():
            raise FileNotFoundError(f"Gaussian point cloud PLY not found: {path}")

        ply = PlyData.read(str(path))
        vertex = ply["vertex"]
        names = vertex.data.dtype.names
        if names is None:
            raise ValueError(f"PLY file {path} does not contain vertex properties")

        positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)
        opacity = vertex["opacity"][..., np.newaxis]

        f_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1)[:, :, np.newaxis]
        rest_names = _sorted_property_names(names, "f_rest_")
        rest = np.stack([vertex[name] for name in rest_names], axis=-1)
        rest = rest.reshape(rest.shape[0], 3, -1)
        sh_coeffs = np.concatenate([f_dc, rest], axis=-1)

        scale_names = _sorted_property_names(names, "scale_")
        scaling = np.stack([vertex[name] for name in scale_names], axis=-1)
        rot_names = _sorted_property_names(names, "rot_")
        rotation = np.stack([vertex[name] for name in rot_names], axis=-1)

        tensors = GaussianAttributeSet(
            positions=torch.as_tensor(positions, dtype=dtype, device=device),
            sh_coeffs=torch.as_tensor(sh_coeffs, dtype=dtype, device=device),
            opacity=torch.as_tensor(opacity, dtype=dtype, device=device),
            scaling=torch.as_tensor(scaling, dtype=dtype, device=device),
            rotation=torch.as_tensor(rotation, dtype=dtype, device=device),
        )
        return cls(tensors)


__all__ = [
    "GaussianAttributeSet",
    "GaussianTeacherFeatures",
]
