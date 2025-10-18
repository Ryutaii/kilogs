"""Teacher-side embedding helpers for feature distillation.

This module encapsulates optional transformations that compress or adapt
teacher feature tensors before they are compared against student projections.
It currently supports identity passthrough and VAE-based compression.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "TeacherEmbeddingConfig",
    "TeacherEmbedding",
    "build_teacher_embedding",
]


def _resolve_device(spec: Optional[str]) -> torch.device:
    if spec is None or spec.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _resolve_dtype(name: Optional[str]) -> torch.dtype:
    key = (name or "float32").lower()
    if key in {"float", "float32", "fp32"}:
        return torch.float32
    if key in {"float16", "fp16", "half"}:
        return torch.float16
    if key in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if key in {"float64", "double", "fp64"}:
        return torch.float64
    raise ValueError(f"Unsupported dtype specifier '{name}' for teacher embedding")


@dataclass
class TeacherEmbeddingConfig:
    """Configuration for optional teacher feature embeddings."""

    type: str = "identity"
    checkpoint: Optional[Path] = None
    stats_path: Optional[Path] = None
    latent_dim: Optional[int] = None
    standardize: Optional[bool] = None
    device: Optional[str] = None
    dtype: str = "float32"
    resolved_input_dim: Optional[int] = None
    resolved_output_dim: Optional[int] = None


class TeacherEmbedding:
    """Base class for teacher feature embeddings."""

    def __init__(self, *, embedding_type: str, input_dim: int, output_dim: int, device: torch.device, dtype: torch.dtype):
        self.embedding_type = embedding_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.dtype = dtype

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {
            "type": self.embedding_type,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
        }


class IdentityEmbedding(TeacherEmbedding):
    """Passthrough embedding that returns features as-is."""

    def __init__(self, input_dim: int, *, device: torch.device, dtype: torch.dtype):
        super().__init__(embedding_type="identity", input_dim=input_dim, output_dim=input_dim, device=device, dtype=dtype)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        return features


class _FeatureVAE(nn.Module):
    """Minimal VAE wrapper matching the training-side architecture."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)


class VAEEmbedding(TeacherEmbedding):
    """Teacher embedding backed by a pre-trained VAE encoder."""

    def __init__(self, cfg: TeacherEmbeddingConfig, input_dim: int):
        device = _resolve_device(cfg.device)
        dtype = _resolve_dtype(cfg.dtype)
        checkpoint_path = cfg.checkpoint
        if checkpoint_path is None:
            raise ValueError("Teacher embedding type 'vae' requires a checkpoint path")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        ckpt_cfg = checkpoint.get("config", {})
        hidden_dim = int(ckpt_cfg.get("hidden_dim", 256))
        latent_dim = int(ckpt_cfg.get("latent_dim", cfg.latent_dim or 32))
        input_dim_ckpt = int(ckpt_cfg.get("input_dim", input_dim))
        if input_dim_ckpt != input_dim:
            raise ValueError(
                "VAE checkpoint input_dim mismatch: checkpoint expects "
                f"{input_dim_ckpt}, but aggregated features have dim {input_dim}"
            )
        if cfg.latent_dim is not None and cfg.latent_dim != latent_dim:
            raise ValueError(
                f"Requested latent_dim {cfg.latent_dim} disagrees with checkpoint latent_dim {latent_dim}"
            )

        super().__init__(
            embedding_type="vae",
            input_dim=input_dim,
            output_dim=latent_dim,
            device=device,
            dtype=dtype,
        )

        self.model = _FeatureVAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device=device, dtype=torch.float32)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        # Determine normalization strategy
        ckpt_standardized = bool(ckpt_cfg.get("standardized", False))
        self.standardize_inputs = cfg.standardize if cfg.standardize is not None else ckpt_standardized
        self._mean: Optional[torch.Tensor]
        self._std: Optional[torch.Tensor]

        if self.standardize_inputs:
            stats_path = cfg.stats_path
            if stats_path is None:
                raise ValueError("stats_path is required when loading a standardized VAE embedding")
            stats = np.load(stats_path)
            self._mean = torch.from_numpy(stats["mean"].astype(np.float32)).to(device)
            self._std = torch.from_numpy(stats["std"].astype(np.float32)).clamp_min_(1e-6).to(device)
        else:
            mean_cfg = ckpt_cfg.get("mean")
            std_cfg = ckpt_cfg.get("std")
            if mean_cfg is not None and std_cfg is not None:
                self._mean = torch.tensor(mean_cfg, dtype=torch.float32, device=device)
                self._std = torch.tensor(std_cfg, dtype=torch.float32, device=device).clamp_min_(1e-6)
            else:
                self._mean = None
                self._std = None

        self._last_logvar: Optional[torch.Tensor] = None

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        input_device = features.device
        work = features.to(self.device, dtype=torch.float32)
        valid_mask = torch.any(work != 0.0, dim=-1)
        latents = torch.zeros(work.shape[0], self.output_dim, device=self.device, dtype=torch.float32)

        if valid_mask.any():
            with torch.no_grad():
                selected = work[valid_mask]
                if self.standardize_inputs and self._mean is not None and self._std is not None:
                    selected = (selected - self._mean) / self._std
                elif not self.standardize_inputs and self._mean is not None and self._std is not None:
                    # Optional centering if stats are available
                    selected = (selected - self._mean) / self._std
                mu, logvar = self.model.encode(selected)
                latents[valid_mask] = mu
                self._last_logvar = logvar
        else:
            self._last_logvar = None

        return latents.to(input_device, dtype=self.dtype)

    def describe(self) -> Dict[str, Any]:
        base = super().describe()
        base.update(
            {
                "standardize": self.standardize_inputs,
                "has_stats": bool(self._mean is not None and self._std is not None),
            }
        )
        return base


def build_teacher_embedding(cfg: Optional[TeacherEmbeddingConfig], input_dim: int) -> Optional[TeacherEmbedding]:
    if cfg is None:
        return None
    embedding_type = cfg.type.lower()
    if embedding_type in {"identity", "none"}:
        return IdentityEmbedding(input_dim, device=_resolve_device(cfg.device), dtype=_resolve_dtype(cfg.dtype))
    if embedding_type == "vae":
        return VAEEmbedding(cfg, input_dim)
    raise ValueError(f"Unsupported teacher embedding type '{cfg.type}'")
