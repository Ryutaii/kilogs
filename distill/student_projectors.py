"""Project student network activations into a feature space for distillation.

Stage 2 introduces feature-level supervision between the 3D-GS teacher and the
student neural renderer.  This module provides utilities to retrieve the most
recent pre-activation tensors produced by the student model and to pass them
through lightweight projection heads before computing feature losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:  # pragma: no cover
    from .lego_response_distill import StudentModel


class _HasLatentCache(Protocol):
    """Protocol for student implementations exposing cached activations."""

    last_pre_activation: Optional[torch.Tensor]
    last_input: Optional[torch.Tensor]


@dataclass(frozen=True)
class ProjectorConfig:
    """Configuration for the student feature projector head."""

    input_dim: int = 4
    hidden_dim: int = 64
    output_dim: int = 32
    activation: str = "relu"
    use_layer_norm: bool = False
    dropout: float = 0.0


@dataclass
class FeatureAdapterConfig:
    """Configuration for optional teacher/student adapters in feature space."""

    type: str = "linear"
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    activation: str = "identity"
    use_layer_norm: bool = False
    dropout: float = 0.0


def _resolve_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    key = name.lower()
    if key == "relu":
        return torch.relu
    if key == "silu":
        return torch.nn.functional.silu
    if key in {"gelu", "geglu"}:
        return torch.nn.functional.gelu
    if key == "tanh":
        return torch.tanh
    if key == "identity" or key == "none":
        return lambda x: x
    raise ValueError(f"Unsupported activation '{name}' for StudentFeatureProjector")


class _ActivationModule(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self._activation = _resolve_activation(name)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:  # noqa: D401 - simple proxy
        return self._activation(tensor)


class StudentFeatureProjector(nn.Module):
    """Tiny MLP that projects student activations into a common feature space."""

    def __init__(self, cfg: ProjectorConfig):
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = []
        in_dim = cfg.input_dim

        layers.append(nn.Linear(in_dim, cfg.hidden_dim))
        if cfg.use_layer_norm:
            layers.append(nn.LayerNorm(cfg.hidden_dim))
        layers.append(_ActivationModule(cfg.activation))
        if cfg.dropout > 0.0:
            layers.append(nn.Dropout(p=cfg.dropout))
        layers.append(nn.Linear(cfg.hidden_dim, cfg.output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        flat = features.reshape(-1, features.shape[-1])
        projected = self.network(flat)
        return projected.reshape(*features.shape[:-1], -1)


class FeatureAdapter(nn.Module):
    """Single-layer adapter to reconcile teacher/student feature dimensions."""

    def __init__(self, cfg: FeatureAdapterConfig):
        super().__init__()
        if cfg.type.lower() not in {"linear", "identity"}:
            raise ValueError(f"Unsupported adapter type '{cfg.type}'")
        if cfg.input_dim is None or cfg.output_dim is None:
            raise ValueError("FeatureAdapterConfig requires 'input_dim' and 'output_dim'")

        self.cfg = cfg
        layers: list[nn.Module] = []
        layers.append(nn.Linear(cfg.input_dim, cfg.output_dim))
        if cfg.use_layer_norm:
            layers.append(nn.LayerNorm(cfg.output_dim))
        activation_key = cfg.activation.lower()
        if activation_key not in {"identity", "none"}:
            layers.append(_ActivationModule(cfg.activation))
        if cfg.dropout > 0.0:
            layers.append(nn.Dropout(p=cfg.dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        last_dim = features.shape[-1]
        flat = features.reshape(-1, last_dim)
        adapted = self.network(flat)
        return adapted.reshape(*features.shape[:-1], adapted.shape[-1])


def extract_student_features(
    model: "StudentModel",
    *,
    source: str = "penultimate",
    activation: str = "post",
    dim: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return cached student activations following the requested policy.

    Parameters
    ----------
    model:
        Wrapper containing the concrete student implementation.
    source:
        Which cached tensor to favour. Supported values include ``"penultimate"``
        (default), ``"preactivation"``/``"linear"`` for the final layer logits,
        and ``"input"`` for the raw coordinates fed to the network.
    activation:
        When ``source="penultimate"``, choose between the post-activation
        (``"post"``) or pre-activation (``"pre"``) tensor. Any other value falls
        back to post-activation when available.
    dim:
        Optional hard expectation for the feature dimensionality. When provided
        and the cached tensor does not match, a warning is emitted and ``None``
        is returned to signal misconfiguration.
    """

    impl = getattr(model, "impl", model)

    source_key = (source or "penultimate").lower()
    act_key = (activation or "post").lower()

    penultimate_post = getattr(impl, "last_penultimate_post", None)
    penultimate_pre = getattr(impl, "last_penultimate_pre", None)
    penultimate_generic = getattr(impl, "last_penultimate", None)
    last_logits = getattr(impl, "last_pre_activation", None)
    last_input = getattr(impl, "last_input", None)

    def _first_available(*candidates: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        for tensor in candidates:
            if tensor is not None:
                return tensor
        return None

    features: Optional[torch.Tensor]

    if source_key in {"penultimate", "penult", "hidden"}:
        if act_key in {"pre", "raw", "none", "linear"}:
            features = _first_available(penultimate_pre, penultimate_generic)
        elif act_key == "both":
            features = _first_available(penultimate_post, penultimate_pre, penultimate_generic)
        else:
            features = _first_available(penultimate_post, penultimate_generic, penultimate_pre)
        if features is None:
            features = last_logits
    elif source_key in {"preactivation", "pre_activation", "pre", "logits", "linear"}:
        features = last_logits
    elif source_key in {"input", "inputs", "coords", "coordinates"}:
        features = last_input
    elif source_key in {"pre_penultimate", "penultimate_pre"}:
        features = _first_available(penultimate_pre, penultimate_generic)
    elif source_key in {"penultimate_post", "hidden_post"}:
        features = _first_available(penultimate_post, penultimate_generic)
    else:
        # Graceful fallback: try penultimate tensors, then logits.
        features = _first_available(penultimate_post, penultimate_generic, last_logits)

    if features is not None and dim is not None and features.shape[-1] != dim:
        print(
            "[feature_pipeline] student feature dimension mismatch: "
            f"expected {dim}, got {features.shape[-1]}; ignoring cached tensor."
        )
        features = None

    return features, last_input


__all__ = [
    "ProjectorConfig",
    "StudentFeatureProjector",
    "FeatureAdapterConfig",
    "FeatureAdapter",
    "extract_student_features",
]
