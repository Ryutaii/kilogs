"""Utilities for Stage 2 feature distillation.

This module provides a small helper that turns the feature-loss knobs defined in
``lego_response_distill.LossConfig`` into ready-to-use PyTorch tensors.
It intentionally keeps the public surface narrow so that we can iterate on the
teacher / student feature plumbing without touching the training loop logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Protocol

import torch


class _FeatureLossConfig(Protocol):
    """Protocol mirroring the feature-loss fields on ``LossConfig``.

    We use a ``Protocol`` here to avoid importing ``lego_response_distill``
    directly, which would introduce a circular dependency during module import.
    Any object that exposes the attributes below will be accepted.
    """

    feature_weight: float
    feature_type: str
    feature_cosine_weight: float
    feature_warmup_steps: int
    feature_schedule: str
    feature_schedule_duration: int
    feature_target_weight: Optional[float]
    feature_target_cosine_weight: Optional[float]


@dataclass
class FeatureLossBreakdown:
    """Named return value for feature-loss computation."""

    recon: torch.Tensor
    cosine: torch.Tensor
    total: torch.Tensor


def _reduce_loss(tensor: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "l1":
        return tensor.abs().mean()
    if reduction == "l2":
        return (tensor ** 2).mean()
    if reduction in {"smooth_l1", "huber"}:
        return torch.nn.functional.smooth_l1_loss(tensor, torch.zeros_like(tensor))
    raise ValueError(f"Unsupported feature loss type: {reduction}")


def _cosine_distance(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    teacher_flat = teacher.reshape(teacher.shape[0], -1)
    student_flat = student.reshape(student.shape[0], -1)
    teacher_norm = torch.nn.functional.normalize(teacher_flat, dim=-1)
    student_norm = torch.nn.functional.normalize(student_flat, dim=-1)
    # cosine *similarity* in [-1, 1]; convert to distance in [0, 2]
    distance = 1.0 - (teacher_norm * student_norm).sum(dim=-1)
    return distance.mean()


class FeatureDistiller:
    """Coordinate feature-based supervision between teacher and student."""

    def __init__(
        self,
        loss_cfg: Optional[_FeatureLossConfig] = None,
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        loss_cfg = loss_cfg or SimpleNamespace(
            feature_weight=0.0,
            feature_type="l2",
            feature_cosine_weight=0.0,
            feature_warmup_steps=0,
            feature_schedule="none",
            feature_schedule_duration=0,
            feature_target_weight=None,
            feature_target_cosine_weight=None,
        )
        self.device = device
        self.cfg = loss_cfg
        self.current_recon_weight: float = 0.0
        self.current_cosine_weight: float = 0.0

    @property
    def enabled(self) -> bool:
        return any(
            weight > 0.0
            for weight in (
                float(getattr(self.cfg, "feature_weight", 0.0) or 0.0),
                float(getattr(self.cfg, "feature_cosine_weight", 0.0) or 0.0),
                float(getattr(self.cfg, "feature_target_weight", 0.0) or 0.0),
                float(getattr(self.cfg, "feature_target_cosine_weight", 0.0) or 0.0),
            )
        )

    def maybe_move(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None or self.device is None:
            return tensor
        return tensor.to(self.device)

    def _effective_weights(self, global_step: int) -> tuple[float, float]:
        warmup_steps = max(int(getattr(self.cfg, "feature_warmup_steps", 0) or 0), 0)
        if global_step < warmup_steps:
            return 0.0, 0.0

        start_recon = float(getattr(self.cfg, "feature_weight", 0.0) or 0.0)
        start_cos = float(getattr(self.cfg, "feature_cosine_weight", 0.0) or 0.0)

        target_recon = getattr(self.cfg, "feature_target_weight", None)
        target_cos = getattr(self.cfg, "feature_target_cosine_weight", None)
        if target_recon is None:
            target_recon = start_recon
        if target_cos is None:
            target_cos = start_cos

        schedule = str(getattr(self.cfg, "feature_schedule", "none") or "none").lower()
        duration = max(int(getattr(self.cfg, "feature_schedule_duration", 0) or 0), 0)

        def _apply_schedule(start: float, target: float) -> float:
            if duration <= 0 or schedule in {"none", "constant"}:
                return float(target)
            progress = min(max((global_step - warmup_steps) / max(duration, 1), 0.0), 1.0)
            if schedule == "linear":
                return float(start + (target - start) * progress)
            if schedule == "cosine":
                return float(start + (target - start) * 0.5 * (1.0 - math.cos(math.pi * progress)))
            return float(target)

        recon_weight = _apply_schedule(start_recon, float(target_recon))
        cosine_weight = _apply_schedule(start_cos, float(target_cos))

        return recon_weight, cosine_weight

    def terminal_reached(self, global_step: int) -> bool:
        if not self.enabled:
            return False

        warmup_steps = max(int(getattr(self.cfg, "feature_warmup_steps", 0) or 0), 0)
        schedule = str(getattr(self.cfg, "feature_schedule", "none") or "none").lower()
        duration = max(int(getattr(self.cfg, "feature_schedule_duration", 0) or 0), 0)
        terminal_step = warmup_steps
        if duration > 0 and schedule not in {"none", "constant"}:
            terminal_step = warmup_steps + duration

        if global_step < terminal_step:
            return False

        start_recon = float(getattr(self.cfg, "feature_weight", 0.0) or 0.0)
        start_cos = float(getattr(self.cfg, "feature_cosine_weight", 0.0) or 0.0)
        target_recon_raw = getattr(self.cfg, "feature_target_weight", None)
        target_cos_raw = getattr(self.cfg, "feature_target_cosine_weight", None)
        final_recon = float(target_recon_raw) if target_recon_raw is not None else start_recon
        final_cos = float(target_cos_raw) if target_cos_raw is not None else start_cos

        recon_weight, cosine_weight = self._effective_weights(global_step)

        def _within(goal: float, current: float, start_value: float) -> bool:
            tolerance = max(abs(goal) * 1e-3, 1e-6)
            if goal >= start_value:
                return current >= goal - tolerance
            return current <= goal + tolerance

        recon_ok = _within(final_recon, recon_weight, start_recon)
        cos_ok = _within(final_cos, cosine_weight, start_cos)

        return recon_ok and cos_ok

    def should_run(self, global_step: int) -> bool:
        if not self.enabled:
            self.current_recon_weight = 0.0
            self.current_cosine_weight = 0.0
            return False
        recon_weight, cosine_weight = self._effective_weights(global_step)
        self.current_recon_weight = recon_weight
        self.current_cosine_weight = cosine_weight
        return (recon_weight > 0.0) or (cosine_weight > 0.0)

    def prepare_teacher_batch(
        self,
        teacher_features: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Detach & freeze teacher features before the training loop consumes them."""

        prepared: Dict[str, torch.Tensor] = {}
        for name, feat in teacher_features.items():
            prepared[name] = self.maybe_move(feat).detach()
        return prepared

    def __call__(
        self,
        teacher_features: Mapping[str, torch.Tensor],
        student_features: Mapping[str, torch.Tensor],
        *,
        mask: Optional[torch.Tensor] = None,
        global_step: int = 0,
    ) -> FeatureLossBreakdown:
        """Compute the weighted feature loss.

        Parameters
        ----------
        teacher_features: mapping from feature names to tensors produced by the teacher.
        student_features: mapping from feature names to tensors produced by the student.
        mask: optional {0,1} tensor broadcastable to the feature shapes to ignore invalid rays.
        global_step: training step; used for warm-up gating.
        """

        base_device = self.device or next(iter(student_features.values())).device

        if not self.should_run(global_step):
            zero = torch.zeros((), device=base_device)
            return FeatureLossBreakdown(recon=zero, cosine=zero, total=zero)

        recon_weight = torch.tensor(self.current_recon_weight, device=base_device, dtype=torch.float32)
        cosine_weight = torch.tensor(self.current_cosine_weight, device=base_device, dtype=torch.float32)

        if teacher_features.keys() != student_features.keys():
            missing_teacher = set(student_features) - set(teacher_features)
            missing_student = set(teacher_features) - set(student_features)
            raise KeyError(
                "Teacher/student feature key mismatch: "
                f"missing_teacher={missing_teacher}, missing_student={missing_student}"
            )

        if mask is not None:
            mask = mask.to(next(iter(student_features.values())).device)

        recon_losses: List[torch.Tensor] = []
        for name in teacher_features.keys():
            teacher_feat = teacher_features[name]
            student_feat = student_features[name]

            if teacher_feat.shape != student_feat.shape:
                raise ValueError(
                    f"Feature '{name}' shape mismatch: teacher {teacher_feat.shape} vs student {student_feat.shape}"
                )

            if mask is not None:
                broadcast_mask = mask
                while broadcast_mask.dim() < teacher_feat.dim():
                    broadcast_mask = broadcast_mask.unsqueeze(-1)
                teacher_feat = teacher_feat * broadcast_mask
                student_feat = student_feat * broadcast_mask

            delta = student_feat - teacher_feat
            recon_losses.append(_reduce_loss(delta, self.cfg.feature_type))

        recon_loss = (
            torch.stack(recon_losses).mean() if recon_losses else torch.zeros((), device=base_device)
        )
        recon_loss = recon_loss.to(base_device)

        cosine_loss = torch.zeros((), device=recon_loss.device)
        if cosine_weight.item() > 0.0 and teacher_features:
            cosine_terms = []
            for name in teacher_features.keys():
                cosine_terms.append(
                    _cosine_distance(student_features[name], teacher_features[name]).to(recon_loss.device)
                )
            cosine_loss = torch.stack(cosine_terms).mean()

        total = recon_loss * recon_weight + cosine_loss * cosine_weight
        return FeatureLossBreakdown(recon=recon_loss, cosine=cosine_loss, total=total)


__all__ = [
    "FeatureDistiller",
    "FeatureLossBreakdown",
]
