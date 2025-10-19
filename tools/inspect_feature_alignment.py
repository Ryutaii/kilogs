#!/usr/bin/env python3
"""Inspect teacher vs. student feature alignment for Stage 2 distillation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from distill.feature_distillation import FeatureDistiller
from distill.feature_embeddings import build_teacher_embedding
from distill.lego_response_distill import (
    LegoRayDataset,
    StudentModel,
    _build_gaussian_cell_features,
    parse_config,
    sample_along_rays,
    _exclusive_cumprod_last,
    set_seed,
)
from distill.student_projectors import (
    ProjectorConfig as StudentProjectorConfig,
    StudentFeatureProjector,
    extract_student_features,
)
from distill.teacher_features import GaussianTeacherFeatures


def _tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    if tensor.numel() == 0:
        return {"numel": 0}
    return {
        "numel": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def inspect_alignment(
    config_path: Path,
    checkpoint_path: Path,
    *,
    num_rays: int,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    (
        _experiment_cfg,
        data_cfg,
        teacher_cfg,
        student_cfg,
        _train_cfg,
        loss_cfg,
        _logging_cfg,
        feature_cfg,
        _feature_aux_cfg,
    ) = parse_config(config_path)

    # Deterministic guard for analysis runs
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    os.environ.setdefault("PYTHONHASHSEED", str(int(_experiment_cfg.seed)))
    try:
        set_seed(int(_experiment_cfg.seed))
    except Exception:
        pass

    dataset = LegoRayDataset(data_cfg)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    student_model = StudentModel(student_cfg).to(torch_device)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device)
    student_model.load_state_dict(checkpoint["model_state"])
    student_model.eval()

    feature_pipeline_active = feature_cfg.enabled and (
        loss_cfg.feature_weight > 0.0 or loss_cfg.feature_cosine_weight > 0.0
    )

    projector: Optional[StudentFeatureProjector] = None
    feature_distiller: Optional[FeatureDistiller] = None
    gaussian_cell_features: Optional[torch.Tensor] = None
    teacher_embedding_info: Optional[Dict[str, Any]] = None
    resolved_teacher_dim: Optional[int] = None

    if feature_pipeline_active:
        projector_output_dim = feature_cfg.projector_output_dim

        wants_gaussian_features = feature_cfg.teacher_mode.startswith("gaussian") or bool(feature_cfg.teacher_components)

        if wants_gaussian_features:
            gaussian_teacher = GaussianTeacherFeatures.from_ply(teacher_cfg.checkpoint)
            bbox_min = torch.tensor(data_cfg.bbox_min, dtype=torch.float32)
            bbox_max = torch.tensor(data_cfg.bbox_max, dtype=torch.float32)
            gaussian_cell_features = _build_gaussian_cell_features(
                gaussian_teacher,
                student_cfg.grid_resolution,
                bbox_min,
                bbox_max,
                mode=feature_cfg.teacher_mode,
                components=feature_cfg.teacher_components,
            )
            if gaussian_cell_features.numel() > 0:
                resolved_teacher_dim = int(gaussian_cell_features.shape[-1])
                embedding_cfg = feature_cfg.teacher_embedding
                if embedding_cfg is not None:
                    try:
                        embedding_impl = build_teacher_embedding(embedding_cfg, resolved_teacher_dim)
                    except Exception as err:
                        print(f"[inspect_feature_alignment] Failed to initialise teacher embedding: {err}")
                    else:
                        transformed = embedding_impl.transform(gaussian_cell_features)
                        gaussian_cell_features = transformed
                        resolved_teacher_dim = int(transformed.shape[-1]) if transformed.numel() > 0 else 0
                        teacher_embedding_info = embedding_impl.describe()
                gaussian_cell_features = gaussian_cell_features.to(torch_device)
                if projector_output_dim != resolved_teacher_dim and resolved_teacher_dim is not None:
                    print(
                        "[inspect_feature_alignment] projector_output_dim="
                        f"{projector_output_dim} mismatches teacher feature dim {resolved_teacher_dim}; overriding to match checkpoint."
                    )
                    projector_output_dim = resolved_teacher_dim

        projector_cfg = StudentProjectorConfig(
            input_dim=4,
            hidden_dim=feature_cfg.projector_hidden_dim,
            output_dim=projector_output_dim,
            activation=feature_cfg.projector_activation,
            use_layer_norm=feature_cfg.projector_use_layer_norm,
        )
        projector = StudentFeatureProjector(projector_cfg).to(torch_device)
        projector_state = checkpoint.get("feature_projector_state")
        if projector_state is not None:
            projector.load_state_dict(projector_state)
        projector.eval()

        feature_distiller = FeatureDistiller(loss_cfg, device=torch_device)

    batch = dataset.sample_random_rays(num_rays, torch_device)

    rays_o = batch["rays_o"]
    rays_d = batch["rays_d"]
    near = batch["near"]
    far = batch["far"]

    bbox_min = torch.tensor(data_cfg.bbox_min, dtype=torch.float32, device=torch_device)
    bbox_max = torch.tensor(data_cfg.bbox_max, dtype=torch.float32, device=torch_device)
    bbox_extent = bbox_max - bbox_min

    with torch.no_grad():
        pts, z_vals = sample_along_rays(
            rays_o=rays_o,
            rays_d=rays_d,
            near=near,
            far=far,
            num_samples=data_cfg.samples_per_ray,
            perturb=False,
        )
        pts_norm = (pts - bbox_min) / bbox_extent
        pts_norm = pts_norm.clamp(0.0, 1.0)
        pts_flat = pts_norm.view(-1, 3)

        student_rgb_samples, student_sigma_samples = student_model(pts_flat)
        student_rgb_samples = student_rgb_samples.view(-1, data_cfg.samples_per_ray, 3)
        student_sigma_samples = student_sigma_samples.view(-1, data_cfg.samples_per_ray)

        deltas = z_vals[:, 1:] - z_vals[:, :-1]
        delta_last = torch.full((deltas.shape[0], 1), 1e10, device=torch_device)
        deltas = torch.cat([deltas, delta_last], dim=-1)

        alpha = 1.0 - torch.exp(-student_sigma_samples * deltas)
        transmittance = _exclusive_cumprod_last(1.0 - alpha + 1e-10)
        weights = alpha * transmittance

        feature_stats: Dict[str, Any] = {
            "feature_pipeline_active": feature_pipeline_active,
            "mask_fraction": None,
            "feature_loss": None,
        }
        feature_stats["teacher_feature_dim"] = resolved_teacher_dim
        if teacher_embedding_info is not None:
            feature_stats["teacher_embedding"] = teacher_embedding_info

        if feature_pipeline_active and projector is not None and feature_distiller is not None:
            student_pre, _ = extract_student_features(student_model)
            if student_pre is None or student_pre.numel() == 0:
                feature_stats["warning"] = "Student implementation did not expose cached activations."
            else:
                try:
                    student_pre = student_pre.view(
                        student_rgb_samples.shape[0],
                        data_cfg.samples_per_ray,
                        student_pre.shape[-1],
                    )
                except RuntimeError:
                    feature_stats["warning"] = "Unable to reshape student pre-activations; skipping alignment."
                else:
                    teacher_feature_tensor: Optional[torch.Tensor] = None
                    student_projected: Optional[torch.Tensor] = None
                    feature_mask: Optional[torch.Tensor] = None
                    primary_weight_sum_stats: Optional[Dict[str, float]] = None

                    gaussian_mode = feature_cfg.teacher_mode
                    gaussian_enabled = (
                        gaussian_cell_features is not None
                        and (gaussian_mode.startswith("gaussian") or bool(feature_cfg.teacher_components))
                    )
                    if gaussian_enabled:
                        impl = getattr(student_model, "impl", student_model)
                        cell_indices = getattr(impl, "last_linear_indices", None)
                        expected = student_pre.numel() // student_pre.shape[-1]
                        if cell_indices is not None and cell_indices.numel() == expected:
                            cell_indices = cell_indices.to(torch_device)
                            cells_per_ray = cell_indices.view(student_rgb_samples.shape[0], data_cfg.samples_per_ray)

                            primary_sample = torch.argmax(weights, dim=-1)
                            primary_cells = cells_per_ray.gather(-1, primary_sample.unsqueeze(-1)).squeeze(-1)
                            same_cell_mask = cells_per_ray == primary_cells.unsqueeze(-1)
                            primary_weights = weights * same_cell_mask
                            summed_weights = primary_weights.sum(dim=-1)
                            summed_safe = summed_weights.unsqueeze(-1).clamp_min(1e-6)
                            primary_weights_norm = primary_weights / summed_safe

                            zero_weight_mask = summed_weights <= 1e-6
                            if torch.any(zero_weight_mask):
                                same_cell_float = same_cell_mask.float()
                                same_cell_count = same_cell_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
                                uniform_weights = same_cell_float / same_cell_count
                                primary_weights_norm[zero_weight_mask] = uniform_weights[zero_weight_mask]
                                summed_weights = summed_weights.masked_fill(zero_weight_mask, 1.0)

                            gathered = gaussian_cell_features.index_select(0, cell_indices)
                            gathered = gathered.view(
                                student_rgb_samples.shape[0],
                                data_cfg.samples_per_ray,
                                gathered.shape[-1],
                            )
                            teacher_feature_tensor = torch.sum(
                                primary_weights_norm[..., None] * gathered,
                                dim=-2,
                            )
                            student_cell_features = torch.sum(
                                primary_weights_norm[..., None] * student_pre,
                                dim=-2,
                            )
                            student_projected = projector(student_cell_features)

                            if feature_cfg.boundary_mask_threshold is not None:
                                boundary_threshold = float(feature_cfg.boundary_mask_threshold)
                                feature_mask = (summed_weights >= boundary_threshold).to(student_projected.device)
                            else:
                                feature_mask = None

                            weight_stats = _tensor_stats(summed_weights)
                            primary_weight_sum_stats = weight_stats if weight_stats["numel"] > 0 else None
                        else:
                            feature_stats["warning"] = (
                                "Gaussian feature alignment failed (cell index mismatch); falling back to RGB supervision."
                            )
                            gaussian_enabled = False

                    if not gaussian_enabled:
                        weighted_features = torch.sum(weights[..., None] * student_pre, dim=-2)
                        student_projected = projector(weighted_features)
                        teacher_feature_tensor = batch["teacher_rgb"].to(student_projected.device)
                        feature_mask = None

                    if teacher_feature_tensor is not None and student_projected is not None:
                        teacher_feature_tensor = teacher_feature_tensor.to(student_projected.device)
                        mask_fraction = None
                        if feature_mask is not None:
                            mask_fraction = float(feature_mask.float().mean().item())
                        feature_stats["mask_fraction"] = mask_fraction
                        feature_stats["primary_weight_sum_stats"] = primary_weight_sum_stats
                        feature_stats["teacher_stats"] = _tensor_stats(teacher_feature_tensor)
                        feature_stats["student_stats"] = _tensor_stats(student_projected)

                        breakdown = feature_distiller(
                            {"primary": teacher_feature_tensor.detach()},
                            {"primary": student_projected},
                            mask=feature_mask,
                            global_step=loss_cfg.feature_warmup_steps + 1,
                        )
                        feature_stats["feature_loss"] = {
                            "recon": float(breakdown.recon.item()),
                            "cosine": float(breakdown.cosine.item()),
                            "total": float(breakdown.total.item()),
                        }

                        diff = (student_projected - teacher_feature_tensor).detach()
                        feature_stats["delta_stats"] = _tensor_stats(diff)
                    else:
                        feature_stats["warning"] = feature_stats.get("warning") or "Feature tensors unavailable."
                        if primary_weight_sum_stats is not None:
                            feature_stats["primary_weight_sum_stats"] = primary_weight_sum_stats
        else:
            feature_stats["info"] = (
                "Feature pipeline disabled or projector state missing; only response losses are active."
            )

    return feature_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Training config YAML used for the checkpoint")
    parser.add_argument("checkpoint", type=Path, help="Checkpoint to inspect")
    parser.add_argument("--num-rays", type=int, default=8192, help="Number of random rays to sample")
    parser.add_argument("--device", type=str, default=None, help="Torch device override (default: auto)")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save JSON report")
    args = parser.parse_args()

    stats = inspect_alignment(
        args.config,
        args.checkpoint,
        num_rays=args.num_rays,
        device=args.device,
    )

    print(json.dumps(stats, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as fp:
            json.dump(stats, fp, indent=2)


if __name__ == "__main__":
    main()
