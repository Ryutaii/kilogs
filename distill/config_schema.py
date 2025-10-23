"""Lightweight schema validation utilities for distillation YAML configs."""
from __future__ import annotations

import numbers
from typing import Any, Dict, Iterable, Mapping, Sequence


_TOP_LEVEL_REQUIRED = {"experiment", "data", "teacher", "student", "train", "loss", "logging"}
_TOP_LEVEL_OPTIONAL = {"feature_pipeline", "feature_targets", "feature_aux_student", "dataset", "background"}


def _ensure_mapping(section: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(section, Mapping):
        raise ValueError(f"Section '{name}' must be a mapping; got {type(section).__name__}")
    return section


def _check_unknown_keys(section: Mapping[str, Any], allowed: Iterable[str], name: str) -> None:
    allowed_set = set(allowed)
    unknown = set(section.keys()) - allowed_set
    if unknown:
        raise ValueError(f"Unknown keys in '{name}': {sorted(unknown)}")


def _ensure_int(value: Any, name: str, *, min_value: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"'{name}' must be an integer")
    if min_value is not None and value < min_value:
        raise ValueError(f"'{name}' must be ≥ {min_value}")


def _ensure_float(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"'{name}' must be numeric")


def _ensure_list(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"'{name}' must be a list")
    return value


def _validate_experiment(section: Mapping[str, Any]) -> None:
    _check_unknown_keys(section, {"name", "seed", "output_dir", "progress_desc"}, "experiment")
    if "name" not in section or "seed" not in section or "output_dir" not in section:
        raise ValueError("'experiment' must define name, seed, and output_dir")
    _ensure_int(section["seed"], "experiment.seed")


def _validate_data(section: Mapping[str, Any]) -> None:
    allowed = {
        "dataset_root",
        "teacher_outputs",
        "teacher_depth_dir",
        "camera_json",
        "background_color",
        "batch_size",
        "ray_chunk",
        "near",
        "far",
        "samples_per_ray",
        "bbox_min",
        "bbox_max",
        "perturb",
    }
    _check_unknown_keys(section, allowed, "data")
    for required_key in ("dataset_root", "teacher_outputs", "camera_json", "background_color", "batch_size"):
        if required_key not in section:
            raise ValueError(f"'data.{required_key}' is required")
    _ensure_int(section["batch_size"], "data.batch_size", min_value=1)
    if "ray_chunk" in section:
        _ensure_int(section["ray_chunk"], "data.ray_chunk", min_value=1)
    if "samples_per_ray" in section:
        _ensure_int(section["samples_per_ray"], "data.samples_per_ray", min_value=1)
    background = _ensure_list(section["background_color"], "data.background_color")
    if len(background) != 3:
        raise ValueError("'data.background_color' must contain exactly 3 entries")
    for idx, value in enumerate(background):
        _ensure_float(value, f"data.background_color[{idx}]")


def _validate_teacher(section: Mapping[str, Any]) -> None:
    _check_unknown_keys(section, {"type", "checkpoint", "render_stats"}, "teacher")
    for key in ("type", "checkpoint", "render_stats"):
        if key not in section:
            raise ValueError(f"'teacher.{key}' is required")


def _validate_student(section: Mapping[str, Any]) -> None:
    allowed = {
        "type",
        "grid_resolution",
        "hidden_dim",
        "num_layers",
        "activation",
        "density_bias",
        "color_bias",
        "regularization_weight",
        "enable_boundary_blend",
        "boundary_blend_margin",
        "hash_levels",
        "hash_features_per_level",
        "hash_log2_hashmap_size",
        "hash_base_resolution",
        "hash_per_level_scale",
        "pos_encoding",
        "pos_L",
        "dir_encoding",
        "dir_L",
        "skips",
        "mlp_hidden",
        "sigma_activation",
        "sigma_bias",
    }
    _check_unknown_keys(section, allowed, "student")
    if "type" not in section:
        raise ValueError("'student.type' is required")


def _validate_train(section: Mapping[str, Any]) -> None:
    allowed = {
        "max_steps",
        "eval_interval",
        "checkpoint_interval",
        "lr",
        "lr_decay_steps",
        "lr_decay_gamma",
        "gradient_clip_norm",
        "ema_decay",
        "phases",
        "lr_schedule",
        "lr_milestones",
        "lr_values",
    "lr_schedule_min_lr",
        "lr_min",
        "lr_final",
        "lr_schedule_steps",
        "lr_warmup_steps",
        "promotion_gates",
        "promotion_min_mask_fraction",
        "promotion_feature_dim",
        "promotion_min_feature_scale",
        "promotion_min_feature_ratio",
        "promotion_min_opacity_ratio",
        "promotion_projector_in_dim",
        "promotion_require_feature_schedule_terminal",
        "promotion_require_opacity_schedule_terminal",
        "promotion_exit_code",
        "effective_weight_avg_window",
    }
    _check_unknown_keys(section, allowed, "train")
    if "max_steps" not in section:
        raise ValueError("'train.max_steps' is required")
    _ensure_int(section["max_steps"], "train.max_steps", min_value=1)
    if "phases" in section:
        phases = _ensure_list(section["phases"], "train.phases")
        for idx, phase in enumerate(phases):
            phase_map = _ensure_mapping(phase, f"train.phases[{idx}]")
            _check_unknown_keys(
                phase_map,
                {
                    "name",
                    "duration",
                    "end_step",
                    "optimize",
                    "train_modules",
                    "mask_override",
                    "feature_weight_scale",
                    "freeze_student",
                    "force_feature_on",
                    "phase_feature_scale",
                },
                f"train.phases[{idx}]",
            )


def _validate_loss(section: Mapping[str, Any]) -> None:
    allowed = {
        "color",
        "opacity",
        "depth",
        "feature",
        "feature_l2",
        "feature_cos",
        "distillation_temperature",
        "feature_schedule",
        "feature_schedule_duration",
        "feature_target_weight",
        "feature_target_cosine_weight",
        "alpha_guard",
    }
    _check_unknown_keys(section, allowed, "loss")
    for key in ("color", "opacity"):
        if key not in section:
            raise ValueError(f"'loss.{key}' is required")
    color = _ensure_mapping(section["color"], "loss.color")
    _check_unknown_keys(color, {"type", "weight", "eps", "epsilon"}, "loss.color")
    if "weight" in color:
        _ensure_float(color["weight"], "loss.color.weight")
    opacity_allowed = {
        "type",
        "weight",
        "target",
        "target_weight",
        "start_weight",
        "warmup_steps",
        "schedule",
        "schedule_duration",
        "background_threshold",
        "temperature",
        "lambda",
        "warmup_step",
        "schedule_mode",
        "hysteresis",
        "enable_hysteresis",
        "warm_start_offset",
        "max_weight",
    }
    opacity = _ensure_mapping(section["opacity"], "loss.opacity")
    _check_unknown_keys(opacity, opacity_allowed, "loss.opacity")
    if "weight" in opacity:
        _ensure_float(opacity["weight"], "loss.opacity.weight")
    if "warmup_steps" in opacity:
        _ensure_int(opacity["warmup_steps"], "loss.opacity.warmup_steps", min_value=0)
    if "depth" in section and section["depth"] is not None:
        depth = _ensure_mapping(section["depth"], "loss.depth")
        _check_unknown_keys(depth, {"type", "weight", "alpha_threshold"}, "loss.depth")
    if "feature" in section and section["feature"] is not None:
        feature_loss = _ensure_mapping(section["feature"], "loss.feature")
        feature_allowed = {
            "type",
            "weight",
            "cosine_weight",
            "warmup_steps",
            "schedule",
            "schedule_duration",
            "target_weight",
            "target_cosine_weight",
        }
        _check_unknown_keys(feature_loss, feature_allowed, "loss.feature")

    alpha_guard_section = section.get("alpha_guard")
    if alpha_guard_section is not None:
        alpha_guard_map = _ensure_mapping(alpha_guard_section, "loss.alpha_guard")
        _check_unknown_keys(
            alpha_guard_map,
            {
                "enabled",
                "check_interval",
                "penalty_hi",
                "penalty_lo",
                "tighten_rate",
                "relax_rate",
                "lambda_floor",
                "lambda_cap",
                "weight_floor",
                "weight_cap",
                "band_weight",
                "fraction_hi_weight",
                "fraction_lo_weight",
                "initial_weight",
                "avg_window",
                "min_target_weight",
                "warmup_enforce_steps",
                "enforce_warmup_steps",
                "adjustment_smoothing",
            },
            "loss.alpha_guard",
        )


def _validate_feature_pipeline(section: Mapping[str, Any]) -> None:
    allowed = {
        "enabled",
        "teacher_mode",
        "compare_space",
        "allow_dim_mismatch",
        "projector_input_dim",
        "projector_hidden_dim",
        "projector_output_dim",
        "projector_activation",
        "projector_use_layer_norm",
        "projector_dropout",
        "boundary_mask_threshold",
        "boundary_mask_soft_transition",
        "boundary_mask_soft_mode",
        "boundary_mask_soft_floor",
        "mask_controller",
        "student_head",
        "student_adapter",
        "teacher_adapter",
        "student_projector",
        "projector",
        "teacher_components",
        "student_features",
        "teacher_embedding",
    }
    _check_unknown_keys(section, allowed, "feature_pipeline")
    if "mask_controller" in section and section["mask_controller"] is not None:
        mask_controller = _ensure_mapping(section["mask_controller"], "feature_pipeline.mask_controller")
        mask_allowed = {
            "enabled",
            "activation_step",
            "activation_offset",
            "min_activation_step",
            "ramp_duration",
            "initial_threshold",
            "min_threshold",
            "min_fraction",
            "relaxation",
            "soft_transition_step",
            "cap_threshold",
            "emergency_fraction",
            "recovery_fraction",
        }
        _check_unknown_keys(mask_controller, mask_allowed, "feature_pipeline.mask_controller")
    for adapter_key in ("student_head", "student_adapter", "teacher_adapter"):
        if adapter_key in section and section[adapter_key] is not None:
            adapter = _ensure_mapping(section[adapter_key], f"feature_pipeline.{adapter_key}")
            _check_unknown_keys(
                adapter,
                {
                    "type",
                    "input_dim",
                    "in_dim",
                    "hidden_dim",
                    "output_dim",
                    "out_dim",
                    "activation",
                    "act",
                    "use_layer_norm",
                    "norm",
                    "dropout",
                },
                f"feature_pipeline.{adapter_key}",
            )
    if "student_projector" in section and section["student_projector"] is not None:
        projector = _ensure_mapping(section["student_projector"], "feature_pipeline.student_projector")
        _check_unknown_keys(projector, {"input_dim", "hidden_dim", "output_dim", "activation", "use_layer_norm", "dropout"}, "feature_pipeline.student_projector")
    if "projector" in section and section["projector"] is not None:
        projector = _ensure_mapping(section["projector"], "feature_pipeline.projector")
        _check_unknown_keys(
            projector,
            {"in_dim", "out_dim", "hidden_dim", "activation", "use_layer_norm", "dropout"},
            "feature_pipeline.projector",
        )


def _validate_logging(section: Mapping[str, Any]) -> None:
    _check_unknown_keys(
        section,
        {
            "tensorboard",
            "csv",
            "render_preview_interval",
            "log_interval",
            "metrics_interval",
            "tensorboard_flush_secs",
            "tensorboard_axis",
        },
        "logging",
    )
    for key in ("tensorboard", "csv"):
        if key not in section:
            raise ValueError(f"'logging.{key}' is required")
    if "log_interval" in section:
        _ensure_int(section["log_interval"], "logging.log_interval", min_value=1)
    if "metrics_interval" in section:
        _ensure_int(section["metrics_interval"], "logging.metrics_interval", min_value=1)
    if "tensorboard_flush_secs" in section:
        _ensure_int(section["tensorboard_flush_secs"], "logging.tensorboard_flush_secs", min_value=1)
    axis_mode = section.get("tensorboard_axis")
    if axis_mode is not None:
        if not isinstance(axis_mode, str):
            raise ValueError("logging.tensorboard_axis must be a string if provided")
        axis_mode_normalized = axis_mode.strip().lower()
        if axis_mode_normalized not in {"time", "step", "elapsed"}:
            raise ValueError("logging.tensorboard_axis must be one of {'time', 'step', 'elapsed'}")


def _validate_feature_targets(section: Mapping[str, Any]) -> None:
    _check_unknown_keys(section, {"l2_target", "cos_target"}, "feature_targets")


def _validate_feature_aux_student(section: Mapping[str, Any]) -> None:
    _check_unknown_keys(
        section,
        {"enabled", "source", "loss", "normalize", "weight", "patch"},
        "feature_aux_student",
    )

    weight_block = section.get("weight")
    if weight_block is not None:
        weight_map = _ensure_mapping(weight_block, "feature_aux_student.weight")
        _check_unknown_keys(
            weight_map,
            {"start", "target", "warmup_steps", "schedule", "schedule_duration"},
            "feature_aux_student.weight",
        )
        if "start" in weight_map:
            _ensure_float(weight_map["start"], "feature_aux_student.weight.start")
        if "target" in weight_map:
            _ensure_float(weight_map["target"], "feature_aux_student.weight.target")
        if "warmup_steps" in weight_map:
            _ensure_int(weight_map["warmup_steps"], "feature_aux_student.weight.warmup_steps", min_value=0)
        if "schedule_duration" in weight_map:
            _ensure_int(
                weight_map["schedule_duration"],
                "feature_aux_student.weight.schedule_duration",
                min_value=0,
            )

    patch_block = section.get("patch")
    if patch_block is not None:
        patch_map = _ensure_mapping(patch_block, "feature_aux_student.patch")
        _check_unknown_keys(patch_map, {"rays_per_patch", "stride"}, "feature_aux_student.patch")
        if "rays_per_patch" in patch_map:
            _ensure_int(
                patch_map["rays_per_patch"],
                "feature_aux_student.patch.rays_per_patch",
                min_value=1,
            )
        if "stride" in patch_map:
            _ensure_int(patch_map["stride"], "feature_aux_student.patch.stride", min_value=1)


def validate_config_dict(cfg: Dict[str, Any]) -> None:
    """Validate config dictionary, raising ValueError on first schema violation."""

    root = _ensure_mapping(cfg, "root")
    required_missing = _TOP_LEVEL_REQUIRED - set(root.keys())
    if required_missing:
        raise ValueError(f"Missing required top-level sections: {sorted(required_missing)}")
    allowed_top = _TOP_LEVEL_REQUIRED | _TOP_LEVEL_OPTIONAL
    extra_top = set(root.keys()) - allowed_top
    if extra_top:
        raise ValueError(f"Unknown top-level sections: {sorted(extra_top)}")

    _validate_experiment(_ensure_mapping(root["experiment"], "experiment"))
    _validate_data(_ensure_mapping(root["data"], "data"))
    _validate_teacher(_ensure_mapping(root["teacher"], "teacher"))
    _validate_student(_ensure_mapping(root["student"], "student"))
    _validate_train(_ensure_mapping(root["train"], "train"))
    _validate_loss(_ensure_mapping(root["loss"], "loss"))
    _validate_logging(_ensure_mapping(root["logging"], "logging"))
    if "feature_pipeline" in root and root["feature_pipeline"] is not None:
        _validate_feature_pipeline(_ensure_mapping(root["feature_pipeline"], "feature_pipeline"))
    if "feature_targets" in root and root["feature_targets"] is not None:
        _validate_feature_targets(_ensure_mapping(root["feature_targets"], "feature_targets"))
    if "feature_aux_student" in root and root["feature_aux_student"] is not None:
        _validate_feature_aux_student(_ensure_mapping(root["feature_aux_student"], "feature_aux_student"))


__all__ = ["validate_config_dict"]
