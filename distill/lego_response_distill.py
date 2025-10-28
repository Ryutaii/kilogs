import argparse
import atexit
import csv
import ctypes
import io
import json
import math
import hashlib
import statistics
import os
import random
import shutil
import signal
import sys
import time
import zlib

print(f"[debug] importing lego_response_distill from {__file__}", flush=True)
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple, Callable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from distill.config_schema import validate_config_dict
from imageio import v2 as imageio
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:  # pragma: no cover - optional dependency wiring
    from distill.quicklook import generate_quicklook, QuicklookGenerationError
except Exception:  # noqa: BLE001 - fall back when matplotlib unavailable
    generate_quicklook = None  # type: ignore[assignment]

    class QuicklookGenerationError(RuntimeError):
        """Placeholder raised when quicklook support is unavailable."""

METRICS_SCHEMA_VERSION = 4

_DEFAULT_PROMOTION_GATES: Tuple[int, int, int] = (10000, 20000, 50000)

_SIGINT_GUARD_STATE = {
    "installed": False,
    "armed": False,
    "original": None,
}

_TEARDOWN_HOOK: Optional[Callable[[str, int], None]] = None


def _register_teardown_hook(callback: Optional[Callable[[str, int], None]]) -> None:
    global _TEARDOWN_HOOK
    _TEARDOWN_HOOK = callback


def _invoke_teardown(reason: str, exit_code: int = 130) -> None:
    hook = _TEARDOWN_HOOK
    if hook is not None:
        try:
            hook(reason, exit_code)
        except Exception as err:  # pragma: no cover - last resort logging
            print(f"[teardown] hook raised {err}", file=sys.stderr)


def _invoke_teardown_for_test(reason: str = "TEST_SIGINT", exit_code: int = 130) -> None:
    _invoke_teardown(reason, exit_code)


def _clean_tensorboard_events(log_dir: Path) -> None:
    if not log_dir.exists():
        return
    patterns = ("events.out.tfevents.*", "*.tfevents.*")
    for pattern in patterns:
        for candidate in log_dir.glob(pattern):
            if candidate.is_file():
                try:
                    candidate.unlink()
                except OSError:
                    pass


_TQDM_DISABLE_ENV_KEYS = (
    "TQDM_DISABLE",
    "DISABLE_TQDM",
    "TQDM_DEFAULT_DISABLE",
)

_TTY_FALLBACK = None


def _resolve_tty_stream():
    global _TTY_FALLBACK
    if _TTY_FALLBACK is not None:
        return _TTY_FALLBACK
    candidate_streams = [getattr(sys, name, None) for name in ("__stdout__", "__stderr__", "__stdin__")]
    for stream in candidate_streams:
        if stream is None:
            continue
        try:
            if not stream.isatty():
                continue
        except Exception:  # pragma: no cover - conservative
            continue
        writable = True
        write_callable = getattr(stream, "write", None)
        if write_callable is None:
            writable = False
        try:
            writable_flag = getattr(stream, "writable", None)
            if callable(writable_flag):
                writable = bool(writable_flag())
        except Exception:
            writable = False
        if not writable:
            continue
        try:
            stream.write("")
            stream.flush()
        except Exception:
            continue
        _TTY_FALLBACK = stream
        return _TTY_FALLBACK

    tty_paths = []
    for stream in candidate_streams:
        if stream is None:
            continue
        try:
            fd = stream.fileno()
            if fd >= 0:
                tty_paths.append(os.ttyname(fd))
        except Exception:
            continue
    try:
        ctermid_path = os.ctermid()
        if ctermid_path:
            tty_paths.append(ctermid_path)
    except Exception:
        pass
    tty_paths.append("/dev/tty")

    for path in tty_paths:
        if not path:
            continue
        try:
            tty = open(path, "w", buffering=1, encoding="utf-8", errors="ignore")
        except Exception:
            continue
        try:
            tty.write("")
            tty.flush()
        except Exception:
            try:
                tty.close()
            except Exception:
                pass
            continue
        _TTY_FALLBACK = tty
        return tty

    _TTY_FALLBACK = None
    return None


class _ProgressStreamProxy:
    """Wrap a stream to convince tqdm it can render in-place even when piped."""

    def __init__(self, stream):
        self._stream = stream
        self._tty = _resolve_tty_stream()

    def write(self, data):
        if not data:
            return 0
        if isinstance(data, bytes):
            try:
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                text = data.decode(errors="ignore")
        else:
            try:
                text = str(data)
            except Exception:
                text = ""
        tty = self._tty
        tty_usable = False
        if tty is not None:
            try:
                tty.write(text)
                tty.flush()
                tty_usable = True
            except Exception:
                tty_usable = False
        if "\n" not in text and tty_usable:
            return len(text)
        return self._stream.write(text)

    def flush(self):
        if self._tty is not None:
            try:
                self._tty.flush()
            except Exception:
                pass
        return self._stream.flush()

    def isatty(self):
        return True

    def fileno(self):
        try:
            return self._stream.fileno()
        except (AttributeError, io.UnsupportedOperation):
            try:
                return sys.__stdout__.fileno()  # type: ignore[attr-defined]
            except Exception:
                return 1

    def __getattr__(self, name):  # pragma: no cover - simple delegation
        return getattr(self._stream, name)


class _TTYOnlyStream:
    """File-like wrapper that forwards tqdm output exclusively to a TTY."""

    def __init__(self, tty_stream):
        self._tty = tty_stream

    def write(self, data):
        if not data:
            return 0
        if isinstance(data, bytes):
            text = data.decode("utf-8", errors="ignore")
        else:
            text = str(data)
        try:
            written = self._tty.write(text)
        except Exception:
            return 0
        return written if written is not None else len(text)

    def flush(self):
        try:
            self._tty.flush()
        except Exception:
            pass

    def isatty(self):
        return True

    @property
    def encoding(self):  # pragma: no cover - compatibility shim
        return getattr(self._tty, "encoding", "utf-8")

    def close(self):  # pragma: no cover - tqdm compatibility
        pass


def _install_sigint_guard() -> None:
    if _SIGINT_GUARD_STATE["installed"]:
        return
    original_handler = signal.getsignal(signal.SIGINT)

    def _handler(signum, frame):  # type: ignore[override]
        if not _SIGINT_GUARD_STATE["armed"]:
            _SIGINT_GUARD_STATE["armed"] = True
            print(
                "[signal] Ctrl+C detected. Press again to confirm interruption (first signal ignored).",
                flush=True,
            )
            return
        _invoke_teardown("SIGINT", exit_code=130)
        handler = _SIGINT_GUARD_STATE["original"]
        if callable(handler):
            handler(signum, frame)
        elif handler in {signal.SIG_IGN, None}:
            sys.exit(130)
        else:
            signal.default_int_handler(signum, frame)

    _SIGINT_GUARD_STATE["installed"] = True
    _SIGINT_GUARD_STATE["original"] = original_handler
    signal.signal(signal.SIGINT, _handler)


def _normalise_disable_flag(flag_value: Optional[object]) -> Optional[bool]:
    if flag_value is None:
        return None
    if isinstance(flag_value, bool):
        return flag_value
    if isinstance(flag_value, (int, float)):
        return bool(flag_value)
    if isinstance(flag_value, str):
        normalised = flag_value.strip().lower()
        if normalised in {"0", "false", "off", "no"}:
            return False
        if normalised in {"1", "true", "on", "yes"}:
            return True
    return bool(flag_value)


def _coerce_bool(value: Optional[object], default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"1", "true", "yes", "y", "on"}:
            return True
        if normalised in {"0", "false", "no", "n", "off"}:
            return False
    return default


def create_progress(iterable=None, **kwargs):
    """Ensure tqdm bars render even when stdout is wrapped or env-disabled."""

    for env_key in _TQDM_DISABLE_ENV_KEYS:
        env_value = os.environ.get(env_key)
        if env_value:
            if _normalise_disable_flag(env_value):
                os.environ[env_key] = "0"
            else:
                os.environ[env_key] = "0"

    requested_disable = _normalise_disable_flag(kwargs.pop("disable", None))

    defaults = {
        "dynamic_ncols": True,
        "disable": False if requested_disable is None else requested_disable,
        "mininterval": 0.1,
        "maxinterval": 1.0,
        "smoothing": 0.0,
        "ascii": True,
        "file": sys.stdout,
    }
    defaults.update(kwargs)
    defaults.setdefault(
        "bar_format",
        "{desc} [{n_fmt}/{total_fmt}]: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    progress_stream = defaults.get("file", sys.stdout)
    needs_proxy = True
    try:
        needs_proxy = not bool(progress_stream.isatty())  # type: ignore[attr-defined]
    except Exception:
        needs_proxy = True
    tty_only_stream = None
    if needs_proxy:
        tty_candidate = _resolve_tty_stream()
        writable_tty = None
        if tty_candidate is not None:
            try:
                tty_candidate.write("")
                tty_candidate.flush()
                writable_tty = tty_candidate
            except Exception:
                writable_tty = None
        if writable_tty is not None:
            tty_only_stream = _TTYOnlyStream(writable_tty)
            defaults["file"] = tty_only_stream
        else:
            defaults["disable"] = True
            progress_stream = _ProgressStreamProxy(progress_stream)
            defaults["file"] = progress_stream
    else:
        defaults["file"] = progress_stream

    if not defaults["disable"]:
        defaults["disable"] = False

    bar = tqdm(iterable, **defaults)
    if getattr(bar, "disable", False) and not defaults["disable"]:
        try:
            bar.disable = False
            bar.refresh()
        except Exception:
            pass
    if tty_only_stream is not None:
        original_write = bar.write

        def _tty_safe_write(line, *, file=None, end="\n", nolock=False):
            target = file if file is not None else sys.stdout
            return tqdm.write(line, file=target, end=end, nolock=nolock)

        bar.write = _tty_safe_write  # type: ignore[assignment]
        bar._kilogs_tty_only_stream = tty_only_stream  # type: ignore[attr-defined]
        bar._kilogs_write_original = original_write  # type: ignore[attr-defined]
    return bar

def _quat_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
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

try:
    import tinycudann as tcnn  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tcnn = None

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent

for path in (CURRENT_DIR, PROJECT_ROOT, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

# Ensure kilonerf repositories (and compiled extensions) are importable before
# we attempt to load the student implementation. Without this, kilonerf_cuda
# may fail to resolve libc10.so before the fallback logic has a chance to add
# the CUDA directory to sys.path, leaving _HAS_KILONERF_CUDA False even though
# the extension is available.
KILONERF_DIR = REPO_ROOT / "kilonerf"
KILONERF_CUDA_DIR = KILONERF_DIR / "cuda"
for extra_path in (KILONERF_DIR, KILONERF_CUDA_DIR):
    if extra_path.exists():
        extra_path_str = str(extra_path)
        if extra_path_str not in sys.path:
            sys.path.append(extra_path_str)


def _preload_torch_shared_objects() -> None:
    torch_lib_dir = Path(torch.__file__).resolve().parent
    candidate_dirs = [torch_lib_dir, torch_lib_dir / "lib"]
    loaded_any = False
    for directory in candidate_dirs:
        if not directory.is_dir():
            continue
        for name in ("libc10.so", "libtorch_cpu.so", "libtorch_cuda.so", "libtorch.so"):
            lib_path = directory / name
            if not lib_path.exists():
                continue
            try:
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                loaded_any = True
            except OSError:
                continue
        if loaded_any:
            break


_preload_torch_shared_objects()

from distill.feature_distillation import FeatureDistiller, FeatureLossBreakdown
from distill.feature_embeddings import TeacherEmbeddingConfig, build_teacher_embedding
from distill.student_projectors import (
    FeatureAdapter,
    FeatureAdapterConfig,
    ProjectorConfig as StudentProjectorConfig,
    StudentFeatureProjector,
    extract_student_features,
)
from distill.teacher_features import GaussianTeacherFeatures

_KILONERF_IMPORT_ERROR = None

try:
    from kilonerf.multi_modules import MultiNetwork, _HAS_KILONERF_CUDA, kilonerf_cuda  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    kilonerf_dir = REPO_ROOT / "kilonerf"
    if kilonerf_dir.exists():
        kilonerf_path = str(kilonerf_dir)
        if kilonerf_path not in sys.path:
            sys.path.append(kilonerf_path)
        kilonerf_cuda_dir = kilonerf_dir / "cuda"
        if kilonerf_cuda_dir.exists():
            cuda_path = str(kilonerf_cuda_dir)
            if cuda_path not in sys.path:
                sys.path.append(cuda_path)
        try:
            from multi_modules import MultiNetwork, _HAS_KILONERF_CUDA, kilonerf_cuda  # type: ignore
        except ImportError as fallback_exc:  # pragma: no cover - optional dependency
            MultiNetwork = None
            kilonerf_cuda = None  # type: ignore
            _HAS_KILONERF_CUDA = False
            _KILONERF_IMPORT_ERROR = fallback_exc
        else:  # pragma: no cover - executed when fallback succeeds
            _KILONERF_IMPORT_ERROR = None
    else:
        MultiNetwork = None
        kilonerf_cuda = None  # type: ignore
        _HAS_KILONERF_CUDA = False
        _KILONERF_IMPORT_ERROR = exc
else:  # pragma: no cover - executed when primary import succeeds
    _KILONERF_IMPORT_ERROR = None


_KILONERF_RUNTIME_INITIALISED = False


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    output_dir: Path
    progress_desc: Optional[str] = None


@dataclass
class DataConfig:
    dataset_root: Path
    teacher_outputs: Path
    teacher_depth_dir: Optional[Path]
    camera_json: Path
    background_color: Tuple[float, float, float]
    batch_size: int
    ray_chunk: int
    near: float
    far: float
    samples_per_ray: int
    bbox_min: Tuple[float, float, float]
    bbox_max: Tuple[float, float, float]
    perturb: bool
    max_frames: Optional[int] = None
    frame_indices: Tuple[int, ...] = tuple()


@dataclass
class TeacherConfig:
    type: str
    checkpoint: Path
    render_stats: Path


@dataclass
class StudentConfig:
    type: str
    grid_resolution: Tuple[int, int, int]
    hidden_dim: int
    num_layers: int
    activation: str
    density_bias: float
    color_bias: float
    regularization_weight: float
    enable_boundary_blend: bool = False
    boundary_blend_margin: float = 0.05
    hash_levels: int = 16
    hash_features_per_level: int = 2
    hash_log2_hashmap_size: int = 19
    hash_base_resolution: int = 16
    hash_per_level_scale: float = 1.5
    pos_encoding: str = "none"
    pos_L: int = 0
    dir_encoding: str = "none"
    dir_L: int = 0
    skips: Tuple[int, ...] = tuple()
    mlp_hidden: Tuple[int, ...] = tuple()
    sigma_activation: str = "relu"
    sigma_bias: Optional[float] = None


@dataclass
class TrainPhaseConfig:
    name: str
    start_step: int
    end_step: int
    optimize: Tuple[str, ...] = ("student", "projector", "student_adapter", "teacher_adapter")
    mask_override: Optional[str] = None
    feature_weight_scale: float = 1.0

    def contains(self, step: int) -> bool:
        return self.start_step < step <= self.end_step


@dataclass
class AlphaGuardConfig:
    enabled: bool = True
    check_interval: int = 200
    penalty_hi: float = 0.15
    penalty_lo: float = 0.05
    tighten_rate: float = 0.9
    relax_rate: float = 1.02
    lambda_floor: float = 0.1
    lambda_cap: float = 1.0
    weight_floor: float = 0.12
    weight_cap: float = 0.45
    band_weight: float = 1.0
    fraction_hi_weight: float = 1.5
    fraction_lo_weight: float = 1.0
    initial_weight: float = 0.2
    avg_window: int = 256
    min_target_weight: float = 0.05
    warmup_enforce_steps: int = 0
    adjustment_smoothing: float = 0.25
    hysteresis_margin: float = 0.02
    min_update_samples: int = 2
    max_lambda_delta: float = 0.05
    max_target_adjustment_delta: float = 0.1
    max_penalty_weight_delta: float = 0.05


@dataclass
class MaskPrefailConfig:
    enabled: bool = True
    window: int = 64
    p5_drop_rate: float = 5e-4
    min_drop_rate: float = 1e-4
    threshold_scale: float = 0.8
    soft_floor_delta: float = 0.03
    variance_ceiling: float = 2.5e-5
    cooldown_steps: int = 200


@dataclass
class TrainConfig:
    max_steps: int
    eval_interval: int
    checkpoint_interval: int
    lr: float
    lr_decay_steps: int
    lr_decay_gamma: float
    gradient_clip_norm: float
    ema_decay: float
    lr_schedule: str
    lr_schedule_milestones: Tuple[int, ...]
    lr_schedule_values: Tuple[float, ...]
    lr_schedule_min_lr: Optional[float]
    lr_schedule_steps: Optional[int]
    lr_warmup_steps: int
    phases: Tuple[TrainPhaseConfig, ...] = tuple()
    promotion_gates: Tuple[int, ...] = tuple()
    promotion_min_mask_fraction: float = 0.0
    promotion_feature_dim: Optional[int] = None
    promotion_min_feature_scale: float = 0.0
    promotion_exit_code: int = 12
    effective_weight_avg_window: int = 256
    promotion_min_feature_ratio: float = 0.55
    promotion_min_opacity_ratio: float = 0.6
    promotion_projector_in_dim: Optional[int] = None
    promotion_require_feature_schedule_terminal: bool = True
    promotion_require_opacity_schedule_terminal: bool = True
    alpha_guard: AlphaGuardConfig = field(default_factory=AlphaGuardConfig)
    mask_prefail: MaskPrefailConfig = field(default_factory=MaskPrefailConfig)
    input_guard_notice_interval: float = 0.0


@dataclass(frozen=True)
class PromotionGateResolution:
    gates: Tuple[int, ...]
    auto_filled: bool
    trimmed: Tuple[int, ...]


def _resolve_promotion_gates(train_cfg: TrainConfig) -> PromotionGateResolution:
    configured: List[int] = []
    for gate in train_cfg.promotion_gates:
        try:
            gate_int = int(gate)
        except (TypeError, ValueError):
            raise TrainingAbort(
                f"Invalid promotion gate value '{gate}' (must be positive integer)",
                exit_code=train_cfg.promotion_exit_code,
            )
        if gate_int > 0:
            configured.append(gate_int)
    unique_sorted = tuple(sorted(set(configured)))
    max_steps = int(train_cfg.max_steps)
    if max_steps > 0:
        trimmed = tuple(g for g in unique_sorted if g > max_steps)
        filtered = tuple(g for g in unique_sorted if g <= max_steps)
    else:
        trimmed = tuple()
        filtered = unique_sorted
    if filtered:
        return PromotionGateResolution(filtered, False, trimmed)
    defaults = tuple(g for g in _DEFAULT_PROMOTION_GATES if max_steps <= 0 or g <= max_steps)
    if defaults:
        return PromotionGateResolution(defaults, True, trimmed)
    return PromotionGateResolution(tuple(), False, trimmed)


class TrainingAbort(Exception):
    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class PromotionGateFailure(TrainingAbort):
    pass


@dataclass
class LossConfig:
    color_weight: float
    opacity_weight: float
    color_type: str = "l2"
    color_eps: float = 1e-3
    color_secondary_type: Optional[str] = None
    color_secondary_weight: float = 0.0
    color_secondary_eps: float = 1e-3
    opacity_type: str = "l1"
    temperature: float = 1.0
    opacity_temperature: Optional[float] = None
    opacity_lambda: float = 1.0
    depth_weight: float = 0.0
    depth_type: str = "l1"
    depth_alpha_threshold: float = 0.0
    feature_weight: float = 0.0
    feature_type: str = "l2"
    feature_cosine_weight: float = 0.0
    feature_warmup_steps: int = 0
    feature_schedule: str = "none"
    feature_schedule_duration: int = 0
    feature_target_weight: Optional[float] = None
    feature_target_cosine_weight: Optional[float] = None
    opacity_target: Optional[float] = None
    opacity_target_weight: float = 0.0
    opacity_target_background_threshold: float = 0.05
    opacity_target_start_weight: Optional[float] = None
    opacity_target_warmup_steps: int = 0
    opacity_target_schedule: str = "none"
    opacity_target_schedule_duration: int = 0
    opacity_target_warm_start_offset: int = 0
    background_color: Optional[Tuple[float, float, float]] = None
    opacity_target_hysteresis: bool = True
    opacity_target_max_weight: Optional[float] = None
    opacity_mean_target: Optional[float] = None
    opacity_mean_weight: float = 0.0


@dataclass
class OpacityTargetScheduler:
    config: LossConfig
    last_weight: Optional[float] = None
    max_weight: Optional[float] = None
    warm_start_offset: int = 0

    def __post_init__(self) -> None:
        offset = getattr(self.config, "opacity_target_warm_start_offset", 0)
        try:
            parsed = int(offset)
        except (TypeError, ValueError):
            parsed = 0
        self.warm_start_offset = max(parsed, 0)

    def state_dict(self) -> Dict[str, Optional[float]]:
        return {
            "last_weight": self.last_weight,
            "max_weight": self.max_weight,
        }

    def load_state_dict(self, state: Optional[Dict[str, Optional[float]]]) -> None:
        if not state:
            return
        last = state.get("last_weight") if isinstance(state, dict) else None
        max_val = state.get("max_weight") if isinstance(state, dict) else None
        self.last_weight = float(last) if last is not None else None
        self.max_weight = float(max_val) if max_val is not None else None
        if (
            self.max_weight is not None
            and self.last_weight is not None
            and self.max_weight + 1e-6 < self.last_weight
        ):
            raise ValueError(
                "OpacityTargetScheduler state invalid: max_weight decreased across resume"
            )

    def compute(self, step: int) -> float:
        cfg = self.config
        target_weight = float(cfg.opacity_target_weight or 0.0)
        start_weight_cfg = getattr(cfg, "opacity_target_start_weight", None)
        start_weight = float(start_weight_cfg) if start_weight_cfg is not None else target_weight
        start_weight = max(0.0, start_weight)

        warmup_cfg = int(getattr(cfg, "opacity_target_warmup_steps", 0) or 0)
        warmup_cfg = max(0, warmup_cfg)

        schedule_mode = str(getattr(cfg, "opacity_target_schedule", "none") or "none").lower()
        schedule_duration = int(getattr(cfg, "opacity_target_schedule_duration", 0) or 0)
        schedule_duration = max(0, schedule_duration)

        increasing = target_weight >= start_weight
        lower_bound = min(start_weight, target_weight)
        upper_bound = max(start_weight, target_weight)

        cap_raw = getattr(cfg, "opacity_target_max_weight", None)
        cap_value: Optional[float]
        if cap_raw is None:
            cap_value = None
        else:
            try:
                cap_value = float(cap_raw)
            except (TypeError, ValueError):
                cap_value = None
            if cap_value is not None and cap_value < 0.0:
                cap_value = 0.0
        if cap_value is not None:
            start_weight = min(start_weight, cap_value)
            target_weight = min(target_weight, cap_value)
            upper_bound = min(upper_bound, cap_value)

        if not increasing:
            self.max_weight = None

        def _apply_monotonic(candidate: float, allow_hysteresis: bool = True) -> float:
            value = max(0.0, float(candidate))
            if cap_value is not None:
                value = min(value, cap_value)
            if increasing:
                value = max(value, start_weight)
                if self.last_weight is not None:
                    value = max(value, self.last_weight)
                value = min(value, upper_bound)
                if allow_hysteresis and cfg.opacity_target_hysteresis:
                    current_max = self.max_weight if self.max_weight is not None else value
                    current_max = min(max(current_max, value), upper_bound)
                    self.max_weight = current_max
                    value = max(value, current_max)
            else:
                value = min(value, start_weight)
                if self.last_weight is not None:
                    value = min(value, self.last_weight)
                value = max(value, lower_bound)
            return value

        effective_step = step + self.warm_start_offset

        if effective_step <= warmup_cfg:
            warmup_weight = _apply_monotonic(start_weight)
            self.last_weight = warmup_weight
            return warmup_weight

        if schedule_duration <= 0 or schedule_mode in {"none", "constant", "fixed"}:
            weight = max(0.0, target_weight)
        else:
            progress = (effective_step - warmup_cfg) / float(max(schedule_duration, 1))
            progress = min(max(progress, 0.0), 1.0)

            if schedule_mode in {"linear", "lin"}:
                weight = start_weight + (target_weight - start_weight) * progress
            elif schedule_mode in {"cos", "cosine", "cosine_decay"}:
                weight = start_weight + (target_weight - start_weight) * 0.5 * (1.0 - math.cos(math.pi * progress))
            else:
                weight = start_weight + (target_weight - start_weight) * progress
        regulated_weight = _apply_monotonic(weight)
        self.last_weight = regulated_weight
        return regulated_weight

    def prime_to(self, step: int) -> None:
        if step <= 0:
            return
        if self.last_weight is not None:
            return
        for current in range(1, step + 1):
            self.compute(current)

    def terminal_reached(self, step: int) -> bool:
        cfg = self.config
        warmup = max(int(getattr(cfg, "opacity_target_warmup_steps", 0) or 0), 0)
        schedule_duration = max(int(getattr(cfg, "opacity_target_schedule_duration", 0) or 0), 0)
        schedule_mode = str(getattr(cfg, "opacity_target_schedule", "none") or "none").lower()
        target_weight = float(getattr(cfg, "opacity_target_weight", 0.0) or 0.0)
        start_weight_cfg = getattr(cfg, "opacity_target_start_weight", None)
        start_weight = float(start_weight_cfg) if start_weight_cfg is not None else target_weight
        start_weight = max(0.0, start_weight)
        increasing = target_weight >= start_weight
        lower_bound = min(start_weight, target_weight)
        upper_bound = max(start_weight, target_weight)

        terminal_step = warmup
        if schedule_duration > 0 and schedule_mode not in {"none", "constant", "fixed"}:
            terminal_step = warmup + schedule_duration

        effective_step = step + self.warm_start_offset

        if effective_step < terminal_step:
            return False
        if self.last_weight is None:
            return False

        expected = upper_bound if increasing else lower_bound
        tolerance = max(abs(expected) * 1e-3, 1e-6)
        if increasing:
            return self.last_weight >= expected - tolerance
        return self.last_weight <= expected + tolerance


@dataclass
class MaskThresholdController:
    base_threshold: Optional[float]
    base_soft_transition: float
    schedule: Sequence[Tuple[int, Optional[float]]]
    min_threshold: float = 0.10
    relaxation: float = 0.05
    min_fraction: float = 0.20
    soft_transition_step: float = 0.05

    _threshold_override: Optional[float] = None
    _soft_override: Optional[float] = None

    def __post_init__(self) -> None:
        self.schedule = tuple(sorted(self.schedule, key=lambda item: item[0]))

    def state_dict(self) -> Dict[str, Optional[float]]:
        return {
            "threshold_override": self._threshold_override,
            "soft_override": self._soft_override,
        }

    def load_state_dict(self, state: Optional[Dict[str, Optional[float]]]) -> None:
        if not state:
            return
        self._threshold_override = state.get("threshold_override")
        self._soft_override = state.get("soft_override")

    def _scheduled_threshold(self, step: int) -> Optional[float]:
        current: Optional[float] = None
        for boundary, value in self.schedule:
            if step >= boundary:
                current = value
            else:
                break
        return current

    def current_threshold(self, step: int) -> Optional[float]:
        scheduled = self._scheduled_threshold(step)
        if self._threshold_override is None:
            return scheduled
        if scheduled is None:
            return self._threshold_override
        return min(scheduled, self._threshold_override)

    def current_soft_transition(self) -> float:
        if self._soft_override is not None:
            return self._soft_override
        return self.base_soft_transition

    def observe(self, step: int, mask_fraction: Optional[float]) -> None:
        if mask_fraction is None or math.isnan(mask_fraction):
            return
        if mask_fraction >= self.min_fraction:
            return
        extreme_low = mask_fraction < (self.min_fraction * 0.25)
        current = self.current_threshold(step)
        candidate = current
        if candidate is None and self.base_threshold is not None:
            candidate = self.base_threshold
        if candidate is not None:
            new_threshold = max(self.min_threshold, candidate - self.relaxation)
            if extreme_low:
                new_threshold = self.min_threshold
            self._threshold_override = new_threshold
        soft_current = self.current_soft_transition()
        updated_soft = soft_current + self.soft_transition_step
        if extreme_low:
            updated_soft = max(updated_soft, soft_current + self.soft_transition_step * 2.0)
        self._soft_override = min(updated_soft, 1.0)

    def force_minimum(self) -> None:
        self._threshold_override = self.min_threshold
        soft_current = self.current_soft_transition()
        self._soft_override = min(max(soft_current, self.base_soft_transition) + self.soft_transition_step, 1.0)

    def force_threshold(self, threshold: float) -> None:
        clamped = max(self.min_threshold, float(threshold))
        self._threshold_override = clamped
        soft_current = self.current_soft_transition()
        baseline = max(self.base_soft_transition, 0.0)
        self._soft_override = min(max(soft_current, baseline) + self.soft_transition_step, 1.0)

    def apply_prefail(self, step: int, *, threshold_scale: float, soft_delta: float) -> float:
        current = self.current_threshold(step)
        if current is None:
            if self.base_threshold is not None:
                current = float(self.base_threshold)
            else:
                current = float(self.min_threshold)
        scaled = max(self.min_threshold, float(current) * float(threshold_scale))
        self._threshold_override = scaled
        soft_current = self.current_soft_transition()
        self._soft_override = min(max(soft_current, 0.0) + float(soft_delta), 1.0)
        return scaled

    def relax_towards_schedule(self, step: int, *, immediate: bool = False) -> None:
        scheduled = self._scheduled_threshold(step)
        target_threshold = scheduled if scheduled is not None else self.base_threshold

        if immediate:
            self._threshold_override = None
        elif self._threshold_override is not None:
            if target_threshold is None:
                self._threshold_override = None
            else:
                step_size = max(self.relaxation, 0.0)
                candidate = self._threshold_override + step_size
                if candidate >= target_threshold:
                    self._threshold_override = None
                else:
                    self._threshold_override = candidate

        base_soft = max(0.0, float(self.base_soft_transition))
        if self._soft_override is None:
            return
        if base_soft <= 0.0:
            self._soft_override = None
            return

        decrement = self.soft_transition_step * (2.0 if immediate else 1.0)
        decrement = max(decrement, 0.0)
        new_soft = max(base_soft, self._soft_override - decrement)
        if new_soft <= base_soft + 1e-6:
            self._soft_override = base_soft
        else:
            self._soft_override = min(new_soft, 1.0)


@dataclass
class LoggingConfig:
    tensorboard: Path
    csv: Path
    render_preview_interval: int
    scalar_interval: int = 100
    tensorboard_flush_secs: Optional[int] = None
    tensorboard_axis: str = "step"


@dataclass
class MaskControllerConfig:
    enabled: bool = True
    activation_step: Optional[int] = None
    activation_offset: int = 3000
    min_activation_step: int = 7000
    ramp_duration: int = 2000
    initial_threshold: Optional[float] = None
    min_threshold: float = 0.03
    min_fraction: float = 0.20
    relaxation: float = 0.05
    soft_transition_step: float = 0.05
    cap_threshold: Optional[float] = 0.30
    emergency_fraction: float = 0.05
    recovery_fraction: float = 0.15


@dataclass
class FeaturePipelineConfig:
    enabled: bool = False
    teacher_mode: str = "rgb"
    teacher_components: Tuple[str, ...] = ()
    projector_input_dim: int = 4
    projector_hidden_dim: int = 64
    projector_output_dim: int = 3
    projector_activation: str = "relu"
    projector_use_layer_norm: bool = False
    projector_dropout: float = 0.0
    compare_space: str = "auto"
    allow_dim_mismatch: bool = False
    student_feature_source: str = "penultimate"
    student_feature_activation: str = "post"
    student_feature_dim: Optional[int] = None
    student_head: Optional[FeatureAdapterConfig] = None
    teacher_adapter: Optional[FeatureAdapterConfig] = None
    boundary_mask_threshold: Optional[float] = 0.75
    boundary_mask_soft_transition: float = 0.0
    boundary_mask_soft_mode: str = "linear"
    boundary_mask_soft_floor: float = 0.0
    mask_controller: Optional[MaskControllerConfig] = None
    teacher_embedding: Optional[TeacherEmbeddingConfig] = None
    resolved_teacher_dim: Optional[int] = None
    resolved_student_dim: Optional[int] = None
    resolved_teacher_raw_dim: Optional[int] = None
    resolved_comparison_dim: Optional[int] = None
    resolved_embedding_type: Optional[str] = None


@dataclass
class FeatureAuxStudentConfig:
    enabled: bool = False
    source: str = "penultimate_post"
    loss: str = "patch_cosine"
    weight_start: float = 0.0
    weight_target: float = 0.0
    weight_warmup_steps: int = 0
    weight_schedule: str = "none"
    weight_schedule_duration: int = 0
    patch_rays: int = 16
    patch_stride: int = 1
    normalize: Optional[str] = None


def _enforce_student_space_policy(
    cfg: FeaturePipelineConfig,
    *,
    exit_code: int,
    expected_dim: int = 128,
) -> None:
    if not cfg.enabled:
        return

    compare_space = str(getattr(cfg, "compare_space", "")).strip().lower()
    if compare_space != "student":
        if compare_space:
            print(
                "[feature_pipeline] student-space policy bypassed for compare_space="
                f"'{compare_space}'"
            )
        return

    def _require_dim(value: Optional[int], label: str) -> None:
        if value is None:
            raise TrainingAbort(
                f"Feature pipeline {label} not set; expected {expected_dim}.",
                exit_code=exit_code,
            )
        actual = int(value)
        if actual != expected_dim:
            raise TrainingAbort(
                f"Feature pipeline {label}={actual} differs from required {expected_dim}.",
                exit_code=exit_code,
            )

    _require_dim(getattr(cfg, "projector_input_dim", None), "projector_input_dim")
    _require_dim(getattr(cfg, "projector_output_dim", None), "projector_output_dim")
    resolved_student = getattr(cfg, "resolved_comparison_dim", None) or getattr(cfg, "resolved_student_dim", None)
    _require_dim(resolved_student, "student_feature_dim")
    _require_dim(getattr(cfg, "resolved_teacher_dim", None), "teacher_feature_dim")

_GAUSSIAN_MODE_COMPONENTS = {
    "gaussian_dc": ("sh_dc",),
    "gaussian_sh": ("sh",),
    "gaussian_dc_opacity": ("sh_dc", "opacity"),
    "gaussian_sh_opacity": ("sh", "opacity"),
    "gaussian_sh_opacity_logscale": ("sh", "opacity", "log_scale"),
    "gaussian_sh_opacity_rotation": ("sh", "opacity", "rotation_quat"),
    "gaussian_full": ("sh", "opacity", "log_scale", "rotation_quat", "position_normalized"),
    "gaussian_all": (
        "sh",
        "opacity",
        "opacity_logit",
        "log_scale",
        "scale",
        "rotation_quat",
        "rotation_matrix",
        "covariance_matrix",
        "position",
        "position_normalized",
    ),
}


def _resolve_gaussian_components(mode: str, components: Sequence[str]) -> Tuple[str, ...]:
    if components:
        return tuple(str(comp) for comp in components)
    key = mode.lower()
    if key in _GAUSSIAN_MODE_COMPONENTS:
        return _GAUSSIAN_MODE_COMPONENTS[key]
    if key.startswith("gaussian_") and key not in _GAUSSIAN_MODE_COMPONENTS:
        raise ValueError(
            f"Unsupported gaussian teacher feature mode '{mode}'. Known modes: {sorted(_GAUSSIAN_MODE_COMPONENTS)}"
        )
    return ()


def _parse_feature_adapter_config(raw: Optional[dict]) -> Optional[FeatureAdapterConfig]:
    if not isinstance(raw, dict):
        return None

    def _maybe_int(value: Optional[object]) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    activation = str(raw.get("activation", raw.get("act", "identity")))
    norm_value = raw.get("use_layer_norm", raw.get("norm"))
    if isinstance(norm_value, str):
        norm_key = norm_value.lower()
        use_layer_norm = norm_key in {"layernorm", "layer_norm", "ln", "true", "1", "yes"}
    else:
        use_layer_norm = bool(norm_value)

    return FeatureAdapterConfig(
        type=str(raw.get("type", "linear")),
        input_dim=_maybe_int(raw.get("in_dim", raw.get("input_dim"))),
        output_dim=_maybe_int(raw.get("out_dim", raw.get("output_dim"))),
        activation=activation,
        use_layer_norm=use_layer_norm,
        dropout=float(raw.get("dropout", 0.0) or 0.0),
    )


def _build_gaussian_cell_features(
    gaussians: GaussianTeacherFeatures,
    grid_resolution: Tuple[int, int, int],
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    *,
    mode: str = "gaussian_dc",
    components: Sequence[str] = (),
) -> torch.Tensor:
    """Aggregate Gaussian-derived features per KiloNeRF cell."""

    positions = gaussians.attributes.positions
    sh_coeffs = gaussians.attributes.sh_coeffs
    opacity = gaussians.attributes.opacity
    scaling = gaussians.attributes.scaling
    rotation = gaussians.attributes.rotation

    if positions.numel() == 0:
        gx, gy, gz = grid_resolution
        resolved_components = _resolve_gaussian_components(mode, components)
        if not resolved_components:
            raise ValueError("No gaussian feature components resolved for empty teacher features.")
        dummy_dim = 0
        if "sh" in resolved_components:
            dummy_dim += sh_coeffs.shape[1] * sh_coeffs.shape[2]
        if "sh_dc" in resolved_components:
            dummy_dim += sh_coeffs.shape[1]
        if "opacity" in resolved_components or "opacity_logit" in resolved_components:
            dummy_dim += 1
        if "log_scale" in resolved_components or "scale" in resolved_components:
            dummy_dim += 3
        if "rotation_quat" in resolved_components:
            dummy_dim += 4
        if "covariance_matrix" in resolved_components:
            dummy_dim += 9
        if "rotation_matrix" in resolved_components:
            dummy_dim += 9
        if "position" in resolved_components or "position_normalized" in resolved_components:
            dummy_dim += 3
        feature_dim = dummy_dim
        return positions.new_zeros((gx * gy * gz, feature_dim))

    bbox_min = bbox_min.to(device=positions.device, dtype=positions.dtype)
    bbox_max = bbox_max.to(device=positions.device, dtype=positions.dtype)
    extent = (bbox_max - bbox_min).clamp_min(1e-6)

    norm_pos = (positions - bbox_min) / extent
    norm_pos = norm_pos.clamp(0.0, 1.0 - 1e-6)

    resolved_components = _resolve_gaussian_components(mode, components)
    if not resolved_components:
        raise ValueError(
            "Gaussian feature aggregation requires either a recognised teacher_mode or explicit teacher_components."
        )

    feature_parts = []
    for comp in resolved_components:
        if comp == "sh":
            feature_parts.append(sh_coeffs.reshape(sh_coeffs.shape[0], -1))
        elif comp == "sh_dc":
            feature_parts.append(sh_coeffs[:, :, 0])
        elif comp == "opacity":
            feature_parts.append(torch.sigmoid(opacity))
        elif comp == "opacity_logit":
            feature_parts.append(opacity)
        elif comp == "log_scale":
            feature_parts.append(scaling)
        elif comp == "scale":
            feature_parts.append(torch.exp(scaling))
        elif comp == "rotation_quat":
            feature_parts.append(torch.nn.functional.normalize(rotation, dim=-1))
        elif comp == "rotation_matrix":
            rot_mats = _quat_to_rotation_matrix(rotation)
            feature_parts.append(rot_mats.reshape(rot_mats.shape[0], -1))
        elif comp == "covariance_matrix":
            cov = gaussians.covariance_matrices()
            feature_parts.append(cov.reshape(cov.shape[0], -1))
        elif comp == "position":
            feature_parts.append(positions)
        elif comp == "position_normalized":
            feature_parts.append(norm_pos)
        else:
            raise ValueError(f"Unsupported gaussian feature component '{comp}'")

    feature_values = torch.cat(feature_parts, dim=-1)

    gx, gy, gz = (int(v) for v in grid_resolution)
    grid_scale = torch.tensor([gx, gy, gz], device=positions.device, dtype=positions.dtype)
    indices = torch.floor(norm_pos * grid_scale).to(torch.int64)
    indices[:, 0].clamp_(0, gx - 1)
    indices[:, 1].clamp_(0, gy - 1)
    indices[:, 2].clamp_(0, gz - 1)

    lin_idx = (
        indices[:, 0]
        + indices[:, 1] * gx
        + indices[:, 2] * gx * gy
    )

    num_cells = gx * gy * gz
    device = positions.device
    feature_dim = feature_values.shape[-1]
    features_sum = torch.zeros((num_cells, feature_dim), device=device, dtype=positions.dtype)
    counts = torch.zeros((num_cells,), device=device, dtype=positions.dtype)

    features_sum.index_add_(0, lin_idx, feature_values)
    ones = torch.ones_like(lin_idx, dtype=positions.dtype, device=device)
    counts.index_add_(0, lin_idx, ones)

    mask = counts > 0
    counts_safe = counts.clone()
    counts_safe[mask] = 1.0 / counts_safe[mask]
    counts_safe = counts_safe.unsqueeze(-1)
    aggregated = features_sum * counts_safe
    aggregated[~mask] = 0.0
    return aggregated


def _compute_aux_weight(cfg: FeatureAuxStudentConfig, step: int) -> float:
    if not cfg.enabled:
        return 0.0

    warmup = max(int(cfg.weight_warmup_steps), 0)
    if step < warmup:
        return 0.0

    start_weight = float(cfg.weight_start)
    target_weight = float(cfg.weight_target)

    if abs(start_weight) <= 0.0 and abs(target_weight) <= 0.0:
        return 0.0

    schedule = (cfg.weight_schedule or "none").lower()
    duration = max(int(cfg.weight_schedule_duration), 0)

    if duration <= 0 or schedule in {"none", "constant", "fixed"}:
        return max(0.0, target_weight)

    progress = (step - warmup) / max(duration, 1)
    progress = min(max(progress, 0.0), 1.0)

    if schedule in {"linear", "lin"}:
        weight = start_weight + (target_weight - start_weight) * progress
    elif schedule in {"cos", "cosine", "cosine_decay"}:
        weight = start_weight + (target_weight - start_weight) * 0.5 * (1.0 - math.cos(math.pi * progress))
    else:
        weight = target_weight

    return max(0.0, float(weight))


def _compute_patch_cosine_loss(
    features: torch.Tensor,
    mask: Optional[torch.Tensor],
    patch_size: int,
    stride: int,
    normalize_mode: Optional[str],
) -> torch.Tensor:
    if features.numel() == 0:
        return torch.zeros((), device=features.device, dtype=features.dtype)

    patch_size = max(1, int(patch_size))
    stride = max(1, int(stride))

    working = features
    mask_vector: Optional[torch.Tensor] = None
    if mask is not None:
        mask_vector = mask.to(features.device, dtype=features.dtype).view(-1)
        working = working * mask_vector.unsqueeze(-1)

    norm_key = (normalize_mode or "none").lower()
    if norm_key in {"layernorm", "ln"}:
        working = F.layer_norm(working, (working.shape[-1],))
    elif norm_key in {"l2", "unit", "normalize"}:
        working = F.normalize(working, dim=-1, eps=1e-6)

    batch = working.shape[0]
    if batch < 2:
        return torch.zeros((), device=features.device, dtype=features.dtype)

    patch_size = min(patch_size, batch)
    if patch_size <= 1:
        return torch.zeros((), device=features.device, dtype=features.dtype)

    patches = working.unfold(0, patch_size, stride)
    mask_patches: Optional[torch.Tensor] = None
    if patches.numel() == 0:
        patches = working.unsqueeze(0)
        if mask_vector is not None:
            mask_patches = mask_vector.unsqueeze(0)
    elif mask_vector is not None:
        mask_patches = mask_vector.unfold(0, patch_size, stride)
        if mask_patches.numel() == 0:
            mask_patches = mask_vector.unsqueeze(0)

    patch_weights: Optional[torch.Tensor] = None
    if mask_vector is not None and mask_patches is not None:
        patch_weights = mask_patches.sum(dim=-1, keepdim=True)
        valid = patch_weights.squeeze(-1) > 1e-6
        if valid.any():
            patches = patches[valid]
            patch_weights = patch_weights[valid]
        else:
            return torch.zeros((), device=features.device, dtype=features.dtype)

    normed = F.normalize(patches, dim=-1, eps=1e-6)
    centers = normed.mean(dim=1, keepdim=True)
    centers = F.normalize(centers, dim=-1, eps=1e-6)
    cosine_distance = 1.0 - (normed * centers).sum(dim=-1)

    if patch_weights is not None:
        weights = patch_weights.squeeze(-1).clamp_min(1e-6)
        return (cosine_distance * weights).sum() / weights.sum()

    return cosine_distance.mean()


class LegoRayDataset(Dataset):
    """Dataset exposing per-frame teacher supervision and camera parameters."""

    def __init__(self, config: DataConfig):
        self.config = config
        with open(config.camera_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        frames: List[dict] = list(meta["frames"])
        total_frames = len(frames)

        if config.frame_indices:
            selected: List[dict] = []
            for candidate in config.frame_indices:
                index = candidate
                if index < 0:
                    index = total_frames + index
                if index < 0 or index >= total_frames:
                    raise IndexError(
                        f"Frame index {candidate} out of range for dataset with {total_frames} frames"
                    )
                selected.append(frames[index])
            frames = selected

        if config.max_frames is not None:
            frames = frames[: config.max_frames]

        self.frames = frames
        self.camera_angle_x = float(meta["camera_angle_x"])
        self.background = torch.tensor(config.background_color, dtype=torch.float32)
        self.near = float(config.near)
        self.far = float(config.far)

        teacher_imgs: List[torch.Tensor] = []
        teacher_alpha: List[torch.Tensor] = []
        teacher_depths: List[Optional[torch.Tensor]] = []
        c2w_mats: List[torch.Tensor] = []

        frame_progress = create_progress(
            self.frames,
            desc="Loading teacher frames",
            unit="frame",
            leave=False,
        )

        manual_frame_progress = None
        if getattr(frame_progress, "disable", False):
            manual_frame_progress = {
                "total": max(len(self.frames), 1),
                "last_len": 0,
                "start": time.perf_counter(),
            }

            def _render_frame_progress(count: int) -> None:
                total = manual_frame_progress["total"]
                completion = min(max(count / float(total), 0.0), 1.0)
                elapsed = time.perf_counter() - manual_frame_progress["start"]
                if completion > 0.0:
                    remaining = elapsed * (1.0 - completion) / completion
                else:
                    remaining = 0.0
                line = (
                    f"Loading teacher frames {count}/{total} ({completion * 100:5.1f}%) "
                    f"ETA {str(timedelta(seconds=int(remaining)))}"
                )
                pad = max(manual_frame_progress["last_len"] - len(line), 0)
                sys.stdout.write("\r" + line + (" " * pad))
                sys.stdout.flush()
                manual_frame_progress["last_len"] = len(line)
            manual_frame_progress["render"] = _render_frame_progress

        for frame_idx, frame in enumerate(frame_progress):
            base = Path(config.teacher_outputs) / frame["file_path"]
            candidate_paths = [
                base.with_suffix(suffix)
                for suffix in (".npy", ".npz", ".png", ".exr")
            ]
            rgba_np = None
            for path in candidate_paths:
                if path.exists():
                    if path.suffix == ".npy":
                        rgba_np = np.load(path)
                    elif path.suffix == ".npz":
                        rgba_np = np.load(path)["arr_0"]
                    else:
                        rgba_np = imageio.imread(path)
                    break
            if rgba_np is None:
                raise FileNotFoundError(f"Teacher frame not found for {base}")

            rgba = torch.tensor(np.asarray(rgba_np, dtype=np.float32), dtype=torch.float32)
            if rgba.max() > 1.0:
                rgba = rgba / 255.0
            if rgba.shape[-1] == 3:
                alpha_channel = torch.ones_like(rgba[..., :1])
                rgba = torch.cat([rgba, alpha_channel], dim=-1)

            teacher_imgs.append(rgba[..., :3].float())
            teacher_alpha.append(rgba[..., 3:4].float())
            teacher_depths.append(self._load_optional_depth(base, frame_idx))
            c2w_mats.append(torch.tensor(frame["transform_matrix"], dtype=torch.float32))

            if manual_frame_progress is not None:
                manual_frame_progress["render"](frame_idx + 1)

        frame_progress.close()
        if manual_frame_progress is not None:
            sys.stdout.write("\n")
            sys.stdout.flush()

        self.teacher_rgb = torch.stack(teacher_imgs)
        self.teacher_alpha = torch.stack(teacher_alpha)
        self.teacher_depth = teacher_depths
        self.has_depth = any(depth is not None for depth in teacher_depths)
        self.c2w_mats = c2w_mats
        self.height = int(self.teacher_rgb.shape[1])
        self.width = int(self.teacher_rgb.shape[2])
        self.num_pixels = self.height * self.width
        self.focal = 0.5 * self.width / np.tan(0.5 * self.camera_angle_x)
        self.bbox_min = torch.tensor(config.bbox_min, dtype=torch.float32)
        self.bbox_max = torch.tensor(config.bbox_max, dtype=torch.float32)

        tqdm.write(
            "[dataset] Loaded teacher supervision frames: "
            f"{len(self.frames)} @ {self.height}x{self.width}, has_depth={self.has_depth}"
        )

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):  # type: ignore[override]
        return {
            "rgb": self.teacher_rgb[idx],
            "alpha": self.teacher_alpha[idx],
            "c2w": self.c2w_mats[idx],
            "index": idx,
        }

    def _load_optional_depth(self, base: Path, frame_idx: int) -> Optional[torch.Tensor]:
        """Attempt to load a depth map corresponding to a teacher frame."""

        candidate_dirs = {base.parent}
        if base.parent.name == "renders":
            candidate_dirs.add(base.parent.parent / "depth")
            candidate_dirs.add(base.parent.parent / "depths")
        if self.config.teacher_depth_dir is not None:
            candidate_dirs.add(self.config.teacher_depth_dir)

        stem = base.stem
        possible_indices: List[str] = [stem]
        if "_" in stem:
            suffix = stem.split("_")[-1]
            if suffix.isdigit():
                possible_indices.append(f"{int(suffix):05d}")
        possible_indices.append(f"{frame_idx:05d}")

        candidates: List[Path] = []
        suffixes = (".npy", ".npz", ".png", ".exr")
        for directory in candidate_dirs:
            for key in possible_indices:
                for suffix in suffixes:
                    candidates.append(directory / f"{key}{suffix}")
                    candidates.append(directory / f"{key}_depth{suffix}")

        for path in candidates:
            if not path.exists():
                continue

            if path.suffix == ".npy":
                depth_np = np.load(path)
            elif path.suffix == ".npz":
                depth_np = np.load(path)["arr_0"]
            else:
                depth_np = imageio.imread(path)

            depth = torch.tensor(np.asarray(depth_np, dtype=np.float32), dtype=torch.float32)
            if depth.ndim == 3:
                depth = depth[..., 0]
            if depth.dtype != torch.float32:
                depth = depth.float()
            max_val = float(depth.max().item()) if depth.numel() > 0 else 0.0
            min_val = float(depth.min().item()) if depth.numel() > 0 else 0.0
            # Heuristic: Gaussian splatting depth buffers are often normalized to [0, 1]
            # in view space. If values stay within a small positive range, rescale them
            # to match the ray-marched near/far distances used by the student.
            if max_val > 0.0 and max_val <= 1.5 and min_val >= -1e-4:
                depth = depth * (self.far - self.near) + self.near
            return depth

        return None

    def sample_random_rays(
        self,
        num_rays: int,
        device: torch.device,
        oversample_factor: int = 2,
    ) -> Dict[str, torch.Tensor]:
        if num_rays <= 0:
            raise ValueError("num_rays must be positive")

        frame_idx = int(torch.randint(0, len(self.frames), (1,), device="cpu"))
        frame_rgb = self.teacher_rgb[frame_idx].to(device)
        frame_alpha = self.teacher_alpha[frame_idx].to(device)
        frame_depth_tensor = self.teacher_depth[frame_idx] if self.teacher_depth else None
        if frame_depth_tensor is not None:
            frame_depth = frame_depth_tensor.to(device)
        else:
            frame_depth = None

        c2w = self.c2w_mats[frame_idx].to(device)
        bbox_min = self.bbox_min.to(device)
        bbox_max = self.bbox_max.to(device)

        rays_o_list: List[torch.Tensor] = []
        rays_d_list: List[torch.Tensor] = []
        near_list: List[torch.Tensor] = []
        far_list: List[torch.Tensor] = []
        rgb_list: List[torch.Tensor] = []
        alpha_list: List[torch.Tensor] = []
        depth_list: List[torch.Tensor] = []

        frame_rgb_flat = frame_rgb.view(-1, 3)
        frame_alpha_flat = frame_alpha.view(-1, 1)
        frame_depth_flat = frame_depth.view(-1, 1) if frame_depth is not None else None

        collected = 0
        attempts = 0
        max_attempts = 16

        while collected < num_rays and attempts < max_attempts:
            remaining = num_rays - collected
            sample_count = max(remaining * oversample_factor, remaining)
            pixel_indices = torch.randint(0, self.num_pixels, (sample_count,), device=device)

            j = torch.div(pixel_indices, self.width, rounding_mode="floor")
            i = pixel_indices - j * self.width
            dirs = torch.stack(
                [
                    (i.float() - self.width * 0.5) / self.focal,
                    -(j.float() - self.height * 0.5) / self.focal,
                    -torch.ones_like(i, dtype=torch.float32),
                ],
                dim=-1,
            )

            rays_d = torch.matmul(dirs, c2w[:3, :3].T)
            rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
            rays_o = c2w[:3, 3].expand_as(rays_d)

            near, far, valid_mask = intersect_rays_aabb(rays_o, rays_d, bbox_min, bbox_max)
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
            if valid_indices.numel() == 0:
                attempts += 1
                continue

            take = min(valid_indices.numel(), remaining)
            selected = valid_indices[:take]
            selected_pixels = pixel_indices[selected]

            rays_o_list.append(rays_o[selected])
            rays_d_list.append(rays_d[selected])
            near_list.append(near[selected])
            far_list.append(far[selected])
            rgb_list.append(frame_rgb_flat[selected_pixels])
            alpha_list.append(frame_alpha_flat[selected_pixels])
            if frame_depth_flat is not None:
                depth_list.append(frame_depth_flat[selected_pixels])

            collected += take
            attempts += 1

        if collected == 0:
            raise RuntimeError("Failed to sample valid rays within the bounding box.")

        rays_o_out = torch.cat(rays_o_list, dim=0)
        rays_d_out = torch.cat(rays_d_list, dim=0)
        near_out = torch.cat(near_list, dim=0)
        far_out = torch.cat(far_list, dim=0)
        rgb_out = torch.cat(rgb_list, dim=0)
        alpha_out = torch.cat(alpha_list, dim=0)
        if frame_depth_flat is not None and depth_list:
            depth_out = torch.cat(depth_list, dim=0)
        else:
            depth_out = None

        depth_valid_out: Optional[torch.Tensor] = None
        if depth_out is not None:
            depth_out = depth_out.view(-1, 1)
            valid_mask = torch.isfinite(depth_out.squeeze(-1)) & (depth_out.squeeze(-1) > 0.0)
            if valid_mask.ndim == 0:
                valid_mask = valid_mask.unsqueeze(0)
            near_expanded = near_out.unsqueeze(-1)
            far_expanded = far_out.unsqueeze(-1)
            depth_out = torch.where(valid_mask.unsqueeze(-1), depth_out, far_expanded)
            depth_out = torch.clamp(depth_out, min=near_expanded, max=far_expanded)
            depth_valid_out = valid_mask.unsqueeze(-1).float()

        return {
            "frame_index": frame_idx,
            "rays_o": rays_o_out,
            "rays_d": rays_d_out,
            "near": near_out,
            "far": far_out,
            "teacher_rgb": rgb_out,
            "teacher_alpha": alpha_out,
            "teacher_depth": depth_out,
            "teacher_depth_valid_mask": depth_valid_out,
        }


def get_camera_rays(height: int, width: int, focal: float, c2w: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    i_coords, j_coords = torch.meshgrid(
        torch.arange(width, dtype=torch.float32, device=device),
        torch.arange(height, dtype=torch.float32, device=device),
        indexing="xy",
    )
    dirs = torch.stack(
        [
            (i_coords - width * 0.5) / focal,
            -(j_coords - height * 0.5) / focal,
            -torch.ones_like(i_coords),
        ],
        dim=-1,
    )
    c2w_rot = c2w[:3, :3]
    rays_d = torch.sum(dirs[..., None, :] * c2w_rot, dim=-1)
    rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
    rays_o = c2w[:3, 3].expand_as(rays_d)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)


def intersect_rays_aabb(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-ray entry/exit distances with an axis-aligned bounding box."""

    dir_safe = torch.where(
        torch.abs(rays_d) > eps,
        rays_d,
        torch.sign(rays_d) * eps + (rays_d == 0).float() * eps,
    )
    inv_d = 1.0 / dir_safe

    t0 = (bbox_min - rays_o) * inv_d
    t1 = (bbox_max - rays_o) * inv_d

    t_min = torch.minimum(t0, t1)
    t_max = torch.maximum(t0, t1)

    near = t_min.max(dim=-1).values
    far = t_max.min(dim=-1).values

    valid = far > near
    near = near.clamp_min(0.0)
    far = torch.maximum(far, near + 1e-4)
    return near, far, valid


def sample_along_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: torch.Tensor,
    far: torch.Tensor,
    num_samples: int,
    perturb: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = rays_o.device
    near = near.unsqueeze(-1)
    far = far.unsqueeze(-1)

    t_vals = torch.linspace(0.0, 1.0, steps=num_samples, device=device).unsqueeze(0)
    z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = z_vals.expand(rays_o.shape[0], num_samples)

    if perturb:
        mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
        lower = torch.cat([z_vals[:, :1], mids], dim=-1)
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., None]
    return pts, z_vals


def _exclusive_cumprod_last(values: torch.Tensor) -> torch.Tensor:
    """Deterministic exclusive cumulative product over the last dimension.

    For an input ``values`` with shape ``(..., N)``, returns a tensor ``T`` with the
    same shape such that ``T[..., i] = prod(values[..., :i])`` (empty product = 1).
    A simple Python loop keeps autograd support while avoiding the non-deterministic
    CUDA kernels used by ``torch.cumprod``.
    """

    if values.size(-1) == 0:
        return values.clone()

    outputs = []
    running = torch.ones_like(values[..., 0])
    for idx in range(values.shape[-1]):
        outputs.append(running.unsqueeze(-1))
        running = running * values[..., idx]
    return torch.cat(outputs, dim=-1)


class _SimpleMLPStudent(torch.nn.Module):
    def __init__(self, cfg: StudentConfig):
        super().__init__()
        hidden = cfg.hidden_dim
        layers: List[torch.nn.Module] = []
        in_dim = 3
        for _ in range(cfg.num_layers):
            layers.append(torch.nn.Linear(in_dim, hidden))
            layers.append(torch.nn.ReLU(inplace=True))
            in_dim = hidden
        layers.append(torch.nn.Linear(in_dim, 4))
        self.mlp = torch.nn.Sequential(*layers)
        self.register_buffer("density_bias", torch.tensor(cfg.density_bias))
        self.register_buffer("color_bias", torch.tensor(cfg.color_bias))
        self.last_input: Optional[torch.Tensor] = None
        self.last_pre_activation: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        ray_directions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.last_input = x
        out = self.mlp(x)
        self.last_pre_activation = out
        rgb = torch.sigmoid(out[..., :3] + self.color_bias)
        sigma = torch.relu(out[..., 3:] + self.density_bias)
        return rgb, sigma


class _HashGridStudent(torch.nn.Module):
    def __init__(self, cfg: StudentConfig):
        super().__init__()
        if tcnn is None:
            raise ImportError(
                "tinycudann is required for hash-grid students. Install it via pip (pip install tinycudann)."
            )

        encoding_config: Dict[str, float | int | str] = {
            "otype": "HashGrid",
            "n_levels": int(cfg.hash_levels),
            "n_features_per_level": int(cfg.hash_features_per_level),
            "log2_hashmap_size": int(cfg.hash_log2_hashmap_size),
            "base_resolution": int(cfg.hash_base_resolution),
            "per_level_scale": float(cfg.hash_per_level_scale),
        }
        activation_lookup = {
            "relu": "ReLU",
            "silu": "SiLU",
            "leaky_relu": "LeakyReLU",
            "none": "None",
        }
        activation_name = activation_lookup.get(cfg.activation.lower(), cfg.activation)

        network_config: Dict[str, float | int | str] = {
            "otype": "CutlassMLP",
            "activation": activation_name,
            "output_activation": "None",
            "n_neurons": int(cfg.hidden_dim),
            "n_hidden_layers": max(int(cfg.num_layers), 1),
        }

        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=4,
            encoding_config=encoding_config,
            network_config=network_config,
        )
        self.register_buffer("density_bias", torch.tensor(cfg.density_bias))
        self.register_buffer("color_bias", torch.tensor(cfg.color_bias))
        self.last_input: Optional[torch.Tensor] = None
        self.last_pre_activation: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        ray_directions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not x.is_contiguous():
            x = x.contiguous()
        self.last_input = x
        out = self.model(x)
        self.last_pre_activation = out
        rgb = torch.sigmoid(out[..., :3] + self.color_bias)
        sigma = torch.relu(out[..., 3:] + self.density_bias)
        return rgb, sigma


class _KiloNeRFStudent(torch.nn.Module):
    """Canonical KiloNeRF multi-network student backed by the official implementation."""

    _SUPPORTED_ACTIVATIONS = {"relu", "leaky_relu", "tanh", "sigmoid"}
    _STREAM_POOL_SIZE = 16

    def __init__(self, cfg: StudentConfig):
        super().__init__()
        if MultiNetwork is None:
            raise ImportError(
                "The kilonerf package could not be imported. Ensure the repository root is on PYTHONPATH "
                "and that kilonerf dependencies are installed."
            ) from _KILONERF_IMPORT_ERROR

        if not _HAS_KILONERF_CUDA:
            raise RuntimeError(
                "KiloNeRF CUDA extensions are required. Please build the kilonerf_cuda module before running "
                "the canonical student. Refer to kilonerf/README for build instructions."
            )

        global _KILONERF_RUNTIME_INITIALISED  # pylint: disable=global-statement
        if not _KILONERF_RUNTIME_INITIALISED:
            if torch.cuda.is_available() and not torch.cuda.is_initialized():
                torch.cuda.init()
            kilonerf_cuda.init_stream_pool(self._STREAM_POOL_SIZE)
            kilonerf_cuda.init_magma()
            _KILONERF_RUNTIME_INITIALISED = True

        self.grid_resolution = tuple(int(v) for v in cfg.grid_resolution)
        if len(self.grid_resolution) != 3:
            raise ValueError("grid_resolution must be a 3-element tuple for kilo_uniform_mlp students")
        self.num_cells = int(self.grid_resolution[0] * self.grid_resolution[1] * self.grid_resolution[2])
        if self.num_cells <= 0:
            raise ValueError("grid_resolution must produce at least one cell")

        activation = cfg.activation.lower()
        if activation not in self._SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"Activation '{cfg.activation}' is not supported by the canonical KiloNeRF implementation. "
                f"Choose one of {sorted(self._SUPPORTED_ACTIVATIONS)}."
            )

        hidden_dim = int(cfg.hidden_dim)
        num_layers = max(int(cfg.num_layers), 1)

        pos_encoding = str(getattr(cfg, "pos_encoding", "none") or "none").lower()
        pos_levels = max(int(getattr(cfg, "pos_L", 0) or 0), 0)
        if pos_encoding in {"fourier", "positional", "sin", "sinusoidal"} and pos_levels > 0:
            freq_bands = torch.pow(2.0, torch.arange(pos_levels, dtype=torch.float32))
            self.register_buffer("position_freq_bands", freq_bands, persistent=False)
            position_channels = 3 + 3 * 2 * pos_levels
        else:
            pos_encoding = "none"
            pos_levels = 0
            self.register_buffer("position_freq_bands", torch.empty(0, dtype=torch.float32), persistent=False)
            position_channels = 3

        self.position_encoding = pos_encoding
        self.pos_levels = pos_levels
        self.position_channels = position_channels

        dir_encoding = str(getattr(cfg, "dir_encoding", "none") or "none").lower()
        dir_levels = max(int(getattr(cfg, "dir_L", 0) or 0), 0)
        if dir_encoding in {"fourier", "positional", "sin", "sinusoidal"} and dir_levels > 0:
            dir_freq_bands = torch.pow(2.0, torch.arange(dir_levels, dtype=torch.float32))
            self.register_buffer("direction_freq_bands", dir_freq_bands, persistent=False)
            direction_channels = 3 + 3 * 2 * dir_levels
        elif dir_encoding in {"raw", "identity", "linear"}:
            dir_levels = 0
            self.register_buffer("direction_freq_bands", torch.empty(0, dtype=torch.float32), persistent=False)
            direction_channels = 3
        else:
            dir_encoding = "none"
            dir_levels = 0
            self.register_buffer("direction_freq_bands", torch.empty(0, dtype=torch.float32), persistent=False)
            direction_channels = 0

        self.direction_encoding = dir_encoding
        self.dir_levels = dir_levels
        self.direction_channels = direction_channels

        skip_candidates: Tuple[int, ...] = tuple(
            int(v) for v in getattr(cfg, "skips", tuple()) if isinstance(v, (int, float))
        )
        self.refeed_position_index: Optional[int] = None
        if num_layers > 1 and skip_candidates:
            primary_layer = max(int(skip_candidates[0]), 1)
            target_layer = min(max(primary_layer, 2), num_layers)
            self.refeed_position_index = target_layer - 2
        else:
            self.refeed_position_index = None

        self.sigma_activation = str(getattr(cfg, "sigma_activation", "relu") or "relu").lower()
        sigma_bias_value = float(cfg.sigma_bias if cfg.sigma_bias is not None else cfg.density_bias)

        gx, gy, gz = self.grid_resolution
        self.linear_index_multipliers = (1, gx, gx * gy)
        grid_resolution_tensor = torch.tensor([gx, gy, gz], dtype=torch.float32)
        cell_size = 1.0 / grid_resolution_tensor

        grid_x = torch.arange(gx, dtype=torch.float32)
        grid_y = torch.arange(gy, dtype=torch.float32)
        grid_z = torch.arange(gz, dtype=torch.float32)
        mesh = torch.stack(torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij"), dim=-1)
        origins = mesh.reshape(-1, 3) / grid_resolution_tensor

        self.register_buffer("grid_resolution_float", grid_resolution_tensor)
        self.register_buffer("cell_size", cell_size)
        self.register_buffer("cell_origins", origins)
        self.register_buffer("domain_mins", origins)
        self.register_buffer("domain_maxs", origins + cell_size)

        implementation = "multimatmul_differentiable"
        self.network = MultiNetwork(
            num_networks=self.num_cells,
            num_position_channels=self.position_channels,
            num_direction_channels=self.direction_channels,
            num_output_channels=4,
            hidden_layer_size=hidden_dim,
            num_hidden_layers=num_layers,
            refeed_position_index=self.refeed_position_index,
            late_feed_direction=True,
            direction_layer_size=hidden_dim,
            nonlinearity=activation,
            linear_implementation=implementation,
            use_same_initialization_for_all_networks=False,
            weight_initialization_method="kaiming_uniform",
            bias_initialization_method="standard",
            alpha_rgb_initalization="updated_yenchenlin",
            use_hard_parameter_sharing_for_color=False,
            view_dependent_dropout_probability=-1,
            use_view_independent_color=False,
        )

        self.register_buffer("density_bias", torch.tensor(float(cfg.density_bias)))
        self.register_buffer("sigma_bias", torch.tensor(sigma_bias_value))
        self.register_buffer("color_bias", torch.tensor(float(cfg.color_bias)))
        self.hidden_activation = str(cfg.activation).lower()
        self.last_input: Optional[torch.Tensor] = None
        self.last_pre_activation: Optional[torch.Tensor] = None
        self.last_linear_indices: Optional[torch.Tensor] = None
        self.last_penultimate_pre: Optional[torch.Tensor] = None
        self.last_penultimate_post: Optional[torch.Tensor] = None
        self.last_penultimate: Optional[torch.Tensor] = None
        self.last_ray_directions: Optional[torch.Tensor] = None
        self.last_encoded_directions: Optional[torch.Tensor] = None
        self._penultimate_sorted_pre: Optional[torch.Tensor] = None
        self._penultimate_sorted_post: Optional[torch.Tensor] = None
        self._feature_hook = None
        feature_linear = getattr(self.network, "feature_linear", None)
        if feature_linear is not None:
            self._feature_hook = feature_linear.register_forward_hook(self._capture_penultimate)
        self.boundary_blend_enabled = bool(getattr(cfg, "enable_boundary_blend", False))
        margin = float(getattr(cfg, "boundary_blend_margin", 0.05))
        self.boundary_blend_margin = max(0.0, min(0.5, margin))

    def _apply_hidden_activation(self, tensor: torch.Tensor) -> torch.Tensor:
        activation = self.hidden_activation
        if activation in {"relu", "leaky_relu"}:
            negative_slope = 0.0 if activation == "relu" else 0.01
            return F.leaky_relu(tensor, negative_slope=negative_slope)
        if activation in {"silu", "swish"}:
            return F.silu(tensor)
        if activation == "gelu":
            return F.gelu(tensor)
        if activation == "tanh":
            return torch.tanh(tensor)
        if activation in {"identity", "none", "linear"}:
            return tensor
        return F.relu(tensor)

    def _capture_penultimate(self, _module: torch.nn.Module, _inputs, output: torch.Tensor):
        self._penultimate_sorted_pre = output
        self._penultimate_sorted_post = self._apply_hidden_activation(output)
        return output

    def _encode_positions(self, coords: torch.Tensor) -> torch.Tensor:
        if self.position_encoding != "fourier" or self.pos_levels <= 0 or self.position_freq_bands.numel() == 0:
            return coords
        if coords.numel() == 0:
            return coords.new_empty((coords.shape[0], self.position_channels))
        freq = self.position_freq_bands.to(device=coords.device, dtype=coords.dtype)
        scaled = coords.unsqueeze(-1) * freq  # (..., 3, L)
        scaled = scaled * math.pi
        sin_terms = torch.sin(scaled)
        cos_terms = torch.cos(scaled)
        sin_flat = sin_terms.reshape(coords.shape[0], -1)
        cos_flat = cos_terms.reshape(coords.shape[0], -1)
        return torch.cat([coords, sin_flat, cos_flat], dim=-1)

    def _encode_directions(
        self,
        directions: Optional[torch.Tensor],
        *,
        count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if count == 0:
            return torch.zeros((0, self.direction_channels), device=device, dtype=dtype)
        if self.direction_channels == 0:
            return torch.zeros((count, 0), device=device, dtype=dtype)
        if directions is None:
            return torch.zeros((count, self.direction_channels), device=device, dtype=dtype)
        if directions.ndim != 2 or directions.shape[-1] != 3:
            raise ValueError("ray_directions must be shaped (N, 3)")
        if directions.shape[0] != count:
            raise ValueError("ray_directions count must match sample count")
        dirs = directions.to(device=device, dtype=dtype)
        dirs = F.normalize(dirs, dim=-1, eps=1e-6)
        if self.direction_encoding != "fourier" or self.dir_levels <= 0 or self.direction_freq_bands.numel() == 0:
            return dirs
        freq = self.direction_freq_bands.to(device=device, dtype=dtype)
        scaled = dirs.unsqueeze(-1) * freq * math.pi
        sin_terms = torch.sin(scaled)
        cos_terms = torch.cos(scaled)
        sin_flat = sin_terms.reshape(dirs.shape[0], -1)
        cos_flat = cos_terms.reshape(dirs.shape[0], -1)
        return torch.cat([dirs, sin_flat, cos_flat], dim=-1)

    def _activate_sigma(self, tensor: torch.Tensor) -> torch.Tensor:
        activation = self.sigma_activation
        if activation in {"relu", "rectified"}:
            return torch.relu(tensor)
        if activation == "softplus":
            return F.softplus(tensor)
        if activation in {"shifted_softplus", "softplus_shifted"}:
            value = F.softplus(tensor) - math.log(2.0)
            return torch.clamp(value, min=0.0)
        if activation in {"identity", "linear", "none"}:
            return torch.clamp(tensor, min=0.0)
        return torch.relu(tensor)

    def _query_network(
        self,
        lin_idx: torch.Tensor,
        coords: torch.Tensor,
        ray_directions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if lin_idx.numel() == 0:
            return coords.new_zeros((0, 4))

        device = coords.device
        dtype = coords.dtype

        sorted_perm = torch.argsort(lin_idx)
        lin_sorted = lin_idx[sorted_perm]
        coords_sorted = coords[sorted_perm]

        domain_mins = self.domain_mins.index_select(0, lin_sorted).to(device=device, dtype=dtype)
        domain_maxs = self.domain_maxs.index_select(0, lin_sorted).to(device=device, dtype=dtype)
        local_coords = 2.0 * (coords_sorted - domain_mins) / (domain_maxs - domain_mins) - 1.0

        batch_size_per_network = torch.zeros(self.num_cells, device=lin_sorted.device, dtype=torch.long)
        if lin_sorted.numel() > 0:
            unique_cells, counts = torch.unique_consecutive(lin_sorted, return_counts=True)
            batch_size_per_network[unique_cells] = counts
        batch_size_cpu = batch_size_per_network.to(torch.device("cpu"))

        encoded_positions = self._encode_positions(local_coords)
        dirs_sorted: Optional[torch.Tensor] = None
        if ray_directions is not None:
            if ray_directions.ndim != 2 or ray_directions.shape[-1] != 3:
                raise ValueError("ray_directions must be shaped (N, 3)")
            if ray_directions.shape[0] != coords.shape[0]:
                raise ValueError("ray_directions count must match coords count")
            dirs_sorted = ray_directions[sorted_perm]
        encoded_directions = self._encode_directions(
            dirs_sorted,
            count=coords_sorted.shape[0],
            device=device,
            dtype=dtype,
        )
        self._penultimate_sorted_pre = None
        self._penultimate_sorted_post = None
        raw_sorted = self.network([encoded_positions, encoded_directions], batch_size_cpu)

        outputs = torch.empty_like(raw_sorted)
        outputs[sorted_perm] = raw_sorted
        penultimate_pre_unsorted: Optional[torch.Tensor] = None
        penultimate_post_unsorted: Optional[torch.Tensor] = None
        if self._penultimate_sorted_pre is not None:
            penultimate_pre_unsorted = torch.empty_like(self._penultimate_sorted_pre)
            penultimate_pre_unsorted[sorted_perm] = self._penultimate_sorted_pre
        if self._penultimate_sorted_post is not None:
            penultimate_post_unsorted = torch.empty_like(self._penultimate_sorted_post)
            penultimate_post_unsorted[sorted_perm] = self._penultimate_sorted_post
        if encoded_directions.numel() == 0:
            encoded_unsorted = encoded_directions
        else:
            encoded_unsorted = torch.empty_like(encoded_directions)
            encoded_unsorted[sorted_perm] = encoded_directions
        self.last_penultimate_pre = penultimate_pre_unsorted
        self.last_penultimate_post = penultimate_post_unsorted
        if penultimate_post_unsorted is not None:
            self.last_penultimate = penultimate_post_unsorted
        elif penultimate_pre_unsorted is not None:
            self.last_penultimate = penultimate_pre_unsorted
        else:
            self.last_penultimate = None
        self.last_encoded_directions = encoded_unsorted
        return outputs

    def forward(
        self,
        x: torch.Tensor,
        ray_directions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.shape[-1] != 3:
            raise ValueError("Input to kilo_uniform_mlp student must be shaped (N, 3)")

        self.last_ray_directions = None
        self.last_encoded_directions = None
        if x.numel() == 0:
            return x.new_zeros((0, 3)), x.new_zeros((0, 1))

        device = x.device
        dtype = x.dtype

        ray_dirs: Optional[torch.Tensor]
        if ray_directions is None:
            ray_dirs = None
        else:
            if ray_directions.ndim != 2 or ray_directions.shape[-1] != 3:
                raise ValueError("ray_directions must be shaped (N, 3)")
            if ray_directions.shape[0] != x.shape[0]:
                raise ValueError("ray_directions count must match input sample count")
            ray_dirs = F.normalize(ray_directions.to(device=device, dtype=dtype), dim=-1, eps=1e-6)

        coords = x.clamp(0.0, 1.0 - 1e-6)
        grid_res = self.grid_resolution_float.to(device=device, dtype=dtype)
        indices = torch.floor(coords * grid_res).to(torch.int64)
        gx, gy, gz = self.grid_resolution
        indices[:, 0] = indices[:, 0].clamp_(0, gx - 1)
        indices[:, 1] = indices[:, 1].clamp_(0, gy - 1)
        indices[:, 2] = indices[:, 2].clamp_(0, gz - 1)

        lin_idx = (
            indices[:, 0]
            + self.linear_index_multipliers[1] * indices[:, 1]
            + self.linear_index_multipliers[2] * indices[:, 2]
        )

        self.last_input = x
        self.last_linear_indices = lin_idx

        base_outputs = self._query_network(lin_idx, coords, ray_dirs)
        primary_penultimate = self.last_penultimate
        primary_penultimate_pre = self.last_penultimate_pre
        primary_penultimate_post = self.last_penultimate_post
        self.last_pre_activation = base_outputs
        self.last_ray_directions = ray_dirs

        rgb = torch.sigmoid(base_outputs[:, :3] + self.color_bias)
        sigma = self._activate_sigma(base_outputs[:, 3:] + self.sigma_bias)

        if self.boundary_blend_enabled and self.boundary_blend_margin > 0.0:
            margin = self.boundary_blend_margin
            gx, gy, gz = self.grid_resolution
            grid_res_tensor = self.grid_resolution_float.to(device=device, dtype=dtype)
            cell_pos = coords * grid_res_tensor
            frac = cell_pos - indices.to(dtype=dtype)

            def _blend_with_neighbor(
                idx_axis: int,
                direction: int,
                mask_tensor: torch.Tensor,
                weight_tensor: torch.Tensor,
                current_rgb: torch.Tensor,
                current_sigma: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                if mask_tensor is None or mask_tensor.numel() == 0:
                    return current_rgb, current_sigma
                sample_idx = torch.nonzero(mask_tensor, as_tuple=False).squeeze(-1)
                if sample_idx.numel() == 0:
                    return current_rgb, current_sigma
                neighbor_indices = indices.index_select(0, sample_idx).clone()
                neighbor_indices[:, idx_axis] += direction
                if idx_axis == 0:
                    valid = (neighbor_indices[:, idx_axis] >= 0) & (neighbor_indices[:, idx_axis] < gx)
                elif idx_axis == 1:
                    valid = (neighbor_indices[:, idx_axis] >= 0) & (neighbor_indices[:, idx_axis] < gy)
                else:
                    valid = (neighbor_indices[:, idx_axis] >= 0) & (neighbor_indices[:, idx_axis] < gz)
                if not torch.any(valid):
                    return current_rgb, current_sigma
                sample_idx = sample_idx[valid]
                neighbor_indices = neighbor_indices[valid]
                if sample_idx.numel() == 0:
                    return current_rgb, current_sigma
                lin_neighbor = (
                    neighbor_indices[:, 0]
                    + self.linear_index_multipliers[1] * neighbor_indices[:, 1]
                    + self.linear_index_multipliers[2] * neighbor_indices[:, 2]
                )

                neighbor_coords = coords.index_select(0, sample_idx)
                neighbor_dirs = None if ray_dirs is None else ray_dirs.index_select(0, sample_idx)
                neighbor_outputs = self._query_network(lin_neighbor, neighbor_coords, neighbor_dirs)
                neighbor_rgb = torch.sigmoid(neighbor_outputs[:, :3] + self.color_bias)
                neighbor_sigma = self._activate_sigma(neighbor_outputs[:, 3:] + self.sigma_bias)

                w = weight_tensor.index_select(0, sample_idx).unsqueeze(-1).clamp(0.0, 1.0)
                base_rgb = current_rgb.index_select(0, sample_idx)
                base_sigma = current_sigma.index_select(0, sample_idx)
                blended_rgb_vals = base_rgb * (1.0 - w) + neighbor_rgb * w
                blended_sigma_vals = base_sigma * (1.0 - w) + neighbor_sigma * w

                rgb_delta_vals = blended_rgb_vals - base_rgb
                sigma_delta_vals = blended_sigma_vals - base_sigma

                rgb_delta = torch.zeros_like(current_rgb)
                sigma_delta = torch.zeros_like(current_sigma)
                rgb_delta.index_add_(0, sample_idx, rgb_delta_vals)
                sigma_delta.index_add_(0, sample_idx, sigma_delta_vals)

                updated_rgb = current_rgb + rgb_delta
                updated_sigma = current_sigma + sigma_delta
                return updated_rgb, updated_sigma

            frac_x = frac[:, 0]
            frac_y = frac[:, 1]
            frac_z = frac[:, 2]

            lower_x = frac_x < margin
            upper_x = frac_x > (1.0 - margin)
            lower_y = frac_y < margin
            upper_y = frac_y > (1.0 - margin)
            lower_z = frac_z < margin
            upper_z = frac_z > (1.0 - margin)

            if torch.any(lower_x):
                weights = ((margin - frac_x).clamp_min(0.0) / margin)
                rgb, sigma = _blend_with_neighbor(0, -1, lower_x, weights, rgb, sigma)
            if torch.any(upper_x):
                weights = ((frac_x - (1.0 - margin)).clamp_min(0.0) / margin)
                rgb, sigma = _blend_with_neighbor(0, +1, upper_x, weights, rgb, sigma)
            if torch.any(lower_y):
                weights = ((margin - frac_y).clamp_min(0.0) / margin)
                rgb, sigma = _blend_with_neighbor(1, -1, lower_y, weights, rgb, sigma)
            if torch.any(upper_y):
                weights = ((frac_y - (1.0 - margin)).clamp_min(0.0) / margin)
                rgb, sigma = _blend_with_neighbor(1, +1, upper_y, weights, rgb, sigma)
            if torch.any(lower_z):
                weights = ((margin - frac_z).clamp_min(0.0) / margin)
                rgb, sigma = _blend_with_neighbor(2, -1, lower_z, weights, rgb, sigma)
            if torch.any(upper_z):
                weights = ((frac_z - (1.0 - margin)).clamp_min(0.0) / margin)
                rgb, sigma = _blend_with_neighbor(2, +1, upper_z, weights, rgb, sigma)

        self.last_penultimate_pre = primary_penultimate_pre
        self.last_penultimate_post = primary_penultimate_post
        if primary_penultimate_post is not None:
            self.last_penultimate = primary_penultimate_post
        else:
            self.last_penultimate = primary_penultimate
        return rgb, sigma


class StudentModel(torch.nn.Module):
    """Factory wrapper that selects the appropriate student implementation."""

    def __init__(self, cfg: StudentConfig):
        super().__init__()
        if cfg.type == "hash_mlp":
            self.impl: torch.nn.Module = _HashGridStudent(cfg)
        elif cfg.type == "kilo_uniform_mlp":
            self.impl = _KiloNeRFStudent(cfg)
        else:
            self.impl = _SimpleMLPStudent(cfg)

    def forward(
        self,
        x: torch.Tensor,
        ray_directions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.impl(x, ray_directions=ray_directions)


def parse_config(path: Path):
    with open(path, "r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)

    try:
        validate_config_dict(raw)
    except Exception as err:
        print(f"[schema] Config validation failed for {path}: {err}")
        raise SystemExit(12)

    def _coerce_optional_int(value: Optional[object]) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _coerce_float(value: Optional[object], default: Optional[float] = None) -> Optional[float]:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    experiment_section = raw["experiment"]
    experiment = ExperimentConfig(
        name=experiment_section["name"],
        seed=experiment_section["seed"],
        output_dir=Path(experiment_section["output_dir"]),
        progress_desc=experiment_section.get("progress_desc"),
    )
    data_section = raw["data"]
    max_frames_raw = data_section.get("max_frames")
    max_frames_value = None
    if max_frames_raw is not None:
        try:
            max_frames_value = int(max_frames_raw)
        except (TypeError, ValueError):
            max_frames_value = None
        if max_frames_value is not None and max_frames_value <= 0:
            max_frames_value = None

    frame_indices_raw = data_section.get("frame_indices")
    frame_indices_tuple: Tuple[int, ...] = tuple()
    if isinstance(frame_indices_raw, (list, tuple)):
        parsed_indices: List[int] = []
        for entry in frame_indices_raw:
            try:
                parsed = int(entry)
            except (TypeError, ValueError):
                continue
            parsed_indices.append(parsed)
        frame_indices_tuple = tuple(parsed_indices)
    data = DataConfig(
        dataset_root=Path(data_section["dataset_root"]),
        teacher_outputs=Path(data_section["teacher_outputs"]),
        teacher_depth_dir=Path(data_section["teacher_depth_dir"]) if "teacher_depth_dir" in data_section else None,
        camera_json=Path(data_section["camera_json"]),
        background_color=tuple(data_section["background_color"]),
        batch_size=int(data_section["batch_size"]),
        ray_chunk=int(data_section.get("ray_chunk", data_section["batch_size"])),
        near=float(data_section.get("near", 2.0)),
        far=float(data_section.get("far", 6.0)),
        samples_per_ray=int(data_section.get("samples_per_ray", 128)),
        bbox_min=tuple(float(v) for v in data_section.get("bbox_min", (-1.5, -1.5, -1.5))),
        bbox_max=tuple(float(v) for v in data_section.get("bbox_max", (1.5, 1.5, 1.5))),
        perturb=bool(data_section.get("perturb", True)),
        max_frames=max_frames_value,
        frame_indices=frame_indices_tuple,
    )
    teacher = TeacherConfig(
        type=raw["teacher"]["type"],
        checkpoint=Path(raw["teacher"]["checkpoint"]),
        render_stats=Path(raw["teacher"]["render_stats"]),
    )
    student_section = raw["student"]
    grid_resolution_raw = student_section.get("grid_resolution", (128, 128, 128))
    grid_resolution = tuple(int(v) for v in grid_resolution_raw)

    mlp_hidden_raw = student_section.get("mlp_hidden")
    if isinstance(mlp_hidden_raw, (list, tuple)):
        mlp_hidden_values = []
        for value in mlp_hidden_raw:
            parsed = _coerce_optional_int(value)
            if parsed is not None and parsed > 0:
                mlp_hidden_values.append(parsed)
        mlp_hidden = tuple(mlp_hidden_values)
    else:
        mlp_hidden = tuple()

    hidden_default = mlp_hidden[0] if mlp_hidden else 64
    num_layers_default = len(mlp_hidden) if mlp_hidden else 4
    hidden_dim = int(student_section.get("hidden_dim", hidden_default))
    num_layers = int(student_section.get("num_layers", num_layers_default))

    density_bias_raw = student_section.get("density_bias")
    sigma_bias_raw = student_section.get("sigma_bias", density_bias_raw)
    density_bias = float(density_bias_raw if density_bias_raw is not None else -1.0)
    sigma_bias_value = _coerce_float(sigma_bias_raw, density_bias)
    if sigma_bias_value is None:
        sigma_bias_value = density_bias

    skips_raw = student_section.get("skips", [])
    if isinstance(skips_raw, (list, tuple)):
        skips_values: List[int] = []
        for candidate in skips_raw:
            parsed = _coerce_optional_int(candidate)
            if parsed is not None:
                skips_values.append(parsed)
        skips_tuple = tuple(skips_values)
    else:
        parsed_skip = _coerce_optional_int(skips_raw)
        skips_tuple = (parsed_skip,) if parsed_skip is not None else tuple()

    pos_encoding = str(student_section.get("pos_encoding", "none") or "none")
    dir_encoding = str(student_section.get("dir_encoding", "none") or "none")
    try:
        pos_L_value = int(student_section.get("pos_L", 0) or 0)
    except (TypeError, ValueError):
        pos_L_value = 0
    pos_L_value = max(pos_L_value, 0)
    try:
        dir_L_value = int(student_section.get("dir_L", 0) or 0)
    except (TypeError, ValueError):
        dir_L_value = 0
    dir_L_value = max(dir_L_value, 0)
    sigma_activation_value = str(student_section.get("sigma_activation", student_section.get("density_activation", "relu")) or "relu")

    student = StudentConfig(
        type=student_section["type"],
        grid_resolution=grid_resolution,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=student_section.get("activation", "relu"),
        density_bias=density_bias,
        color_bias=float(student_section.get("color_bias", 0.0)),
        regularization_weight=float(student_section.get("regularization_weight", 0.0)),
        enable_boundary_blend=bool(student_section.get("enable_boundary_blend", False)),
        boundary_blend_margin=float(student_section.get("boundary_blend_margin", 0.05)),
        hash_levels=int(student_section.get("hash_levels", 16)),
        hash_features_per_level=int(student_section.get("hash_features_per_level", 2)),
        hash_log2_hashmap_size=int(student_section.get("hash_log2_hashmap_size", 19)),
        hash_base_resolution=int(student_section.get("hash_base_resolution", 16)),
        hash_per_level_scale=float(student_section.get("hash_per_level_scale", 1.5)),
        pos_encoding=pos_encoding,
        pos_L=pos_L_value,
        dir_encoding=dir_encoding,
        dir_L=dir_L_value,
        skips=skips_tuple,
        mlp_hidden=mlp_hidden,
        sigma_activation=sigma_activation_value,
        sigma_bias=sigma_bias_value,
    )
    train_section = raw["train"]
    max_steps_value = int(train_section.get("max_steps", 0))

    def _normalise_optimize_list(value: Optional[object]) -> Tuple[str, ...]:
        default_groups = ("student", "projector", "student_adapter", "teacher_adapter")
        if value is None:
            return default_groups
        if isinstance(value, str):
            tokens = [token.strip().lower() for token in value.replace(",", " ").split() if token.strip()]
            return tuple(tokens) if tokens else default_groups
        if isinstance(value, (list, tuple)):
            tokens = [str(token).strip().lower() for token in value if str(token).strip()]
            return tuple(tokens) if tokens else default_groups
        return default_groups

    phases_raw = train_section.get("phases", [])
    phase_configs: List[TrainPhaseConfig] = []
    if isinstance(phases_raw, list):
        current_start = 0
        for idx, phase_entry in enumerate(phases_raw):
            if not isinstance(phase_entry, dict):
                continue
            name = str(phase_entry.get("name", f"phase_{idx + 1}"))
            duration_val = phase_entry.get("duration")
            end_step_val = phase_entry.get("end_step")
            if end_step_val is not None:
                try:
                    end_step = int(end_step_val)
                except (TypeError, ValueError):
                    continue
            elif duration_val is not None:
                try:
                    end_step = current_start + int(duration_val)
                except (TypeError, ValueError):
                    continue
            elif max_steps_value > 0:
                end_step = max_steps_value
            else:
                continue
            if max_steps_value > 0:
                end_step = min(end_step, max_steps_value)
            if end_step <= current_start:
                continue
            optimize_groups = _normalise_optimize_list(phase_entry.get("optimize", phase_entry.get("train_modules")))
            mask_override_val = phase_entry.get("mask_override")
            mask_override = str(mask_override_val).lower() if isinstance(mask_override_val, str) else None
            feature_scale_raw = phase_entry.get("feature_weight_scale", 1.0)
            try:
                feature_scale = float(feature_scale_raw)
            except (TypeError, ValueError):
                feature_scale = 1.0
            feature_scale = max(feature_scale, 0.0)
            phase_configs.append(
                TrainPhaseConfig(
                    name=name,
                    start_step=current_start,
                    end_step=end_step,
                    optimize=optimize_groups,
                    mask_override=mask_override,
                    feature_weight_scale=feature_scale,
                )
            )
            current_start = end_step
            if max_steps_value > 0 and current_start >= max_steps_value:
                break

    lr_schedule_raw = train_section.get("lr_schedule", train_section.get("lr_decay_mode", "none"))
    lr_schedule = str(lr_schedule_raw or "none")

    lr_milestones_raw = train_section.get("lr_milestones", ())
    if isinstance(lr_milestones_raw, (int, float)):
        lr_milestones_raw = [lr_milestones_raw]
    lr_milestones = tuple(int(m) for m in lr_milestones_raw)

    lr_values_raw = train_section.get("lr_values", train_section.get("lr_milestone_values", ()))
    if isinstance(lr_values_raw, (int, float)):
        lr_values_raw = [lr_values_raw]
    lr_values = tuple(float(v) for v in lr_values_raw)

    lr_min_raw = train_section.get("lr_min", train_section.get("lr_final"))
    lr_min = float(lr_min_raw) if lr_min_raw is not None else None

    lr_schedule_steps_raw = train_section.get("lr_schedule_steps", train_section.get("lr_decay_steps"))
    lr_schedule_steps = int(lr_schedule_steps_raw) if lr_schedule_steps_raw is not None else None

    lr_warmup_steps = int(train_section.get("lr_warmup_steps", train_section.get("lr_schedule_warmup_steps", 0)) or 0)

    promotion_default: Tuple[int, ...] = tuple()
    if max_steps_value <= 0 or max_steps_value >= min(_DEFAULT_PROMOTION_GATES):
        promotion_default = tuple(
            gate for gate in _DEFAULT_PROMOTION_GATES if max_steps_value <= 0 or gate <= max_steps_value
        )
    promotion_raw = train_section.get("promotion_gates", promotion_default)
    promotion_gates: Tuple[int, ...] = tuple()
    if promotion_raw not in (None, False):
        parsed: List[int] = []
        if isinstance(promotion_raw, (int, float)):
            parsed = [int(promotion_raw)]
        elif isinstance(promotion_raw, str):
            if promotion_raw.strip().lower() in {"none", "disable", "disabled"}:
                parsed = []
            else:
                tokens = [token.strip() for token in promotion_raw.replace(",", " ").split() if token.strip()]
                for token in tokens:
                    try:
                        parsed.append(int(token))
                    except ValueError:
                        continue
        elif isinstance(promotion_raw, (list, tuple)):
            for value in promotion_raw:
                try:
                    parsed.append(int(value))
                except (TypeError, ValueError):
                    continue
        candidates: Set[int] = set()
        for gate in parsed:
            if gate <= 0:
                continue
            if max_steps_value > 0 and gate > max_steps_value:
                continue
            candidates.add(int(gate))
        if candidates:
            promotion_gates = tuple(sorted(candidates))

    promotion_min_mask_fraction = float(train_section.get("promotion_min_mask_fraction", 0.25) or 0.0)
    promotion_feature_dim_raw = train_section.get("promotion_feature_dim")
    promotion_feature_dim = None
    if promotion_feature_dim_raw not in (None, "auto", "AUTO"):
        try:
            promotion_feature_dim = int(promotion_feature_dim_raw)
        except (TypeError, ValueError):
            promotion_feature_dim = None
    promotion_min_feature_scale = float(train_section.get("promotion_min_feature_scale", 0.0) or 0.0)
    promotion_min_feature_ratio_raw = train_section.get("promotion_min_feature_ratio")
    if promotion_min_feature_ratio_raw is not None:
        try:
            promotion_min_feature_ratio = float(promotion_min_feature_ratio_raw)
        except (TypeError, ValueError):
            promotion_min_feature_ratio = 0.55
    else:
        promotion_min_feature_ratio = 0.55
    promotion_min_opacity_ratio_raw = train_section.get("promotion_min_opacity_ratio")
    if promotion_min_opacity_ratio_raw is not None:
        try:
            promotion_min_opacity_ratio = float(promotion_min_opacity_ratio_raw)
        except (TypeError, ValueError):
            promotion_min_opacity_ratio = 0.6
    else:
        promotion_min_opacity_ratio = 0.6
    promotion_projector_in_dim_raw = train_section.get("promotion_projector_in_dim")
    promotion_projector_in_dim = None
    if promotion_projector_in_dim_raw not in (None, "auto", "AUTO"):
        try:
            promotion_projector_in_dim = int(promotion_projector_in_dim_raw)
        except (TypeError, ValueError):
            promotion_projector_in_dim = None
    promotion_exit_code = int(train_section.get("promotion_exit_code", 12) or 12)
    effective_weight_avg_window = int(train_section.get("effective_weight_avg_window", 256) or 256)
    effective_weight_avg_window = max(effective_weight_avg_window, 1)
    require_feature_terminal = _coerce_bool(
        train_section.get("promotion_require_feature_schedule_terminal"),
        True,
    )
    require_opacity_terminal = _coerce_bool(
        train_section.get("promotion_require_opacity_schedule_terminal"),
        True,
    )

    loss_raw = raw["loss"]
    color_raw = loss_raw["color"]
    opacity_raw = loss_raw["opacity"]
    depth_raw = loss_raw.get("depth", {})
    feature_raw = raw.get("feature_pipeline", {})

    alpha_guard_section = loss_raw.get("alpha_guard")
    if alpha_guard_section is None:
        alpha_guard_section = opacity_raw.get("alpha_guard")
    if not isinstance(alpha_guard_section, dict):
        alpha_guard_section = {}
    try:
        alpha_guard_warmup_enforce = int(
            alpha_guard_section.get("warmup_enforce_steps", alpha_guard_section.get("enforce_warmup_steps", 0)) or 0
        )
    except (TypeError, ValueError):
        alpha_guard_warmup_enforce = 0
    try:
        alpha_guard_min_target_weight = float(alpha_guard_section.get("min_target_weight", 0.05) or 0.0)
    except (TypeError, ValueError):
        alpha_guard_min_target_weight = 0.05
    try:
        alpha_guard_adjustment_smoothing = float(alpha_guard_section.get("adjustment_smoothing", 0.25) or 0.0)
    except (TypeError, ValueError):
        alpha_guard_adjustment_smoothing = 0.25

    hysteresis_margin_raw = alpha_guard_section.get("hysteresis_margin")
    if hysteresis_margin_raw is None:
        alpha_guard_hysteresis_margin = 0.02
    else:
        try:
            alpha_guard_hysteresis_margin = float(hysteresis_margin_raw)
        except (TypeError, ValueError):
            alpha_guard_hysteresis_margin = 0.02
    alpha_guard_hysteresis_margin = max(0.0, float(alpha_guard_hysteresis_margin))

    min_update_samples_raw = alpha_guard_section.get("min_update_samples")
    if min_update_samples_raw is None:
        alpha_guard_min_update_samples = 2
    else:
        try:
            alpha_guard_min_update_samples = int(min_update_samples_raw)
        except (TypeError, ValueError):
            alpha_guard_min_update_samples = 2
    if alpha_guard_min_update_samples <= 0:
        alpha_guard_min_update_samples = 1

    max_lambda_delta_raw = alpha_guard_section.get("max_lambda_delta")
    if max_lambda_delta_raw is None:
        alpha_guard_max_lambda_delta = 0.05
    else:
        try:
            alpha_guard_max_lambda_delta = float(max_lambda_delta_raw)
        except (TypeError, ValueError):
            alpha_guard_max_lambda_delta = 0.05
    alpha_guard_max_lambda_delta = max(0.0, float(alpha_guard_max_lambda_delta))

    max_target_adjustment_delta_raw = alpha_guard_section.get("max_target_adjustment_delta")
    if max_target_adjustment_delta_raw is None:
        alpha_guard_max_target_adjustment_delta = 0.1
    else:
        try:
            alpha_guard_max_target_adjustment_delta = float(max_target_adjustment_delta_raw)
        except (TypeError, ValueError):
            alpha_guard_max_target_adjustment_delta = 0.1
    alpha_guard_max_target_adjustment_delta = max(0.0, float(alpha_guard_max_target_adjustment_delta))

    max_penalty_weight_delta_raw = alpha_guard_section.get("max_penalty_weight_delta")
    if max_penalty_weight_delta_raw is None:
        alpha_guard_max_penalty_weight_delta = 0.05
    else:
        try:
            alpha_guard_max_penalty_weight_delta = float(max_penalty_weight_delta_raw)
        except (TypeError, ValueError):
            alpha_guard_max_penalty_weight_delta = 0.05
    alpha_guard_max_penalty_weight_delta = max(0.0, float(alpha_guard_max_penalty_weight_delta))

    alpha_guard_cfg = AlphaGuardConfig(
        enabled=bool(alpha_guard_section.get("enabled", True)),
        check_interval=int(alpha_guard_section.get("check_interval", 200) or 200),
        penalty_hi=float(alpha_guard_section.get("penalty_hi", 0.15) or 0.0),
        penalty_lo=float(alpha_guard_section.get("penalty_lo", 0.05) or 0.0),
        tighten_rate=float(alpha_guard_section.get("tighten_rate", 0.9) or 0.9),
        relax_rate=float(alpha_guard_section.get("relax_rate", 1.02) or 1.0),
        lambda_floor=float(alpha_guard_section.get("lambda_floor", 0.1) or 0.0),
        lambda_cap=float(alpha_guard_section.get("lambda_cap", 1.0) or 1.0),
        weight_floor=float(alpha_guard_section.get("weight_floor", 0.12) or 0.12),
        weight_cap=float(alpha_guard_section.get("weight_cap", 0.45) or 0.45),
        band_weight=float(alpha_guard_section.get("band_weight", 1.0) or 0.0),
        fraction_hi_weight=float(alpha_guard_section.get("fraction_hi_weight", 1.5) or 0.0),
        fraction_lo_weight=float(alpha_guard_section.get("fraction_lo_weight", 1.0) or 0.0),
        initial_weight=float(alpha_guard_section.get("initial_weight", 0.2) or 0.0),
        avg_window=max(1, int(alpha_guard_section.get("avg_window", 256) or 256)),
        min_target_weight=alpha_guard_min_target_weight,
        warmup_enforce_steps=alpha_guard_warmup_enforce,
        adjustment_smoothing=alpha_guard_adjustment_smoothing,
        hysteresis_margin=alpha_guard_hysteresis_margin,
        min_update_samples=alpha_guard_min_update_samples,
        max_lambda_delta=alpha_guard_max_lambda_delta,
        max_target_adjustment_delta=alpha_guard_max_target_adjustment_delta,
        max_penalty_weight_delta=alpha_guard_max_penalty_weight_delta,
    )

    mask_prefail_section = feature_raw.get("mask_prefail")
    if not isinstance(mask_prefail_section, dict):
        mask_prefail_section = {}
    mask_prefail_cfg = MaskPrefailConfig(
        enabled=bool(mask_prefail_section.get("enabled", True)),
        window=max(4, int(mask_prefail_section.get("window", 64) or 64)),
        p5_drop_rate=float(mask_prefail_section.get("p5_drop_rate", 5e-4) or 0.0),
        min_drop_rate=float(mask_prefail_section.get("min_drop_rate", 1e-4) or 0.0),
        threshold_scale=float(mask_prefail_section.get("threshold_scale", 0.8) or 0.8),
        soft_floor_delta=float(mask_prefail_section.get("soft_floor_delta", 0.03) or 0.0),
        variance_ceiling=float(mask_prefail_section.get("variance_ceiling", 2.5e-5) or 0.0),
        cooldown_steps=int(mask_prefail_section.get("cooldown_steps", 200) or 200),
    )

    train_cfg = TrainConfig(
        max_steps=max_steps_value,
        eval_interval=int(train_section.get("eval_interval", 0)),
        checkpoint_interval=int(train_section.get("checkpoint_interval", 0)),
        lr=float(train_section.get("lr", 0.0)),
        lr_decay_steps=int(train_section.get("lr_decay_steps", 0)),
        lr_decay_gamma=float(train_section.get("lr_decay_gamma", 1.0)),
        gradient_clip_norm=float(train_section.get("gradient_clip_norm", 1.0)),
        ema_decay=float(train_section.get("ema_decay", 0.999)),
        lr_schedule=lr_schedule,
        lr_schedule_milestones=lr_milestones,
        lr_schedule_values=lr_values,
        lr_schedule_min_lr=lr_min,
        lr_schedule_steps=lr_schedule_steps,
        lr_warmup_steps=lr_warmup_steps,
        phases=tuple(phase_configs),
        promotion_gates=promotion_gates,
        promotion_min_mask_fraction=promotion_min_mask_fraction,
        promotion_feature_dim=promotion_feature_dim,
        promotion_min_feature_scale=promotion_min_feature_scale,
        promotion_min_feature_ratio=promotion_min_feature_ratio,
        promotion_min_opacity_ratio=promotion_min_opacity_ratio,
        promotion_projector_in_dim=promotion_projector_in_dim,
    promotion_require_feature_schedule_terminal=require_feature_terminal,
    promotion_require_opacity_schedule_terminal=require_opacity_terminal,
        promotion_exit_code=promotion_exit_code,
        effective_weight_avg_window=effective_weight_avg_window,
        alpha_guard=alpha_guard_cfg,
        mask_prefail=mask_prefail_cfg,
        input_guard_notice_interval=float(train_section.get("input_guard_notice_interval", 0.0) or 0.0),
    )
    feature_section = loss_raw.get("feature", {}) if isinstance(loss_raw.get("feature"), dict) else {}
    feature_schedule_raw = feature_section.get("schedule", loss_raw.get("feature_schedule", "none"))
    feature_schedule = str(feature_schedule_raw or "none")
    schedule_duration_raw = feature_section.get(
        "schedule_duration",
        loss_raw.get("feature_schedule_duration", 0),
    )
    schedule_duration = int(schedule_duration_raw or 0)

    target_weight_raw = feature_section.get("target_weight", loss_raw.get("feature_target_weight"))
    target_cos_raw = feature_section.get("target_cosine_weight", loss_raw.get("feature_target_cosine_weight"))
    feature_targets_section = raw.get("feature_targets", {})
    if isinstance(feature_targets_section, dict):
        if target_weight_raw is None:
            target_weight_raw = feature_targets_section.get("l2_target") or feature_targets_section.get("l2")
        if target_cos_raw is None:
            target_cos_raw = (
                feature_targets_section.get("cos_target")
                or feature_targets_section.get("cosine_target")
                or feature_targets_section.get("cos")
            )

    target_weight = float(target_weight_raw) if target_weight_raw is not None else None
    target_cos_weight = float(target_cos_raw) if target_cos_raw is not None else None
    background_color_list = raw["data"].get("background_color")
    background_color_tuple: Optional[Tuple[float, float, float]] = None
    if background_color_list is not None and len(background_color_list) == 3:
        background_color_tuple = (
            float(background_color_list[0]),
            float(background_color_list[1]),
            float(background_color_list[2]),
        )

    opacity_target_val = opacity_raw.get("target")
    opacity_target: Optional[float] = None
    if opacity_target_val is not None:
        try:
            opacity_target = float(opacity_target_val)
        except (TypeError, ValueError):
            opacity_target = None

    opacity_target_start_weight_raw = opacity_raw.get("start_weight", opacity_raw.get("initial_weight"))
    if opacity_target_start_weight_raw is not None:
        try:
            opacity_target_start_weight = float(opacity_target_start_weight_raw)
        except (TypeError, ValueError):
            opacity_target_start_weight = None
    else:
        opacity_target_start_weight = None

    opacity_warmup_raw = opacity_raw.get("warmup_steps", opacity_raw.get("warmup_step"))
    try:
        opacity_target_warmup_steps = int(opacity_warmup_raw)
    except (TypeError, ValueError):
        opacity_target_warmup_steps = 0
    opacity_target_warmup_steps = max(opacity_target_warmup_steps, 0)

    opacity_schedule_raw = opacity_raw.get("schedule", opacity_raw.get("schedule_mode"))
    opacity_target_schedule = str(opacity_schedule_raw or "none")

    warm_start_offset_raw = opacity_raw.get("warm_start_offset")
    try:
        opacity_target_warm_start_offset = int(warm_start_offset_raw) if warm_start_offset_raw is not None else 0
    except (TypeError, ValueError):
        opacity_target_warm_start_offset = 0
    opacity_target_warm_start_offset = max(opacity_target_warm_start_offset, 0)

    opacity_schedule_duration_raw = opacity_raw.get("schedule_duration", opacity_raw.get("schedule_steps"))
    try:
        opacity_target_schedule_duration = int(opacity_schedule_duration_raw)
    except (TypeError, ValueError):
        opacity_target_schedule_duration = 0
    opacity_target_schedule_duration = max(opacity_target_schedule_duration, 0)

    hysteresis_raw = opacity_raw.get("hysteresis", opacity_raw.get("enable_hysteresis"))
    if hysteresis_raw is None:
        opacity_target_hysteresis = True
    else:
        opacity_target_hysteresis = bool(hysteresis_raw)

    color_eps_value = _coerce_float(color_raw.get("eps", color_raw.get("epsilon")), 1e-3)
    if color_eps_value is None or color_eps_value <= 0.0:
        color_eps_value = 1e-3

    opacity_max_weight = _coerce_float(opacity_raw.get("max_weight"))
    if opacity_max_weight is not None and opacity_max_weight < 0.0:
        opacity_max_weight = 0.0

    color_secondary_type_raw = color_raw.get("secondary_type")
    color_secondary_type = str(color_secondary_type_raw).strip().lower() if color_secondary_type_raw is not None else None
    if color_secondary_type == "":
        color_secondary_type = None
    color_secondary_weight_value = _coerce_float(color_raw.get("secondary_weight"), 0.0)
    if color_secondary_weight_value is None:
        color_secondary_weight_value = 0.0
    color_secondary_weight_value = float(min(max(color_secondary_weight_value, 0.0), 1.0))
    color_secondary_eps_value = _coerce_float(color_raw.get("secondary_eps"), color_eps_value)
    if color_secondary_eps_value is None or color_secondary_eps_value <= 0.0:
        color_secondary_eps_value = color_eps_value

    opacity_mean_target = _coerce_float(opacity_raw.get("mean_target"))
    opacity_mean_weight = _coerce_float(opacity_raw.get("mean_weight"), 0.0)
    if opacity_mean_weight is None:
        opacity_mean_weight = 0.0
    opacity_mean_weight = max(0.0, float(opacity_mean_weight))

    loss_cfg = LossConfig(
        color_weight=float(color_raw["weight"]),
        opacity_weight=float(opacity_raw["weight"]),
        color_type=color_raw.get("type", "l2"),
        color_eps=float(color_eps_value),
        color_secondary_type=color_secondary_type,
        color_secondary_weight=float(color_secondary_weight_value),
        color_secondary_eps=float(color_secondary_eps_value),
        opacity_type=opacity_raw.get("type", "l1"),
        temperature=float(loss_raw.get("distillation_temperature", 1.0)),
        opacity_temperature=(
            float(opacity_raw["temperature"]) if "temperature" in opacity_raw and opacity_raw["temperature"] is not None else None
        ),
        opacity_lambda=float(opacity_raw.get("lambda", 1.0) or 1.0),
        depth_weight=float(depth_raw.get("weight", 0.0) or 0.0),
        depth_type=depth_raw.get("type", "l1"),
        depth_alpha_threshold=float(depth_raw.get("alpha_threshold", 0.0) or 0.0),
        feature_weight=float(feature_section.get("weight", loss_raw.get("feature_weight", 0.0)) or 0.0),
        feature_type=feature_section.get("type", loss_raw.get("feature_type", "l2")),
        feature_cosine_weight=float(feature_section.get("cosine_weight", loss_raw.get("feature_cosine_weight", 0.0)) or 0.0),
        feature_warmup_steps=int(
            feature_section.get(
                "warmup_steps",
                feature_section.get(
                    "warmup_step",
                    loss_raw.get("feature_warmup_steps", loss_raw.get("feature_warmup_step", 0)),
                ),
            )
            or 0
        ),
        feature_schedule=feature_schedule,
        feature_schedule_duration=schedule_duration,
        feature_target_weight=target_weight,
        feature_target_cosine_weight=target_cos_weight,
        opacity_target=opacity_target,
        opacity_target_weight=float(opacity_raw.get("target_weight", 0.0) or 0.0),
        opacity_target_background_threshold=float(opacity_raw.get("background_threshold", 0.05) or 0.0),
        opacity_target_start_weight=opacity_target_start_weight,
        opacity_target_warmup_steps=opacity_target_warmup_steps,
        opacity_target_schedule=opacity_target_schedule,
        opacity_target_schedule_duration=opacity_target_schedule_duration,
    opacity_target_warm_start_offset=opacity_target_warm_start_offset,
        background_color=background_color_tuple,
        opacity_target_hysteresis=opacity_target_hysteresis,
        opacity_target_max_weight=opacity_max_weight,
        opacity_mean_target=opacity_mean_target,
        opacity_mean_weight=float(opacity_mean_weight),
    )
    feature_raw = raw.get("feature_pipeline", {})
    threshold_raw = feature_raw.get("boundary_mask_threshold", 0.75)
    components_raw = feature_raw.get("teacher_components")
    if isinstance(components_raw, (list, tuple)):
        teacher_components = tuple(str(comp) for comp in components_raw)
    elif isinstance(components_raw, str):
        teacher_components = (components_raw,)
    else:
        teacher_components = ()

    embedding_raw = feature_raw.get("teacher_embedding")
    embedding_cfg: Optional[TeacherEmbeddingConfig] = None
    if isinstance(embedding_raw, dict):
        checkpoint_val = embedding_raw.get("checkpoint")
        stats_val = embedding_raw.get("stats_path")
        latent_val = embedding_raw.get("latent_dim")
        embedding_cfg = TeacherEmbeddingConfig(
            type=str(embedding_raw.get("type", "identity")),
            checkpoint=Path(checkpoint_val) if checkpoint_val else None,
            stats_path=Path(stats_val) if stats_val else None,
            latent_dim=int(latent_val) if latent_val is not None else None,
            standardize=embedding_raw.get("standardize"),
            device=str(embedding_raw.get("device", "auto")),
            dtype=str(embedding_raw.get("dtype", "float32")),
        )

    projector_section = feature_raw.get("student_projector")
    legacy_projector = feature_raw.get("projector")
    if not isinstance(projector_section, dict):
        if isinstance(legacy_projector, dict):
            projector_section = {
                "input_dim": legacy_projector.get("input_dim", legacy_projector.get("in_dim")),
                "hidden_dim": legacy_projector.get("hidden_dim"),
                "output_dim": legacy_projector.get("output_dim", legacy_projector.get("out_dim")),
                "activation": legacy_projector.get("activation"),
                "use_layer_norm": legacy_projector.get("use_layer_norm"),
                "dropout": legacy_projector.get("dropout"),
            }
        else:
            projector_section = {}

    projector_hidden = projector_section.get("hidden_dim")
    if projector_hidden is None:
        projector_hidden = feature_raw.get("projector_hidden_dim", 64)

    projector_input = projector_section.get("input_dim")
    if projector_input is None:
        projector_input = feature_raw.get("projector_input_dim", 4)

    projector_output = projector_section.get("output_dim")
    if projector_output is None:
        projector_output = feature_raw.get("projector_output_dim", 3)

    projector_activation = projector_section.get("activation")
    if projector_activation is None:
        projector_activation = feature_raw.get("projector_activation", "relu")

    projector_ln = projector_section.get("use_layer_norm")
    if projector_ln is None:
        projector_ln = feature_raw.get("projector_use_layer_norm", False)

    projector_dropout = projector_section.get("dropout")
    if projector_dropout is None:
        projector_dropout = feature_raw.get("projector_dropout", 0.0)
    compare_space = str(feature_raw.get("compare_space", "auto"))
    allow_dim_mismatch = bool(feature_raw.get("allow_dim_mismatch", False))
    student_features_raw = feature_raw.get("student_features")
    if not isinstance(student_features_raw, dict):
        student_features_raw = {}
    student_feature_source = str(student_features_raw.get("source", "penultimate"))
    student_feature_activation = str(student_features_raw.get("activation", student_features_raw.get("mode", "post")))
    student_feature_dim_raw = student_features_raw.get("dim", student_features_raw.get("dimension"))
    try:
        student_feature_dim = int(student_feature_dim_raw) if student_feature_dim_raw is not None else None
    except (TypeError, ValueError):
        student_feature_dim = None
    student_head_raw = feature_raw.get("student_head")
    if student_head_raw is None and feature_raw.get("student_adapter") is not None:
        student_head_raw = feature_raw.get("student_adapter")
    student_head_cfg = _parse_feature_adapter_config(student_head_raw)
    teacher_adapter_cfg = _parse_feature_adapter_config(feature_raw.get("teacher_adapter"))

    mask_controller_raw = feature_raw.get("mask_controller")
    if isinstance(mask_controller_raw, dict):
        mask_controller_cfg = MaskControllerConfig(
            enabled=bool(mask_controller_raw.get("enabled", True)),
            activation_step=_coerce_optional_int(mask_controller_raw.get("activation_step")),
            activation_offset=int(mask_controller_raw.get("activation_offset", 3000) or 0),
            min_activation_step=int(mask_controller_raw.get("min_activation_step", 7000) or 0),
            ramp_duration=max(0, int(mask_controller_raw.get("ramp_duration", 2000) or 0)),
            initial_threshold=_coerce_float(mask_controller_raw.get("initial_threshold")),
            min_threshold=float(mask_controller_raw.get("min_threshold", 0.03) or 0.0),
            min_fraction=float(mask_controller_raw.get("min_fraction", 0.20) or 0.0),
            relaxation=float(mask_controller_raw.get("relaxation", 0.05) or 0.0),
            soft_transition_step=float(mask_controller_raw.get("soft_transition_step", 0.05) or 0.0),
            cap_threshold=_coerce_float(mask_controller_raw.get("cap_threshold"), 0.30),
            emergency_fraction=float(mask_controller_raw.get("emergency_fraction", 0.05) or 0.0),
            recovery_fraction=float(mask_controller_raw.get("recovery_fraction", 0.15) or 0.0),
        )
    else:
        mask_controller_cfg = MaskControllerConfig()

    feature_cfg = FeaturePipelineConfig(
        enabled=bool(feature_raw.get("enabled", False)),
        teacher_mode=str(feature_raw.get("teacher_mode", "rgb")),
        teacher_components=teacher_components,
        projector_input_dim=int(projector_input),
        projector_hidden_dim=int(projector_hidden),
        projector_output_dim=int(projector_output),
        projector_activation=str(projector_activation),
        projector_use_layer_norm=bool(projector_ln),
        projector_dropout=float(projector_dropout or 0.0),
        compare_space=compare_space,
        allow_dim_mismatch=allow_dim_mismatch,
        student_feature_source=student_feature_source.lower(),
        student_feature_activation=student_feature_activation.lower(),
        student_feature_dim=student_feature_dim,
        student_head=student_head_cfg,
        teacher_adapter=teacher_adapter_cfg,
        boundary_mask_threshold=None if threshold_raw is None else float(threshold_raw),
        boundary_mask_soft_transition=float(feature_raw.get("boundary_mask_soft_transition", 0.0) or 0.0),
        boundary_mask_soft_mode=str(feature_raw.get("boundary_mask_soft_mode", "linear") or "linear"),
        boundary_mask_soft_floor=float(feature_raw.get("boundary_mask_soft_floor", 0.0) or 0.0),
        mask_controller=mask_controller_cfg,
        teacher_embedding=embedding_cfg,
    )
    feature_aux_raw = raw.get("feature_aux_student")
    feature_aux_cfg = FeatureAuxStudentConfig()
    if isinstance(feature_aux_raw, dict) and feature_aux_raw:
        weight_block = feature_aux_raw.get("weight")
        if not isinstance(weight_block, dict):
            weight_block = {}
        patch_block = feature_aux_raw.get("patch")
        if not isinstance(patch_block, dict):
            patch_block = {}

        normalize_value = feature_aux_raw.get("normalize")
        if normalize_value is not None:
            normalize_str = str(normalize_value).lower()
            if normalize_str in {"", "none", "false", "0"}:
                normalize_value = None
            else:
                normalize_value = normalize_str

        feature_aux_cfg = FeatureAuxStudentConfig(
            enabled=bool(feature_aux_raw.get("enabled", True)),
            source=str(feature_aux_raw.get("source", "penultimate_post") or "penultimate_post").lower(),
            loss=str(feature_aux_raw.get("loss", "patch_cosine") or "patch_cosine").lower(),
            weight_start=float(weight_block.get("start", weight_block.get("start_weight", 0.0)) or 0.0),
            weight_target=float(weight_block.get("target", weight_block.get("target_weight", 0.0)) or 0.0),
            weight_warmup_steps=max(
                int(weight_block.get("warmup_steps", weight_block.get("warmup_step", 0)) or 0),
                0,
            ),
            weight_schedule=str(weight_block.get("schedule", "none") or "none").lower(),
            weight_schedule_duration=max(
                int(weight_block.get("schedule_duration", weight_block.get("duration", 0)) or 0),
                0,
            ),
            patch_rays=max(int(patch_block.get("rays_per_patch", 16) or 16), 1),
            patch_stride=max(int(patch_block.get("stride", 1) or 1), 1),
            normalize=normalize_value if normalize_value is None else str(normalize_value),
        )

    logging_raw = raw["logging"]
    scalar_interval = int(logging_raw.get("log_interval", logging_raw.get("metrics_interval", 100)) or 100)
    if scalar_interval <= 0:
        scalar_interval = 100
    flush_secs_value = logging_raw.get("tensorboard_flush_secs")
    flush_secs = None
    if flush_secs_value is not None:
        try:
            flush_secs_candidate = int(flush_secs_value)
        except (TypeError, ValueError):
            flush_secs_candidate = None
        if flush_secs_candidate is not None and flush_secs_candidate > 0:
            flush_secs = flush_secs_candidate

    axis_raw_value = logging_raw.get("tensorboard_axis")
    axis_mode = "step"
    if isinstance(axis_raw_value, str):
        candidate = axis_raw_value.strip().lower()
        if candidate in {"step", "time", "elapsed"}:
            axis_mode = candidate
    elif axis_raw_value is not None:
        try:
            candidate = str(axis_raw_value).strip().lower()
        except Exception:
            candidate = ""
        if candidate in {"step", "time", "elapsed"}:
            axis_mode = candidate

    logging_cfg = LoggingConfig(
        tensorboard=Path(logging_raw["tensorboard"]),
        csv=Path(logging_raw["csv"]),
        render_preview_interval=int(logging_raw["render_preview_interval"]),
        scalar_interval=scalar_interval,
        tensorboard_flush_secs=flush_secs,
        tensorboard_axis=axis_mode,
    )
    return experiment, data, teacher, student, train_cfg, loss_cfg, logging_cfg, feature_cfg, feature_aux_cfg


def set_seed(
    seed: int,
    *,
    strict_pythonhash: bool = True,
    require_cublas_determinism: bool = True,
) -> None:
    workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    deterministic_configs = {":16:8", ":4096:8"}
    if require_cublas_determinism:
        if workspace_config not in deterministic_configs:
            raise SystemExit(
                "[seed] CUBLAS_WORKSPACE_CONFIG is %s; set to ':16:8' or ':4096:8' before launching to enable deterministic cuBLAS kernels." % (workspace_config or "unset")
            )
        print(
            f"[seed] CUBLAS_WORKSPACE_CONFIG={workspace_config} (deterministic cuBLAS kernels enabled)."
        )
    else:
        if workspace_config in deterministic_configs:
            print(
                f"[seed] CUBLAS_WORKSPACE_CONFIG={workspace_config} (deterministic cuBLAS kernels enabled)."
            )
        elif workspace_config:
            print(
                "[seed] Warning: CUBLAS_WORKSPACE_CONFIG=%s may be non-deterministic; unset or use ':16:8' / ':4096:8' if reproducibility drifts." % workspace_config
            )
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            workspace_config = ":4096:8"
            print("[seed] CUBLAS_WORKSPACE_CONFIG unset; forcing ':4096:8' for deterministic cuBLAS kernels.")

    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed != str(seed):
        message = f"[seed] PYTHONHASHSEED is {hash_seed or 'unset'}; expected {seed} for reproducibility."
        if strict_pythonhash:
            raise SystemExit(message + " Export PYTHONHASHSEED before launching.")
        print(message + " Setting PYTHONHASHSEED in-process for evaluation.")
        os.environ["PYTHONHASHSEED"] = str(seed)
        hash_seed = str(seed)
    print(f"[seed] PYTHONHASHSEED aligned to {hash_seed}.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except Exception as err:
        print(f"[seed] Failed to enforce deterministic algorithms: {err}")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False


def compute_losses(
    student_rgb: torch.Tensor,
    student_sigma: torch.Tensor,
    teacher_rgb: torch.Tensor,
    teacher_alpha: torch.Tensor,
    cfg: LossConfig,
    student_depth: Optional[torch.Tensor] = None,
    teacher_depth: Optional[torch.Tensor] = None,
    depth_mask: Optional[torch.Tensor] = None,
    background_rgb: Optional[torch.Tensor] = None,
    opacity_target_weight_override: Optional[float] = None,
    opacity_lambda_override: Optional[float] = None,
):
    losses: Dict[str, torch.Tensor] = {}

    def _compute_color_core(loss_type: str, eps_value: float) -> torch.Tensor:
        if loss_type == "l2":
            return torch.nn.functional.mse_loss(student_rgb, teacher_rgb)
        if loss_type == "l1":
            return torch.nn.functional.l1_loss(student_rgb, teacher_rgb)
        if loss_type == "charbonnier":
            eps_local = float(eps_value or 1e-3)
            diff_local = student_rgb - teacher_rgb
            return torch.sqrt(diff_local * diff_local + eps_local * eps_local).mean()
        raise ValueError(f"Unsupported color loss type: {loss_type}")

    color_primary = _compute_color_core(cfg.color_type, float(getattr(cfg, "color_eps", 1e-3) or 1e-3))
    color_loss = color_primary
    if cfg.color_secondary_type and float(cfg.color_secondary_weight) > 0.0:
        secondary_core = _compute_color_core(cfg.color_secondary_type, float(cfg.color_secondary_eps or cfg.color_eps))
        blend = float(cfg.color_secondary_weight)
        blend = min(max(blend, 0.0), 1.0)
        color_loss = (1.0 - blend) * color_loss + blend * secondary_core
        losses["color_secondary_component"] = cfg.color_weight * secondary_core
    losses["color"] = cfg.color_weight * color_loss

    student_opacity = student_sigma
    if cfg.opacity_temperature is not None and cfg.opacity_temperature > 0.0:
        lambda_value = float(cfg.opacity_lambda)
        if opacity_lambda_override is not None:
            lambda_value = float(opacity_lambda_override)
        sigma_scaled = torch.clamp(student_sigma, min=0.0) * lambda_value
        student_opacity = 1.0 - torch.exp(-sigma_scaled * float(cfg.opacity_temperature))

    if cfg.opacity_type == "l2":
        opacity_loss = torch.nn.functional.mse_loss(student_opacity, teacher_alpha)
    elif cfg.opacity_type == "l1":
        opacity_loss = torch.nn.functional.l1_loss(student_opacity, teacher_alpha)
    else:
        raise ValueError(f"Unsupported opacity loss type: {cfg.opacity_type}")
    losses["opacity"] = cfg.opacity_weight * opacity_loss

    if cfg.opacity_mean_target is not None and cfg.opacity_mean_weight > 0.0:
        mean_target_value = float(cfg.opacity_mean_target)
        mean_residual = (student_opacity.mean() - mean_target_value).abs()
        losses["opacity_mean"] = cfg.opacity_mean_weight * mean_residual

    if (
        cfg.depth_weight > 0.0
        and student_depth is not None
        and teacher_depth is not None
    ):
        depth_pred = student_depth
        depth_target = teacher_depth
        elementwise_diff: torch.Tensor
        if cfg.depth_type == "l2":
            elementwise_diff = (depth_pred - depth_target) ** 2
        elif cfg.depth_type == "l1":
            elementwise_diff = (depth_pred - depth_target).abs()
        elif cfg.depth_type in {"smooth_l1", "huber"}:
            elementwise_diff = torch.nn.functional.smooth_l1_loss(
                depth_pred,
                depth_target,
                reduction="none",
            )
        else:
            raise ValueError(f"Unsupported depth loss type: {cfg.depth_type}")

        if depth_mask is not None:
            mask = depth_mask.to(elementwise_diff.device, dtype=elementwise_diff.dtype)
            while mask.dim() < elementwise_diff.dim():
                mask = mask.unsqueeze(-1)
            weighted = elementwise_diff * mask
            denom = mask.sum().clamp_min(1e-6)
            depth_loss = weighted.sum() / denom
        else:
            depth_loss = elementwise_diff.mean()

        losses["depth"] = cfg.depth_weight * depth_loss

    total_loss = torch.zeros((), device=student_rgb.device)
    for key, value in losses.items():
        if key == "color_secondary_component":
            continue
        total_loss = total_loss + value

    effective_opacity_target_weight = float(
        cfg.opacity_target_weight if opacity_target_weight_override is None else opacity_target_weight_override
    )
    if cfg.opacity_target is not None and effective_opacity_target_weight > 0.0:
        target_value = float(cfg.opacity_target)
        target_tensor = torch.full_like(student_opacity, target_value)

        background_tensor = background_rgb
        if background_tensor is None and cfg.background_color is not None:
            background_tensor = torch.tensor(
                cfg.background_color,
                dtype=student_rgb.dtype,
                device=student_rgb.device,
            )

        background_mask: Optional[torch.Tensor] = None
        if (
            background_tensor is not None
            and cfg.opacity_target_background_threshold >= 0.0
        ):
            bg = background_tensor.view(1, 1, -1) if background_tensor.dim() > 1 else background_tensor.view(1, -1)
            if teacher_rgb.dim() == 2:
                rgb_diff = torch.abs(teacher_rgb - bg.squeeze(0))
            else:
                rgb_diff = torch.abs(teacher_rgb - bg)
            diff_metric = rgb_diff.max(dim=-1, keepdim=True).values
            background_mask = (diff_metric <= cfg.opacity_target_background_threshold).to(student_opacity.dtype)

        residual = torch.nn.functional.relu(student_opacity - target_tensor)
        if background_mask is not None:
            weighted_residual = residual * background_mask
            denom = background_mask.sum().clamp_min(1.0)
        else:
            weighted_residual = residual
            denom = torch.tensor(float(weighted_residual.numel()), device=weighted_residual.device)
        regulariser = (weighted_residual.square().sum() / denom).clamp_min(0.0)
        losses["opacity_target"] = effective_opacity_target_weight * regulariser
        total_loss = total_loss + losses["opacity_target"]
    elif cfg.opacity_target is not None:
        losses["opacity_target"] = torch.zeros((), device=student_rgb.device)

    return total_loss, losses

def train(
    config_path: Path,
    resume_path: Optional[Path] = None,
    *,
    overfit_mode: Optional[str] = None,
    overfit_steps: Optional[int] = None,
    overfit_lr: Optional[float] = None,
    max_steps_override: Optional[int] = None,
):
    (
        experiment,
        data_cfg,
        teacher_cfg,
        student_cfg,
        train_cfg,
        loss_cfg,
        logging_cfg,
        feature_cfg,
        feature_aux_cfg,
    ) = parse_config(config_path)

    if max_steps_override is not None:
        try:
            override_steps = int(max_steps_override)
        except (TypeError, ValueError) as err:
            raise TrainingAbort(
                f"Invalid max-steps override '{max_steps_override}': {err}", exit_code=12
            ) from err
        if override_steps <= 0:
            raise TrainingAbort("max-steps override must be a positive integer", exit_code=12)
        original_max = int(train_cfg.max_steps)
        if original_max > 0 and override_steps < original_max:
            print(
                f"[config] max_steps override {override_steps} is below config max_steps {original_max}; using override regardless.",
                flush=True,
            )
        if override_steps != train_cfg.max_steps:
            phase_list = list(train_cfg.phases)
            if phase_list and override_steps > 0:
                updated = []
                for phase in phase_list:
                    end_step = phase.end_step
                    if original_max > 0 and end_step == original_max and override_steps > original_max:
                        updated.append(
                            TrainPhaseConfig(
                                name=phase.name,
                                start_step=phase.start_step,
                                end_step=override_steps,
                                optimize=phase.optimize,
                                mask_override=phase.mask_override,
                                feature_weight_scale=phase.feature_weight_scale,
                            )
                        )
                    else:
                        updated.append(phase)
                train_cfg.phases = tuple(updated)
            print(
                f"[config] Overriding train.max_steps {train_cfg.max_steps} -> {override_steps}",
                flush=True,
            )
            train_cfg.max_steps = override_steps

    resume_checkpoint_path: Optional[Path] = None
    if resume_path is not None:
        resume_checkpoint_path = Path(resume_path)
        if not resume_checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint_path}")

    experiment.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = experiment.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logging_cfg.csv.parent.mkdir(parents=True, exist_ok=True)
    if logging_cfg.tensorboard.is_file():
        logging_cfg.tensorboard.parent.mkdir(parents=True, exist_ok=True)
    else:
        logging_cfg.tensorboard.mkdir(parents=True, exist_ok=True)

    _clean_tensorboard_events(logging_cfg.tensorboard if logging_cfg.tensorboard.is_dir() else logging_cfg.tensorboard.parent)

    debug_log_path = logging_cfg.tensorboard.parent / "tb_debug.log"
    debug_log_path.parent.mkdir(parents=True, exist_ok=True)

    def debug_log(message: str) -> None:
        if os.getenv("KILOGS_DEBUG_TB", "0") != "1":
            return
        try:
            timestamp = datetime.utcnow().isoformat()
            with debug_log_path.open("a", encoding="utf-8") as fp:
                fp.write(f"{timestamp} {message}\n")
        except Exception:
            pass

    writer: Optional[SummaryWriter] = None
    tb_flush_env = os.getenv("KILOGS_TENSORBOARD_FLUSH_SECS")
    writer_kwargs = {}
    if tb_flush_env:
        try:
            writer_kwargs["flush_secs"] = max(1, int(tb_flush_env))
        except ValueError:
            pass
    elif logging_cfg.tensorboard_flush_secs is not None:
        writer_kwargs["flush_secs"] = max(1, int(logging_cfg.tensorboard_flush_secs))
    try:
        writer = SummaryWriter(log_dir=str(logging_cfg.tensorboard), **writer_kwargs)
        writer_dir = getattr(writer, "log_dir", str(logging_cfg.tensorboard))
        print(f"[logging] TensorBoard writer initialised → {writer_dir}", flush=True)
    except Exception as err:  # pragma: no cover - optional dependency issues
        print(f"[logging] Failed to initialise TensorBoard writer: {err}")
        writer = None

    tensorboard_axis_mode = (logging_cfg.tensorboard_axis or "step").strip().lower()
    if tensorboard_axis_mode not in {"step", "time", "elapsed"}:
        tensorboard_axis_mode = "step"
    logging_cfg.tensorboard_axis = tensorboard_axis_mode
    tensorboard_use_walltime = tensorboard_axis_mode in {"time", "elapsed"}
    tensorboard_use_elapsed = tensorboard_axis_mode == "elapsed"
    train_start_wall = time.time()

    if writer is not None:
        def _close_tensorboard_writer() -> None:
            try:
                writer.flush()
            except Exception as err:
                debug_log(f"tb_flush_error_teardown err={err}")
            try:
                writer.close()
            except Exception as err:
                debug_log(f"tb_close_error_teardown err={err}")

        atexit.register(_close_tensorboard_writer)

    strict_pythonhash = _coerce_bool(os.getenv("KILOGS_STRICT_PYHASH"), default=True)
    if not strict_pythonhash:
        expected_hash = str(experiment.seed)
        current_hash = os.environ.get("PYTHONHASHSEED")
        if current_hash != expected_hash:
            print(
                f"[seed] Aligning PYTHONHASHSEED to {expected_hash} (previously {current_hash or 'unset'}).",
                flush=True,
            )
            os.environ["PYTHONHASHSEED"] = expected_hash
    require_cublas_determinism = _coerce_bool(
        os.getenv("KILOGS_REQUIRE_CUBLAS_DETERMINISM"),
        default=True,
    )
    set_seed(
        experiment.seed,
        strict_pythonhash=strict_pythonhash,
        require_cublas_determinism=require_cublas_determinism,
    )

    dataset = LegoRayDataset(data_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model = StudentModel(student_cfg).to(device)

    resolved_overfit_mode: Optional[str] = None
    if overfit_mode:
        candidate = overfit_mode.strip().lower()
        valid_modes = {"projector", "student", "all"}
        if candidate not in valid_modes:
            raise ValueError(f"Unrecognised overfit_mode '{overfit_mode}'. Expected one of {sorted(valid_modes)}.")
        resolved_overfit_mode = candidate

    overfit_enabled = resolved_overfit_mode is not None
    if overfit_enabled:
        if overfit_steps is not None and overfit_steps > 0:
            if train_cfg.max_steps <= 0:
                train_cfg.max_steps = int(overfit_steps)
            else:
                train_cfg.max_steps = min(int(overfit_steps), train_cfg.max_steps)
        else:
            if train_cfg.max_steps <= 0:
                train_cfg.max_steps = 512
            else:
                train_cfg.max_steps = min(train_cfg.max_steps, 512)
        if train_cfg.max_steps <= 0:
            raise ValueError("Overfit mode requires max_steps > 0. Provide overfit_steps or set train.max_steps in the config.")
        if overfit_lr is not None and overfit_lr > 0.0:
            train_cfg.lr = float(overfit_lr)
    if train_cfg.max_steps <= 0:
        raise ValueError("train.max_steps must be positive.")

    feature_pipeline_active = feature_cfg.enabled and any(
        weight > 0.0
        for weight in (
            loss_cfg.feature_weight,
            loss_cfg.feature_cosine_weight,
            loss_cfg.feature_target_weight or 0.0,
            loss_cfg.feature_target_cosine_weight or 0.0,
        )
    )
    feature_cfg.boundary_mask_soft_mode = (feature_cfg.boundary_mask_soft_mode or "linear").lower()
    feature_cfg.boundary_mask_soft_floor = max(0.0, min(feature_cfg.boundary_mask_soft_floor, 1.0))
    feature_projector: Optional[StudentFeatureProjector] = None
    feature_distiller: Optional[FeatureDistiller] = None
    student_feature_adapter: Optional[FeatureAdapter] = None
    teacher_feature_adapter: Optional[FeatureAdapter] = None
    student_adapter_warned = False
    teacher_adapter_warned = False
    gaussian_cell_features: Optional[torch.Tensor] = None
    gaussian_feature_warning_emitted = False
    gaussian_feature_dim: Optional[int] = None
    gaussian_feature_dim_reported = False
    feature_aux_enabled = False
    feature_aux_loss_mode = (feature_aux_cfg.loss or "patch_cosine").lower()
    feature_aux_supported_losses = {"patch_cosine"}
    if feature_aux_cfg.enabled and feature_pipeline_active:
        if feature_aux_loss_mode in feature_aux_supported_losses:
            feature_aux_enabled = True
        else:
            print(
                f"[feature_aux] Unsupported auxiliary loss '{feature_aux_cfg.loss}'. Skipping auxiliary supervision.",
                flush=True,
            )
    elif feature_aux_cfg.enabled and not feature_pipeline_active:
        print(
            "[feature_aux] Auxiliary student-space loss requested but feature pipeline is disabled; skipping.",
            flush=True,
        )
    feature_aux_source_warned = False
    feature_mask_fraction_missing_warned = False
    if feature_pipeline_active:
        wants_gaussian_features = feature_cfg.teacher_mode.startswith("gaussian") or bool(feature_cfg.teacher_components)

        if wants_gaussian_features:
            try:
                gaussian_teacher = GaussianTeacherFeatures.from_ply(teacher_cfg.checkpoint)
            except (FileNotFoundError, ImportError) as err:
                print(f"[feature_pipeline] Failed to load Gaussian teacher features: {err}")
            else:
                impl = getattr(student_model, "impl", student_model)
                if isinstance(impl, _KiloNeRFStudent):
                    bbox_min_tensor = torch.tensor(data_cfg.bbox_min, dtype=torch.float32)
                    bbox_max_tensor = torch.tensor(data_cfg.bbox_max, dtype=torch.float32)
                    gaussian_cell_features = _build_gaussian_cell_features(
                        gaussian_teacher,
                        impl.grid_resolution,
                        bbox_min_tensor,
                        bbox_max_tensor,
                        mode=feature_cfg.teacher_mode,
                        components=feature_cfg.teacher_components,
                    )
                    gaussian_feature_dim = int(gaussian_cell_features.shape[-1]) if gaussian_cell_features.numel() > 0 else 0
                    if gaussian_feature_dim and not gaussian_feature_dim_reported:
                        print(
                            f"[feature_pipeline] loaded gaussian teacher features with dimension {gaussian_feature_dim}"
                        )
                        gaussian_feature_dim_reported = True
                    if (
                        gaussian_cell_features is not None
                        and gaussian_feature_dim > 0
                        and feature_cfg.teacher_embedding is not None
                    ):
                        try:
                            embedding_impl = build_teacher_embedding(feature_cfg.teacher_embedding, gaussian_feature_dim)
                        except Exception as err:  # pragma: no cover - configuration dependent
                            print(f"[feature_pipeline] Failed to initialise teacher embedding: {err}")
                        else:
                            feature_cfg.teacher_embedding.resolved_input_dim = gaussian_feature_dim
                            transformed = embedding_impl.transform(gaussian_cell_features)
                            gaussian_cell_features = transformed
                            gaussian_feature_dim = int(transformed.shape[-1]) if transformed.numel() > 0 else 0
                            feature_cfg.teacher_embedding.resolved_output_dim = gaussian_feature_dim
                            feature_cfg.resolved_embedding_type = embedding_impl.embedding_type
                            desc = embedding_impl.describe()
                            print(
                                "[feature_pipeline] teacher embedding initialised "
                                f"(type={desc['type']}, input_dim={desc['input_dim']}, output_dim={desc['output_dim']}, standardize={desc.get('standardize')})"
                            )
                else:
                    print(
                        "[feature_pipeline] Gaussian teacher mode currently supports only KiloNeRF students; "
                        "falling back to RGB supervision."
                    )

        projector_output_dim = feature_cfg.projector_output_dim
        teacher_raw_dim: Optional[int] = None
        teacher_adapter_cfg = feature_cfg.teacher_adapter
        adapter_handles_alignment = False
        if teacher_adapter_cfg is not None:
            adapter_target_dim = teacher_adapter_cfg.output_dim
            if adapter_target_dim is None:
                adapter_target_dim = projector_output_dim
            if adapter_target_dim is not None:
                adapter_handles_alignment = True
        if gaussian_feature_dim is not None and gaussian_feature_dim > 0:
            teacher_raw_dim = gaussian_feature_dim
            feature_cfg.resolved_teacher_raw_dim = teacher_raw_dim
            if projector_output_dim != teacher_raw_dim:
                if feature_cfg.allow_dim_mismatch:
                    print(
                        f"[feature_pipeline] projector_output_dim={projector_output_dim} mismatches teacher feature dim {teacher_raw_dim}; "
                        "keeping student dimension due to allow_dim_mismatch."
                    )
                elif adapter_handles_alignment:
                    print(
                        f"[feature_pipeline] projector_output_dim={projector_output_dim} mismatches teacher feature dim {teacher_raw_dim}; "
                        "keeping student dimension because teacher adapter will align the features."
                    )
                else:
                    print(
                        f"[feature_pipeline] projector_output_dim={projector_output_dim} mismatches teacher feature dim {teacher_raw_dim}; "
                        "overriding to match teacher."
                    )
                    projector_output_dim = teacher_raw_dim
        projector_cfg = StudentProjectorConfig(
            input_dim=feature_cfg.projector_input_dim,
            hidden_dim=feature_cfg.projector_hidden_dim,
            output_dim=projector_output_dim,
            activation=feature_cfg.projector_activation,
            use_layer_norm=feature_cfg.projector_use_layer_norm,
            dropout=feature_cfg.projector_dropout,
        )
        feature_cfg.projector_output_dim = projector_output_dim
        feature_projector = StudentFeatureProjector(projector_cfg).to(device)
        feature_distiller = FeatureDistiller(loss_cfg, device=device)

        if overfit_enabled and resolved_overfit_mode in {"projector", "all"}:
            # Ensure feature loss is live so projector-only overfits retain gradients.
            distill_cfg = feature_distiller.cfg
            forced_adjustment = False
            warmup_steps = getattr(distill_cfg, "feature_warmup_steps", 0)
            if warmup_steps and warmup_steps > 0:
                distill_cfg.feature_warmup_steps = 0
                forced_adjustment = True
            base_weight = float(getattr(distill_cfg, "feature_weight", 0.0) or 0.0)
            target_weight = getattr(distill_cfg, "feature_target_weight", None)
            if base_weight <= 0.0 and target_weight is not None and float(target_weight) > 0.0:
                distill_cfg.feature_weight = float(target_weight)
                forced_adjustment = True
            base_cos = float(getattr(distill_cfg, "feature_cosine_weight", 0.0) or 0.0)
            target_cos = getattr(distill_cfg, "feature_target_cosine_weight", None)
            if base_cos <= 0.0 and target_cos is not None and float(target_cos) > 0.0:
                distill_cfg.feature_cosine_weight = float(target_cos)
                forced_adjustment = True
            schedule_mode = str(getattr(distill_cfg, "feature_schedule", "none") or "none").lower()
            if schedule_mode not in {"none", "constant"}:
                distill_cfg.feature_schedule = "none"
                distill_cfg.feature_schedule_duration = 0
                forced_adjustment = True
            if forced_adjustment:
                print("[overfit] Feature supervision forced on for projector overfit diagnostics.")

        student_feature_dim = projector_cfg.output_dim
        feature_cfg.resolved_student_dim = student_feature_dim

        if feature_cfg.student_head is not None:
            head_cfg = feature_cfg.student_head
            if head_cfg.input_dim is None:
                head_cfg.input_dim = student_feature_dim
            if head_cfg.output_dim is None:
                head_cfg.output_dim = teacher_raw_dim or student_feature_dim
            student_feature_adapter = FeatureAdapter(head_cfg).to(device)
            student_feature_dim = int(head_cfg.output_dim)
            feature_cfg.resolved_student_dim = student_feature_dim
            print(
                "[feature_pipeline] student head adapter in/out="
                f"({head_cfg.input_dim}->{head_cfg.output_dim})"
            )

        teacher_feature_dim = teacher_raw_dim
        if feature_cfg.teacher_adapter is not None:
            adapter_cfg = feature_cfg.teacher_adapter
            if adapter_cfg.input_dim is None:
                if teacher_feature_dim is None:
                    raise ValueError(
                        "feature_pipeline.teacher_adapter requires teacher features but none are available"
                    )
                adapter_cfg.input_dim = teacher_feature_dim
            if adapter_cfg.output_dim is None:
                adapter_cfg.output_dim = student_feature_dim
            teacher_feature_adapter = FeatureAdapter(adapter_cfg).to(device)
            teacher_feature_dim = int(adapter_cfg.output_dim)
            print(
                "[feature_pipeline] teacher adapter in/out="
                f"({adapter_cfg.input_dim}->{adapter_cfg.output_dim})"
            )

        comparison_dim = student_feature_dim
        feature_cfg.resolved_comparison_dim = comparison_dim
        if teacher_feature_dim is None:
            teacher_feature_dim = comparison_dim
        feature_cfg.resolved_teacher_dim = teacher_feature_dim

        if teacher_feature_dim != comparison_dim:
            raise ValueError(
                "Teacher and student feature dimensions remain misaligned after adapter configuration: "
                f"teacher={teacher_feature_dim}, student={comparison_dim}"
            )

        if (
            feature_cfg.resolved_teacher_raw_dim is not None
            and feature_cfg.resolved_teacher_raw_dim != feature_cfg.resolved_teacher_dim
        ):
            print(
                "[feature_pipeline] teacher features adapted "
                f"{feature_cfg.resolved_teacher_raw_dim}->{feature_cfg.resolved_teacher_dim}"
            )

        print(
            "[feature_pipeline] comparison feature dim="
            f"{feature_cfg.resolved_comparison_dim}, projector in/out="
            f"({projector_cfg.input_dim}->{projector_cfg.output_dim})"
        )

        policy_exit_code = int(train_cfg.promotion_exit_code or 12)
        _enforce_student_space_policy(
            feature_cfg,
            exit_code=policy_exit_code,
            expected_dim=128,
        )

    mask_controller: Optional[MaskThresholdController] = None
    mask_controller_activation_step: Optional[int] = None
    mask_emergency_fraction = 0.05
    mask_recovery_fraction = 0.15
    if feature_pipeline_active:
        mask_ctrl_cfg = feature_cfg.mask_controller or MaskControllerConfig()
        if mask_ctrl_cfg.enabled:
            configured_threshold = feature_cfg.boundary_mask_threshold
            if configured_threshold is None:
                configured_threshold = 0.40
            if mask_ctrl_cfg.cap_threshold is not None:
                configured_threshold = min(configured_threshold, mask_ctrl_cfg.cap_threshold)
            configured_threshold = max(0.0, configured_threshold)
            if configured_threshold > 0.0:
                initial_threshold = mask_ctrl_cfg.initial_threshold
                if initial_threshold is None:
                    initial_threshold = max(0.06, min(configured_threshold, 0.12))
                else:
                    initial_threshold = max(0.0, min(initial_threshold, configured_threshold))
                explicit_activation = mask_ctrl_cfg.activation_step
                if explicit_activation is not None:
                    mask_activation_step = max(0, int(explicit_activation))
                else:
                    activation_offset = max(0, int(mask_ctrl_cfg.activation_offset))
                    min_activation = max(0, int(mask_ctrl_cfg.min_activation_step))
                    base_step = int(loss_cfg.feature_warmup_steps) + activation_offset
                    mask_activation_step = max(base_step, min_activation)
                ramp_duration = max(0, int(mask_ctrl_cfg.ramp_duration))
                ramp_completion_step = mask_activation_step + ramp_duration
                mask_controller = MaskThresholdController(
                    base_threshold=configured_threshold,
                    base_soft_transition=float(feature_cfg.boundary_mask_soft_transition),
                    schedule=(
                        (0, None),
                        (mask_activation_step, initial_threshold),
                        (ramp_completion_step, configured_threshold),
                    ),
                    min_threshold=max(0.0, float(mask_ctrl_cfg.min_threshold)),
                    relaxation=max(0.0, float(mask_ctrl_cfg.relaxation)),
                    min_fraction=max(0.0, min(float(mask_ctrl_cfg.min_fraction), 1.0)),
                    soft_transition_step=max(0.0, float(mask_ctrl_cfg.soft_transition_step)),
                )
                mask_controller_activation_step = mask_activation_step
                mask_emergency_fraction = max(0.0, min(float(mask_ctrl_cfg.emergency_fraction), 1.0))
                mask_recovery_fraction = max(
                    mask_emergency_fraction,
                    min(float(mask_ctrl_cfg.recovery_fraction), 1.0),
                )
        if mask_controller is None:
            mask_ctrl_cfg = feature_cfg.mask_controller or MaskControllerConfig()
            mask_emergency_fraction = max(0.0, min(float(mask_ctrl_cfg.emergency_fraction), 1.0))
            mask_recovery_fraction = max(
                mask_emergency_fraction,
                min(float(mask_ctrl_cfg.recovery_fraction), 1.0),
            )

    mask_low_fraction_threshold = 0.05
    mask_low_fraction_required_steps = 200
    mask_emergency_hold_steps = 200
    mask_low_fraction_streak = 0
    mask_emergency_active_flag = False
    mask_emergency_release_counter = 0
    mask_emergency_activation_total = 0

    alpha_guard_cfg = train_cfg.alpha_guard or AlphaGuardConfig()
    alpha_guard_enabled = bool(alpha_guard_cfg.enabled)

    def _clamp_penalty_weight(value: float) -> float:
        lower = max(0.0, float(alpha_guard_cfg.weight_floor))
        upper = float(alpha_guard_cfg.weight_cap)
        if upper <= 0.0 or upper < lower:
            upper = max(lower, 0.45)
        return min(max(float(value), lower), upper)

    def _clamp_target_adjustment(value: float) -> float:
        return min(max(float(value), 0.05), 5.0)

    def _clamp_lambda(value: float) -> float:
        lower = max(0.0, float(alpha_guard_cfg.lambda_floor))
        upper = float(alpha_guard_cfg.lambda_cap)
        if upper <= 0.0 or upper < lower:
            upper = lower if lower > 0.0 else max(lower, 1.0)
        return min(max(float(value), lower), upper)

    def _smooth_transition(current: float, target: float, smoothing: float) -> float:
        if not math.isfinite(current):
            current = float(target)
        if not math.isfinite(target):
            return current
        smoothing = max(0.0, min(float(smoothing), 1.0))
        if smoothing <= 0.0:
            return float(target)
        return current + (target - current) * smoothing

    def _apply_rate_limit(current: float, proposed: float, max_delta: float) -> float:
        if not math.isfinite(proposed):
            return float(current)
        if max_delta <= 0.0 or not math.isfinite(max_delta):
            return float(proposed)
        if not math.isfinite(current):
            current = float(proposed)
        delta = float(proposed) - float(current)
        if delta > max_delta:
            return float(current) + float(max_delta)
        if delta < -max_delta:
            return float(current) - float(max_delta)
        return float(proposed)

    opacity_lambda_runtime = float(loss_cfg.opacity_lambda)
    opacity_target_adjustment = 1.0
    alpha_guard_initial_weight_raw = alpha_guard_cfg.initial_weight
    if alpha_guard_initial_weight_raw is None:
        alpha_guard_initial_weight_raw = 0.2
    alpha_guard_initial_weight = float(alpha_guard_initial_weight_raw)
    alpha_penalty_weight = _clamp_penalty_weight(alpha_guard_initial_weight)
    alpha_guard_acc_penalty = 0.0
    alpha_guard_sample_count = 0
    alpha_guard_avg_penalty = 0.0
    alpha_guard_last_update_step = 0
    alpha_guard_window = max(1, int(alpha_guard_cfg.avg_window))
    alpha_guard_penalty_history: Deque[float] = deque()
    alpha_guard_penalty_history_sum = 0.0
    alpha_penalty_weight_target = alpha_penalty_weight
    opacity_target_adjustment_target = opacity_target_adjustment
    opacity_lambda_target = _clamp_lambda(opacity_lambda_runtime)
    opacity_lambda_runtime = opacity_lambda_target
    alpha_guard_smoothing = max(0.0, float(getattr(alpha_guard_cfg, "adjustment_smoothing", 0.0) or 0.0))
    alpha_guard_min_target = max(0.0, float(getattr(alpha_guard_cfg, "min_target_weight", 0.0) or 0.0))
    opacity_warm_start_offset = max(0, int(getattr(loss_cfg, "opacity_target_warm_start_offset", 0) or 0))
    alpha_guard_enforce_steps = int(getattr(alpha_guard_cfg, "warmup_enforce_steps", 0) or 0)
    if alpha_guard_enforce_steps <= 0:
        alpha_guard_enforce_steps = max(int(loss_cfg.opacity_target_warmup_steps or 0) + opacity_warm_start_offset, 0)
    else:
        alpha_guard_enforce_steps = max(alpha_guard_enforce_steps + opacity_warm_start_offset, 0)
    alpha_guard_hysteresis_margin = max(
        0.0,
        float(getattr(alpha_guard_cfg, "hysteresis_margin", 0.0) or 0.0),
    )
    alpha_guard_min_update_samples = max(
        1,
        int(getattr(alpha_guard_cfg, "min_update_samples", 1) or 1),
    )
    alpha_guard_max_lambda_delta = max(
        0.0,
        float(getattr(alpha_guard_cfg, "max_lambda_delta", 0.0) or 0.0),
    )
    alpha_guard_max_adjustment_delta = max(
        0.0,
        float(getattr(alpha_guard_cfg, "max_target_adjustment_delta", 0.0) or 0.0),
    )
    alpha_guard_max_penalty_delta = max(
        0.0,
        float(getattr(alpha_guard_cfg, "max_penalty_weight_delta", 0.0) or 0.0),
    )
    alpha_guard_tighten_streak = 0
    alpha_guard_relax_streak = 0
    alpha_guard_last_direction: Optional[str] = None

    try:
        alpha_quantile_interval_env = os.getenv("KILOGS_ALPHA_QUANTILE_INTERVAL", "0") or "0"
        alpha_quantile_interval_requested = int(alpha_quantile_interval_env)
    except ValueError:
        alpha_quantile_interval_requested = 0
    alpha_quantile_interval = max(1, alpha_quantile_interval_requested or 1)
    try:
        alpha_quantile_sample_fraction = float(os.getenv("KILOGS_ALPHA_QUANTILE_SAMPLE_FRAC", "0.05"))
    except ValueError:
        alpha_quantile_sample_fraction = 0.05
    alpha_quantile_sample_fraction = min(max(alpha_quantile_sample_fraction, 0.0), 1.0)
    try:
        alpha_quantile_min_samples = max(1, int(os.getenv("KILOGS_ALPHA_QUANTILE_MIN", "2048")))
    except ValueError:
        alpha_quantile_min_samples = 2048
    try:
        alpha_quantile_history_window = max(1, int(os.getenv("KILOGS_ALPHA_TAIL_WINDOW", "1000")))
    except ValueError:
        alpha_quantile_history_window = 1000
    alpha_quantile_last_values: Dict[str, torch.Tensor] = {
        "p50": torch.tensor(float("nan"), device=device),
        "p90": torch.tensor(float("nan"), device=device),
        "p99": torch.tensor(float("nan"), device=device),
    }
    alpha_quantile_last_refresh_step: Optional[int] = None
    alpha_quantile_refresh_count = 0
    alpha_quantile_nan_events = 0
    alpha_quantile_history: Deque[Tuple[int, float]] = deque()
    alpha_spread_last = torch.tensor(float("nan"), device=device)
    alpha_tail_slope_last = torch.tensor(float("nan"), device=device)
    if "KILOGS_ALPHA_LEAK_THRESHOLD" not in os.environ:
        raise SystemExit("[alpha] Required env KILOGS_ALPHA_LEAK_THRESHOLD is unset. Export it before launch for reproducible leak diagnostics.")
    if "KILOGS_ALPHA_HALO_THRESHOLD" not in os.environ:
        raise SystemExit("[alpha] Required env KILOGS_ALPHA_HALO_THRESHOLD is unset. Export it before launch for reproducible halo diagnostics.")
    try:
        alpha_leak_threshold = float(os.environ["KILOGS_ALPHA_LEAK_THRESHOLD"])
    except ValueError as err:
        raise SystemExit("[alpha] KILOGS_ALPHA_LEAK_THRESHOLD must parse as float.") from err
    try:
        alpha_halo_threshold = float(os.environ["KILOGS_ALPHA_HALO_THRESHOLD"])
    except ValueError as err:
        raise SystemExit("[alpha] KILOGS_ALPHA_HALO_THRESHOLD must parse as float.") from err
    try:
        alpha_issue_streak_window = max(1, int(os.getenv("KILOGS_ALPHA_ISSUE_STREAK", "8")))
    except ValueError:
        alpha_issue_streak_window = 8
    alpha_leak_streak = 0
    alpha_halo_streak = 0
    alpha_last_issue_code = 0

    mask_prefail_cfg = train_cfg.mask_prefail or MaskPrefailConfig()
    mask_prefail_window = max(4, int(mask_prefail_cfg.window))
    mask_prefail_enabled = bool(mask_prefail_cfg.enabled) and mask_controller is not None
    mask_prefail_history: Deque[Tuple[int, float, float]] = deque(maxlen=mask_prefail_window)
    mask_prefail_active_flag = False
    mask_prefail_trigger_step = -1
    mask_prefail_activation_total = 0
    mask_prefail_drop_rate_recent = float("nan")
    mask_prefail_min_rate_recent = float("nan")
    mask_prefail_last_threshold = float("nan")
    mask_prefail_variance_recent = float("nan")
    mask_prefail_cooldown_steps = max(0, int(mask_prefail_cfg.cooldown_steps))
    mask_prefail_cooldown_until = -1

    last_feature_mask_fraction_value: Optional[float] = None
    last_feature_src_dim_logged: Optional[int] = None
    last_projector_out_dim_logged: Optional[int] = None

    feature_on_steps = 0
    opacity_on_steps = 0

    default_optimize_groups = ("student", "projector", "student_adapter", "teacher_adapter")
    phase_schedule: List[TrainPhaseConfig] = list(train_cfg.phases)
    phase_mask_override: Optional[str] = None
    phase_feature_loss_scale = 1.0
    current_phase_index = -1
    current_phase: Optional[TrainPhaseConfig] = None
    forced_optimize_groups: Optional[Tuple[str, ...]] = None
    forced_mask_override: Optional[str] = None
    if resolved_overfit_mode == "projector":
        forced_optimize_groups = ("projector", "student_adapter", "teacher_adapter")
        forced_mask_override = "disabled"
    elif resolved_overfit_mode == "student":
        forced_optimize_groups = ("student",)
    elif resolved_overfit_mode == "all":
        forced_optimize_groups = default_optimize_groups

    mask_override_codes = {
        "inherit": 0.0,
        "disabled": 1.0,
        "off": 1.0,
        "full": 2.0,
        "ones": 2.0,
    }
    if phase_schedule:
        summary = ", ".join(
            f"{phase.name}:{phase.start_step + 1}-{phase.end_step}"
            for phase in phase_schedule
        )
        print(f"[phase] Loaded schedule: {summary}")

    trainable_parameters: List[torch.nn.Parameter] = list(student_model.parameters())
    if feature_pipeline_active and feature_projector is not None:
        trainable_parameters += list(feature_projector.parameters())
    if student_feature_adapter is not None:
        trainable_parameters += list(student_feature_adapter.parameters())
    if teacher_feature_adapter is not None:
        trainable_parameters += list(teacher_feature_adapter.parameters())
    optimizer = torch.optim.Adam(trainable_parameters, lr=train_cfg.lr)

    opacity_scheduler = OpacityTargetScheduler(loss_cfg)
    start_step = 0

    if resume_checkpoint_path is not None:
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        student_model.load_state_dict(checkpoint["model_state"])
        if feature_pipeline_active and feature_projector is not None and "feature_projector_state" in checkpoint:
            feature_projector.load_state_dict(checkpoint["feature_projector_state"])
        if student_feature_adapter is not None and "student_feature_adapter_state" in checkpoint:
            student_feature_adapter.load_state_dict(checkpoint["student_feature_adapter_state"])
        if teacher_feature_adapter is not None and "teacher_feature_adapter_state" in checkpoint:
            teacher_feature_adapter.load_state_dict(checkpoint["teacher_feature_adapter_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint.get("step", 0))
        scheduler_state = checkpoint.get("opacity_scheduler_state")
        if scheduler_state:
            opacity_scheduler.load_state_dict(scheduler_state)
            last_weight = opacity_scheduler.last_weight
            max_weight = opacity_scheduler.max_weight
            if (
                last_weight is not None
                and max_weight is not None
                and (max_weight + 1e-9) < last_weight
            ):
                raise TrainingAbort(
                    "Checkpoint opacity scheduler state is inconsistent (max_weight decreased).",
                    exit_code=12,
                )
        if opacity_scheduler.last_weight is None and start_step > 0:
            opacity_scheduler.prime_to(start_step)
            print(
                "[resume] opacity scheduler state missing in checkpoint; reconstructed history up to step "
                f"{start_step}."
            )
        if mask_controller is not None:
            mask_state = checkpoint.get("mask_controller_state")
            if mask_state:
                mask_controller.load_state_dict(mask_state)
        emergency_state = checkpoint.get("mask_emergency_state")
        if emergency_state:
            mask_low_fraction_streak = int(emergency_state.get("low_streak", 0) or 0)
            mask_emergency_active_flag = bool(emergency_state.get("active", False))
            mask_emergency_release_counter = int(emergency_state.get("release_counter", 0) or 0)
            mask_emergency_activation_total = int(emergency_state.get("activation_total", 0) or 0)
        cumulative_state = checkpoint.get("effective_activation_steps")
        if isinstance(cumulative_state, dict):
            feature_on_steps = int(cumulative_state.get("feature_on", feature_on_steps) or feature_on_steps)
            opacity_on_steps = int(cumulative_state.get("opacity_on", opacity_on_steps) or opacity_on_steps)
        alpha_guard_state = checkpoint.get("alpha_guard_state")
        if isinstance(alpha_guard_state, dict):
            opacity_lambda_runtime = float(alpha_guard_state.get("lambda_runtime", opacity_lambda_runtime))
            opacity_target_adjustment = _clamp_target_adjustment(
                alpha_guard_state.get("target_adjustment", opacity_target_adjustment)
            )
            alpha_penalty_weight = _clamp_penalty_weight(alpha_guard_state.get("penalty_weight", alpha_penalty_weight))
            alpha_guard_avg_penalty = float(alpha_guard_state.get("avg_penalty", alpha_guard_avg_penalty))
            alpha_guard_last_update_step = int(alpha_guard_state.get("last_update_step", start_step) or start_step)
            direction_raw = alpha_guard_state.get("last_direction")
            if isinstance(direction_raw, str):
                normalized_direction = direction_raw.strip().lower()
                if normalized_direction in {"tighten", "relax"}:
                    alpha_guard_last_direction = normalized_direction
                else:
                    alpha_guard_last_direction = None
            tighten_streak_raw = alpha_guard_state.get("tighten_streak")
            if tighten_streak_raw is not None:
                try:
                    alpha_guard_tighten_streak = max(0, int(tighten_streak_raw))
                except (TypeError, ValueError):
                    alpha_guard_tighten_streak = 0
            relax_streak_raw = alpha_guard_state.get("relax_streak")
            if relax_streak_raw is not None:
                try:
                    alpha_guard_relax_streak = max(0, int(relax_streak_raw))
                except (TypeError, ValueError):
                    alpha_guard_relax_streak = 0
            history_raw = alpha_guard_state.get("history")
            if isinstance(history_raw, (list, tuple)):
                alpha_guard_penalty_history.clear()
                alpha_guard_penalty_history_sum = 0.0
                for item in history_raw[-alpha_guard_window:]:
                    try:
                        value = float(item)
                    except (TypeError, ValueError):
                        continue
                    alpha_guard_penalty_history.append(value)
                    alpha_guard_penalty_history_sum += value
                if alpha_guard_penalty_history:
                    alpha_guard_avg_penalty = alpha_guard_penalty_history_sum / len(alpha_guard_penalty_history)
        mask_prefail_state = checkpoint.get("mask_prefail_state")
        if isinstance(mask_prefail_state, dict):
            history_raw = mask_prefail_state.get("history", [])
            mask_prefail_history.clear()
            for item in history_raw:
                try:
                    step_val = int(item[0])
                    min_val = float(item[1])
                    p5_val = float(item[2])
                except (TypeError, ValueError, IndexError):
                    continue
                mask_prefail_history.append((step_val, min_val, p5_val))
            mask_prefail_active_flag = bool(mask_prefail_state.get("active", mask_prefail_active_flag))
            mask_prefail_trigger_step = int(mask_prefail_state.get("trigger_step", mask_prefail_trigger_step))
            mask_prefail_activation_total = int(mask_prefail_state.get("activation_total", mask_prefail_activation_total))
            mask_prefail_last_threshold = float(mask_prefail_state.get("last_threshold", mask_prefail_last_threshold))
            mask_prefail_drop_rate_recent = float(mask_prefail_state.get("drop_rate", mask_prefail_drop_rate_recent))
            mask_prefail_min_rate_recent = float(mask_prefail_state.get("min_rate", mask_prefail_min_rate_recent))
            mask_prefail_variance_recent = float(mask_prefail_state.get("variance", mask_prefail_variance_recent))
            mask_prefail_cooldown_until = int(mask_prefail_state.get("cooldown_until", mask_prefail_cooldown_until))
        print(f"[resume] Loaded checkpoint '{resume_checkpoint_path}' at step {start_step}")

    if alpha_guard_last_update_step == 0:
        alpha_guard_last_update_step = start_step

    base_lr = float(train_cfg.lr)
    warmup_steps = max(int(train_cfg.lr_warmup_steps), 0)
    schedule_mode = str(train_cfg.lr_schedule or "none").lower()
    milestone_steps = tuple(sorted(train_cfg.lr_schedule_milestones))
    milestone_values = train_cfg.lr_schedule_values
    min_lr_config = train_cfg.lr_schedule_min_lr
    schedule_total = train_cfg.lr_schedule_steps if train_cfg.lr_schedule_steps is not None else train_cfg.max_steps
    schedule_total = max(int(schedule_total), 1)

    def compute_learning_rate(step: int) -> float:
        if step <= 0:
            return 0.0 if warmup_steps > 0 else base_lr

        if warmup_steps > 0 and step < warmup_steps:
            return base_lr * float(step) / float(warmup_steps)

        effective_step = max(step - warmup_steps, 0)

        if schedule_mode in {"none", "constant", "fixed"}:
            return base_lr

        if schedule_mode in {"cos", "cosine", "cosine_decay"}:
            decay_span = max(schedule_total - warmup_steps, 1)
            clipped = min(effective_step, decay_span)
            min_lr = min_lr_config if min_lr_config is not None else base_lr * float(train_cfg.lr_decay_gamma)
            min_lr = max(0.0, float(min_lr))
            cosine = 0.5 * (1.0 + math.cos(math.pi * clipped / decay_span))
            return min_lr + (base_lr - min_lr) * cosine

        if schedule_mode in {"step", "milestone", "piecewise"}:
            lr_value = base_lr
            if milestone_values:
                pairs = sorted(zip(milestone_steps, milestone_values))
                for milestone, value in pairs:
                    if step >= milestone:
                        lr_value = float(value)
            else:
                lr_value = base_lr
                for milestone in milestone_steps:
                    if step >= milestone:
                        lr_value *= float(train_cfg.lr_decay_gamma)
            return max(0.0, lr_value)

        return base_lr

    initial_lr = compute_learning_rate(start_step if start_step > 0 else 0)
    for group in optimizer.param_groups:
        group["lr"] = initial_lr

    quicklook_interval = max(int(logging_cfg.render_preview_interval or 0), 0)
    quicklook_error_logged = False

    def save_checkpoint(step: int) -> Path:
        ckpt_path = checkpoint_dir / f"step_{step:06d}.pth"
        state = {
            "step": step,
            "model_state": student_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        if feature_pipeline_active and feature_projector is not None:
            state["feature_projector_state"] = feature_projector.state_dict()
        if student_feature_adapter is not None:
            state["student_feature_adapter_state"] = student_feature_adapter.state_dict()
        if teacher_feature_adapter is not None:
            state["teacher_feature_adapter_state"] = teacher_feature_adapter.state_dict()
        state["opacity_scheduler_state"] = opacity_scheduler.state_dict()
        if mask_controller is not None:
            state["mask_controller_state"] = mask_controller.state_dict()
        state["mask_emergency_state"] = {
            "low_streak": mask_low_fraction_streak,
            "active": mask_emergency_active_flag,
            "release_counter": mask_emergency_release_counter,
            "activation_total": mask_emergency_activation_total,
        }
        state["effective_activation_steps"] = {
            "feature_on": feature_on_steps,
            "opacity_on": opacity_on_steps,
        }
        state["alpha_guard_state"] = {
            "lambda_runtime": float(opacity_lambda_runtime),
            "target_adjustment": float(opacity_target_adjustment),
            "penalty_weight": float(alpha_penalty_weight),
            "avg_penalty": float(alpha_guard_avg_penalty),
            "last_update_step": int(alpha_guard_last_update_step),
            "history": [float(value) for value in alpha_guard_penalty_history],
            "last_direction": str(alpha_guard_last_direction or ""),
            "tighten_streak": int(alpha_guard_tighten_streak),
            "relax_streak": int(alpha_guard_relax_streak),
        }
        state["mask_prefail_state"] = {
            "history": [
                (int(step), float(min_val), float(p5_val))
                for step, min_val, p5_val in mask_prefail_history
            ],
            "active": bool(mask_prefail_active_flag),
            "trigger_step": int(mask_prefail_trigger_step),
            "activation_total": int(mask_prefail_activation_total),
            "last_threshold": float(mask_prefail_last_threshold),
            "drop_rate": float(mask_prefail_drop_rate_recent),
            "min_rate": float(mask_prefail_min_rate_recent),
            "variance": float(mask_prefail_variance_recent),
            "cooldown_until": int(mask_prefail_cooldown_until),
        }
        torch.save(state, ckpt_path)
        return ckpt_path

    moving_average_window = max(int(train_cfg.effective_weight_avg_window), 1)
    moving_average_specs: Tuple[Tuple[str, str, str], ...] = (
        (
            "opacity_target_weight_effective",
            f"opacity_target_weight_effective_avg_{moving_average_window}",
            "opacity/target_weight_effective_avg",
        ),
        (
            "feature_weight_effective",
            f"feature_weight_effective_avg_{moving_average_window}",
            "feature/weight_effective_avg",
        ),
        (
            "feature_cos_weight_effective",
            f"feature_cos_weight_effective_avg_{moving_average_window}",
            "feature/cos_weight_effective_avg",
        ),
    )
    moving_average_buffers = {
        spec[0]: deque(maxlen=moving_average_window) for spec in moving_average_specs
    }
    gate_resolution = _resolve_promotion_gates(train_cfg)
    if gate_resolution.trimmed:
        print(
            f"[gate] Dropped promotion gates beyond max_steps={train_cfg.max_steps}: {gate_resolution.trimmed}",
            flush=True,
        )
    if gate_resolution.auto_filled and gate_resolution.gates:
        print(
            f"[gate] promotion_gates unspecified; defaulting to {gate_resolution.gates}",
            flush=True,
        )
    train_cfg.promotion_gates = gate_resolution.gates

    if not feature_pipeline_active:
        if train_cfg.promotion_gates:
            print(
                "[gate] feature pipeline disabled; ignoring promotion_gates",
                flush=True,
            )
        train_cfg.promotion_gates = tuple()

    promotion_active = bool(train_cfg.promotion_gates)
    if promotion_active and train_cfg.promotion_min_mask_fraction <= 0.0:
        raise TrainingAbort(
            "promotion_min_mask_fraction must be positive when promotion gates are active.",
            exit_code=train_cfg.promotion_exit_code,
        )
    if promotion_active:
        if train_cfg.promotion_min_feature_ratio <= 0.0 or train_cfg.promotion_min_opacity_ratio <= 0.0:
            raise TrainingAbort(
                "promotion feature/opacity ratios must be positive when promotion gates are active.",
                exit_code=train_cfg.promotion_exit_code,
            )

    if train_cfg.promotion_feature_dim is not None:
        promotion_expected_feature_dim = int(train_cfg.promotion_feature_dim)
    else:
        if feature_cfg.projector_input_dim is None:
            raise TrainingAbort(
                "feature.projector_input_dim must be specified when promotion gates are active.",
                exit_code=train_cfg.promotion_exit_code,
            )
        promotion_expected_feature_dim = int(feature_cfg.projector_input_dim)

    if train_cfg.promotion_projector_in_dim is not None:
        promotion_expected_projector_in = int(train_cfg.promotion_projector_in_dim)
    else:
        if feature_cfg.projector_output_dim is None:
            raise TrainingAbort(
                "feature.projector_output_dim must be specified when promotion gates are active.",
                exit_code=train_cfg.promotion_exit_code,
            )
        promotion_expected_projector_in = int(feature_cfg.projector_output_dim)

    promotion_pending: Set[int] = {
        gate
        for gate in train_cfg.promotion_gates
        if gate > start_step and (train_cfg.max_steps <= 0 or gate <= train_cfg.max_steps)
    }
    promotion_min_mask_fraction = max(0.0, float(train_cfg.promotion_min_mask_fraction))
    promotion_min_feature_scale = max(0.0, float(train_cfg.promotion_min_feature_scale))

    known_metric_keys: List[str] = []
    logged_steps_tb: Set[int] = set()

    def write_metrics_csv(
        step: int,
        total_loss_val: float,
        raw_metrics: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        nonlocal known_metric_keys
        csv_path = logging_cfg.csv
        metrics_floats: Dict[str, float] = {}
        for key, value in raw_metrics.items():
            if isinstance(value, torch.Tensor):
                try:
                    metrics_floats[key] = float(value.detach().cpu().item())
                except (AttributeError, ValueError):
                    metrics_floats[key] = float("nan")
            else:
                try:
                    metrics_floats[key] = float(value)
                except (TypeError, ValueError):
                    metrics_floats[key] = float("nan")

        metric_keys_sorted = sorted(metrics_floats.keys())
        prefix = ["timestamp", "step", "total"]
        suffix = ["_eor_checksum"]
        existing_header: Optional[List[str]] = None
        existing_row_maps: List[Dict[str, str]] = []

        if csv_path.exists():
            try:
                with csv_path.open("r", newline="", encoding="utf-8") as existing_fp:
                    csv_reader = csv.reader(existing_fp)
                    existing_header = next(csv_reader, None)
                    if existing_header is not None:
                        if existing_header[: len(prefix)] == prefix and existing_header[-1:] == suffix:
                            if not known_metric_keys:
                                known_metric_keys = list(existing_header[len(prefix) : -1])
                            for row in csv_reader:
                                if not row:
                                    continue
                                row_map: Dict[str, str] = {}
                                for idx, column in enumerate(existing_header):
                                    if idx < len(row):
                                        row_map[column] = row[idx]
                                existing_row_maps.append(row_map)
                        else:
                            backup_path = csv_path.with_suffix(csv_path.suffix + ".legacy")
                            try:
                                csv_path.rename(backup_path)
                            except OSError as err:
                                print(f"[logging] Failed to rotate metrics CSV: {err}")
                            else:
                                print(f"[logging] Rotated metrics CSV header mismatch → '{backup_path.name}'")
                            existing_header = None
                            existing_row_maps = []
            except Exception as err:
                print(f"[logging] Failed to read metrics CSV: {err}")
                existing_header = None
                existing_row_maps = []

        if not known_metric_keys:
            known_metric_keys = list(metric_keys_sorted)
        else:
            new_keys = [key for key in metric_keys_sorted if key not in known_metric_keys]
            if new_keys:
                known_metric_keys = list(known_metric_keys) + new_keys

        headers = prefix + list(known_metric_keys) + suffix

        def _serialize_row(row_map: Dict[str, str]) -> List[str]:
            row_values_existing: List[str] = [
                row_map.get("timestamp", ""),
                row_map.get("step", ""),
                row_map.get("total", "nan"),
            ]
            for key in known_metric_keys:
                row_values_existing.append(row_map.get(key, "nan"))
            payload_existing = "|".join(row_values_existing)
            checksum_existing = format(zlib.crc32(payload_existing.encode("utf-8")) & 0xFFFFFFFF, "08x")
            row_values_existing.append(checksum_existing)
            return row_values_existing

        existing_row_maps = [row for row in existing_row_maps if row.get("step") != str(step)]
        rows_to_write: List[List[str]] = [_serialize_row(row_map) for row_map in existing_row_maps]

        row_values: List[str] = [
            datetime.utcnow().isoformat(),
            str(step),
            f"{total_loss_val:.6f}" if math.isfinite(total_loss_val) else "nan",
        ]
        for key in known_metric_keys:
            value = metrics_floats.get(key, float("nan"))
            row_values.append(f"{value:.6f}" if math.isfinite(value) else "nan")
        payload = "|".join(row_values)
        checksum = format(zlib.crc32(payload.encode("utf-8")) & 0xFFFFFFFF, "08x")
        row_values.append(checksum)
        rows_to_write.append(row_values)

        try:
            with csv_path.open("w", newline="", encoding="utf-8") as fp:
                csv_writer = csv.writer(fp)
                csv_writer.writerow(headers)
                csv_writer.writerows(rows_to_write)
        except Exception as err:
            print(f"[logging] Failed to write metrics CSV: {err}")

        debug_log(
            "csv_append step=%s total=%s keys=%s" % (
                step,
                total_loss_val,
                len(known_metric_keys),
            )
        )

        return metrics_floats

    def emit_tensorboard_scalars(
        step: int,
        total_loss_val: float,
        metrics_floats: Dict[str, float],
        *,
        full: bool,
        base: bool = True,
    ) -> None:
        if writer is None:
            return

        debug_log(
            "tb_emit_invoke step=%s full=%s" % (
                step,
                full,
            )
        )

        metrics_source = metrics_floats or {}

        walltime_override: Optional[float] = None
        if tensorboard_use_walltime:
            elapsed_monotonic = max(time.perf_counter() - train_start_time, 0.0)
            if tensorboard_use_elapsed:
                walltime_override = train_start_wall + elapsed_monotonic
            else:
                walltime_override = time.time()

        def _coerce_numeric(value: Optional[float]) -> Optional[float]:
            if value is None:
                return None
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(numeric):
                return None
            return numeric

        def _add_scalar(tag: str, value: Optional[float]) -> None:
            numeric = _coerce_numeric(value)
            if numeric is None:
                return
            try:
                if walltime_override is not None:
                    writer.add_scalar(tag, numeric, step, walltime=walltime_override)
                else:
                    writer.add_scalar(tag, numeric, step)
            except Exception as err:
                debug_log(f"tb_scalar_error tag={tag} step={step} err={err}")

        if base:
            _add_scalar("_/heartbeat", 1.0)
            _add_scalar("loss/total", total_loss_val)

        if full:
            for loss_key in ("color", "opacity", "depth", "feature_recon", "feature_cosine"):
                _add_scalar(f"loss/{loss_key}", metrics_source.get(loss_key))

            scalar_map = {
                "alpha_guard_penalty": "opacity/alpha_guard_penalty",
                "opacity_target_weight_effective": "opacity/target_weight_effective",
            }
            for metric_key, tb_tag in scalar_map.items():
                _add_scalar(tb_tag, metrics_source.get(metric_key))

        try:
            writer.flush()
        except Exception as err:
            debug_log(f"tb_flush_error step={step} err={err}")

    def snapshot_recent_metrics(rows: int = 1000) -> None:
        csv_path = logging_cfg.csv
        if not csv_path.exists():
            return
        try:
            with csv_path.open("r", encoding="utf-8") as src:
                lines = src.readlines()
        except Exception as err:
            progress.write(f"[logging] Failed to read metrics CSV for bailout snapshot: {err}")
            return
        if not lines:
            return
        header = lines[:1]
        body = lines[1:]
        if not body:
            content = header
        else:
            rows_to_keep = body[-min(len(body), max(1, rows)) :]
            content = header + rows_to_keep
        bailout_path = csv_path.with_suffix(csv_path.suffix + ".bailout")
        try:
            with bailout_path.open("w", encoding="utf-8") as dst:
                dst.writelines(content)
        except Exception as err:
            progress.write(f"[logging] Failed to write bailout metrics CSV: {err}")

    def _update_alpha_quantiles(
        opacity_tensor: torch.Tensor,
        *,
        step: int,
        force: bool = False,
    ) -> None:
        nonlocal alpha_quantile_last_refresh_step
        nonlocal alpha_quantile_last_values
        nonlocal alpha_quantile_refresh_count
        nonlocal alpha_quantile_nan_events
        nonlocal alpha_spread_last
        nonlocal alpha_tail_slope_last
        nonlocal alpha_quantile_history

        if opacity_tensor is None or opacity_tensor.numel() == 0:
            return

        should_refresh = force
        if not should_refresh:
            if alpha_quantile_last_refresh_step is None:
                should_refresh = True
            else:
                should_refresh = (step - alpha_quantile_last_refresh_step) >= alpha_quantile_interval
        if not should_refresh and step <= start_step + 5:
            should_refresh = True
        if not should_refresh and step >= train_cfg.max_steps:
            should_refresh = True
        if not should_refresh:
            return

        success = False
        with torch.no_grad():
            try:
                flat = opacity_tensor.detach().reshape(-1).to(dtype=torch.float32)
                sample_count = flat.numel()
                if sample_count == 0:
                    raise RuntimeError("alpha tensor empty")
                if alpha_quantile_sample_fraction > 0.0 and sample_count > alpha_quantile_min_samples:
                    target = int(math.ceil(sample_count * alpha_quantile_sample_fraction))
                    target = max(alpha_quantile_min_samples, target)
                    target = min(target, sample_count)
                    if target < sample_count:
                        if target * 2 >= sample_count:
                            indices = torch.randperm(sample_count, device=flat.device, dtype=torch.int64)[:target]
                        else:
                            indices = torch.randint(0, sample_count, (target,), device=flat.device, dtype=torch.int64)
                        flat = flat.index_select(0, indices)
                        sample_count = target
                quantile_points = torch.tensor([0.50, 0.90, 0.99], device=flat.device, dtype=flat.dtype)
                quantiles = torch.quantile(flat, quantile_points)
                quantiles = torch.sort(torch.clamp(quantiles, min=0.0, max=1.0))[0]
                if (
                    hasattr(torch, "distributed")
                    and torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    reduced = quantiles.to(dtype=torch.float64)
                    torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
                    world_size = torch.distributed.get_world_size()
                    if world_size > 0:
                        reduced = reduced / float(world_size)
                    quantiles = reduced.to(dtype=opacity_tensor.dtype)
                else:
                    quantiles = quantiles.to(dtype=opacity_tensor.dtype)
                p50, p90, p99 = quantiles
                p90 = torch.maximum(p90, p50)
                p99 = torch.maximum(p99, p90)
                for value in (p50, p90, p99):
                    if not torch.isfinite(value):
                        raise RuntimeError("non-finite alpha quantile")
                alpha_quantile_last_values = {
                    "p50": p50.detach(),
                    "p90": p90.detach(),
                    "p99": p99.detach(),
                }
                alpha_quantile_last_refresh_step = step
                alpha_quantile_refresh_count += 1
                alpha_spread_last = torch.clamp(p99 - p50, min=0.0).detach()
                alpha_quantile_history.append((step, float(p99.detach().cpu().item())))
                while alpha_quantile_history and (step - alpha_quantile_history[0][0]) > alpha_quantile_history_window:
                    alpha_quantile_history.popleft()
                if len(alpha_quantile_history) >= 2:
                    hist_start_step, hist_start_val = alpha_quantile_history[0]
                    hist_end_step, hist_end_val = alpha_quantile_history[-1]
                    denom = max(hist_end_step - hist_start_step, 1)
                    slope_value = (hist_end_val - hist_start_val) / float(denom)
                    alpha_tail_slope_last = torch.tensor(
                        slope_value,
                        device=device,
                        dtype=opacity_tensor.dtype,
                    )
                else:
                    alpha_tail_slope_last = torch.tensor(float("nan"), device=device, dtype=opacity_tensor.dtype)
                success = True
            except Exception as err:
                debug_log(f"alpha_quantile_refresh_failed step={step} err={err}")
        if not success:
            alpha_quantile_nan_events += 1
        else:
            alpha_quantile_last_values = {
                key: value.to(device=device, dtype=opacity_tensor.dtype)
                for key, value in alpha_quantile_last_values.items()
            }
            alpha_spread_last = alpha_spread_last.to(device=device, dtype=opacity_tensor.dtype)
            alpha_tail_slope_last = alpha_tail_slope_last.to(device=device, dtype=opacity_tensor.dtype)

    def update_moving_averages(metrics: Dict[str, torch.Tensor]) -> None:
        for base_key, avg_key, _ in moving_average_specs:
            buffer = moving_average_buffers.get(base_key)
            if buffer is None:
                continue
            value_tensor = metrics.get(base_key)
            if value_tensor is not None:
                try:
                    value = float(value_tensor.detach().cpu().item())
                except (AttributeError, ValueError):
                    value = float("nan")
                if math.isfinite(value):
                    buffer.append(value)
            if buffer and len(buffer) > 0:
                avg_value = sum(buffer) / len(buffer)
                metrics[avg_key] = torch.tensor(avg_value, device=device)
            else:
                metrics.setdefault(avg_key, torch.tensor(float("nan"), device=device))

    def compute_mask_prefail_rates(history: Deque[Tuple[int, float, float]]) -> Tuple[Optional[float], Optional[float]]:
        if len(history) < 2:
            return None, None
        start_step, start_min, start_p5 = history[0]
        end_step, end_min, end_p5 = history[-1]
        span = max(end_step - start_step, 1)
        p5_rate = (end_p5 - start_p5) / span
        min_rate = (end_min - start_min) / span
        return p5_rate, min_rate

    def enforce_promotion_gate(step: int, metrics: Dict[str, torch.Tensor]) -> None:
        if step not in promotion_pending:
            return

        def _metric_float(key: str) -> float:
            tensor = metrics.get(key)
            if tensor is None:
                return float("nan")
            try:
                return float(tensor.detach().cpu().item())
            except (AttributeError, ValueError):
                return float("nan")

        mask_fraction = _metric_float("feature_mask_fraction")
        feature_dim_val = _metric_float("feature_src_dim")
        phase_feature_scale_val = _metric_float("phase_feature_scale")
        projector_out_dim_val = _metric_float("projector_out_dim")
        feature_on_steps_val = _metric_float("feature_on_steps")
        opacity_on_steps_val = _metric_float("opacity_on_steps")
        feature_terminal_val = _metric_float("feature_schedule_terminal")
        opacity_terminal_val = _metric_float("opacity_schedule_terminal")
        step_denominator = max(step, 1)

        def _normalise_activation_ratio(raw_value: float) -> float:
            if not math.isfinite(raw_value):
                return float("nan")
            # Recent logging switched to store activation ratios directly in [0, 1].
            # Historical checkpoints used absolute counts, so fall back to the old behaviour
            # when the value is clearly larger than a single step.
            if raw_value <= 1.0 + 1e-6:
                return raw_value
            if step_denominator <= 0:
                return float("nan")
            return raw_value / step_denominator

        feature_on_ratio = _normalise_activation_ratio(feature_on_steps_val)
        opacity_on_ratio = _normalise_activation_ratio(opacity_on_steps_val)

        failure_reason: Optional[str] = None
        def _terminal_flag(value: float) -> bool:
            return math.isfinite(value) and value >= 0.5

        if not math.isfinite(mask_fraction) or mask_fraction < promotion_min_mask_fraction:
            failure_reason = (
                f"feature_mask_fraction={mask_fraction:.4f} below gate requirement {promotion_min_mask_fraction:.2f}"
            )
        elif promotion_expected_feature_dim is not None and (
            not math.isfinite(feature_dim_val)
            or int(round(feature_dim_val)) != int(promotion_expected_feature_dim)
        ):
            failure_reason = (
                f"feature_src_dim={feature_dim_val:.0f} differs from expected {promotion_expected_feature_dim}"
            )
        elif promotion_expected_projector_in is not None and (
            not math.isfinite(projector_out_dim_val)
            or int(round(projector_out_dim_val)) != int(promotion_expected_projector_in)
        ):
            failure_reason = (
                f"projector_out_dim={projector_out_dim_val:.0f} differs from expected {promotion_expected_projector_in}"
            )
        elif not math.isfinite(phase_feature_scale_val) or phase_feature_scale_val <= promotion_min_feature_scale:
            failure_reason = (
                f"phase_feature_scale={phase_feature_scale_val:.4f} ≤ minimum {promotion_min_feature_scale:.4f}"
            )
        elif train_cfg.promotion_min_feature_ratio > 0.0 and (
            not math.isfinite(feature_on_ratio)
            or feature_on_ratio < float(train_cfg.promotion_min_feature_ratio)
        ):
            failure_reason = (
                f"feature_on_ratio={feature_on_ratio:.3f} below {train_cfg.promotion_min_feature_ratio:.3f}"
            )
        elif train_cfg.promotion_min_opacity_ratio > 0.0 and (
            not math.isfinite(opacity_on_ratio)
            or opacity_on_ratio < float(train_cfg.promotion_min_opacity_ratio)
        ):
            failure_reason = (
                f"opacity_on_ratio={opacity_on_ratio:.3f} below {train_cfg.promotion_min_opacity_ratio:.3f}"
            )
        elif (
            feature_pipeline_active
            and train_cfg.promotion_require_feature_schedule_terminal
            and not _terminal_flag(feature_terminal_val)
        ):
            failure_reason = (
                f"feature_schedule_terminal={feature_terminal_val:.2f} indicates scheduler incomplete"
            )
        elif (
            train_cfg.promotion_require_opacity_schedule_terminal
            and not _terminal_flag(opacity_terminal_val)
        ):
            failure_reason = (
                f"opacity_schedule_terminal={opacity_terminal_val:.2f} indicates scheduler incomplete"
            )

        if failure_reason is not None:
            snapshot_recent_metrics()
            promotion_pending.discard(step)
            raise PromotionGateFailure(
                f"Promotion gate failed at step {step}: {failure_reason}",
                exit_code=train_cfg.promotion_exit_code,
            )

        promotion_pending.discard(step)
        feature_sched_status = "Y" if _terminal_flag(feature_terminal_val) else "N"
        opacity_sched_status = "Y" if _terminal_flag(opacity_terminal_val) else "N"
        progress.write(
            "[gate] Promotion gate passed at step "
            f"{step} (mask={mask_fraction:.3f}, feature_dim={int(round(feature_dim_val))}, "
            f"projector_dim={int(round(projector_out_dim_val))}, feature_scale={phase_feature_scale_val:.3f}, "
            f"feature_ratio={feature_on_ratio:.2f}, opacity_ratio={opacity_on_ratio:.2f}, "
            f"feature_sched={feature_sched_status}, opacity_sched={opacity_sched_status})"
        )

    bbox_min = torch.tensor(data_cfg.bbox_min, dtype=torch.float32, device=device)
    bbox_max = torch.tensor(data_cfg.bbox_max, dtype=torch.float32, device=device)
    bbox_extent = bbox_max - bbox_min
    background = dataset.background.to(device)

    ray_perturb_enabled = bool(data_cfg.perturb) and not overfit_enabled

    overfit_static_batch: Optional[Dict[str, torch.Tensor]] = None
    if overfit_enabled:
        try:
            overfit_static_batch_raw = dataset.sample_random_rays(data_cfg.batch_size, device)
        except RuntimeError as err:
            raise RuntimeError(f"Failed to obtain overfit batch: {err}") from err
        overfit_static_batch = {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in overfit_static_batch_raw.items()
        }
        print(
            f"[overfit] Using fixed batch of {data_cfg.batch_size} rays in {resolved_overfit_mode} mode "
            f"for {train_cfg.max_steps} steps (lr={train_cfg.lr:.3g})."
        )

    global_step = start_step
    train_start_time = time.perf_counter()
    train_start_wall = time.time()
    progress_label = experiment.progress_desc or experiment.name or "response"
    if overfit_enabled and resolved_overfit_mode is not None:
        progress_label = f"{progress_label} overfit[{resolved_overfit_mode}]"

    def _compute_progress_total() -> int:
        if train_cfg.max_steps <= 0:
            # fallback to at least one step when max_steps is unset
            return max(global_step, 1)
        remaining = max(train_cfg.max_steps - start_step, 0)
        return remaining if remaining > 0 else train_cfg.max_steps

    def _make_progress(disable: bool) -> "tqdm":
        completed = max(global_step - start_step, 0)
        total_steps = _compute_progress_total()
        initial = min(completed, total_steps)
        return create_progress(
            total=total_steps,
            desc=f"({progress_label})",
            unit="step",
            leave=True,
            initial=initial,
            disable=disable,
        )

    progress_initially_hidden = False
    progress = _make_progress(progress_initially_hidden)

    try:
        step_time_ema_alpha = float(os.getenv("KILOGS_PROGRESS_EMA_ALPHA", "0.1"))
    except ValueError:
        step_time_ema_alpha = 0.1
    step_time_ema_alpha = min(max(step_time_ema_alpha, 0.01), 1.0)
    step_time_ema: Optional[float] = None

    health_enabled = os.getenv("KILOGS_HEALTHCHECK", "0") == "1"
    try:
        health_fail_steps = max(1, int(os.getenv("KILOGS_HEALTHCHECK_MAX_STEPS", "5000") or 5000))
    except ValueError:
        health_fail_steps = 5000
    try:
        health_warn_steps = max(1, int(os.getenv("KILOGS_HEALTHCHECK_WARN_STEPS", "2000") or 2000))
    except ValueError:
        health_warn_steps = 2000
    health_warn_steps = min(health_warn_steps, health_fail_steps)
    try:
        health_median_window = max(1, int(os.getenv("KILOGS_HEALTHCHECK_MEDIAN_WINDOW", "400") or 400))
    except ValueError:
        health_median_window = 400
    health_record_limit = max(health_warn_steps, health_fail_steps)
    health_failfast = os.getenv("KILOGS_HEALTHCHECK_FAILFAST", "1") != "0"
    health_records: List[Tuple[int, Dict[str, float]]] = []
    health_warn_emitted = False
    health_fail_evaluated = False
    health_monitor_finished = False

    def _fmt_float(value: float, digits: int = 3) -> str:
        if not math.isfinite(value):
            return "nan"
        fmt = f"{{:.{digits}f}}"
        return fmt.format(value)

    def evaluate_health_snapshot(stage: str) -> Tuple[bool, str, List[str]]:
        if not health_records:
            return True, "no records", []

        # Use a rolling median over the most recent records so transient spikes do not trigger false alarms.
        window = health_records[-min(len(health_records), health_median_window) :]

        def collect(key: str) -> List[float]:
            values: List[float] = []
            for _, snap in window:
                raw = snap.get(key)
                if raw is None:
                    continue
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            return values

        def median_or_nan(values: List[float]) -> float:
            if not values:
                return float("nan")
            try:
                return float(statistics.median(values))
            except statistics.StatisticsError:
                return float("nan")

        mask_med = median_or_nan(collect("feature_mask_fraction"))
        feature_src_vals = collect("feature_src_dim")
        projector_vals = collect("projector_out_dim")
        opacity_weight_med = median_or_nan(collect("opacity_target_weight_effective"))
        alpha_hi_med = median_or_nan(collect("alpha_fraction_ge95"))
        alpha_lo_med = median_or_nan(collect("alpha_fraction_le05"))

        issues: List[str] = []
        if not math.isfinite(mask_med) or mask_med < 0.25:
            issues.append(f"mask_med={_fmt_float(mask_med)} < 0.25")

        def _dims_ok(values: List[float], expected: int, label: str) -> bool:
            if not values:
                issues.append(f"{label} missing")
                return False
            rounded = {int(round(v)) for v in values}
            if rounded != {expected}:
                issues.append(f"{label}={sorted(rounded)} != {expected}")
                return False
            return True

        _dims_ok(feature_src_vals, 128, "feature_src_dim")
        _dims_ok(projector_vals, 52, "projector_out_dim")

        if not math.isfinite(opacity_weight_med) or opacity_weight_med <= 0.0:
            issues.append("opacity_target_weight_effective ≤ 0")
        if math.isfinite(alpha_hi_med) and alpha_hi_med > 0.80:
            issues.append(f"alpha_fraction_ge95 median {_fmt_float(alpha_hi_med)} > 0.80")
        if math.isfinite(alpha_lo_med) and alpha_lo_med > 0.80:
            issues.append(f"alpha_fraction_le05 median {_fmt_float(alpha_lo_med)} > 0.80")

        summary = (
            f"mask_med={_fmt_float(mask_med)}; src_med={_fmt_float(median_or_nan(feature_src_vals), 0)}; "
            f"proj_med={_fmt_float(median_or_nan(projector_vals), 0)}; alpha_hi_med={_fmt_float(alpha_hi_med)}; "
            f"alpha_lo_med={_fmt_float(alpha_lo_med)}; opacity_w_med={_fmt_float(opacity_weight_med)}"
        )
        ok = not issues
        return ok, summary, issues

    guard_notice_interval = max(0.0, float(getattr(train_cfg, "input_guard_notice_interval", 0.0)))
    last_input_guard_notice = time.perf_counter()

    projector_only_override_active = False
    projector_only_override_cache: Dict[str, object] = {}

    def _activate_projector_only_override() -> None:
        nonlocal projector_only_override_active, projector_only_override_cache
        if projector_only_override_active:
            return
        if not feature_pipeline_active or feature_distiller is None:
            return
        cfg = feature_distiller.cfg
        projector_only_override_cache = {
            "feature_warmup_steps": getattr(cfg, "feature_warmup_steps", 0),
            "feature_weight": float(getattr(cfg, "feature_weight", 0.0) or 0.0),
            "feature_target_weight": getattr(cfg, "feature_target_weight", None),
            "feature_cosine_weight": float(getattr(cfg, "feature_cosine_weight", 0.0) or 0.0),
            "feature_target_cosine_weight": getattr(cfg, "feature_target_cosine_weight", None),
            "feature_schedule": getattr(cfg, "feature_schedule", "none"),
            "feature_schedule_duration": getattr(cfg, "feature_schedule_duration", 0),
        }

        def _positive_or_fallback(value: Optional[float], fallback: float) -> float:
            if value is None:
                return fallback
            numeric = float(value)
            return numeric if numeric > 0.0 else fallback

        cfg.feature_warmup_steps = 0
        forced_weight = _positive_or_fallback(
            cfg.feature_target_weight,
            _positive_or_fallback(cfg.feature_weight, 0.05),
        )
        cfg.feature_weight = forced_weight
        cfg.feature_target_weight = forced_weight
        forced_cos = _positive_or_fallback(
            cfg.feature_target_cosine_weight,
            _positive_or_fallback(cfg.feature_cosine_weight, 0.0),
        )
        cfg.feature_cosine_weight = forced_cos
        cfg.feature_target_cosine_weight = forced_cos
        cfg.feature_schedule = "none"
        cfg.feature_schedule_duration = 0
        projector_only_override_active = True
        if progress is not None:
            progress.write("[phase] Projector-only warmup forcing feature loss on.")

    def _deactivate_projector_only_override() -> None:
        nonlocal projector_only_override_active, projector_only_override_cache
        if not projector_only_override_active:
            return
        if not feature_pipeline_active or feature_distiller is None:
            projector_only_override_active = False
            projector_only_override_cache = {}
            return
        cfg = feature_distiller.cfg
        warmup = projector_only_override_cache.get("feature_warmup_steps")
        if warmup is not None:
            cfg.feature_warmup_steps = int(warmup)
        if "feature_weight" in projector_only_override_cache:
            cfg.feature_weight = float(projector_only_override_cache["feature_weight"])
        if "feature_target_weight" in projector_only_override_cache:
            cfg.feature_target_weight = projector_only_override_cache["feature_target_weight"]
        if "feature_cosine_weight" in projector_only_override_cache:
            cfg.feature_cosine_weight = float(projector_only_override_cache["feature_cosine_weight"])
        if "feature_target_cosine_weight" in projector_only_override_cache:
            cfg.feature_target_cosine_weight = projector_only_override_cache["feature_target_cosine_weight"]
        if "feature_schedule" in projector_only_override_cache:
            cfg.feature_schedule = projector_only_override_cache["feature_schedule"]
        if "feature_schedule_duration" in projector_only_override_cache:
            cfg.feature_schedule_duration = int(projector_only_override_cache["feature_schedule_duration"])
        projector_only_override_active = False
        projector_only_override_cache = {}
        if progress is not None:
            progress.write("[phase] Projector-only warmup feature loss restored to config values.")

    def get_phase_for_step(step: int) -> Tuple[int, Optional[TrainPhaseConfig]]:
        for idx, phase_cfg in enumerate(phase_schedule):
            if phase_cfg.contains(step):
                return idx, phase_cfg
        return -1, None

    def apply_phase(new_index: int, phase_cfg: Optional[TrainPhaseConfig]) -> None:
        nonlocal phase_mask_override, phase_feature_loss_scale, current_phase_index, current_phase
        if new_index == current_phase_index and phase_cfg is current_phase:
            return
        active_groups = set(default_optimize_groups)
        mask_override_local: Optional[str] = None
        feature_scale_local = 1.0
        if phase_cfg is not None:
            active_groups = set(phase_cfg.optimize) if phase_cfg.optimize else set(default_optimize_groups)
            mask_override_local = phase_cfg.mask_override
            feature_scale_local = max(float(phase_cfg.feature_weight_scale), 0.0)
        if forced_optimize_groups is not None:
            active_groups = set(forced_optimize_groups)
        if "all" in active_groups:
            active_groups = set(default_optimize_groups)
        module_groups = {
            "student": (student_model,),
            "projector": (feature_projector,),
            "student_adapter": (student_feature_adapter,),
            "teacher_adapter": (teacher_feature_adapter,),
        }
        for name, modules in module_groups.items():
            enable = name in active_groups
            for module in modules:
                if module is None:
                    continue
                for param in module.parameters():
                    param.requires_grad = enable
        projector_only_phase = ("projector" in active_groups) and ("student" not in active_groups)
        if projector_only_phase:
            _activate_projector_only_override()
        else:
            _deactivate_projector_only_override()
        phase_mask_override = forced_mask_override or mask_override_local
        phase_feature_loss_scale = feature_scale_local
        current_phase_index = new_index
        current_phase = phase_cfg
        phase_name = phase_cfg.name if phase_cfg is not None else "default"
        if phase_cfg is not None:
            window = f"{phase_cfg.start_step + 1}-{phase_cfg.end_step}"
        else:
            window = f"{start_step + 1}-{train_cfg.max_steps}"
        progress.write(
            f"[phase] Enter {phase_name} [{window}] optimize={sorted(active_groups)} "
            f"mask_override={phase_mask_override or 'inherit'} feature_scale={phase_feature_loss_scale:.3f}"
        )

    def update_phase_for_step(step: int) -> None:
        phase_index, phase_cfg = get_phase_for_step(step)
        apply_phase(phase_index, phase_cfg)

    manual_progress_active = False
    manual_progress_last_len = 0
    suppress_manual_progress = progress_initially_hidden
    manual_progress_width = 40
    manual_progress_stream = getattr(progress, "_kilogs_tty_only_stream", None)
    if manual_progress_stream is None:
        manual_progress_stream = _resolve_tty_stream()
    if manual_progress_stream is None:
        manual_progress_stream = sys.stdout

    def _write_manual(text: str) -> None:
        target = manual_progress_stream
        try:
            target.write(text)
            target.flush()
        except Exception:
            try:
                sys.stdout.write(text)
                sys.stdout.flush()
            except Exception:
                pass

    def emit_manual_progress(step: int) -> None:
        nonlocal manual_progress_active, manual_progress_last_len
        completion = 0.0 if train_cfg.max_steps <= 0 else step / float(train_cfg.max_steps)
        completion = max(0.0, min(completion, 1.0))
        bar_length = manual_progress_width
        filled = int(round(bar_length * completion))
        filled = min(max(filled, 0), bar_length)
        bar_chars = "█" * filled + "░" * (bar_length - filled)
        elapsed_seconds = time.perf_counter() - train_start_time
        steps_completed = max(step - start_step, 1)
        average_step_time = elapsed_seconds / steps_completed if steps_completed > 0 else 0.0
        remaining_steps = max(train_cfg.max_steps - step, 0)
        eta_seconds = int(max(average_step_time * remaining_steps, 0.0))
        eta_text = str(timedelta(seconds=eta_seconds))
        elapsed_text = str(timedelta(seconds=int(elapsed_seconds)))
        line = (
            f"({progress_label}) [{step}/{train_cfg.max_steps}]: "
            f"{completion * 100:3.0f}%|{bar_chars}| {step}/{train_cfg.max_steps} [{elapsed_text}<{eta_text}]"
        )
        _write_manual("\r" + line)
        manual_progress_last_len = len(line)
        manual_progress_active = True

    original_progress_write = progress.write

    def progress_write_wrapper(message: str, *, file=None, end="\n", nolock=False):
        if getattr(progress, "disable", False) and manual_progress_active:
            clear_line = "\r" + " " * manual_progress_last_len + "\r"
            _write_manual(clear_line)
        result = original_progress_write(message, file=file, end=end, nolock=nolock)
        if getattr(progress, "disable", False) and manual_progress_active:
            emit_manual_progress(global_step)
        return result

    progress.write = progress_write_wrapper  # type: ignore[assignment]

    student_feature_missing_warned = False
    mask_fraction_emergency_warned = False
    first_forward_logged = False

    if global_step < train_cfg.max_steps:
        update_phase_for_step(max(global_step + 1, 1))

    if global_step >= train_cfg.max_steps:
        progress.write(
            f"[resume] Checkpoint step {global_step} ≥ max_steps ({train_cfg.max_steps}); skipping training loop."
        )
        progress.close()
        if writer is not None:
            writer.flush()
            writer.close()
        return

    raw_log_interval = os.getenv("KILOGS_LOG_INTERVAL")
    if raw_log_interval:
        try:
            scalar_log_interval = max(1, int(raw_log_interval))
        except ValueError:
            scalar_log_interval = max(1, logging_cfg.scalar_interval)
    else:
        scalar_log_interval = max(1, logging_cfg.scalar_interval)
    debug_log(
        "log_interval_init=%s start_step=%s" % (
            scalar_log_interval,
            start_step,
        )
    )

    if alpha_quantile_interval_requested > 0:
        alpha_quantile_interval = max(1, alpha_quantile_interval_requested)
    else:
        alpha_quantile_interval = max(1, scalar_log_interval)

    abort_exc: Optional[TrainingAbort] = None
    try:
        while global_step < train_cfg.max_steps:
            debug_log(f"loop_enter prev_step={global_step}")
            step_start_time = time.perf_counter()
            if guard_notice_interval > 0.0 and step_start_time - last_input_guard_notice >= guard_notice_interval:
                progress.write("[guard] Monitoring only; please avoid typing into this terminal.")
                last_input_guard_notice = step_start_time
            global_step += 1
            debug_log(f"loop_step global_step={global_step}")
            if global_step == start_step + 1:
                progress.write(f"[debug_start] start_step={start_step}")
            update_phase_for_step(global_step)
            if train_cfg.max_steps > 0:
                target_n = max(global_step - start_step, 0)
            else:
                target_n = max(global_step, 0)
            try:
                current_n = int(getattr(progress, "n", 0))
            except Exception:
                current_n = 0
            delta = max(target_n - current_n, 0)
            if delta and not getattr(progress, "disable", False):
                progress.update(delta)
            if getattr(progress, "disable", False) and not suppress_manual_progress:
                emit_manual_progress(global_step)
            if not getattr(progress, "disable", False):
                try:
                    progress.refresh()
                except Exception:  # pragma: no cover - tqdm safety
                    pass
            current_lr = compute_learning_rate(global_step)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            if overfit_static_batch is not None:
                batch = {
                    key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in overfit_static_batch.items()
                }
            else:
                try:
                    batch = dataset.sample_random_rays(data_cfg.batch_size, device)
                except RuntimeError as err:
                    progress.write(f"Ray sampling failed at step {global_step}: {err}")
                    continue

            rays_o = batch["rays_o"]
            rays_d = batch["rays_d"]
            teacher_rgb = batch["teacher_rgb"]
            teacher_alpha = batch["teacher_alpha"]
            teacher_depth = batch.get("teacher_depth")
            teacher_depth_valid_mask = batch.get("teacher_depth_valid_mask")
            near_selected = batch["near"]
            far_selected = batch["far"]

            pts, z_vals = sample_along_rays(
                rays_o=rays_o,
                rays_d=rays_d,
                near=near_selected,
                far=far_selected,
                num_samples=data_cfg.samples_per_ray,
                perturb=ray_perturb_enabled,
            )
            pts_norm = (pts - bbox_min) / bbox_extent
            pts_norm = pts_norm.clamp(0.0, 1.0)
            pts_flat = pts_norm.view(-1, 3)
            ray_dirs = F.normalize(rays_d, dim=-1, eps=1e-6)
            ray_dirs_flat = (
                ray_dirs[:, None, :]
                .expand(-1, data_cfg.samples_per_ray, -1)
                .reshape(-1, 3)
            )

            if not first_forward_logged:
                progress.write("[init] Entering first KiloNeRF forward pass (may take several minutes).")
            student_rgb_samples, student_sigma_samples = student_model(pts_flat, ray_dirs_flat)
            if not first_forward_logged:
                progress.write("[init] First KiloNeRF forward pass finished; continuing with training loop.")
                if progress_initially_hidden and getattr(progress, "disable", False):
                    progress.close()
                    progress = _make_progress(False)
                    suppress_manual_progress = False
                    progress_initially_hidden = False
                first_forward_logged = True
            student_rgb_samples = student_rgb_samples.view(-1, data_cfg.samples_per_ray, 3)
            student_sigma_samples = student_sigma_samples.view(-1, data_cfg.samples_per_ray)

            deltas = z_vals[:, 1:] - z_vals[:, :-1]
            delta_last = torch.full((deltas.shape[0], 1), 1e10, device=device)
            deltas = torch.cat([deltas, delta_last], dim=-1)

            alpha = 1.0 - torch.exp(-student_sigma_samples * deltas)
            transmittance = _exclusive_cumprod_last(1.0 - alpha + 1e-10)
            weights = alpha * transmittance

            rgb_map = torch.sum(weights[..., None] * student_rgb_samples, dim=-2)
            opacity_map = weights.sum(dim=-1, keepdim=True)
            rgb_map = rgb_map + (1.0 - opacity_map) * background
            depth_map = torch.sum(weights * z_vals, dim=-1, keepdim=True)

            student_rgb = rgb_map
            student_sigma = opacity_map

            depth_mask = None
            depth_valid_fraction: Optional[torch.Tensor] = None
            depth_stat_tensors: Dict[str, torch.Tensor] = {}
            combined_valid_tensor: Optional[torch.Tensor] = None
            if (
                teacher_depth is not None
                and loss_cfg.depth_weight > 0.0
            ):
                mask_components: List[torch.Tensor] = []
                if teacher_depth_valid_mask is not None:
                    mask_components.append(teacher_depth_valid_mask.to(device).float())
                if loss_cfg.depth_alpha_threshold > 0.0:
                    mask_components.append((teacher_alpha > loss_cfg.depth_alpha_threshold).float())
                if mask_components:
                    depth_mask = mask_components[0]
                    for extra_mask in mask_components[1:]:
                        depth_mask = depth_mask * extra_mask

                combined_valid = torch.isfinite(teacher_depth)
                combined_valid = combined_valid & (teacher_depth > 0.0)
                if depth_mask is not None:
                    combined_valid = combined_valid & (depth_mask > 0.0)
                combined_valid_tensor = combined_valid

                if combined_valid.numel() > 0:
                    depth_valid_fraction = combined_valid.float().mean()
                if combined_valid.any():
                    depth_values = teacher_depth[combined_valid]
                    depth_stat_tensors = {
                        "teacher_depth_min": depth_values.min(),
                        "teacher_depth_max": depth_values.max(),
                        "teacher_depth_mean": depth_values.mean(),
                    }
                else:
                    nan_tensor = torch.tensor(float("nan"), device=device)
                    depth_stat_tensors = {
                        "teacher_depth_min": nan_tensor,
                        "teacher_depth_max": nan_tensor,
                        "teacher_depth_mean": nan_tensor,
                    }

            student_depth_for_loss = depth_map
            teacher_depth_for_loss = teacher_depth
            depth_range_tensor: Optional[torch.Tensor] = None
            if (
                teacher_depth is not None
                and loss_cfg.depth_weight > 0.0
            ):
                near_expanded = near_selected.view(-1, 1)
                far_expanded = far_selected.view(-1, 1)
                depth_range = (far_expanded - near_expanded).clamp_min(1e-5)
                depth_range_tensor = depth_range

                student_depth_for_loss = torch.clamp(
                    (depth_map - near_expanded) / depth_range,
                    min=0.0,
                    max=1.0,
                )
                teacher_depth_for_loss = torch.clamp(
                    (teacher_depth - near_expanded) / depth_range,
                    min=0.0,
                    max=1.0,
                )

            base_opacity_target_weight = opacity_scheduler.compute(global_step)
            if base_opacity_target_weight is None:
                base_opacity_target_weight = 0.0
            if alpha_guard_min_target > 0.0:
                if global_step <= alpha_guard_enforce_steps:
                    base_opacity_target_weight = max(base_opacity_target_weight, alpha_guard_min_target)
            opacity_target_weight_base = base_opacity_target_weight
            opacity_target_weight_effective = base_opacity_target_weight * opacity_target_adjustment
            if alpha_guard_min_target > 0.0:
                opacity_target_weight_effective = max(opacity_target_weight_effective, alpha_guard_min_target)

            total_loss, loss_dict = compute_losses(
                student_rgb,
                student_sigma,
                teacher_rgb,
                teacher_alpha,
                loss_cfg,
                student_depth=student_depth_for_loss,
                teacher_depth=teacher_depth_for_loss,
                depth_mask=depth_mask,
                background_rgb=background,
                opacity_target_weight_override=opacity_target_weight_effective,
                opacity_lambda_override=opacity_lambda_runtime,
            )

            alpha_mean_tensor = student_sigma.mean()
            alpha_fraction_ge95 = (student_sigma >= 0.95).float().mean()
            alpha_fraction_le05 = (student_sigma <= 0.05).float().mean()
            alpha_fraction_mid = torch.clamp(1.0 - alpha_fraction_ge95 - alpha_fraction_le05, 0.0, 1.0)
            alpha_band_core = (
                torch.nn.functional.relu(alpha_mean_tensor - 0.45)
                + torch.nn.functional.relu(0.35 - alpha_mean_tensor)
            )
            alpha_band_component = alpha_band_core * float(alpha_guard_cfg.band_weight)
            alpha_hi_excess = torch.nn.functional.relu(alpha_fraction_ge95 - alpha_guard_cfg.penalty_hi)
            alpha_hi_component = alpha_hi_excess * float(alpha_guard_cfg.fraction_hi_weight)
            alpha_lo_excess = torch.nn.functional.relu(alpha_fraction_le05 - alpha_guard_cfg.penalty_lo)
            alpha_lo_component = alpha_lo_excess * float(alpha_guard_cfg.fraction_lo_weight)
            alpha_penalty_core = alpha_band_component + alpha_hi_component + alpha_lo_component
            alpha_guard_penalty_tensor = alpha_penalty_core * alpha_penalty_weight

            total_loss = total_loss + alpha_guard_penalty_tensor
            loss_dict["alpha_guard_penalty"] = alpha_guard_penalty_tensor
            loss_dict["alpha_band_penalty"] = alpha_band_component
            loss_dict["alpha_penalty_core"] = alpha_penalty_core
            loss_dict["alpha_penalty_weight"] = torch.tensor(alpha_penalty_weight, device=device)
            debug_log("loss_ready step=%s" % (global_step,))
            penalty_scalar = float(alpha_penalty_core.detach().cpu().item())
            if len(alpha_guard_penalty_history) >= alpha_guard_window:
                removed_penalty = alpha_guard_penalty_history.popleft()
                alpha_guard_penalty_history_sum -= removed_penalty
            alpha_guard_penalty_history.append(penalty_scalar)
            alpha_guard_penalty_history_sum += penalty_scalar
            if alpha_guard_penalty_history:
                alpha_guard_avg_penalty = alpha_guard_penalty_history_sum / len(alpha_guard_penalty_history)
            else:
                alpha_guard_avg_penalty = penalty_scalar
            if alpha_guard_enabled:
                alpha_guard_acc_penalty += penalty_scalar
                alpha_guard_sample_count += 1
                interval = max(1, int(alpha_guard_cfg.check_interval))
                if (global_step - alpha_guard_last_update_step) >= interval:
                    avg_penalty = alpha_guard_avg_penalty
                    if not math.isfinite(avg_penalty):
                        avg_penalty = 0.0
                    alpha_guard_avg_penalty = avg_penalty
                    tighten_threshold = float(alpha_guard_cfg.penalty_hi)
                    relax_threshold = float(alpha_guard_cfg.penalty_lo)
                    if alpha_guard_last_direction == "relax":
                        tighten_threshold += alpha_guard_hysteresis_margin
                    if alpha_guard_last_direction == "tighten":
                        relax_threshold = max(0.0, relax_threshold - alpha_guard_hysteresis_margin)

                    action: Optional[str] = None
                    if avg_penalty > tighten_threshold:
                        alpha_guard_tighten_streak += 1
                        alpha_guard_relax_streak = 0
                        if alpha_guard_tighten_streak >= alpha_guard_min_update_samples:
                            action = "tighten"
                    elif avg_penalty < relax_threshold:
                        alpha_guard_relax_streak += 1
                        alpha_guard_tighten_streak = 0
                        if alpha_guard_relax_streak >= alpha_guard_min_update_samples:
                            action = "relax"
                    else:
                        alpha_guard_tighten_streak = 0
                        alpha_guard_relax_streak = 0

                    if action == "tighten":
                        proposed_lambda_target = max(
                            alpha_guard_cfg.lambda_floor,
                            opacity_lambda_target * float(alpha_guard_cfg.tighten_rate),
                        )
                        proposed_target_adjustment = _clamp_target_adjustment(
                            opacity_target_adjustment_target * float(alpha_guard_cfg.relax_rate)
                        )
                        proposed_penalty_weight = _clamp_penalty_weight(
                            alpha_penalty_weight_target * float(alpha_guard_cfg.relax_rate)
                        )
                        opacity_lambda_target = _clamp_lambda(
                            _apply_rate_limit(
                                opacity_lambda_target,
                                proposed_lambda_target,
                                alpha_guard_max_lambda_delta,
                            )
                        )
                        opacity_target_adjustment_target = _clamp_target_adjustment(
                            _apply_rate_limit(
                                opacity_target_adjustment_target,
                                proposed_target_adjustment,
                                alpha_guard_max_adjustment_delta,
                            )
                        )
                        alpha_penalty_weight_target = _clamp_penalty_weight(
                            _apply_rate_limit(
                                alpha_penalty_weight_target,
                                proposed_penalty_weight,
                                alpha_guard_max_penalty_delta,
                            )
                        )
                        alpha_guard_last_direction = "tighten"
                        alpha_guard_tighten_streak = 0
                        alpha_guard_relax_streak = 0
                    elif action == "relax":
                        proposed_lambda_target = min(
                            alpha_guard_cfg.lambda_cap,
                            opacity_lambda_target * float(alpha_guard_cfg.relax_rate),
                        )
                        proposed_target_adjustment = _clamp_target_adjustment(
                            opacity_target_adjustment_target * float(alpha_guard_cfg.tighten_rate)
                        )
                        proposed_penalty_weight = _clamp_penalty_weight(
                            alpha_penalty_weight_target * float(alpha_guard_cfg.tighten_rate)
                        )
                        opacity_lambda_target = _clamp_lambda(
                            _apply_rate_limit(
                                opacity_lambda_target,
                                proposed_lambda_target,
                                alpha_guard_max_lambda_delta,
                            )
                        )
                        opacity_target_adjustment_target = _clamp_target_adjustment(
                            _apply_rate_limit(
                                opacity_target_adjustment_target,
                                proposed_target_adjustment,
                                alpha_guard_max_adjustment_delta,
                            )
                        )
                        alpha_penalty_weight_target = _clamp_penalty_weight(
                            _apply_rate_limit(
                                alpha_penalty_weight_target,
                                proposed_penalty_weight,
                                alpha_guard_max_penalty_delta,
                            )
                        )
                        alpha_guard_last_direction = "relax"
                        alpha_guard_tighten_streak = 0
                        alpha_guard_relax_streak = 0
                    alpha_guard_acc_penalty = 0.0
                    alpha_guard_sample_count = 0
                    alpha_guard_last_update_step = global_step

                opacity_lambda_runtime = _clamp_lambda(
                    _smooth_transition(opacity_lambda_runtime, opacity_lambda_target, alpha_guard_smoothing)
                )
                opacity_target_adjustment = _clamp_target_adjustment(
                    _smooth_transition(
                        opacity_target_adjustment,
                        opacity_target_adjustment_target,
                        alpha_guard_smoothing,
                    )
                )
                alpha_penalty_weight = _clamp_penalty_weight(
                    _smooth_transition(alpha_penalty_weight, alpha_penalty_weight_target, alpha_guard_smoothing)
                )

            feature_breakdown = None
            log_feature_mask_fraction: Optional[torch.Tensor] = None
            feature_src_dim_value: Optional[int] = None
            projector_out_dim_value: Optional[int] = None
            feature_aux_loss_unscaled: Optional[torch.Tensor] = None
            feature_aux_weight_value: float = 0.0
            feature_aux_feature_dim_value: Optional[int] = None
            if global_step <= start_step + 5:
                debug_log(
                    "log_feature_gate step=%s active=%s" % (
                        global_step,
                        feature_pipeline_active,
                    )
                )
            if (
                feature_pipeline_active
                and feature_projector is not None
                and feature_distiller is not None
                and feature_distiller.enabled
            ):
                student_pre, _ = extract_student_features(
                    student_model,
                    source=feature_cfg.student_feature_source,
                    activation=feature_cfg.student_feature_activation,
                    dim=feature_cfg.student_feature_dim,
                )
                if student_pre is None:
                    if not student_feature_missing_warned:
                        progress.write(
                            "[feature_pipeline] student features unavailable for configured source; "
                            "feature loss will be skipped until activations become available."
                        )
                        student_feature_missing_warned = True
                elif student_feature_missing_warned:
                    progress.write("[feature_pipeline] student features detected again; resuming feature losses when enabled.")
                    student_feature_missing_warned = False
                feature_mask_fraction_value: Optional[float] = None
                mask_threshold_for_log: Optional[float] = None
                prefail_min_candidate: Optional[float] = None
                prefail_p5_candidate: Optional[float] = None
                prefail_variance_candidate: Optional[float] = None
    
                if student_pre is not None and student_pre.numel() != 0:
                    try:
                        student_pre = student_pre.view(
                            student_rgb_samples.shape[0],
                            data_cfg.samples_per_ray,
                            student_pre.shape[-1],
                        )
                        expected_student_dim = getattr(feature_projector.cfg, "input_dim", None)
                        if expected_student_dim is not None and student_pre.shape[-1] != expected_student_dim:
                            raise RuntimeError(
                                "student penultimate feature dimensionality mismatch: "
                                f"expected {expected_student_dim}, got {student_pre.shape[-1]} at step {global_step}"
                            )
                    except RuntimeError:
                        student_pre = None
    
                    if student_pre is not None:
                        feature_src_dim_value = int(student_pre.shape[-1])
                        last_feature_src_dim_logged = feature_src_dim_value
                        projector_out_dim_value = int(feature_projector.cfg.output_dim)
                        last_projector_out_dim_logged = projector_out_dim_value
                        teacher_feature_tensor: Optional[torch.Tensor] = None
                        student_projected: Optional[torch.Tensor] = None
                        feature_mask_weights: Optional[torch.Tensor] = None
                        feature_mask_fraction_tensor: Optional[torch.Tensor] = None
                        feature_mask_weight_mean_tensor: Optional[torch.Tensor] = None
                        feature_mask_weight_min_tensor: Optional[torch.Tensor] = None
                        feature_mask_weight_max_tensor: Optional[torch.Tensor] = None
                        skip_mask_updates = False
                        mask_override_mode = (phase_mask_override or "inherit").lower()
                        student_aux_base_features: Optional[torch.Tensor] = None
    
                        gaussian_mode = feature_cfg.teacher_mode
                        gaussian_enabled = (
                            gaussian_cell_features is not None
                            and (gaussian_mode.startswith("gaussian") or bool(feature_cfg.teacher_components))
                        )
    
                        if gaussian_enabled:
                            impl = getattr(student_model, "impl", student_model)
                            cell_indices = getattr(impl, "last_linear_indices", None)
                            expected_samples = student_pre.numel() // student_pre.shape[-1]
                            if (
                                cell_indices is not None
                                and cell_indices.numel() == expected_samples
                            ):
                                gaussian_features_device = gaussian_cell_features.to(
                                    device=device,
                                    dtype=student_pre.dtype,
                                )
    
                                cell_indices = cell_indices.to(device)
                                cells_per_ray = cell_indices.view(
                                    student_rgb_samples.shape[0],
                                    data_cfg.samples_per_ray,
                                )
    
                                primary_sample = torch.argmax(weights, dim=-1)
                                primary_cells = cells_per_ray.gather(
                                    -1, primary_sample.unsqueeze(-1)
                                ).squeeze(-1)
    
                                same_cell_mask = cells_per_ray == primary_cells.unsqueeze(-1)
                                primary_weights = weights * same_cell_mask
                                primary_weight_sum_raw = torch.sum(primary_weights, dim=-1)
                                primary_weight_sum = primary_weight_sum_raw.unsqueeze(-1).clamp_min(1e-6)
                                primary_weights_norm = primary_weights / primary_weight_sum
    
                                zero_weight_mask = primary_weight_sum_raw <= 1e-6
                                if torch.any(zero_weight_mask):
                                    same_cell_float = same_cell_mask.float()
                                    same_cell_count = same_cell_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
                                    uniform_weights = same_cell_float / same_cell_count
                                    primary_weights_norm[zero_weight_mask] = uniform_weights[zero_weight_mask]
                                    primary_weight_sum_raw = primary_weight_sum_raw.masked_fill(
                                        zero_weight_mask, 1.0
                                    )
    
                                primary_weight_sum_raw = primary_weight_sum_raw.clamp_min(1e-2)
    
                                if mask_prefail_enabled:
                                    try:
                                        prefail_min_candidate = float(
                                            primary_weight_sum_raw.min().detach().cpu().item()
                                        )
                                    except (RuntimeError, ValueError):
                                        prefail_min_candidate = None
                                    prefail_p5_tensor: Optional[torch.Tensor] = None
                                    try:
                                        prefail_p5_tensor = torch.quantile(primary_weight_sum_raw, 0.05)
                                    except RuntimeError:
                                        flattened = primary_weight_sum_raw.view(-1)
                                        if flattened.numel() > 0:
                                            k_index = max(int(math.floor(0.05 * (flattened.numel() - 1))), 0)
                                            sorted_vals, _ = torch.sort(flattened)
                                            prefail_p5_tensor = sorted_vals[k_index : k_index + 1].mean()
                                    if prefail_p5_tensor is not None:
                                        try:
                                            prefail_p5_candidate = float(prefail_p5_tensor.detach().cpu().item())
                                        except (RuntimeError, ValueError):
                                            prefail_p5_candidate = None
                                    if primary_weight_sum_raw.numel() > 1:
                                        try:
                                            variance_tensor = torch.var(primary_weight_sum_raw, unbiased=False)
                                        except RuntimeError:
                                            variance_tensor = None
                                    else:
                                        variance_tensor = torch.zeros((), device=primary_weight_sum_raw.device)
                                    if variance_tensor is not None:
                                        try:
                                            prefail_variance_candidate = float(variance_tensor.detach().cpu().item())
                                        except (RuntimeError, ValueError):
                                            prefail_variance_candidate = None
    
                                feature_mask_weight_mean_tensor = primary_weight_sum_raw.mean()
                                feature_mask_weight_min_tensor = primary_weight_sum_raw.min()
                                feature_mask_weight_max_tensor = primary_weight_sum_raw.max()
    
                                gathered = gaussian_features_device.index_select(0, cell_indices)
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
    
                                student_aux_base_features = student_cell_features
    
                                student_projected = feature_projector(student_cell_features)
                                threshold_source = mask_controller.current_threshold(global_step) if mask_controller else feature_cfg.boundary_mask_threshold
                                if mask_override_mode in {"disabled", "off", "full"}:
                                    feature_mask_weights = torch.ones_like(primary_weight_sum_raw, device=student_projected.device)
                                    feature_mask_fraction_tensor = torch.tensor(1.0, device=student_projected.device)
                                    mask_threshold_for_log = float("nan")
                                    skip_mask_updates = True
                                elif threshold_source is not None:
                                    boundary_threshold = float(threshold_source)
                                    soft_transition = max(
                                        mask_controller.current_soft_transition() if mask_controller else float(feature_cfg.boundary_mask_soft_transition),
                                        0.0,
                                    )
                                    soft_mode = feature_cfg.boundary_mask_soft_mode
                                    floor = feature_cfg.boundary_mask_soft_floor
                                    if soft_mode == "sigmoid":
                                        scale = max(soft_transition, 1e-6)
                                        logits = (primary_weight_sum_raw - boundary_threshold) / scale
                                        weights = torch.sigmoid(logits)
                                        if floor > 0.0:
                                            weights = weights * (1.0 - floor) + floor
                                        feature_mask_weights = weights
                                    elif soft_mode in {"smooth", "smoothstep"} and soft_transition > 0.0:
                                        lower = boundary_threshold - soft_transition
                                        upper = boundary_threshold + soft_transition
                                        denom = max(upper - lower, 1e-6)
                                        t = ((primary_weight_sum_raw - lower) / denom).clamp(0.0, 1.0)
                                        weights = t * t * (3.0 - 2.0 * t)
                                        if floor > 0.0:
                                            weights = weights * (1.0 - floor) + floor
                                        feature_mask_weights = weights
                                    elif soft_transition > 0.0:
                                        lower = boundary_threshold - soft_transition
                                        transition = max(soft_transition, 1e-6)
                                        weights = (primary_weight_sum_raw - lower) / transition
                                        weights = torch.clamp(weights, 0.0, 1.0)
                                        if floor > 0.0:
                                            weights = weights * (1.0 - floor) + floor
                                        feature_mask_weights = weights
                                    else:
                                        weights = (primary_weight_sum_raw >= boundary_threshold).float()
                                        if floor > 0.0:
                                            weights = weights * (1.0 - floor) + floor
                                        feature_mask_weights = weights
                                    feature_mask_weights = feature_mask_weights.to(student_projected.device)
                                    feature_mask_weights = torch.clamp(feature_mask_weights, 0.0, 1.0)
                                    feature_mask_fraction_tensor = feature_mask_weights.mean()
                                    mask_threshold_for_log = boundary_threshold
                                else:
                                    feature_mask_weights = None
                                    feature_mask_fraction_tensor = None
                            else:
                                gaussian_enabled = False
                                if not gaussian_feature_warning_emitted:
                                    print(
                                        "[feature_pipeline] Unable to align Gaussian features with student cells; "
                                        "falling back to RGB supervision for this step."
                                    )
                                    gaussian_feature_warning_emitted = True
    
                        if not gaussian_enabled:
                            weighted_features = torch.sum(weights[..., None] * student_pre, dim=-2)
                            student_aux_base_features = weighted_features
                            student_projected = feature_projector(weighted_features)
                            if student_projected.shape[-1] == teacher_rgb.shape[-1]:
                                teacher_feature_tensor = teacher_rgb
                            else:
                                teacher_feature_tensor = None
                                feature_mask_weights = None
                                feature_mask_fraction_tensor = None
    
                        if not gaussian_enabled and mask_override_mode in {"disabled", "off", "full"}:
                            skip_mask_updates = True
                            mask_threshold_for_log = float("nan")
                            feature_mask_fraction_tensor = torch.tensor(
                                1.0,
                                device=student_projected.device if student_projected is not None else device,
                            )
    
                        if student_projected is not None and student_feature_adapter is not None:
                            expected_in = student_feature_adapter.cfg.input_dim
                            if student_projected.shape[-1] != expected_in:
                                if not student_adapter_warned:
                                    print(
                                        "[feature_pipeline] student adapter input mismatch: "
                                        f"expected {expected_in}, got {student_projected.shape[-1]}; skipping adaptation."
                                    )
                                    student_adapter_warned = True
                            else:
                                student_projected = student_feature_adapter(student_projected)
    
                        if teacher_feature_tensor is not None and teacher_feature_adapter is not None:
                            expected_teacher_in = teacher_feature_adapter.cfg.input_dim
                            if teacher_feature_tensor.shape[-1] != expected_teacher_in:
                                if not teacher_adapter_warned:
                                    print(
                                        "[feature_pipeline] teacher adapter input mismatch: "
                                        f"expected {expected_teacher_in}, got {teacher_feature_tensor.shape[-1]}; skipping adaptation."
                                    )
                                    teacher_adapter_warned = True
                            else:
                                teacher_feature_tensor = teacher_feature_adapter(teacher_feature_tensor)
    
                        if feature_aux_enabled:
                            aux_weight = _compute_aux_weight(feature_aux_cfg, global_step)
                            feature_aux_weight_value = float(aux_weight)
                            if aux_weight > 0.0:
                                aux_tensor: Optional[torch.Tensor] = None
                                aux_source_key = (feature_aux_cfg.source or "penultimate_post").lower()
    
                                if aux_source_key in {
                                    "projector",
                                    "projected",
                                    "projector_out",
                                    "student_projected",
                                    "student_head",
                                    "adapter",
                                }:
                                    aux_tensor = student_projected
                                elif aux_source_key in {
                                    "penultimate",
                                    "penultimate_post",
                                    "penultimate_pre",
                                    "hidden",
                                    "hidden_post",
                                    "hidden_pre",
                                    "student_pre",
                                    "student_hidden",
                                    "cell",
                                    "cells",
                                    "primary",
                                }:
                                    aux_tensor = student_aux_base_features
                                if (
                                    aux_tensor is None
                                    and aux_source_key
                                    not in {
                                        "projector",
                                        "projected",
                                        "projector_out",
                                        "student_projected",
                                        "student_head",
                                        "adapter",
                                    }
                                ):
                                    aux_raw, _ = extract_student_features(
                                        student_model,
                                        source=feature_aux_cfg.source,
                                    )
                                    if aux_raw is not None:
                                        if (
                                            aux_raw.dim() == 2
                                            and aux_raw.shape[0]
                                            == student_rgb_samples.shape[0] * data_cfg.samples_per_ray
                                        ):
                                            aux_tensor = aux_raw.view(
                                                student_rgb_samples.shape[0],
                                                data_cfg.samples_per_ray,
                                                aux_raw.shape[-1],
                                            )
                                            aux_tensor = torch.sum(weights[..., None] * aux_tensor, dim=-2)
                                        elif aux_raw.dim() == 2 and aux_raw.shape[0] == student_rgb_samples.shape[0]:
                                            aux_tensor = aux_raw
                                        else:
                                            aux_tensor = aux_raw
    
                                if aux_tensor is None or aux_tensor.numel() == 0:
                                    if not feature_aux_source_warned:
                                        print(
                                            f"[feature_aux] Unable to resolve auxiliary features for source '{feature_aux_cfg.source}'; skipping loss until available.",
                                            flush=True,
                                        )
                                        feature_aux_source_warned = True
                                else:
                                    if aux_tensor.dim() >= 3:
                                        aux_tensor = aux_tensor.reshape(aux_tensor.shape[0], -1)
                                    if aux_tensor.dim() != 2:
                                        if not feature_aux_source_warned:
                                            print(
                                                f"[feature_aux] Expected 2D tensor for auxiliary loss, got shape {tuple(aux_tensor.shape)}; skipping.",
                                                flush=True,
                                            )
                                            feature_aux_source_warned = True
                                    else:
                                        aux_mask_tensor = feature_mask_weights
                                        patch_loss = _compute_patch_cosine_loss(
                                            aux_tensor,
                                            mask=aux_mask_tensor,
                                            patch_size=int(feature_aux_cfg.patch_rays),
                                            stride=int(feature_aux_cfg.patch_stride),
                                            normalize_mode=feature_aux_cfg.normalize,
                                        )
                                        feature_aux_loss_unscaled = patch_loss.detach()
                                        feature_aux_feature_dim_value = int(aux_tensor.shape[-1])
                                        loss_aux_weight = torch.tensor(
                                            aux_weight,
                                            device=patch_loss.device,
                                            dtype=patch_loss.dtype,
                                        )
                                        weighted_aux_loss = patch_loss * loss_aux_weight
                                        total_loss = total_loss + weighted_aux_loss
                                        loss_dict["feature_aux"] = weighted_aux_loss
    
                        if teacher_feature_tensor is not None and student_projected is not None:
                            teacher_feature_tensor = teacher_feature_tensor.to(
                                device=student_projected.device,
                                dtype=student_projected.dtype,
                            )
                            feature_breakdown_raw = feature_distiller(
                                {"primary": teacher_feature_tensor.detach()},
                                {"primary": student_projected},
                                mask=feature_mask_weights,
                                global_step=global_step,
                            )
                            if phase_feature_loss_scale != 1.0:
                                feature_breakdown = FeatureLossBreakdown(
                                    recon=feature_breakdown_raw.recon * phase_feature_loss_scale,
                                    cosine=feature_breakdown_raw.cosine * phase_feature_loss_scale,
                                    total=feature_breakdown_raw.total * phase_feature_loss_scale,
                                )
                            else:
                                feature_breakdown = feature_breakdown_raw
                            if feature_breakdown.total.requires_grad:
                                total_loss = total_loss + feature_breakdown.total
                                loss_dict["feature"] = feature_breakdown.total
                            if feature_mask_fraction_tensor is not None:
                                log_feature_mask_fraction = feature_mask_fraction_tensor.detach()
                                feature_mask_fraction_value = float(log_feature_mask_fraction.item())
                                last_feature_mask_fraction_value = feature_mask_fraction_value
                                if mask_controller is not None and not skip_mask_updates:
                                    mask_controller.observe(global_step, feature_mask_fraction_value)
                                    if (
                                        mask_controller_activation_step is not None
                                        and global_step >= mask_controller_activation_step
                                    ):
                                        emergency_fraction = mask_emergency_fraction
                                        recovery_fraction = mask_recovery_fraction
    
                                        if feature_mask_fraction_value < mask_low_fraction_threshold:
                                            mask_low_fraction_streak += 1
                                            mask_emergency_release_counter = 0
                                        else:
                                            mask_low_fraction_streak = 0
                                            if mask_emergency_active_flag:
                                                if feature_mask_fraction_value >= recovery_fraction:
                                                    mask_emergency_release_counter = min(
                                                        mask_emergency_release_counter + 1,
                                                        mask_emergency_hold_steps,
                                                    )
                                                else:
                                                    mask_emergency_release_counter = 0
    
                                        if (
                                            not mask_emergency_active_flag
                                            and mask_low_fraction_streak >= mask_low_fraction_required_steps
                                        ):
                                            base_threshold = mask_controller.current_threshold(global_step)
                                            if base_threshold is None:
                                                base_threshold = mask_controller.base_threshold
                                            if base_threshold is None:
                                                mask_controller.force_minimum()
                                            else:
                                                mask_controller.force_threshold(base_threshold * 0.8)
                                            mask_emergency_active_flag = True
                                            mask_emergency_activation_total += 1
                                            mask_fraction_emergency_warned = True
                                            progress.write(
                                                (
                                                    "[feature_pipeline] Mask coverage stayed below "
                                                    f"{mask_low_fraction_threshold:.2f} for {mask_low_fraction_required_steps} steps; "
                                                    "relaxing threshold temporarily."
                                                )
                                            )
    
                                        if mask_emergency_active_flag:
                                            if feature_mask_fraction_value < emergency_fraction:
                                                mask_controller.force_minimum()
                                                mask_emergency_release_counter = 0
                                            elif mask_emergency_release_counter >= mask_emergency_hold_steps:
                                                mask_controller.relax_towards_schedule(
                                                    global_step,
                                                    immediate=True,
                                                )
                                                mask_emergency_active_flag = False
                                                mask_fraction_emergency_warned = False
                                                mask_emergency_release_counter = 0
                                        else:
                                            if feature_mask_fraction_value >= recovery_fraction:
                                                if mask_fraction_emergency_warned:
                                                    progress.write(
                                                        "[feature_pipeline] Mask coverage recovered; releasing emergency override."
                                                    )
                                                    mask_fraction_emergency_warned = False
                                                mask_controller.relax_towards_schedule(
                                                    global_step,
                                                    immediate=True,
                                                )
                                            elif feature_mask_fraction_value >= mask_controller.min_fraction:
                                                mask_controller.relax_towards_schedule(global_step)
                                            mask_emergency_release_counter = 0
    
                                    if (
                                        mask_prefail_enabled
                                        and prefail_min_candidate is not None
                                        and prefail_p5_candidate is not None
                                    ):
                                        mask_prefail_history.append(
                                            (global_step, float(prefail_min_candidate), float(prefail_p5_candidate))
                                        )
                                        p5_rate, min_rate = compute_mask_prefail_rates(mask_prefail_history)
                                        if p5_rate is not None:
                                            mask_prefail_drop_rate_recent = float(p5_rate)
                                        if min_rate is not None:
                                            mask_prefail_min_rate_recent = float(min_rate)
                                        if prefail_variance_candidate is not None:
                                            mask_prefail_variance_recent = float(prefail_variance_candidate)
                                        trigger_prefail = False
                                        cooldown_active = (
                                            mask_prefail_cooldown_steps > 0 and global_step < mask_prefail_cooldown_until
                                        )
                                        if not mask_prefail_active_flag and not cooldown_active:
                                            if (
                                                p5_rate is not None
                                                and p5_rate <= -float(mask_prefail_cfg.p5_drop_rate)
                                            ):
                                                trigger_prefail = True
                                            if (
                                                min_rate is not None
                                                and min_rate <= -float(mask_prefail_cfg.min_drop_rate)
                                            ):
                                                trigger_prefail = True
                                            variance_ceiling = float(mask_prefail_cfg.variance_ceiling)
                                            if (
                                                prefail_variance_candidate is not None
                                                and variance_ceiling > 0.0
                                                and prefail_variance_candidate >= variance_ceiling
                                            ):
                                                trigger_prefail = True
                                        if trigger_prefail:
                                            applied_threshold = None
                                            if mask_controller is not None:
                                                applied_threshold = mask_controller.apply_prefail(
                                                    global_step,
                                                    threshold_scale=float(mask_prefail_cfg.threshold_scale),
                                                    soft_delta=float(mask_prefail_cfg.soft_floor_delta),
                                                )
                                                mask_prefail_last_threshold = float(applied_threshold)
                                            mask_prefail_active_flag = True
                                            mask_prefail_trigger_step = global_step
                                            mask_prefail_activation_total += 1
                                            if mask_prefail_cooldown_steps > 0:
                                                mask_prefail_cooldown_until = max(
                                                    mask_prefail_cooldown_until,
                                                    global_step + mask_prefail_cooldown_steps,
                                                )
                                            p5_desc = f"{p5_rate:.6f}" if p5_rate is not None else "nan"
                                            min_desc = f"{min_rate:.6f}" if min_rate is not None else "nan"
                                            thresh_desc = (
                                                f"{mask_prefail_last_threshold:.4f}"
                                                if math.isfinite(mask_prefail_last_threshold)
                                                else "n/a"
                                            )
                                            progress.write(
                                                "[feature_pipeline] Mask prefail trigger: "
                                                f"p5_rate={p5_desc}, min_rate={min_desc}, threshold={thresh_desc}"
                                            )
                                        elif mask_prefail_active_flag and mask_controller is not None:
                                            release = False
                                            if (
                                                feature_mask_fraction_value is not None
                                                and feature_mask_fraction_value >= mask_controller.min_fraction + 0.05
                                            ):
                                                release = True
                                            elif p5_rate is not None and p5_rate >= -float(mask_prefail_cfg.p5_drop_rate) * 0.25:
                                                release = True
                                            if release and (global_step - mask_prefail_trigger_step) >= mask_prefail_window // 2:
                                                mask_prefail_active_flag = False
                                                mask_controller.relax_towards_schedule(global_step)
                                                current_threshold = mask_controller.current_threshold(global_step)
                                                if current_threshold is not None:
                                                    mask_prefail_last_threshold = float(current_threshold)
                                                mask_prefail_trigger_step = global_step
                                                if mask_prefail_cooldown_steps > 0:
                                                    mask_prefail_cooldown_until = max(
                                                        mask_prefail_cooldown_until,
                                                        global_step + mask_prefail_cooldown_steps,
                                                    )
    
            if feature_pipeline_active and "feature" not in loss_dict:
                loss_dict["feature"] = torch.zeros((), device=device)
            if feature_aux_enabled and "feature_aux" not in loss_dict:
                loss_dict["feature_aux"] = torch.zeros((), device=device)
    
            total_loss_detached = total_loss.detach()
            loss_is_finite = bool(torch.isfinite(total_loss_detached).item())
            total_val = float(total_loss_detached.cpu().item()) if loss_is_finite else float("nan")

            steps_since_start = max(global_step - start_step, 0)
            should_log_interval = (steps_since_start % scalar_log_interval) == 0
            warmup_span = max(1, min(scalar_log_interval, 5))
            within_log_warmup = steps_since_start <= warmup_span
            should_log_step = (
                should_log_interval
                or global_step in promotion_pending
                or not loss_is_finite
                or global_step == train_cfg.max_steps
                or within_log_warmup
            )
            if scalar_log_interval <= 1:
                should_log_interval = True
                should_log_step = True

            _update_alpha_quantiles(
                student_sigma,
                step=global_step,
                force=should_log_step,
            )

            alpha_quantile_p50 = alpha_quantile_last_values["p50"]
            alpha_quantile_p90 = alpha_quantile_last_values["p90"]
            alpha_quantile_p99 = alpha_quantile_last_values["p99"]
            alpha_spread_tensor = alpha_spread_last
            alpha_tail_slope_tensor = alpha_tail_slope_last

            # Leak/Halo heuristics (clamped to >=0 for interpretability).
            leak_primary = torch.clamp(0.2 - alpha_quantile_p50, min=0.0)
            leak_tail = torch.clamp(-alpha_tail_slope_tensor, min=0.0)
            alpha_leak_indicator = (leak_primary + 0.5 * leak_tail).to(dtype=student_sigma.dtype)
            halo_primary = torch.clamp(alpha_quantile_p99 - 0.985, min=0.0)
            halo_spread = torch.clamp(alpha_spread_tensor - 0.25, min=0.0)
            halo_tail = torch.clamp(alpha_tail_slope_tensor, min=0.0)
            alpha_halo_indicator = (halo_primary + 0.5 * halo_spread + 0.25 * halo_tail).to(dtype=student_sigma.dtype)

            leak_scalar = float(alpha_leak_indicator.detach().cpu().item()) if torch.isfinite(alpha_leak_indicator) else 0.0
            halo_scalar = float(alpha_halo_indicator.detach().cpu().item()) if torch.isfinite(alpha_halo_indicator) else 0.0
            if leak_scalar > alpha_leak_threshold:
                alpha_leak_streak = min(alpha_leak_streak + 1, alpha_issue_streak_window)
            else:
                alpha_leak_streak = max(alpha_leak_streak - 1, 0)
            if halo_scalar > alpha_halo_threshold:
                alpha_halo_streak = min(alpha_halo_streak + 1, alpha_issue_streak_window)
            else:
                alpha_halo_streak = max(alpha_halo_streak - 1, 0)

            issue_code = 0
            issue_strength = 0.0
            if alpha_leak_streak >= alpha_issue_streak_window and alpha_leak_streak >= alpha_halo_streak:
                issue_code = 1
                issue_strength = leak_scalar
            elif alpha_halo_streak >= alpha_issue_streak_window and alpha_halo_streak > alpha_leak_streak:
                issue_code = 2
                issue_strength = halo_scalar
            alpha_last_issue_code = issue_code if issue_code != 0 else alpha_last_issue_code

            if global_step <= start_step + 5:
                debug_log("log_checkpoint_gate step=%s" % (global_step,))
            debug_log("checkpoint_metrics step=%s" % (global_step,))
    
            log_metrics: Dict[str, torch.Tensor] = dict(loss_dict)
            log_metrics["metrics_schema_version"] = torch.tensor(float(METRICS_SCHEMA_VERSION), device=device)
            log_metrics["opacity_target_weight_effective"] = torch.tensor(
                opacity_target_weight_effective, device=device
            )
            log_metrics["opacity_target_weight_base"] = torch.tensor(opacity_target_weight_base, device=device)
            log_metrics["opacity_target_adjustment"] = torch.tensor(opacity_target_adjustment, device=device)
            log_metrics["opacity_lambda_runtime"] = torch.tensor(opacity_lambda_runtime, device=device)
            log_metrics["learning_rate"] = torch.tensor(current_lr, device=device)
            log_metrics["alpha_mean"] = alpha_mean_tensor.detach()
            log_metrics["alpha_fraction_ge95"] = alpha_fraction_ge95.detach()
            log_metrics["alpha_fraction_le05"] = alpha_fraction_le05.detach()
            log_metrics["alpha_fraction_mid"] = alpha_fraction_mid.detach()
            log_metrics["alpha_penalty_core"] = alpha_penalty_core.detach()
            log_metrics["alpha.p50_ray"] = alpha_quantile_p50.detach()
            log_metrics["alpha.p90_ray"] = alpha_quantile_p90.detach()
            log_metrics["alpha.p99_ray"] = alpha_quantile_p99.detach()
            log_metrics["alpha.spread"] = alpha_spread_tensor.detach()
            log_metrics["alpha.tail_slope"] = alpha_tail_slope_tensor.detach()
            log_metrics["alpha.quantile_refresh_count"] = torch.tensor(
                float(alpha_quantile_refresh_count),
                device=device,
            )
            log_metrics["alpha.quantile_nan_events"] = torch.tensor(
                float(alpha_quantile_nan_events),
                device=device,
            )
            # Backwards compatibility for in-flight dashboards expecting legacy keys.
            log_metrics["alpha_quantile_p50"] = log_metrics["alpha.p50_ray"]
            log_metrics["alpha_quantile_p90"] = log_metrics["alpha.p90_ray"]
            log_metrics["alpha_quantile_p99"] = log_metrics["alpha.p99_ray"]
            log_metrics["alpha.leak_indicator"] = alpha_leak_indicator.detach()
            log_metrics["alpha.halo_indicator"] = alpha_halo_indicator.detach()
            log_metrics["alpha.leak_streak"] = torch.tensor(float(alpha_leak_streak), device=device)
            log_metrics["alpha.halo_streak"] = torch.tensor(float(alpha_halo_streak), device=device)
            log_metrics["alpha.issue_code"] = torch.tensor(float(issue_code), device=device)
            log_metrics["alpha.issue_strength"] = torch.tensor(float(issue_strength), device=device)
            log_metrics.setdefault("alpha.issue_last", torch.tensor(float(alpha_last_issue_code), device=device))
            log_metrics["alpha_penalty_weight"] = torch.tensor(alpha_penalty_weight, device=device)
            log_metrics["alpha_guard_avg_penalty"] = torch.tensor(alpha_guard_avg_penalty, device=device)
            log_metrics["alpha_penalty_weight_target"] = torch.tensor(
                alpha_penalty_weight_target,
                device=device,
            )
            log_metrics["opacity_target_adjustment_target"] = torch.tensor(
                opacity_target_adjustment_target,
                device=device,
            )
            log_metrics["opacity_lambda_target"] = torch.tensor(opacity_lambda_target, device=device)
            log_metrics["alpha_guard_tighten_streak"] = torch.tensor(
                float(alpha_guard_tighten_streak),
                device=device,
            )
            log_metrics["alpha_guard_relax_streak"] = torch.tensor(
                float(alpha_guard_relax_streak),
                device=device,
            )
            direction_code: float
            if alpha_guard_last_direction == "tighten":
                direction_code = 1.0
            elif alpha_guard_last_direction == "relax":
                direction_code = -1.0
            else:
                direction_code = 0.0
            log_metrics["alpha_guard_last_direction"] = torch.tensor(direction_code, device=device)
            if feature_src_dim_value is not None:
                log_metrics["feature_src_dim"] = torch.tensor(float(feature_src_dim_value), device=device)
            if projector_out_dim_value is not None:
                log_metrics["projector_out_dim"] = torch.tensor(float(projector_out_dim_value), device=device)
            if feature_aux_enabled:
                log_metrics["feature_aux_weight"] = torch.tensor(feature_aux_weight_value, device=device)
                if feature_aux_loss_unscaled is not None:
                    log_metrics["feature_aux_loss_raw"] = feature_aux_loss_unscaled
                else:
                    log_metrics.setdefault(
                        "feature_aux_loss_raw",
                        torch.tensor(float("nan"), device=device),
                    )
                if feature_aux_feature_dim_value is not None:
                    log_metrics["feature_aux_dim"] = torch.tensor(
                        float(feature_aux_feature_dim_value),
                        device=device,
                    )
            if feature_pipeline_active:
                nan_default = torch.tensor(float("nan"), device=device)
                log_metrics.setdefault("feature_recon", nan_default)
                log_metrics.setdefault("feature_cosine", nan_default)
                log_metrics.setdefault("feature_mask_fraction", nan_default)
                log_metrics.setdefault(
                    "feature_compare_teacher",
                    torch.tensor(
                        1.0 if str(feature_cfg.compare_space).lower() == "teacher" else 0.0,
                        device=device,
                    ),
                )
                log_metrics.setdefault(
                    "feature_source_penultimate",
                    torch.tensor(
                        1.0 if str(feature_cfg.student_feature_source).lower() == "penultimate" else 0.0,
                        device=device,
                    ),
                )
                log_metrics.setdefault(
                    "feature_source_post_activation",
                    torch.tensor(
                        1.0 if str(feature_cfg.student_feature_activation).lower() in {"post", "postrelu", "post_relu"} else 0.0,
                        device=device,
                    ),
                )
                log_metrics.setdefault(
                    "feature_src_available",
                    torch.tensor(1.0 if student_pre is not None else 0.0, device=device),
                )
                if feature_mask_weight_mean_tensor is not None:
                    log_metrics["feature_mask_weight_mean"] = feature_mask_weight_mean_tensor.detach()
                    log_metrics["feature_mask_weight_min"] = feature_mask_weight_min_tensor.detach()
                    log_metrics["feature_mask_weight_max"] = feature_mask_weight_max_tensor.detach()
                else:
                    log_metrics["feature_mask_weight_mean"] = nan_default
                    log_metrics["feature_mask_weight_min"] = nan_default
                    log_metrics["feature_mask_weight_max"] = nan_default
                log_metrics["mask_emergency_active"] = torch.tensor(
                    1.0 if mask_emergency_active_flag else 0.0,
                    device=device,
                )
                log_metrics["mask_emergency_count"] = torch.tensor(
                    float(mask_emergency_activation_total),
                    device=device,
                )
                log_metrics["mask_low_fraction_streak"] = torch.tensor(
                    float(mask_low_fraction_streak),
                    device=device,
                )
                if mask_prefail_enabled:
                    log_metrics["mask_prefail_active"] = torch.tensor(
                        1.0 if mask_prefail_active_flag else 0.0,
                        device=device,
                    )
                    log_metrics["mask_prefail_count"] = torch.tensor(
                        float(mask_prefail_activation_total),
                        device=device,
                    )
                    log_metrics["mask_prefail_drop_rate"] = torch.tensor(
                        mask_prefail_drop_rate_recent,
                        device=device,
                    )
                    log_metrics["mask_prefail_min_rate"] = torch.tensor(
                        mask_prefail_min_rate_recent,
                        device=device,
                    )
                    log_metrics["mask_prefail_threshold"] = torch.tensor(
                        mask_prefail_last_threshold,
                        device=device,
                    )
                    log_metrics["mask_prefail_variance"] = torch.tensor(
                        mask_prefail_variance_recent,
                        device=device,
                    )
                else:
                    log_metrics["mask_prefail_active"] = torch.tensor(0.0, device=device)
                    log_metrics["mask_prefail_count"] = torch.tensor(0.0, device=device)
                    log_metrics["mask_prefail_drop_rate"] = nan_default
                    log_metrics["mask_prefail_min_rate"] = nan_default
                    log_metrics["mask_prefail_threshold"] = nan_default
                    log_metrics["mask_prefail_variance"] = nan_default
                if mask_controller is not None:
                    threshold_value = mask_threshold_for_log
                    if threshold_value is None:
                        threshold_value = mask_controller.current_threshold(global_step)
                    if threshold_value is not None:
                        log_metrics["feature_mask_threshold"] = torch.tensor(threshold_value, device=device)
                    else:
                        log_metrics["feature_mask_threshold"] = nan_default
                    log_metrics["feature_mask_soft_transition"] = torch.tensor(
                        mask_controller.current_soft_transition(),
                        device=device,
                    )
            if (
                teacher_depth is not None
                and loss_cfg.depth_weight > 0.0
            ):
                if depth_valid_fraction is None:
                    depth_valid_fraction = torch.tensor(float("nan"), device=device)
                log_metrics["depth_valid_fraction"] = depth_valid_fraction
                log_metrics.update(depth_stat_tensors)
                if teacher_depth_for_loss is not None:
                    if (
                        combined_valid_tensor is not None
                        and combined_valid_tensor.any()
                    ):
                        norm_values = teacher_depth_for_loss[combined_valid_tensor]
                        log_metrics["teacher_depth_norm_min"] = norm_values.min()
                        log_metrics["teacher_depth_norm_max"] = norm_values.max()
                        log_metrics["teacher_depth_norm_mean"] = norm_values.mean()
                    else:
                        nan_tensor = torch.tensor(float("nan"), device=device)
                        log_metrics["teacher_depth_norm_min"] = nan_tensor
                        log_metrics["teacher_depth_norm_max"] = nan_tensor
                        log_metrics["teacher_depth_norm_mean"] = nan_tensor
                if depth_range_tensor is not None:
                    log_metrics["ray_depth_range_mean"] = depth_range_tensor.mean()
                    log_metrics["ray_depth_range_min"] = depth_range_tensor.min()
                    log_metrics["ray_depth_range_max"] = depth_range_tensor.max()
    
            if feature_breakdown is not None:
                log_metrics["feature_recon"] = feature_breakdown.recon.detach()
                log_metrics["feature_cosine"] = feature_breakdown.cosine.detach()
            if feature_distiller is not None and feature_distiller.enabled:
                log_metrics["feature_weight_effective"] = torch.tensor(
                    feature_distiller.current_recon_weight, device=device
                )
                log_metrics["feature_cos_weight_effective"] = torch.tensor(
                    feature_distiller.current_cosine_weight, device=device
                )
                feature_terminal_value = 1.0 if feature_distiller.terminal_reached(global_step) else 0.0
            else:
                feature_terminal_value = float("nan")
            log_metrics["feature_schedule_terminal"] = torch.tensor(feature_terminal_value, device=device)
            opacity_terminal_value = 1.0 if opacity_scheduler.terminal_reached(global_step) else 0.0
            log_metrics["opacity_schedule_terminal"] = torch.tensor(opacity_terminal_value, device=device)
            feature_effective_val = 0.0
            if "feature_weight_effective" in log_metrics:
                feature_effective_tensor = log_metrics["feature_weight_effective"]
                try:
                    feature_effective_val = float(feature_effective_tensor.detach().cpu().item())
                except (AttributeError, ValueError):
                    feature_effective_val = 0.0
            feature_cos_effective_val = 0.0
            if "feature_cos_weight_effective" in log_metrics:
                feature_cos_tensor = log_metrics["feature_cos_weight_effective"]
                try:
                    feature_cos_effective_val = float(feature_cos_tensor.detach().cpu().item())
                except (AttributeError, ValueError):
                    feature_cos_effective_val = 0.0
            if feature_effective_val > 0.0 or feature_cos_effective_val > 0.0:
                feature_on_steps += 1
            if opacity_target_weight_effective > 0.0:
                opacity_on_steps += 1
            log_metrics["feature_on_steps"] = torch.tensor(float(feature_on_steps), device=device)
            log_metrics["opacity_on_steps"] = torch.tensor(float(opacity_on_steps), device=device)
            if log_feature_mask_fraction is not None:
                log_metrics["feature_mask_fraction"] = log_feature_mask_fraction.to(device)
            log_metrics["phase_index"] = torch.tensor(float(current_phase_index), device=device)
            log_metrics["phase_feature_scale"] = torch.tensor(float(phase_feature_loss_scale), device=device)
            override_value = (phase_mask_override or "inherit").lower()
            override_code = mask_override_codes.get(override_value, -1.0)
            log_metrics["phase_mask_override"] = torch.tensor(float(override_code), device=device)
            log_metrics.setdefault("mask_emergency_active", torch.tensor(0.0, device=device))
            log_metrics.setdefault("mask_emergency_count", torch.tensor(0.0, device=device))
            log_metrics.setdefault("mask_low_fraction_streak", torch.tensor(0.0, device=device))
    
            if feature_pipeline_active:
                fraction_targets = {
                    6500: 0.15,
                    7500: 0.20,
                    8500: 0.25,
                }
                required_fraction = fraction_targets.get(global_step)
                if required_fraction is not None:
                    effective_mask_fraction = feature_mask_fraction_value
                    if effective_mask_fraction is None and last_feature_mask_fraction_value is not None:
                        effective_mask_fraction = last_feature_mask_fraction_value
                        if not feature_mask_fraction_missing_warned:
                            progress.write(
                                "[feature_pipeline] feature_mask_fraction missing for current step; reusing the last recorded value."
                            )
                            feature_mask_fraction_missing_warned = True
                    if effective_mask_fraction is None:
                        if not feature_mask_fraction_missing_warned:
                            progress.write(
                                (
                                    f"[feature_pipeline] feature_mask_fraction unavailable at step {global_step}; "
                                    "skipping coverage check for this step."
                                )
                            )
                            feature_mask_fraction_missing_warned = True
                    else:
                        feature_mask_fraction_missing_warned = False
                        feature_mask_fraction_value = effective_mask_fraction
                        if feature_mask_fraction_value < required_fraction:
                            raise RuntimeError(
                                f"feature mask fraction {feature_mask_fraction_value:.4f} below required {required_fraction:.2f} at step {global_step}"
                            )
                    expected_src_dim = int(feature_cfg.projector_input_dim)
                    if feature_src_dim_value is not None and feature_src_dim_value != expected_src_dim:
                        raise RuntimeError(
                            f"feature_src_dim reported {feature_src_dim_value} but expected {expected_src_dim} at step {global_step}"
                        )
                    expected_teacher_dim = feature_cfg.resolved_teacher_dim
                    if expected_teacher_dim is None:
                        expected_teacher_dim = projector_out_dim_value
                    if expected_teacher_dim is not None and projector_out_dim_value is not None:
                        if projector_out_dim_value != int(expected_teacher_dim):
                            raise RuntimeError(
                                f"projector_out_dim reported {projector_out_dim_value} but expected {int(expected_teacher_dim)} at step {global_step}"
                            )
                    if feature_distiller is not None and global_step > loss_cfg.feature_warmup_steps:
                        if feature_distiller.current_recon_weight <= 0.0:
                            raise RuntimeError(
                                f"feature reconstruction weight inactive at step {global_step} despite warmup completion"
                            )
                        if (
                            (loss_cfg.feature_target_cosine_weight or 0.0) > 0.0
                            and feature_distiller.current_cosine_weight <= 0.0
                        ):
                            raise RuntimeError(
                                f"feature cosine weight inactive at step {global_step} despite warmup completion"
                            )
    
            # Update per-step metrics even if feature pipeline is inactive.
            debug_log("before_moving_avg step=%s" % (global_step,))
            update_moving_averages(log_metrics)
            debug_log("after_moving_avg step=%s" % (global_step,))
    
            debug_log(
                "log_before_csv step=%s finite=%s total=%s" % (
                    global_step,
                    loss_is_finite,
                    total_val,
                )
            )
    
            base_log_metrics: Dict[str, torch.Tensor] = dict(loss_dict)
            for optional_key in ("feature_recon", "feature_cosine"):
                if optional_key in log_metrics:
                    base_log_metrics[optional_key] = log_metrics[optional_key]
            base_log_metrics["metrics_schema_version"] = torch.tensor(float(METRICS_SCHEMA_VERSION), device=device)
            base_log_metrics["learning_rate"] = torch.tensor(current_lr, device=device)
            base_log_metrics["phase_index"] = torch.tensor(float(current_phase_index), device=device)
            base_log_metrics["phase_feature_scale"] = torch.tensor(float(phase_feature_loss_scale), device=device)
            base_log_metrics["opacity_target_weight_effective"] = torch.tensor(opacity_target_weight_effective, device=device)
            base_log_metrics["opacity_target_weight_base"] = torch.tensor(opacity_target_weight_base, device=device)
            base_log_metrics["opacity_target_adjustment"] = torch.tensor(opacity_target_adjustment, device=device)
            base_log_metrics["opacity_lambda_runtime"] = torch.tensor(opacity_lambda_runtime, device=device)
            base_log_metrics.setdefault("alpha.p50_ray", alpha_quantile_p50.detach())
            base_log_metrics.setdefault("alpha.p90_ray", alpha_quantile_p90.detach())
            base_log_metrics.setdefault("alpha.p99_ray", alpha_quantile_p99.detach())
            base_log_metrics.setdefault("alpha.spread", alpha_spread_last.detach())
            base_log_metrics.setdefault("alpha.tail_slope", alpha_tail_slope_last.detach())
            base_log_metrics.setdefault(
                "alpha.quantile_refresh_count",
                torch.tensor(float(alpha_quantile_refresh_count), device=device),
            )
            base_log_metrics.setdefault(
                "alpha.quantile_nan_events",
                torch.tensor(float(alpha_quantile_nan_events), device=device),
            )
            base_log_metrics.setdefault("alpha_quantile_p50", alpha_quantile_p50.detach())
            base_log_metrics.setdefault("alpha_quantile_p90", alpha_quantile_p90.detach())
            base_log_metrics.setdefault("alpha_quantile_p99", alpha_quantile_p99.detach())
            base_log_metrics.setdefault("alpha.leak_indicator", alpha_leak_indicator.detach())
            base_log_metrics.setdefault("alpha.halo_indicator", alpha_halo_indicator.detach())
            base_log_metrics.setdefault(
                "alpha.leak_streak",
                torch.tensor(float(alpha_leak_streak), device=device),
            )
            base_log_metrics.setdefault(
                "alpha.halo_streak",
                torch.tensor(float(alpha_halo_streak), device=device),
            )
            base_log_metrics.setdefault(
                "alpha.issue_code",
                torch.tensor(float(issue_code), device=device),
            )
            base_log_metrics.setdefault(
                "alpha.issue_strength",
                torch.tensor(float(issue_strength), device=device),
            )
            base_log_metrics.setdefault(
                "alpha.issue_last",
                torch.tensor(float(alpha_last_issue_code), device=device),
            )
    
            metrics_floats_for_log = write_metrics_csv(global_step, total_val, base_log_metrics)
            metrics_floats_for_tb = dict(metrics_floats_for_log)
            for optional_key in ("feature_recon", "feature_cosine"):
                value_tensor = log_metrics.get(optional_key)
                if value_tensor is None:
                    continue
                try:
                    value_numeric = float(value_tensor.detach().cpu().item())
                except (AttributeError, ValueError):
                    continue
                if math.isfinite(value_numeric):
                    metrics_floats_for_tb[optional_key] = value_numeric
            debug_log(
                "log_after_csv step=%s keys=%s" % (
                    global_step,
                    len(metrics_floats_for_log),
                )
            )
            if global_step not in logged_steps_tb:
                debug_log("tb_base_emit step=%s" % (global_step,))
                emit_tensorboard_scalars(
                    global_step,
                    total_val,
                    metrics_floats_for_tb,
                    full=False,
                    base=True,
                )
                logged_steps_tb.add(global_step)
                debug_log("tb_base_emit_done step=%s" % (global_step,))
            debug_log(
                "log_state step=%s scalar_interval=%s start=%s" % (
                    global_step,
                    scalar_log_interval,
                    start_step,
                )
            )
            debug_window_limit = start_step + max(200, scalar_log_interval * 4)
            if global_step <= debug_window_limit:
                debug_log(
                    "log_gate step=%s since_start=%s interval_hit=%s warmup=%s should=%s promotion_hit=%s loss_finite=%s" % (
                        global_step,
                        steps_since_start,
                        should_log_interval,
                        within_log_warmup,
                        should_log_step,
                        global_step in promotion_pending,
                        loss_is_finite,
                    )
                )
            if global_step <= start_step + 5:
                debug_log(
                    "log_debug step=%s since_start=%s interval=%s" % (
                        global_step,
                        steps_since_start,
                        scalar_log_interval,
                    )
                )
            emit_metrics = should_log_step or scalar_log_interval <= 1
            if global_step <= start_step + 5:
                debug_log(
                    "log_flags step=%s since_start=%s interval=%s interval_hit=%s warmup=%s should=%s emit=%s" % (
                        global_step,
                        steps_since_start,
                        scalar_log_interval,
                        should_log_interval,
                        within_log_warmup,
                        should_log_step,
                        emit_metrics,
                    )
                )
            if emit_metrics:
                if global_step <= debug_window_limit:
                    debug_log(
                        "log_emit step=%s" % (
                            global_step,
                        )
                    )
                postfix = {
                    "total": f"{total_val:.4f}",
                    "color": f"{loss_dict['color'].item():.4f}",
                    "opacity": f"{loss_dict['opacity'].item():.4f}",
                }
                if "depth" in loss_dict:
                    postfix["depth"] = f"{loss_dict['depth'].item():.4f}"
                if "feature" in loss_dict:
                    postfix["feature"] = f"{loss_dict['feature'].item():.4f}"
                for extra_key in ("depth_valid_fraction", "teacher_depth_min", "teacher_depth_max", "teacher_depth_mean"):
                    if extra_key in log_metrics:
                        postfix[extra_key] = f"{log_metrics[extra_key].item():.4f}"
                for feat_key in ("feature_recon", "feature_cosine"):
                    if feat_key in log_metrics:
                        postfix[feat_key] = f"{log_metrics[feat_key].item():.4f}"
    
                elapsed_sec = time.perf_counter() - train_start_time
                steps_done = max(global_step - start_step, 0)
                remaining_steps = 0
                if train_cfg.max_steps > 0:
                    remaining_steps = max(train_cfg.max_steps - global_step, 0)
                if steps_done > 0 and elapsed_sec > 0.0:
                    if step_time_ema is not None and remaining_steps > 0:
                        eta_seconds = step_time_ema * remaining_steps
                        postfix["eta"] = str(timedelta(seconds=int(eta_seconds)))
                        postfix["step_time"] = f"{step_time_ema:.2f}s"
                    elif remaining_steps > 0:
                        avg_step = elapsed_sec / steps_done
                        eta_seconds = avg_step * remaining_steps
                        postfix["eta"] = str(timedelta(seconds=int(eta_seconds)))
                        postfix["step_time"] = f"{avg_step:.2f}s"
                    else:
                        if step_time_ema is not None:
                            postfix["step_time"] = f"{step_time_ema:.2f}s"
                    postfix["elapsed"] = str(timedelta(seconds=int(elapsed_sec)))
                progress.set_postfix(**postfix)
                try:
                    progress.refresh()
                except Exception:
                    pass
            if health_enabled and not health_monitor_finished and global_step <= health_record_limit:
                def _snapshot_value(key: str, fallback: Optional[float] = None) -> float:
                    value = log_metrics.get(key)
                    if value is None:
                        return float(fallback) if fallback is not None else float("nan")
                    if isinstance(value, torch.Tensor):
                        try:
                            return float(value.detach().cpu().item())
                        except (AttributeError, ValueError):
                            return float("nan")
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return float("nan")
    
                snapshot = {
                    "feature_mask_fraction": _snapshot_value(
                        "feature_mask_fraction",
                        last_feature_mask_fraction_value,
                    ),
                    "feature_src_dim": _snapshot_value(
                        "feature_src_dim",
                        float(last_feature_src_dim_logged) if last_feature_src_dim_logged is not None else None,
                    ),
                    "projector_out_dim": _snapshot_value(
                        "projector_out_dim",
                        float(last_projector_out_dim_logged) if last_projector_out_dim_logged is not None else None,
                    ),
                    "feature_schedule_terminal": _snapshot_value("feature_schedule_terminal", None),
                    "opacity_schedule_terminal": _snapshot_value("opacity_schedule_terminal", None),
                    "feature_weight_effective": _snapshot_value("feature_weight_effective"),
                    "feature_cos_weight_effective": _snapshot_value("feature_cos_weight_effective"),
                    "opacity_target_weight_effective": _snapshot_value("opacity_target_weight_effective", opacity_target_weight_effective),
                    "alpha_fraction_ge95": _snapshot_value("alpha_fraction_ge95"),
                    "alpha_fraction_le05": _snapshot_value("alpha_fraction_le05"),
                    "alpha_mean": _snapshot_value("alpha_mean"),
                    "loss_color": _snapshot_value("color"),
                    "loss_total": total_val,
                    "feature_on_steps": float(feature_on_steps),
                    "opacity_on_steps": float(opacity_on_steps),
                }
                if math.isnan(snapshot["feature_mask_fraction"]) and last_feature_mask_fraction_value is not None:
                    snapshot["feature_mask_fraction"] = float(last_feature_mask_fraction_value)
                health_records.append((global_step, snapshot))
                if not health_warn_emitted and global_step >= health_warn_steps:
                    ok_warn, summary_warn, issues_warn = evaluate_health_snapshot("warn")
                    warn_status = "OK" if ok_warn else "NG"
                    warn_detail = " :: stable" if not issues_warn else f" :: issues: {', '.join(issues_warn)}"
                    progress.write(f"[health][WARN][{warn_status}] {summary_warn}{warn_detail}")
                    health_warn_emitted = True
                if not health_fail_evaluated and global_step >= health_fail_steps:
                    ok_fail, summary_fail, issues_fail = evaluate_health_snapshot("fail")
                    fail_status = "OK" if ok_fail else "NG"
                    fail_detail = " :: stable" if not issues_fail else f" :: issues: {', '.join(issues_fail)}"
                    progress.write(f"[health][FAIL][{fail_status}] {summary_fail}{fail_detail}")
                    health_fail_evaluated = True
                    health_monitor_finished = True
                    if not ok_fail and health_failfast:
                        raise TrainingAbort("Health check snapshot deemed abnormal.", exit_code=12)
    
            if (
                should_log_step
                and quicklook_interval > 0
                and generate_quicklook is not None
                and global_step > 0
                and global_step % quicklook_interval == 0
            ):
                try:
                    quicklook_dir = logging_cfg.csv.parent
                    quicklook_dir.mkdir(parents=True, exist_ok=True)
                    step_image = quicklook_dir / f"quicklook_step_{global_step:06d}.png"
                    recent_window = max(quicklook_interval * 6, 600)
                    rolling_window = max(min(quicklook_interval // 4, 128), 16)
                    generate_quicklook(
                        metrics_csv=logging_cfg.csv,
                        output_path=step_image,
                        recent_steps=recent_window,
                        rolling=rolling_window,
                    )
                    latest_image = quicklook_dir / "quicklook_latest.png"
                    try:
                        shutil.copyfile(step_image, latest_image)
                    except Exception:  # pragma: no cover - best effort copy
                        pass
                except QuicklookGenerationError as err:
                    if not quicklook_error_logged:
                        progress.write(f"[quicklook] generation skipped: {err}")
                        quicklook_error_logged = True
                except Exception as err:  # pragma: no cover - matplotlib missing or similar
                    if not quicklook_error_logged:
                        progress.write(f"[quicklook] unexpected failure: {err}")
                        quicklook_error_logged = True
    
            enforce_promotion_gate(global_step, log_metrics)
    
            if not loss_is_finite:
                snapshot_recent_metrics()
                raise TrainingAbort(f"Non-finite loss detected at step {global_step}", exit_code=86)
    
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, train_cfg.gradient_clip_norm)
            optimizer.step()
    
            steps_since_start_post = max(global_step - start_step, 0)
            should_log_interval_post = (steps_since_start_post % scalar_log_interval) == 0
            warmup_span_post = max(1, min(scalar_log_interval, 5))
            within_log_warmup_post = steps_since_start_post <= warmup_span_post
            emit_tensorboard = (
                scalar_log_interval <= 1
                or should_log_interval_post
                or within_log_warmup_post
                or global_step in promotion_pending
                or not loss_is_finite
                or global_step == train_cfg.max_steps
            )
            emit_tensorboard_scalars(
                global_step,
                total_val,
                metrics_floats_for_tb,
                full=emit_tensorboard,
                base=False,
            )
            logged_steps_tb.add(global_step)
    
            step_end_time = time.perf_counter()
            step_duration = step_end_time - step_start_time
            if step_duration > 0.0:
                if step_time_ema is None:
                    step_time_ema = step_duration
                else:
                    step_time_ema = step_time_ema_alpha * step_duration + (1.0 - step_time_ema_alpha) * step_time_ema
    
            if global_step % train_cfg.checkpoint_interval == 0:
                ckpt_path = save_checkpoint(global_step)
                progress.write(f"Saved checkpoint to {ckpt_path}")
    
    except TrainingAbort as err:
        abort_exc = err

    if abort_exc is None:
        try:
            target_n = max(global_step - start_step, 0)
            total_steps = getattr(progress, "total", None)
            if isinstance(total_steps, (int, float)) and math.isfinite(total_steps):
                target_n = min(target_n, int(total_steps))
            current_n = int(getattr(progress, "n", 0))
            delta = target_n - current_n
            if delta != 0:
                progress.update(delta)
            progress.refresh()
        except Exception:
            pass

    if getattr(progress, "disable", False) and manual_progress_active:
        _write_manual("\n")
        manual_progress_active = False
        manual_progress_last_len = 0

    if abort_exc is not None:
        progress.write(f"[abort] {abort_exc}")
        progress.close()
        if writer is not None:
            writer.flush()
            writer.close()
        sys.exit(abort_exc.exit_code)

    if overfit_enabled and feature_pipeline_active:
        mask_ok = (
            last_feature_mask_fraction_value is not None
            and last_feature_mask_fraction_value >= 0.99
        )
        src_ok = (
            last_feature_src_dim_logged is not None
            and last_feature_src_dim_logged == int(feature_cfg.projector_input_dim)
        )
        proj_ok = (
            last_projector_out_dim_logged is not None
            and last_projector_out_dim_logged == int(feature_cfg.projector_output_dim)
        )
        if not (mask_ok and src_ok and proj_ok):
            progress.write(
                "[overfit] Diagnostic thresholds not met (mask ≥0.99, src_dim match, projector_out_dim match required)."
            )
            progress.close()
            if writer is not None:
                writer.flush()
                writer.close()
            sys.exit(17)

    progress.close()
    final_ckpt = save_checkpoint(global_step)
    progress.write(f"Final checkpoint saved to {final_ckpt}")
    if writer is not None:
        writer.flush()
        writer.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from. If omitted, training starts from scratch.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override train.max_steps from the config (must be positive).",
    )
    parser.add_argument(
        "--overfit-mode",
        type=str,
        choices=["none", "projector", "student", "all"],
        default="none",
        help="Run a single-batch overfit diagnostic in the specified mode (projector/student/all).",
    )
    parser.add_argument(
        "--overfit-steps",
        type=int,
        default=None,
        help="Override the number of training steps when running in overfit mode.",
    )
    parser.add_argument(
        "--overfit-lr",
        type=float,
        default=None,
        help="Override the learning rate when running in overfit mode.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    resume_arg = Path(args.resume) if args.resume else None
    overfit_mode_arg = args.overfit_mode.lower() if args.overfit_mode else "none"
    if overfit_mode_arg == "none":
        overfit_mode_arg = None
    _install_sigint_guard()
    train(
        Path(args.config),
        resume_arg,
        overfit_mode=overfit_mode_arg,
        overfit_steps=args.overfit_steps,
        overfit_lr=args.overfit_lr,
        max_steps_override=args.max_steps,
    )


if __name__ == "__main__":
    main()
