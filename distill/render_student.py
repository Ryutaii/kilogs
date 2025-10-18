import argparse
import json
import sys
import time
import types
from pathlib import Path
from typing import Dict, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm


def create_progress(iterable=None, **kwargs):
    """Wrapper ensuring tqdm always renders even when stdout is not a TTY."""
    defaults = {
        "dynamic_ncols": True,
        "disable": False,
        "mininterval": 0.1,
        "maxinterval": 1.0,
        "smoothing": 0.0,
        "ascii": True,
        "file": sys.stdout,
        "leave": kwargs.get("leave", True),
    }
    defaults.update(kwargs)
    return tqdm(iterable, **defaults)

try:
    import pynvml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
for path in (CURRENT_DIR, PARENT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

import lego_response_distill as lrd  # type: ignore  # pylint: disable=import-error
from lego_response_distill import (  # type: ignore  # pylint: disable=import-error
    ExperimentConfig,
    DataConfig,
    TeacherConfig,
    StudentConfig,
    TrainConfig,
    LossConfig,
    LoggingConfig,
    StudentModel,
    compute_losses,  # noqa: F401  # intentional re-export for future use
    parse_config,
)

# Ensure checkpoints saved when lego_response_distill ran as "__main__" can be deserialized.
main_alias = sys.modules.setdefault("__main__", types.ModuleType("__main__"))
for cls in (
    "ExperimentConfig",
    "DataConfig",
    "TeacherConfig",
    "StudentConfig",
    "TrainConfig",
    "LossConfig",
    "LoggingConfig",
):
    setattr(main_alias, cls, getattr(lrd, cls))


class NvmlPowerLogger:
    """Light-weight NVML sampler that collects average power draw in watts."""

    def __init__(self, device_index: Optional[int], allow_nvml: bool = False):
        self.enabled = False
        self.samples = []
        self._handle = None
        self._initialised = False

        if not allow_nvml or pynvml is None:
            return

        if device_index is None:
            device_index = 0

        try:
            pynvml.nvmlInit()
            self._initialised = True
            device_count = pynvml.nvmlDeviceGetCount()
            if 0 <= device_index < device_count:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self.enabled = True
        except Exception:  # pragma: no cover - NVML failures are non-fatal
            self.enabled = False

    def sample(self) -> None:
        if not self.enabled or self._handle is None:
            return
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            if power_mw is not None:
                self.samples.append(power_mw / 1000.0)
        except Exception:  # pragma: no cover - transient NVML issues
            pass

    @property
    def average(self) -> Optional[float]:
        if not self.samples:
            return None
        return float(sum(self.samples) / len(self.samples))

    def shutdown(self) -> None:
        if self._initialised:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # pragma: no cover
                pass
            self._initialised = False


def load_transform_metadata(camera_json: Path) -> Dict:
    with camera_json.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def get_camera_rays(H: int, W: int, focal: float, c2w: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing="xy",
    )
    dirs = torch.stack(
        [
            (i - W * 0.5) / focal,
            -(j - H * 0.5) / focal,
            -torch.ones_like(i),
        ],
        dim=-1,
    )  # (H, W, 3) camera coords

    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)


def volumetric_render(student: StudentModel,
                      rays_o: torch.Tensor,
                      rays_d: torch.Tensor,
                      near: float,
                      far: float,
                      num_samples: int,
                      chunk_size: int,
                      device: torch.device,
                      bbox_min: torch.Tensor,
                      bbox_max: torch.Tensor,
                      background_color: torch.Tensor,
                      debug: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
    t_vals = torch.linspace(0.0, 1.0, steps=num_samples, device=device)
    z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = z_vals.expand(rays_o.shape[0], num_samples)

    rgb_chunks = []
    rgb_pre_chunks = []
    depth_chunks = []
    opacity_chunks = []

    chunk_index = 0
    for start in range(0, rays_o.shape[0], chunk_size):
        end = start + chunk_size
        ro = rays_o[start:end]
        rd = rays_d[start:end]
        z = z_vals[start:end]

        pts = ro[:, None, :] + rd[:, None, :] * z[..., None]
        pts_norm = (pts - bbox_min) / (bbox_max - bbox_min)
        pts_norm = pts_norm.clamp(0.0, 1.0)
        pts_flat = pts_norm.reshape(-1, 3)

        rgb, sigma = student(pts_flat)
        rgb = rgb.view(-1, num_samples, 3)
        sigma = sigma.view(-1, num_samples)

        deltas = z[:, 1:] - z[:, :-1]
        delta_last = torch.tensor([1e10], device=device).expand(deltas.shape[0], 1)
        deltas = torch.cat([deltas, delta_last], dim=-1)

        alpha = 1.0 - torch.exp(-sigma * deltas)
        transmittance = torch.cumprod(
            torch.cat([torch.ones(alpha.shape[0], 1, device=device), 1.0 - alpha + 1e-10], dim=-1),
            dim=-1,
        )[:, :-1]
        weights = alpha * transmittance

        rgb_pre_map = torch.sum(weights[..., None] * rgb, dim=-2)
        depth_map = torch.sum(weights * z, dim=-1)
        opacity_map = torch.sum(weights, dim=-1)

        rgb_map = rgb_pre_map + (1.0 - opacity_map)[..., None] * background_color

        if debug is not None:
            chunks = debug.setdefault("chunks", [])
            chunk_limit = debug.get("chunk_limit")
            if chunk_limit is None or chunk_limit < 0:
                chunk_limit = float("inf")

            pre_sigma_stats = {}
            impl = getattr(student, "impl", student)
            pre_activation = getattr(impl, "last_pre_activation", None)
            if pre_activation is not None and pre_activation.shape[-1] >= 4:
                pre_sigma = pre_activation[:, 3:4]
                pre_sigma_stats = {
                    "pre_sigma_min": float(pre_sigma.min().item()),
                    "pre_sigma_max": float(pre_sigma.max().item()),
                    "pre_sigma_mean": float(pre_sigma.mean().item()),
                    "pre_sigma_has_nan": bool(torch.isnan(pre_sigma).any().item()),
                }

            if chunk_index < chunk_limit:
                chunk_stats = {
                    "chunk_index": int(chunk_index),
                    "sigma_min": float(sigma.min().item()),
                    "sigma_max": float(sigma.max().item()),
                    "sigma_mean": float(sigma.mean().item()),
                    "sigma_has_nan": bool(torch.isnan(sigma).any().item()),
                    "alpha_min": float(alpha.min().item()),
                    "alpha_max": float(alpha.max().item()),
                    "alpha_mean": float(alpha.mean().item()),
                    "alpha_has_nan": bool(torch.isnan(alpha).any().item()),
                    "opacity_map_min": float(opacity_map.min().item()),
                    "opacity_map_max": float(opacity_map.max().item()),
                    "opacity_map_mean": float(opacity_map.mean().item()),
                    "opacity_map_has_nan": bool(torch.isnan(opacity_map).any().item()),
                }
                chunk_stats.update(pre_sigma_stats)
                chunks.append(chunk_stats)
            chunk_index += 1

        rgb_chunks.append(rgb_map)
        rgb_pre_chunks.append(rgb_pre_map)
        depth_chunks.append(depth_map)
        opacity_chunks.append(opacity_map)

    rgb_full = torch.cat(rgb_chunks, dim=0)
    rgb_pre_full = torch.cat(rgb_pre_chunks, dim=0)
    depth_full = torch.cat(depth_chunks, dim=0)
    opacity_full = torch.cat(opacity_chunks, dim=0)

    if debug is not None:
        debug["final"] = {
            "rgb_min": float(rgb_full.min().item()),
            "rgb_max": float(rgb_full.max().item()),
            "rgb_mean": float(rgb_full.mean().item()),
            "rgb_pre_min": float(rgb_pre_full.min().item()),
            "rgb_pre_max": float(rgb_pre_full.max().item()),
            "rgb_pre_mean": float(rgb_pre_full.mean().item()),
            "rgb_pre_has_nan": bool(torch.isnan(rgb_pre_full).any().item()),
            "depth_min": float(depth_full.min().item()),
            "depth_max": float(depth_full.max().item()),
            "depth_mean": float(depth_full.mean().item()),
            "opacity_min": float(opacity_full.min().item()),
            "opacity_max": float(opacity_full.max().item()),
            "opacity_mean": float(opacity_full.mean().item()),
            "opacity_has_nan": bool(torch.isnan(opacity_full).any().item()),
        }

    return {
        "rgb": rgb_full,
        "rgb_pre_background": rgb_pre_full,
        "depth": depth_full,
        "opacity": opacity_full,
    }


def render_student_scene(args):
    config_path = Path(args.config)
    (
        experiment,
        data_cfg,
        teacher_cfg,
        student_cfg,
        train_cfg,
        loss_cfg,
        logging_cfg,
        _feature_cfg,
        _feature_aux_cfg,
    ) = parse_config(config_path)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    student_model = StudentModel(student_cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint["model_state"]
    if args.allow_mismatched_weights:
        current_state = student_model.state_dict()
        compatible = {k: v for k, v in model_state.items() if k in current_state and current_state[k].shape == v.shape}
        missing_keys = sorted(set(current_state.keys()) - set(compatible.keys()))
        skipped_keys = sorted(set(model_state.keys()) - set(compatible.keys()))
        student_model.load_state_dict(compatible, strict=False)
        if missed := [k for k in missing_keys if k.startswith("impl.")]:
            print(f"[warning] skipped {len(missed)} student parameters due to shape mismatch (e.g. {missed[:3]})")
        if skipped_keys:
            print(f"[warning] ignored {len(skipped_keys)} checkpoint entries that have no matching parameter")
    else:
        student_model.load_state_dict(model_state)
    student_model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = output_dir / "renders"
    depth_dir = output_dir / "depth"
    opacity_dir = output_dir / "opacity"
    rgba_dir = output_dir / "rgba_npz"

    directories = [rgb_dir, depth_dir, opacity_dir]
    if args.store_rgba:
        directories.append(rgba_dir)

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    meta = load_transform_metadata(data_cfg.camera_json)
    camera_angle_x = meta["camera_angle_x"]

    first_stub = Path(meta["frames"][0]["file_path"].replace("./", ""))
    candidate_suffixes = [".png", ".jpg", ".jpeg", ".npy", ".npz"]
    sample_img = None
    for suffix in candidate_suffixes:
        candidate = Path(data_cfg.teacher_outputs) / first_stub.with_suffix(suffix)
        if candidate.exists():
            if suffix == ".npy":
                sample_img = np.load(candidate)
            elif suffix == ".npz":
                sample_img = np.load(candidate)["arr_0"]
            else:
                sample_img = imageio.imread(str(candidate))
            break
    if sample_img is None:
        raise FileNotFoundError(f"Could not locate sample teacher image for {first_stub}")
    H, W = sample_img.shape[0], sample_img.shape[1]
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    background = torch.tensor(args.background_color, device=device, dtype=torch.float32)
    bbox_min = torch.tensor(args.bbox_min, device=device, dtype=torch.float32)
    bbox_max = torch.tensor(args.bbox_max, device=device, dtype=torch.float32)

    total_time = 0.0
    per_view_times = {}

    power_logger = NvmlPowerLogger(
        device.index if device.type == "cuda" else None,
        allow_nvml=args.enable_nvml,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    frames = meta["frames"]
    start_index = max(0, int(args.start_frame))
    if start_index:
        frames = frames[start_index:]
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    debug_frames = max(0, getattr(args, "debug_frames", 0) or 0)
    debug_chunk_limit = getattr(args, "debug_chunk_limit", 1)
    debug_records = []

    with torch.no_grad():
        for frame_idx, frame in enumerate(
            create_progress(frames, desc="Rendering student", unit="frame", leave=True)
        ):
            file_stub = Path(frame["file_path"].replace("./", ""))
            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32, device=device)
            rays_o, rays_d = get_camera_rays(H, W, focal, c2w, device)

            debug_payload = None
            if frame_idx < debug_frames:
                debug_payload = {
                    "frame": str(file_stub),
                    "chunk_limit": debug_chunk_limit,
                }

            if power_logger.enabled:
                power_logger.sample()
            start = time.perf_counter()
            outputs = volumetric_render(
                student=student_model,
                rays_o=rays_o,
                rays_d=rays_d,
                near=args.near,
                far=args.far,
                num_samples=args.num_samples,
                chunk_size=args.chunk,
                device=device,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                background_color=background,
                debug=debug_payload,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            render_time = time.perf_counter() - start
            total_time += render_time
            per_view_times[str(file_stub)] = render_time

            if power_logger.enabled:
                power_logger.sample()

            rgb_tensor = outputs["rgb"].clamp(0.0, 1.0)
            rgb = rgb_tensor.view(H, W, 3).cpu().numpy()
            depth = outputs["depth"].view(H, W).cpu().numpy()
            opacity = outputs["opacity"].clamp(0.0, 1.0).view(H, W).cpu().numpy()

            rgba_data = None
            if args.store_rgba:
                rgb_pre = outputs.get("rgb_pre_background")
                if rgb_pre is None:
                    raise KeyError("volumetric_render did not return 'rgb_pre_background'")
                rgba_data = {
                    "rgb": rgb_pre.view(H, W, 3).cpu().numpy().astype(np.float32),
                    "alpha": opacity.astype(np.float32),
                }

            rgb_scaled = rgb * 255.0
            rgb_scaled = np.minimum(np.maximum(rgb_scaled, 0.0), 255.0)
            rgb_uint8 = np.ascontiguousarray(rgb_scaled.astype(np.uint8))
            (rgb_dir / file_stub.parent).mkdir(parents=True, exist_ok=True)
            (depth_dir / file_stub.parent).mkdir(parents=True, exist_ok=True)
            (opacity_dir / file_stub.parent).mkdir(parents=True, exist_ok=True)

            imageio.imwrite(rgb_dir / file_stub.with_suffix(".png"), rgb_uint8)

            depth_norm = depth / (args.far - args.near)
            depth_uint8 = np.ascontiguousarray((depth_norm.clip(0.0, 1.0) * 255.0).astype(np.uint8))
            imageio.imwrite(depth_dir / file_stub.with_suffix(".png"), depth_uint8)

            opacity_scaled = opacity * 255.0
            opacity_scaled = np.minimum(np.maximum(opacity_scaled, 0.0), 255.0)
            opacity_uint8 = np.ascontiguousarray(opacity_scaled.astype(np.uint8))
            imageio.imwrite(opacity_dir / file_stub.with_suffix(".png"), opacity_uint8)

            if args.store_rgba:
                (rgba_dir / file_stub.parent).mkdir(parents=True, exist_ok=True)
                np.savez_compressed(rgba_dir / Path(file_stub.name).with_suffix(".npz"), **rgba_data)

            if debug_payload is not None:
                debug_payload["rgb_uint8_min"] = int(rgb_uint8.min())
                debug_payload["rgb_uint8_max"] = int(rgb_uint8.max())
                debug_payload["opacity_uint8_min"] = int(opacity_uint8.min())
                debug_payload["opacity_uint8_max"] = int(opacity_uint8.max())
                debug_records.append(debug_payload)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    gpu_mem_peak_mib = None
    gpu_mem_reserved_peak_mib = None
    if device.type == "cuda":
        gpu_mem_peak_mib = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        gpu_mem_reserved_peak_mib = torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)

    power_avg = power_logger.average if power_logger.enabled else None
    power_logger.shutdown()

    num_frames = len(frames)
    stats = {
        "num_frames": num_frames,
        "total_render_time_s": total_time,
        "avg_fps": num_frames / max(total_time, 1e-6),
        "per_view_time_s": per_view_times,
    "start_frame": start_index,
    "chunk_size": args.chunk,
        "num_samples": args.num_samples,
        "near": args.near,
        "far": args.far,
        "bbox_min": args.bbox_min,
        "bbox_max": args.bbox_max,
    }

    if gpu_mem_peak_mib is not None:
        stats["gpu_mem_peak_mib"] = gpu_mem_peak_mib
    if gpu_mem_reserved_peak_mib is not None:
        stats["gpu_mem_reserved_peak_mib"] = gpu_mem_reserved_peak_mib
    if power_avg is not None:
        stats["power_avg_watts"] = power_avg
    stats["nvml_enabled"] = power_logger.enabled

    stats_path = output_dir / "render_stats.json"
    with stats_path.open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)

    if debug_records:
        debug_path = output_dir / "debug_stats.json"
        with debug_path.open("w", encoding="utf-8") as fp:
            json.dump(debug_records, fp, indent=2)

    print(f"Saved renders to {rgb_dir}")
    print(f"Render stats written to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Render student model checkpoints")
    parser.add_argument("--config", type=str, required=True, help="Path to lego_response config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Student checkpoint .pth")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save render outputs")
    parser.add_argument("--num-samples", type=int, default=128, help="Samples per ray")
    parser.add_argument("--chunk", type=int, default=4096, help="Number of rays per chunk")
    parser.add_argument("--near", type=float, default=2.0, help="Near plane distance")
    parser.add_argument("--far", type=float, default=6.0, help="Far plane distance")
    parser.add_argument(
        "--background-color",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        help="Background color for alpha compositing",
    )
    parser.add_argument(
        "--bbox-min",
        type=float,
        nargs=3,
        default=(-1.5, -1.5, -1.5),
        help="Minimum corner of bounding box for coordinate normalization",
    )
    parser.add_argument(
        "--bbox-max",
        type=float,
        nargs=3,
        default=(1.5, 1.5, 1.5),
        help="Maximum corner of bounding box for coordinate normalization",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu override")
    parser.add_argument(
        "--enable-nvml",
        action="store_true",
        help="Collect GPU power stats via NVML (can be unstable on some systems)",
    )
    parser.add_argument(
        "--allow-mismatched-weights",
        action="store_true",
        help="Silently skip checkpoint entries whose shapes do not match the current student architecture",
    )
    parser.add_argument(
        "--store-rgba",
        action="store_true",
        help="Also store pre-background RGB and alpha as compressed .npz",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Index of the first frame to render (0-based)",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to render")
    parser.add_argument(
        "--debug-frames",
        type=int,
        default=0,
        help="Capture debug statistics for the first N frames (default: 0)",
    )
    parser.add_argument(
        "--debug-chunk-limit",
        type=int,
        default=1,
        help="Number of chunks per debug frame to record stats for (default: 1)",
    )
    args = parser.parse_args()
    render_student_scene(args)


if __name__ == "__main__":
    main()
