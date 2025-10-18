import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
for path in (CURRENT_DIR, PARENT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

import torch
from tqdm import tqdm

from lego_response_distill import (
    LegoRayDataset,
    StudentModel,
    get_camera_rays,
    intersect_rays_aabb,
    parse_config,
    sample_along_rays,
)


@torch.no_grad()
def render_valid_rays(
    model: StudentModel,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: torch.Tensor,
    far: torch.Tensor,
    samples_per_ray: int,
    bbox_min: torch.Tensor,
    bbox_extent: torch.Tensor,
    background: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Volume-render the student model for the provided rays."""

    device = rays_o.device
    outputs: List[torch.Tensor] = []
    total = rays_o.shape[0]
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        rays_o_chunk = rays_o[start:end]
        rays_d_chunk = rays_d[start:end]
        near_chunk = near[start:end]
        far_chunk = far[start:end]

        pts, z_vals = sample_along_rays(
            rays_o=rays_o_chunk,
            rays_d=rays_d_chunk,
            near=near_chunk,
            far=far_chunk,
            num_samples=samples_per_ray,
            perturb=False,
        )
        pts_norm = (pts - bbox_min) / bbox_extent
        pts_norm = pts_norm.clamp(0.0, 1.0)
        pts_flat = pts_norm.view(-1, 3)

        student_rgb_samples, student_sigma_samples = model(pts_flat)
        student_rgb_samples = student_rgb_samples.view(-1, samples_per_ray, 3)
        student_sigma_samples = student_sigma_samples.view(-1, samples_per_ray)

        deltas = z_vals[:, 1:] - z_vals[:, :-1]
        delta_last = torch.full((deltas.shape[0], 1), 1e10, device=device)
        deltas = torch.cat([deltas, delta_last], dim=-1)

        alpha = 1.0 - torch.exp(-student_sigma_samples * deltas)
        transmittance = torch.cumprod(
            torch.cat([torch.ones(alpha.shape[0], 1, device=device), 1.0 - alpha + 1e-10], dim=-1),
            dim=-1,
        )[:, :-1]
        weights = alpha * transmittance

        rgb_map = torch.sum(weights[..., None] * student_rgb_samples, dim=-2)
        opacity_map = weights.sum(dim=-1, keepdim=True)
        rgb_map = rgb_map + (1.0 - opacity_map) * background
        outputs.append(rgb_map)

    return torch.cat(outputs, dim=0)


@torch.no_grad()
def evaluate(config_path: Path, checkpoint_path: Path, output_csv: Path | None = None) -> Dict[str, float]:
    (
        experiment_cfg,
        data_cfg,
        _teacher_cfg,
        student_cfg,
        _train_cfg,
        _loss_cfg,
        logging_cfg,
        _feature_cfg,
        _feature_aux_cfg,
    ) = parse_config(config_path)

    dataset = LegoRayDataset(data_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StudentModel(student_cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    background = dataset.background.to(device)
    bbox_min = torch.tensor(data_cfg.bbox_min, dtype=torch.float32, device=device)
    bbox_max = torch.tensor(data_cfg.bbox_max, dtype=torch.float32, device=device)
    bbox_extent = bbox_max - bbox_min

    chunk_size = max(1, data_cfg.ray_chunk)

    per_view_rows: List[Dict[str, float]] = []
    psnr_values: List[float] = []

    progress = tqdm(total=len(dataset), desc="Evaluating", unit="view")
    for idx in range(len(dataset)):
        teacher_rgb = dataset.teacher_rgb[idx].to(device)
        c2w = dataset.c2w_mats[idx].to(device)

        rays_o_all, rays_d_all = get_camera_rays(dataset.height, dataset.width, dataset.focal, c2w, device)
        near_all, far_all, valid_mask = intersect_rays_aabb(rays_o_all, rays_d_all, bbox_min, bbox_max)
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)

        pred_flat = background.repeat(dataset.num_pixels, 1)

        if valid_indices.numel() > 0:
            rays_o = rays_o_all[valid_indices]
            rays_d = rays_d_all[valid_indices]
            near = near_all[valid_indices]
            far = far_all[valid_indices]

            predicted_valid = render_valid_rays(
                model=model,
                rays_o=rays_o,
                rays_d=rays_d,
                near=near,
                far=far,
                samples_per_ray=data_cfg.samples_per_ray,
                bbox_min=bbox_min,
                bbox_extent=bbox_extent,
                background=background,
                chunk_size=chunk_size,
            )
            pred_flat[valid_indices] = predicted_valid

        pred_image = pred_flat.view(dataset.height, dataset.width, 3)
        mse = torch.mean((pred_image - teacher_rgb) ** 2).item()
        mse = max(mse, 1e-12)
        psnr = 20.0 * math.log10(1.0 / math.sqrt(mse))
        psnr_values.append(psnr)

        per_view_rows.append({
            "index": idx,
            "mse": mse,
            "psnr": psnr,
        })
        progress.set_postfix(psnr=f"{psnr:.2f}dB")
        progress.update(1)

    progress.close()

    avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else float("nan")
    avg_mse = sum(row["mse"] for row in per_view_rows) / len(per_view_rows)

    if output_csv is None:
        step_name = checkpoint_path.stem
        output_csv = logging_cfg.csv.parent / f"eval_{step_name}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as fp:
        fp.write("index,mse,psnr\n")
        for row in per_view_rows:
            fp.write(f"{row['index']},{row['mse']:.8f},{row['psnr']:.4f}\n")
        fp.write(f"avg,{avg_mse:.8f},{avg_psnr:.4f}\n")

    return {"avg_psnr": avg_psnr, "avg_mse": avg_mse, "csv": str(output_csv)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a distilled LEGO student checkpoint")
    parser.add_argument("--config", type=Path, required=True, help="Path to the training config YAML")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path to evaluate")
    parser.add_argument("--out", type=Path, help="Optional CSV output path")
    args = parser.parse_args()

    metrics = evaluate(args.config, args.checkpoint, args.out)
    print(f"Average PSNR: {metrics['avg_psnr']:.4f} dB")
    print(f"Average MSE: {metrics['avg_mse']:.8f}")
    print(f"Per-view metrics saved to: {metrics['csv']}")


if __name__ == "__main__":
    main()
