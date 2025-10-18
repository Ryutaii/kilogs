# Stage 2 Feature Distillation Plan

_Last updated: 2025-10-05_

## Goals
- Recover image quality while keeping Stage 1 runtime/memory gains.
- Target LEGO benchmarks first, then generalise to other Blender / Realistic Synthetic scenes.
- Near-term quality targets: PSNR ≥ 17 dB, SSIM ≥ 0.80, LPIPS ≤ 0.23.
- Maintain FPS ≥ 0.4 (chunk 8192) and GPU peak ≤ 1.6 GiB during evaluation.

## Current Baselines
| Variant | PSNR | SSIM | LPIPS | Avg FPS | Peak VRAM | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Stage1B (response only) | 9.61 | 0.761 | 0.264 | 0.40 | 0.77 GiB | chunk 8192, NVML off |
| Stage2 SH+opacity+log-scale | 17.47 | 0.747 | 0.330 | 0.13 | 1.55 GiB | feature success, runtime drop |
| Stage2 gaussian_all (baseline) | 15.05 | 0.784 | 0.262 | 0.12 | 1.55 GiB | feature loss overwhelms |

## Experiment Matrix
| ID | Description | Status |
| --- | --- | --- |
| GA-SMOKE | `gaussian_all` 10k smoketest with warmup + projector 256 | halted @371/10k (abort) |
| GA-WARMUP | 50k run with warmup schedule (new config) | **todo** |
| GA-SOFTMASK | Soft weighting for boundary mask (0.30/0.45/0.60 sweep) | in progress |
| GA-ACTIVE | Introduce active ray resampling (PVD-AL-lite) | pending |
| GA-ABLATE | SH-only / SH+opacity / SH+opacity+log-scale comparisons | pending |
| GA-VAE-DATA | Export gaussian_all feature tensors for VAE training | **todo** |
| GA-VAE-TRAIN | Train latent VAE (latent 16-32) on exported features | **todo** |
| GA-VAE-INTEG | Integrate VAE decoder into Stage2 feature supervision | pending |

## Key Changes Implemented
- `feature_mask_weights`: continuous weights applied to feature loss mask. YAML exposes `boundary_mask_soft_transition`.
- Warmup configs created:
  - `configs/lego_response_stage2_kilo_feature_gaussian_all_capacity128_warmup_smoketest.yaml`
  - `configs/lego_response_stage2_kilo_feature_gaussian_all_capacity128_warmup.yaml`
- Projector hidden dimension raised to 256 for gaussian_all variants.
- Pivot to feature VAE:
  - `tools/export_gaussian_features.py` exports per-cell aggregated teacher features.
  - `tools/train_feature_vae.py` trains MLP VAE (latent 16-32) and logs recon/KL.

## Next Steps
1. Run GA-VAE-DATA: execute export script on latest teacher checkpoint, target ≥100k cells (≈500 MB `.npz`).
2. Run GA-VAE-TRAIN: train VAE (latent 24 default) until recon MSE ≤ 0.0008 and KL stabilises; archive checkpoint & scalers.
3. Wire VAE decoder into Stage2 pipeline (GA-VAE-INTEG) and gate gaussian_all supervision via latent recon loss.
4. Re-run GA-SMOKE with VAE latent supervision; confirm `feature_mask_weight_mean ≥ 0.15` within 10k steps.
5. Resume GA-WARMUP 50k using VAE pipeline and log metrics to `results/lego/metrics_summary.csv`; follow with mask threshold sweep + ablations.

## Diagnostics Checklist
- Monitor `feature_mask_weight_{mean,min,max}` for collapse (<0.05) or saturation (=1.0).
- Inspect `training_metrics.csv` for feature loss plateaus beyond warmup.
- Use `tools/inspect_feature_alignment.py` on checkpoints at steps 10k / 30k / 50k.
- Record power usage with `--enable-nvml` on 200-frame evaluation runs.

## Deliverables
- Updated `metrics_summary.csv` with gaussian_all (warmup) and downstream variants.
- PSNR/FPS/VRAM chart for Stage1 vs Stage2 (SH subset vs gaussian_all).
- Visual residual maps for representative frames before/after warmup.
- Draft figure for publication contrasting Stage1B vs Stage2 gaussian_all.

## Risks & Mitigations
- **Feature over-regularisation**: If PSNR stalls <17 dB, reduce cosine weight to 0 or shorten warmup.
- **Runtime regression**: Prioritise profiling after GA-WARMUP; consider reducing projector depth if FPS <0.1.
- **Memory spikes**: Monitor peak reserved memory; drop chunk size or prune teacher components if >1.6 GiB.
- **Alignment instability**: If `feature_mask_weight_min` remains ≈0, revisit cell index handling or blend margin.

---
_Context: LEGO scene, Stage 2 distillation on KiloNeRF student with gaussian_all teacher features._
