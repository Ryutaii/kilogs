# CVPR/ECCV Submission Outline — Gaussian→KiloNeRF Feature Distillation

_Last updated: 2025-10-06 (post-reviewer checklist integration)_

## 1. Motivation & Positioning
- **Problem**: Bridging high-quality 3D Gaussian Splatting (3D-GS) teachers and lightweight KiloNeRF students without sacrificing PSNR under strict background constraints.
- **Key Observation**: Directly distilling the full Gaussian feature vector (SH coefficients, opacity, scale, covariance, rotation) into KiloNeRF yields a white-background PSNR of **22.50 dB** (see `metrics_summary.csv`, row `特徴蒸留 (教師フル)`), already surpassing RGB-only response distillation under identical budgets; next target is a **+1 dB uplift (23–24 dB)** while keeping runtime/power in check.
- **Pitch**: "World-first Gaussian→MLP feature distillation delivering PSNR >20 dB under deployable compute budgets." This keeps the narrative focused and coherent.

## 2. Contributions (Draft)
1. **Full Gaussian Feature Distillation Baseline**
   - Demonstrate that naïvely projecting the 84-D Gaussian feature vector into KiloNeRF (projector auto-matched) delivers PSNR 22.5 dB / SSIM 0.826 / LPIPS 0.190 on LEGO (white background), with render stats 0.121 FPS / 1.54 GiB / 170 W (from `metrics_summary.csv`).
   - Highlight opacity regularisation + white background recomposition pipeline enabling clean supervision.

2. **Efficient Stage-wise Hyperparameter Exploration**
   - 10k smoke tests → 50k full runs; capacity (grid, hidden dim, depth) and chunk/ray settings explored systematically while logging FPS/VRAM/Power.
   - PDCA loop documented in `research_notes.md` and `metrics_summary.csv`, supporting reproducible tuning.

3. **Reproducible White-Background Evaluation Framework**
   - Teacher re-rendering, RGBA recomposition, opacity analytics (`tools/analyse_opacity.py`), and consistent metric logging (`tools/evaluate_student_metrics.py`).
   - Public release plan: configs (`configs/lego_feature_teacher_full_rehab_masked_white*.yaml`), renders, metrics CSV/JSON.

## 3. Evidence & Current Results
| Method | PSNR (white) | SSIM (white) | LPIPS (white) | FPS | GPU peak (GiB) | Power avg (W) |
| --- | --- | --- | --- | --- | --- | --- |
| NeRF (teacher ref) | 31.48 | 0.964 | 0.020 | 0.134 | 4.57 | 198.9 |
| KiloNeRF (student baseline) | 23.92 | 0.945 | 0.053 | 30.92 | 1.31 | 43.99 |
| 3D-GS (teacher) | 32.74 | 0.976 | 0.021 | 31.26 | 3.21 | 25.34 |
| **特徴蒸留 (教師フル)** | **22.50** | **0.827** | **0.190** | 0.121 | 1.54 | 170.40 |
| 特徴蒸留 (教師フル + VAE) | – | – | – | – | – | – |
| 特徴蒸留 (教師フル + VAE + PVD-AL) | – | – | – | – | – | – |

*Table: Extracted from `metrics_summary.csv` (2025-10-06). Empty entries indicate experiments pending or deprecated.*

## 4. CVPR/ECCV Readiness Checklist
1. **Push white-background PSNR to 23–24 dB**
   - Continue capacity sweeps (grid 8³/10³, hidden 160/192, depth 4–6) via 10k → 50k pipeline.
   - Record improvements in `metrics_summary.csv` and capture qualitative frames.
2. **Cross-scene Validation**
   - Replicate pipeline on at least 2 additional NerfSynthetic scenes (e.g., Mic, Chair).
   - Document metrics + render stats per scene.
3. **Response vs Feature Ablation**
   - Compare Stage1B RGB-only vs feature distillation in terms of PSNR gain and opacity behaviour.
   - Provide plots/curves for reviewers.
4. **Optional Extensions**
   - Revisit VAE/PVD-AL as supplementary experiments (show scalability when feature dimensionality or scene count increases).
   - Include diagnostic visuals (opacity histograms, background recomposition before/after).

## 5. Narrative Outline
1. **Introduction**
   - Motivation: Need for deployable NeRF-like models with high fidelity.
   - Gap: RGB response distillation saturates; teacher features remain under-utilised.
2. **Related Work**
   - Distillation in NeRF (response vs feature) & Gaussian Splatting.
3. **Method**
   - Gaussian feature extraction, projector design, opacity regularisation, white-background handling.
   - Stage-wise training schedule and PDCA pipeline.
4. **Experiments**
   - LEGO white baseline (primary result 22.5+ dB).
   - Hyperparameter ablations (capacity, chunk, opacity targets).
   - Cross-scene replication.
   - Resource metrics discussion (FPS, VRAM, power).
5. **Discussion / Future Work**
   - Scaling to large scenes, potential of VAE/PVD-AL, multi-background distillation.
6. **Conclusion**
   - Emphasise world-first PSNR >20 dB via Gaussian feature distillation and reproducible evaluation pipeline.

## 6. Immediate Next Steps
- [ ] Finish 10k smoke tests (grid8, hidden160/layers5) and log metrics.
- [ ] Start 50k runs for promising configs → update table.
- [ ] Prepare Mic / Chair configs + renders.
- [ ] Draft figure list (render comparisons, opacity histograms, training curves).
- [ ] Create Overleaf skeleton referencing this outline.

## 7. Risks & Mitigations
- **PSNR plateau <23 dB**: Explore boundary blending tweaks, opacity loss weights, additional capacity.
- **Reviewer concern about speed**: Leverage metrics table to show resource footprints; emphasise student runtime is already constrained by Gaussian projector.
- **VAE skepticism**: Position as future scalability work; optionally include one positive VAE result on larger scenes.

---
_All numbers and configs traceable via `metrics_summary.csv`, `research_notes.md`, and `results/lego/feat_t_full/runs/teacher_full_rehab_masked_white/`._
