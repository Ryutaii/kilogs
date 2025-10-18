# Quality Improvement Roadmap

## Baseline snapshot (2025-09-26)
- **Scenario**: LEGO student distilled via response loss (step 200k)
- **Metrics** (student_200k):
  - PSNR: 9.47579
  - SSIM: 0.75573
  - LPIPS: 0.26660
  - Avg FPS: 0.94429
  - GPU Peak (GiB): 0.32102
  - Power Avg (W): 187.39531
- **Artifacts**:
  - Checkpoint: `results/lego/checkpoints/step_200000.pth`
  - Renders: `results/lego/student_200k/`
  - Metrics CSV: `results/lego/student_200k_metrics.csv`

## Stage 1 — Ray/space alignment
Goal: Ensure student samples match the teacher ray geometry.

  - `lego_response_distill.py` now samples per-pixel rays from `transforms_test.json` and performs volumetric integration along stratified samples.
- [ ] Validate by re-rendering a small subset (e.g., 10 views) and comparing against teacher (target: ≥ PSNR 20 dB before feature distillation).

### Stage 1A — Hash student (2025-09-27)
- **20k run**
  - Config: `configs/lego_response_stage1a_20k.yaml`
  - Artifacts: checkpoints under `results/lego/stage1a_hash20k/checkpoints/`; renders at `results/lego/stage1a_hash20k/eval/`; metrics CSV `logs/lego/stage1a_hash20k/eval_step_020000.csv`
  - Metrics: PSNR 9.61 dB, FPS 2.51, GPU Peak 0.29 GiB, Power Avg 192.83 W
- **50k run**
  - Config: `configs/lego_response_stage1a_50k.yaml`
  - Artifacts: `results/lego/stage1a_hash50k/checkpoints/`, renders `results/lego/stage1a_hash50k/eval/`, metrics CSV `logs/lego/stage1a_hash50k/eval_step_050000.csv`
  - Metrics: PSNR 9.61 dB, FPS 2.44, GPU Peak 0.29 GiB, Power Avg 192.83 W
  - Notes: PSNR unchanged vs. 20k → schedule extension alone insufficient; next steps are to add teacher depth supervision and loss shaping before pushing to 100k.
- **50k run + depth regulariser**
  - Config: `configs/lego_response_stage1a_50k_depth.yaml`
  - Artifacts: `results/lego/stage1a_hash50k_depth/checkpoints/`, renders `results/lego/stage1a_hash50k_depth/eval/`, metrics CSV `logs/lego/stage1a_hash50k_depth/eval_step_050000_depth.csv`
  - Metrics: PSNR 9.61 dB (avg 9.6066), FPS 2.55, GPU Peak 0.29 GiB, Power Avg 192.61 W, depth loss ≈10–12 (ほぼ横ばい)
  - Notes: 教師 depth `.npy` は読み込めているが、学生側 depth とスケールが噛み合わずロスが高止まり。Ray depth のレンジ（約0.001–0.39m）に合わせた正規化と α マスク閾値の再調整を次の実験で試す。
    - Debug aids: 学習ループに `depth_valid_fraction` と `teacher_depth_{min,max,mean}` を記録するメトリクスを追加済み。CSV ログおよび tqdm の後置表示で深度カバレッジを即確認できる。
  - Helper script: `python tools/inspect_teacher_depth.py --config configs/lego_response_stage1a_50k_depth.yaml` で教師 depth のレンジとカバレッジを集計し、再正規化の基準に利用できる。
  - Update: 深度損失はレイごとの near/far で正規化した [0,1] 空間で計算するよう変更。`teacher_depth_norm_{min,max,mean}` と `ray_depth_range_{min,max,mean}` もログ化されるため、スケールずれを定量的に確認可能。
  - Depth weight sweep 1 (w=0.05 → baseline)
    - Logs archived as `training_metrics_depth_w005.csv` for比較。
- **50k run + depth w=0.20 / α=0.5**
  - Config: `configs/lego_response_stage1a_50k_depth_w020.yaml`
  - Artifacts: `results/lego/stage1a_hash50k_depth_w020/checkpoints/`, renders `results/lego/stage1a_hash50k_depth_w020/eval/`, metrics CSV `logs/lego/stage1a_hash50k_depth_w020/eval_step_050000_depth_w020.csv`
  - Training CSV: `logs/lego/stage1a_hash50k_depth_w020/training_metrics.csv`
  - Metrics: PSNR 9.61 dB (avg 9.6066), FPS 2.40, GPU Peak 0.29 GiB, Power Avg 193.34 W, depth loss ≈0.02–0.04（正規化後も依然小さめ）
  - Notes: depth weight を 0.2 に引き上げても PSNR は変化なし。`depth_valid_fraction`=1.0 を維持しつつ、`teacher_depth_norm_mean` ≈0.13–0.19。さらなる重み引き上げ or α 閾値調整が必要。
- **50k run + depth w=0.50 / α=0.6**
  - Config: `configs/lego_response_stage1a_50k_depth_w050.yaml`
  - Artifacts: `results/lego/stage1a_hash50k_depth_w050/checkpoints/`, renders `results/lego/stage1a_hash50k_depth_w050/eval/`, metrics CSV `logs/lego/stage1a_hash50k_depth_w050/eval_step_050000_depth_w050.csv`
  - Training CSV: `logs/lego/stage1a_hash50k_depth_w050/training_metrics.csv`
  - Metrics: PSNR 9.61 dB (avg 9.6066), FPS 2.43, GPU Peak 0.29 GiB, Power Avg 193.18 W, depth loss ≈0.06–0.09
  - Notes: depth を 0.5 まで増やしても PSNR は据え置き。正規化深度平均は ≈0.14–0.20、`depth_valid_fraction` は 1.0 を維持。以降は loss 形式の見直し（smooth L1 など）や特徴蒸留との併用を検討。
- **50k run + depth w=0.50 / α=0.6 / smooth L1**
  - Config: `configs/lego_response_stage1a_50k_depth_w050_smoothl1.yaml`
  - Artifacts: `results/lego/stage1a_hash50k_depth_w050_smoothl1/checkpoints/`, renders `results/lego/stage1a_hash50k_depth_w050_smoothl1/eval/`, metrics CSV `logs/lego/stage1a_hash50k_depth_w050_smoothl1/eval_step_050000_depth_w050_smoothl1.csv`
  - Training CSV: `logs/lego/stage1a_hash50k_depth_w050_smoothl1/training_metrics.csv`
  - Metrics: PSNR 9.61 dB (avg 9.6066), FPS 2.44, GPU Peak 0.29 GiB, Power Avg 193.30 W, depth loss ≈0.007
  - Notes: smooth L1 でも RGB 指標は伸びず。深度正規化＆重み調整のみでは打開できない。Stage 1B への移行と Stage 2（特徴蒸留）強化を優先。
- **Stage 1B prep — KiloUniform student (smoke test)**
  - Config: `configs/lego_response_stage1b_kilo_uniform_smoketest.yaml`
  - Artifacts: checkpoints `results/lego/stage1b_kilo_uniform_smoketest/checkpoints/`
  - Notes: Uniform 2×2×2 grid of local MLPs trains with smooth L1 depth supervision (64 steps) confirming pipeline compatibility. Next run: 50k full experiment `configs/lego_response_stage1b_kilo_uniform.yaml`.
- **Stage 1B — KiloUniform student 50k run**
  - Config: `configs/lego_response_stage1b_kilo_uniform.yaml`
  - Artifacts: checkpoints `results/lego/stage1b_kilo_uniform_50k/checkpoints/`, renders `results/lego/stage1b_kilo_uniform_50k/eval/`, metrics CSV `logs/lego/stage1b_kilo_uniform_50k/eval_step_050000_kilo_uniform.csv`
  - Training CSV: `logs/lego/stage1b_kilo_uniform_50k/training_metrics.csv`
  - Metrics: PSNR 9.61 dB (avg 9.6066), FPS 0.31, GPU Peak 0.17 GiB, Power Avg 154.64 W, depth loss ≈0.25
  - Notes: Quality unchanged vs Stage 1A hash while FPS drops ~8×. Confirms response-only supervision saturates; proceed to Stage 2 feature distillation.

## Stage 2 — Feature distillation
Goal: Inject teacher signal beyond RGB supervision by aligning intermediate features.

- [ ] Export teacher-side features:
  - SH coefficients per Gaussian (pre-multiplied color features).
  - Density/amplitude statistics along rays (can be approximated using occupancy tree queries).
- [ ] Design student-side feature taps:
  - Capture hidden activations after positional encoding & first linear layers.
  - Aggregate per-sample outputs before volumetric integration.
- [ ] Add auxiliary losses (e.g., L2 / cosine) between student activations and teacher features mapped along the same ray samples.
- [ ] Tune loss weights and monitor for training stability (log feature loss curves alongside RGB metrics).

## Stage 3 — Evaluation hardening
- [ ] Automate render + metric export via a single script (collects renders, renames frames, runs metrics, writes CSV).
- [ ] Extend metrics to include LPIPS-vgg, GPU peak memory, and power from teacher stats for parity checks.
- [ ] Maintain a dashboard (CSV → Markdown table) to track improvements per stage.

## Suggested cadence
1. **Ray alignment prototype** on 1/8 resolution subset → metric sanity check.
2. **Full-resolution retrain** with aligned rays (reuse 200k schedule, log intermediate checkpoints).
3. **Feature distillation pilot** on aligned setup (start with SH color supervision).
4. Iterate with ablations (ray-only vs ray+feature) before scaling to Matrix City.
