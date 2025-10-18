# Research Notes — LEGO Feature Distillation (teacher‑space, SH52)

**Status:** Active baseline run (teacher space, 52D) — *2025‑10‑17 JST*

---

# kilogs / KiloNeRF – research notes

Last updated: <today>

## Goal

Stable, reproducible pipeline for rendering & evaluating student vs teacher on **LEGO test_white** with KiloNeRF CUDA extensions under WSL + Conda.

---

## Machine / stack snapshot

* **GPU**: NVIDIA RTX A4500 (CUDA 12.x driver; nvcc 11.8 toolchain)
* **OS**: WSL2 Ubuntu
* **Conda env**: `kilogs` (Python 3.10)
* **PyTorch**: 2.0.1 **+cu118** (pip wheels)
* **Torchvision/Torchaudio**: 0.15.2 / 2.0.2 (cu118 wheels)
* **NumPy**: 1.26.4 (pinned)

Why these pins: keep compatibility with the prebuilt **cu118** wheels and with custom CUDA extensions.

---

## Permanent imports (no PYTHONPATH needed)

To make `kilonerf`, `kilonerf_cuda` & friends importable without fiddling with env vars:

* `site-packages/kilonerf_local.pth`:

  ```
  /mnt/d/imaizumi/kilonerf
  /mnt/d/imaizumi
  /mnt/d/imaizumi/nerf
  ```
* `site-packages/kilonerf_cuda_local.pth`:

  ```
  /mnt/d/imaizumi/kilonerf/cuda
  ```

> These `.pth` files append search paths at interpreter start; they survive shell restarts and VSCode terminals.

Optional hardening:

```bash
# make the .pth files read-only (undo with chmod +w or chattr -i)
chmod a-w $(python -c 'import site,os; print("%s/kilonerf_local.pth"%site.getsitepackages()[-1])')
chmod a-w $(python -c 'import site,os; print("%s/kilonerf_cuda_local.pth"%site.getsitepackages()[-1])')
# if you really want them immutable on ext4:
# sudo chattr +i <path-to>/kilonerf_local.pth <path-to>/kilonerf_cuda_local.pth
```

---

## Conda env activation hooks (per-env)

Create `etc/conda/{activate.d,deactivate.d}` scripts inside the `kilogs` env so terminals and VSCode inherit the setup.

**Activate: `.../activate.d/kilogs.sh`**

```bash
# KiloGS / KiloNeRF environment glue
export KILONERF_DIR=${KILONERF_DIR:-/mnt/d/imaizumi/kilonerf}
# Prefer nvcc hint; fall back to /usr/local/cuda-11.8
export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc 2>/dev/null || echo /usr/local/cuda-11.8/bin/nvcc)")")}
# Torch shared libs (libc10.so, etc.)
TORCH_LIB=$(python - <<'PY'
import torch, os
print(os.path.join(os.path.dirname(torch.__file__), 'lib'))
PY
)
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
# Determinism / misc
export PYTHONHASHSEED=2025
export NVIDIA_TF32_OVERRIDE=0
export KILOGS_DISABLE_NVML=1
# Optional: visible confirmation once per terminal
export KILOGS_ENV_READY=1
```

**Deactivate: `.../deactivate.d/kilogs.sh`**

```bash
unset KILONERF_DIR CUDA_HOME KILOGS_DISABLE_NVML NVIDIA_TF32_OVERRIDE KILOGS_ENV_READY
# Strip the torch lib prefix once (best-effort)
TORCH_LIB=$(python - <<'PY'
import torch, os
print(os.path.join(os.path.dirname(torch.__file__), 'lib'))
PY
)
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed "s#${TORCH_LIB}:##")
unset TORCH_LIB
```

> Don’t mark these variables `readonly` here; deactivation needs to unset them. Use the **locked subshell** wrapper (below) when you want immutability within a session.

---

## Locked subshell (immutable env for this session)

If you want variables unchangeable while working:

`bin/kilogs-shell`:

```bash
#!/usr/bin/env bash
set -euo pipefail
source ~/miniconda/etc/profile.d/conda.sh
conda activate kilogs
# Export again then lock within this subshell
export KILONERF_DIR=/mnt/d/imaizumi/kilonerf
export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
TORCH_LIB=$(python - <<'PY'
import torch, os
print(os.path.join(os.path.dirname(torch.__file__), 'lib'))
PY
)
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
export PYTHONHASHSEED=2025 NVIDIA_TF32_OVERRIDE=0 KILOGS_DISABLE_NVML=1
readonly KILONERF_DIR CUDA_HOME LD_LIBRARY_PATH PYTHONHASHSEED NVIDIA_TF32_OVERRIDE KILOGS_DISABLE_NVML
exec "$SHELL" -i
```

Usage:

```bash
chmod +x bin/kilogs-shell
./bin/kilogs-shell   # opens an interactive shell where those vars are readonly
```

---

## PyTorch / CUDA pins (pip)

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
python - <<'PY'
import torch, numpy as np
print('cuda?', torch.cuda.is_available(), 'torch', torch.__version__, 'numpy', np.__version__)
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY
```

## KiloNeRF CUDA extension

* Build (only after torch is installed):

  ```bash
  cd /mnt/d/imaizumi/kilonerf/cuda
  # Preferred: modern pip editable without isolation so it sees torch headers
  python -m pip install -e . --no-build-isolation --no-deps
  ```
* Import test:

  ```bash
  python - <<'PY'
  import kilonerf_cuda, torch
  print('kilonerf_cuda OK?', hasattr(kilonerf_cuda, 'init_stream_pool'))
  print('cuda available?', torch.cuda.is_available())
  PY
  ```

---

## Fixes applied (bugs & patches)

1. **`evaluate_student_metrics.py` → image loading**

   * Original error: `TypeError: expected np.ndarray (got numpy.ndarray)` when numpy 2.0 landed in one path.
   * Resolved by pinning NumPy 1.26.4 and using robust loader when needed.

2. **PSNR tensor reshape**

   * In `/mnt/d/imaizumi/3dgs/utils/image_utils.py`:

     * Replace `.view(img1.shape[0], -1)` with `.reshape(img1.shape[0], -1)` to avoid stride issues.

3. **RGBA `.npz` save path**

   * `distill/render_student.py` flattened save path to avoid nested `rgba_npz/renders/` hiccup:

     * `file_stub.with_suffix(".npz") → Path(file_stub.name).with_suffix(".npz")`

4. **Import paths**

   * Added `.pth` files so `kilonerf`, `run_nerf_helpers`, `kilonerf_cuda` import cleanly without `PYTHONPATH`.

---

## Rendering & evaluation – working recipes

### Render N frames with RGBA (student)

```bash
OUTDIR="results/lego/feat_t_full/runs/teacher_full_rehab_masked_white/renders/step_001000"
CKPT="results/lego/feat_t_full/runs/teacher_full_rehab_masked_white/checkpoints/step_001000.pth"
FRAMES=8  # e.g., 8 or 200

rm -rf "$OUTDIR/rgba_npz"
python distill/render_student.py \
  --config configs/lego_feature_teacher_full_rehab_masked_white.yaml \
  --checkpoint "$CKPT" \
  --output-dir "$OUTDIR" \
  --chunk 8192 --store-rgba --start-frame 0 --max-frames $FRAMES

# Optional: cleanup nested folder if created by older code
rmdir "$OUTDIR/rgba_npz/renders" 2>/dev/null || true
```

### Self-check (teacher vs teacher)

```bash
python tools/evaluate_student_metrics.py \
  teacher/outputs/lego/test_white/ours_30000/renders \
  teacher/outputs/lego/test_white/ours_30000/renders \
  --output-json /tmp/teacher_selfcheck.json
# Expect: PSNR=inf, SSIM=1, LPIPS=0
```

### Student vs Teacher (PNG direct quick check)

```bash
STUD_DIR=$(find "$OUTDIR" -type d -path '*/renders/renders' | head -n1)
python tools/evaluate_student_metrics.py \
  "$STUD_DIR" \
  teacher/outputs/lego/test_white/ours_30000/renders \
  --output-json "$OUTDIR/metrics_png_direct_${FRAMES}.json"
```

### Full pipeline (RGBA recomposition; white bg)

```bash
python tools/run_eval_pipeline.py \
  "$OUTDIR" \
  --teacher-renders teacher/outputs/lego/test_white/ours_30000/renders \
  --background white \
  --method-name FeatureDistill_SH52_teacherSpace_step1000_${FRAMES}f \
  --summary "$OUTDIR/metrics_summary_${FRAMES}.csv" \
  --render-stats "$OUTDIR/render_stats_${FRAMES}.json"
```

**Sample results (so far)**

* 1 frame PNG-direct: PSNR≈8.88, SSIM≈0.749, LPIPS≈0.269
* 8 frames PNG-direct: PSNR≈8.76, SSIM≈0.748, LPIPS≈0.271

---

## VSCode tips

* Workspace `settings.json`:

  ```json
  {
    "python.defaultInterpreterPath": "/home/araki/miniconda/envs/kilogs/bin/python",
    "terminal.integrated.env.linux": {
      "CONDA_DEFAULT_ENV": "kilogs"
    }
  }
  ```
* Optional Task to open locked shell:

  ```json
  {
    "label": "kilogs shell (locked)",
    "type": "shell",
    "command": "${workspaceFolder}/bin/kilogs-shell",
    "problemMatcher": []
  }
  ```

---



## Troubleshooting checklist

* `import kilonerf_cuda` fails → ensure `.pth` file exists **and** `LD_LIBRARY_PATH` includes torch lib dir.
* `CUDA error: no CUDA-capable device` → check `nvidia-smi` and that the pip wheels are **+cu118**.
* `view size is not compatible` → confirm PSNR patch to `.reshape(...)`.
* Evaluation finds zero frames → verify student PNG root (often `*/renders/renders`).

---

## Next

* Scale to 200 frames; repeat at higher checkpoints (5k/10k) and log `metrics_summary_*.csv` rows.
* Optionally run black-background eval and compare.
* If this becomes standard, bake the activation scripts + wrapper into repo under `env/` and symlink from env.


## TL;DR

* **Teacher‑space比較で再構築。** `compare_space: teacher`、教師特徴 **SH+α+logscale = 52D** をそのまま使用。
* **Projector** は **Linear(128→52)**。ログ上でも `projector in/out=(128->52)` を確認。
* **二段フェーズ**: 1–2000 step = projector‑only warmup（実装側で feature 強制ON）→ 2001–50000 step = joint（student+projector）。
* **ログ/パスは `feat_t_full` で統一**：`logs/lego/feat_t_full/...` / `results/lego/feat_t_full/...`。
* 50k 完走後、**白背景 200frames** をレンダ→評価。**運用合格ゲート**: *PSNR ≥ 22.5 dB*, *LPIPS ≤ 0.19*（white）。

---

## Assets & Paths

* **Teacher renders**: `teacher/outputs/lego/test_white/ours_30000`
* **Teacher depth**: `teacher/outputs/lego/test_white/ours_30000/depth`
* **Teacher point cloud**: `teacher/checkpoints/lego/point_cloud/iteration_30000/point_cloud.ply`
* **Config (baseline)**: `configs/lego_feature_teacher_full_rehab_masked_white.yaml`
* **Run dirs**:

  * TensorBoard: `logs/lego/feat_t_full/runs/teacher_full_rehab_masked_white/tensorboard`
  * CSV: `logs/lego/feat_t_full/runs/teacher_full_rehab_masked_white/training_metrics.csv`
  * Checkpoints: `results/lego/feat_t_full/runs/teacher_full_rehab_masked_white/checkpoints/`

---

## Canonical Config (key excerpts)

```yaml
feature_pipeline:
  enabled: true
  teacher_mode: gaussian_sh_opacity_logscale  # 52D
  compare_space: teacher
  projector: { in_dim: 128, out_dim: 52 }

train:
  max_steps: 50000
  phases:
    - { name: projector_warmup, end_step: 2000, optimize: [projector] }
    - { name: joint_training,     end_step: 50000, optimize: [student, projector] }
  lr: 5e-4
  lr_schedule: cosine
  lr_schedule_steps: 50000
  gradient_clip_norm: 1.0
  ema_decay: 0.999

loss:
  color:   { type: l2, weight: 1.0 }
  depth:   { type: smooth_l1, weight: 0.3, alpha_threshold: 0.6 }
  opacity: { type: l1, weight: 0.25, target: 0.05, target_weight: 0.35,
             warmup_steps: 2000, schedule: cosine, schedule_duration: 6000,
             background_threshold: 0.05 }
  feature_l2: { weight: 0.05, warmup: { steps: 4000, type: linear, start_weight: 0.0 } }
  feature_cos:{ weight: 0.01, warmup: { steps: 4000, type: linear, start_weight: 0.0 } }
```

> **Note:** スキーマ外キー（例: `freeze_student`, `force_feature_on`, `phase_feature_scale`）は削除済み。projector-only フェーズ中の feature 強制ONは実装側で自動処理。

---

## How to Run

```bash
cd /mnt/d/imaizumi/kilogs
export PYTHONHASHSEED=2025
conda run --no-capture-output -n kilogs \
  python -m distill.lego_response_distill \
  --config configs/lego_feature_teacher_full_rehab_masked_white.yaml
```

### Stage2 quickwin run（2025-10-18, `recover_v2_stage2`）

このドキュメントで現在運用中の 5k スモークは下記テンプレートで再現可能。

```bash
cd /mnt/d/imaizumi/kilogs
source ~/.bashrc                      # PATH 修正を反映させる（要 1 回）
conda deactivate 2>/dev/null || true
conda activate kilogs

# 進捗ログを細かく書き出し、TensorBoard を 15 秒ごとに flush
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=2025
export KILOGS_LOG_INTERVAL=50
export KILOGS_TENSORBOARD_FLUSH_SECS=15

conda run --no-capture-output -n kilogs \
  python -m distill.lego_response_distill \
  --config configs/generated/lego_feature_teacher_full_quickwin_relaxed_alpha045_recover_v2_stage2.yaml \
  --max-steps 5000
```

**期待される初期ログ**

```
[config] max_steps override 5000 is below config max_steps 50000; using override regardless.
[logging] TensorBoard writer initialised → logs/lego/feat_t_full/runs/teacher_full_quickwin_relaxed_alpha045_recover_v2/tensorboard
[phase] Projector-only warmup forcing feature loss on.
```

**正常起動チェック（ログ）**

* `loaded gaussian teacher features with dimension 52`
* `comparison feature dim=52, projector in/out=(128->52)`
* `student-space policy bypassed for compare_space='teacher'`
* `Enter projector_warmup ... optimize=['projector']` → `Enter joint_training ...`

---

## Monitoring

### TensorBoard

```bash
conda run --no-capture-output -n kilogs \
  tensorboard --logdir logs/lego/feat_t_full/runs/teacher_full_rehab_masked_white/tensorboard \
  --host 127.0.0.1 --port 6006
```

### Stage2 run の TensorBoard

```bash
conda run --no-capture-output -n kilogs \
  tensorboard --logdir logs/lego/feat_t_full/runs/teacher_full_quickwin_relaxed_alpha045_recover_v2/tensorboard \
  --host 127.0.0.1 --port 6006 \
  --load_fast=false
```

> 先に既存のポート 6006 プロセスを落とす場合は `ss -tulpn | grep 6006` → `kill <pid>`。

**Pin推奨スカラー**

* `loss/{total,color,opacity,depth,feature_l2,feature_cos}`
* `feature_mask/fraction`
* `opacity/target_weight_effective`

### CSV quick checks

ファイル: `logs/lego/feat_t_full/runs/teacher_full_rehab_masked_white/training_metrics.csv`

* **マスク健全性**

```bash
awk -F, 'NR==1{for(i=1;i<=NF;i++){if($i=="feature_mask_fraction") mf=i}} NR>1{c++; s+=$mf} END{print "mask_fraction_avg=",s/c}' \
  logs/lego/feat_t_full/runs/teacher_full_rehab_masked_white/training_metrics.csv
```

* **Opacity重みの単調性**

```bash
awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="opacity_target_weight_effective") c=i} NR>2{if($c<prev) d++} {prev=$c} END{print "opacity_weight_drops=",d}' \
  logs/lego/feat_t_full/runs/teacher_full_rehab_masked_white/training_metrics.csv
```

---

## Render & Evaluate (white background)

### Render

```bash
python distill/render_student.py \
  --checkpoint results/lego/feat_t_full/runs/teacher_full_rehab_masked_white/checkpoints/step_050000.pth \
  --chunk 8192 --store-rgba --enable-nvml
```

### Evaluate

```bash
python tools/run_eval_pipeline.py <render_dir> \
  --teacher-renders teacher/outputs/lego/test_white/ours_30000/renders \
  --background white \
  --method-name FeatureDistill_SH52_teacherSpace_50k \
  --summary metrics_summary.csv --render-stats render_stats.json
```

**Success gate (white)**: **PSNR ≥ 22.5 dB**, **LPIPS ≤ 0.19**、SSIM ≥ 0.82 目安。

---

## Troubleshooting (short)

* **スキーマエラー（Unknown keys in train.phases）**: YAML から `freeze_student` / `force_feature_on` / `phase_feature_scale` を削除。
* **tqdmが0%のまま**: 初回 KiloNeRF フォワード/カーネルJIT中は見かけ上停止。`[init] First KiloNeRF forward pass finished` 後に進捗開始。`nvidia-smi` で稼働確認推奨。
* **ログファイルが生成されない**: `tee` で標準出力をファイルへ。例 `PYTHONHASHSEED=2025 stdbuf -oL -eL python ... |& tee logs/.../train.log`。
* **進捗バーが更新されない/ETAが見えない**: `stdbuf` を併用しつつ学習コマンドを直接見る。追加で `KILOGS_LOG_INTERVAL=10` を指定してログ頻度を上げると ETA 更新が早くなる。
* **mask collapse**: `feature_mask_fraction` が極小 (<0.05) を連続する場合は中止。マスク閾値・feature重みスケジュールを緩和して再走。
* **Opacityの再落下**: `opacity_target_weight_effective` が履歴より低下したらスケジューラ/再開状態を点検。ヒステリシス設定の破れを疑う。
* **GPUを見つけられず学習が極端に遅い**: `which python` で `.../envs/kilogs/bin/python` を指しているか確認。誤って `(base)`/`3dgs` のままなら `source ~/miniconda/etc/profile.d/conda.sh && conda activate kilogs` を実行。

---

## Checklists

**Pre‑flight**

* [ ] `PYTHONHASHSEED=2025`
* [ ] Teacher assets 一式が存在
* [ ] `compare_space=teacher`, projector in/out=(128→52) が起動ログに出る

**10k smoke GO**

* [ ] `feature_mask_fraction ≥ 0.25`
* [ ] `opacity_target_weight_effective` 単調↑
* [ ] NaN/Inf なし、color loss 下降

**50k submission**

* [ ] Render 200 frames (white)
* [ ] PSNR ≥ 22.5 / LPIPS ≤ 0.19
* [ ] metrics & config snapshot を保存

---

## Change Log (this doc)

* 2025‑10‑17: 新規作成。teacher‑space SH52 ベースラインの運用ノートを簡潔に再編し、`feat_t_full` にパス統一。
* 2025‑10‑18: 進捗バー停止・CSV未生成の原因が誤環境（3dgs）だった件を追記。`tee`/`stdbuf` の活用と `KILOGS_LOG_INTERVAL` 設定をメモ。
* 2025‑10‑18: Stage2 クイックラン（5k）の再現テンプレ（環境変数・コマンド・TensorBoard 手順）を記録。
* 2025‑10‑18: Stage2 quickwin が `--max-steps 200` で早期終了していたことを確認し、5k 再走と後続レンダ・評価の再実施をToDo化。
* 2025‑10‑19: 学習進捗バーを短縮フォーマット `(特徴蒸留_教師52d) [n/50000]: ...` に統一し、TTY 未接続時も手動更新ラインが同形式で流れるよう `distill/lego_response_distill.py` を整理。
* 2025‑10‑19: TensorBoard 出力が 88B のまま成長しない件を調査。既存 run ディレクトリが過去ログを引きずっていたため、ランごとの退避 (`archive/<timestamp>/`) と `--max-steps` 短縮による動作確認テンプレを整備。

---

## 2025‑10‑18 Run Progress Snapshot

* 既存ログが欠落していたため `logs/lego/feat_t_full/runs/teacher_full_rehab_masked_white/` を再生成。
* `PYTHONHASHSEED=2025 stdbuf -oL -eL python ... |& tee .../train.log` で学習を再開し、リアルタイム進捗＋ファイル出力を両立。
* 進捗バー 0% 停滞は kilogs 環境が有効化されていなかったことが原因 (`which python` → `envs/3dgs`)。
  * 修正: `source ~/miniconda/etc/profile.d/conda.sh` 後に `conda activate kilogs`、再実行。
* `KILOGS_LOG_INTERVAL=10` を設定し、ETA・CSV 更新間隔を短縮。
* 現在は教師フレーム読み込み完了後に学習ステップが進行中（TQDM で 1/50000 以降を確認）。      

---

## 2025-10-18 Quickwin Stage2 スモークラン（5k）

* config: `configs/generated/lego_feature_teacher_full_quickwin_relaxed_alpha045_recover_v2_stage2.yaml`
* run dir: `logs/lego/feat_t_full/runs/teacher_full_quickwin_relaxed_alpha045_recover_v2`
* CSV: `.../training_metrics.csv`（log_interval=50 で 100+ 行見込）
* TensorBoard: `.../tensorboard`（flush 15s）
* 開始コマンドは上記テンプレ使用。`python -m distill.lego_response_distill` 形式に切り替えて `ModuleNotFoundError: distill` を回避。
* 進捗: 2025-10-18 14:22 JST 時点で step 5000 / final checkpoint `results/.../checkpoints/step_005000.pth` を保存。イベントファイルは 3 KB 程度で全スカラー 1 ステップ更新を確認。
* 既存 TensorBoard (`teacher_full_rehab_masked_white`) が 6006 を掴んでいたため kill → 再起動で解消。
* `.bashrc` を修正し `which python` が `envs/kilogs/bin/python` を指すことを確認後に実行。

#### 2025-10-18 夕方レビュー

- 最新ログを再確認したところ CLI 側で `--max-steps 200` を指定してしまい、`train.log` も TensorBoard も 200 step 終了（`step_000200.pth` のみ保存）で止まっていた。
- `logs/lego/feat_t_full/runs/teacher_full_quickwin_relaxed_alpha045_recover_v2/tensorboard/events.out.tfevents.*` は生成済みで、6006 ポートの TensorBoard から閲覧できるがステップ範囲は 0-200 に限定されている。
- **ToDo**: 5k 再走。既存 run ディレクトリを必要に応じて `archive/` へ退避し、`python -m distill.lego_response_distill ... --max-steps 5000` で再実行 → `step_005000.pth` の生成を確認後、レンダリングと評価パイプラインを再開する。

---

## 2025‑10‑19 Progress Bar Stabilization

* `distill/lego_response_distill.py` の `create_progress` と手動レンダラーを揃え、**TTY 接続時は tqdm、ログパイプ経由でも同一フォーマットが残る**ように調整。
* 進捗出力は `(特徴蒸留_教師52d) [48/50000]:   0%|          | 48/50000 [00:05<1:43:05]` 形式に固定。タブ切り替えで改行が増えず、ログファイルでも1行表示を維持。
* 教師フレーム読み込み進捗が初期化フェーズで再び表示されることを確認。`Loaded teacher frames (200)` 行の直後から学習バーが滑らかに更新。

---

## 2025‑10‑19 TensorBoard/CSV Reset Playbook

### 症状

* `.tfevents` が **88 B** のまま増えず、TensorBoard にグラフが描画されない。
* `training_metrics.csv` が過去ランの行を引き継いだまま新行が書かれない（重複ヘッダ・列崩れ）。

### 原因

1. 途中チェックポイントからの再開で `global_step` が 2k〜3k にジャンプ → `append_metrics` がまだ走っていない。
2. 既存 run ディレクトリに旧 `.tfevents` / CSV / checkpoint が残っており、新規ログと混線。

### 対処テンプレ（全リセット）

```bash
cd /mnt/d/imaizumi/kilogs
stamp=$(date +%Y%m%d_%H%M%S)
mkdir -p archive/$stamp
mv logs/lego/feat_t_full/runs/teacher_full_rehab_masked_white/{tensorboard,training_metrics.csv,tb_debug.log,train.log} \
  archive/$stamp/ 2>/dev/null
mv results/lego/feat_t_full/runs/teacher_full_rehab_masked_white/checkpoints \
  archive/$stamp/ 2>/dev/null

PYTHONHASHSEED=2025 KILOGS_LOG_INTERVAL=1 KILOGS_TENSORBOARD_FLUSH_SECS=10 \
stdbuf -oL -eL python -m distill.lego_response_distill \
  --config configs/lego_feature_teacher_full_rehab_masked_white.yaml \
  --max-steps 1000 \
  |& tee logs/lego/feat_t_full/runs/teacher_full_rehab_masked_white/train.log
```

* `--max-steps` を 1000 以下にして動作確認 → TensorBoard で loss が描画されるのを確認。
* 問題なければ `--max-steps` を外し本走。

### デバッグ補助

* `KILOGS_DEBUG_TB=1` で学習ループ先頭や `append_metrics` 呼び出しを `tb_debug.log` に記録。
* `stat logs/.../tensorboard/events.out.tfevents.*` でファイルサイズの増分を監視。
* `tail logs/.../training_metrics.csv` で最新行が追加されているか確認。

### 注意

* CSV はランごとにヘッダが変化するため、過去ファイルと同一ディレクトリで追記させない。
* `tensorboard --logdir <run>/tensorboard` は常に新しいフォルダを指しているか確認（古い `logdir` を開くと空に見える）。

---

## Results — 50k (white)

実行: 2025‑10‑18、既存のレンダ出力を評価（再合成→指標計測）

- run_dir: `results/lego/feat_t_full/runs/teacher_full_rehab_masked_white/renders/step_050000`
- teacher: `teacher/outputs/lego/test_white/ours_30000/renders`
- コマンド:

```bash
python tools/run_eval_pipeline.py \
  results/lego/feat_t_full/runs/teacher_full_rehab_masked_white/renders/step_050000 \
  --teacher-renders teacher/outputs/lego/test_white/ours_30000/renders \
  --background white \
  --method-name FeatureDistill_SH52_teacherSpace_50k \
  --summary results/lego/feat_t_full/runs/teacher_full_rehab_masked_white/renders/step_050000/metrics_summary.csv \
  --render-stats results/lego/feat_t_full/runs/teacher_full_rehab_masked_white/renders/step_050000/render_stats.json
```

計測値（200枚）:

- PSNR: 9.6066
- SSIM: 0.7611
- LPIPS: 0.2643
- avg_fps: 0.0957, GPU peak ≈ 3.57 GiB, power ≈ 187.6 W

判定: 現行ゲート未達（PSNR < 22.5, LPIPS > 0.19）。

Next steps（短期）:

1) 目視チェック（学生レンダ）

```bash
xdg-open results/lego/feat_t_full/runs/teacher_full_rehab_masked_white/renders/step_050000/renders/00000.png 2>/dev/null || true
```

2) 再合成の影響切り分け（プレ合成PNGを直接比較）

```bash
python tools/evaluate_student_metrics.py \
  results/lego/feat_t_full/runs/teacher_full_rehab_masked_white/renders/step_050000/renders \
  teacher/outputs/lego/test_white/ours_30000/renders \
  --output-json results/lego/feat_t_full/runs/teacher_full_rehab_masked_white/renders/step_050000/metrics_direct.json
```

3) カメラ整合性の確認（YAMLの `data.camera_json` が teacher test_white に一致していること）

```bash
grep -n "camera_json" configs/lego_feature_teacher_full_rehab_masked_white.yaml
```

4) チェックポイント/設定の齟齬確認（hidden_dim, grid_resolution, projector入出力次元）

上記の切り分けで問題点を特定し、必要なら再レンダ/再学習を検討。

## 2025-10-18 — WSL Kilogs bring-up (Raphaelログ)
- Torch stack: torch 2.0.1+cu118 / torchvision 0.15.2+cu118 / torchaudio 2.0.2+cu118
- NumPy: 1.26.4（2.x 非互換問題を回避）
- 修正:
  - KiloNeRF CUDA 拡張をビルド＆読み込み（`kilonerf_cuda`）
  - `libc10.so` 対策: `LD_LIBRARY_PATH` に `torch/lib` を追加
  - `sys.path`/`.pth` で `kilonerf` と `nerf` を解決
  - 画像評価: `psnr` の `view` → `.flatten(1)` に置換（メモリ配置非連続対策）
  - `rgba_npz` のファイル名: `file_stub.with_suffix(".npz")` → `Path(file_stub.name).with_suffix(".npz")`
- レンダリング: step_001000 / 8 frames 完了
  - metrics（白背景, teacher=ours_30000）: PSNR ≈ 8.76 / SSIM ≈ 0.748 / LPIPS ≈ 0.271
  - `tools/run_eval_pipeline.py` で PNG 直比較も動作確認
- Git:
  - 初期化・.gitignore 整備（renders/ rgba_npz/ など重量物を除外）
  - ユーザー名/メール設定（ローカル）
  - SSH 鍵 → GitHub 登録 → Private リポへ push 成功
  - タグ: `exp-lego-fdistill-001000-8f-seed2025`
  - オフラインバックアップ: `git bundle` 作成（`~/kilogs-YYYY-MM-DD.bundle`）
- 次のアクション:
  - 200 frames レンダを継続、完了時にタグ `exp-lego-fdistill-001000-200f-seed2025` を付与
  - `exp_snap <タグ名>` で再現スナップ取って push（下のエイリアス参照）


## 2025-10-18 — WSL Kilogs bring-up (Raphaelログ)
- Torch: 2.0.1+cu118 / TV: 0.15.2+cu118 / TA: 2.0.2+cu118 / NumPy: 1.26.4
- CUDA 拡張: `kilonerf_cuda` をビルド・読込成功（libc10.so は LD_LIBRARY_PATH で解決）
- import 解決: `.pth` に /mnt/d/imaizumi/kilonerf /mnt/d/imaizumi /mnt/d/imaizumi/nerf を追加
- PSNR 関数の `.view` → `.flatten(1)` に変更（非連続テンソル対応）
- `rgba_npz` の保存名: `file_stub.with_suffix('.npz')` → `Path(file_stub.name).with_suffix('.npz')`
- 単枚～8枚評価 OK（白背景, teacher ours_30000）：PSNR ≈ 8.76 / SSIM ≈ 0.748 / LPIPS ≈ 0.271
- Git: Private リポに push 済み＋タグ `exp-lego-fdistill-001000-8f-seed2025`、bundle も保存
- 次: 200 frame レンダ完了後に新タグでスナップ（`exp_snap <TAG>` 予定）
