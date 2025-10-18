# LEGO Feature Distill Baseline (PSNR 22.497 dB)

*保証付き再現フロー — 2025-10-14*

---

## 1. 実験サマリ

| 指標 | 値 |
| --- | --- |
| PSNR | **22.497 dB** |
| SSIM | 0.8266 |
| LPIPS (alex) | 0.1903 |
| FPS (render) | 0.121 |
| GPU peak | 1.54 GiB |
| Power avg | 170 W |
| Steps | 50,000 |

この結果は `configs/lego_feature_teacher_full_rehab_masked_white.yaml` をそのまま実行し、200 frame 白背景評価を行った際の確定ラインです。以下に同値を再現するための固定条件を一枚に集約します。**本ラインの教師特徴は 52 次元（SH + opacity + log_scale）であることに注意してください。**

---

## 2. シーン & 教師

- **Scene**: NeRF Synthetic LEGO（白背景評価）
- **Teacher**: 3D Gaussian Splats, `teacher_mode=gaussian_sh_opacity_logscale`
  - 教師出力: `teacher/outputs/lego/test_white/ours_30000`（RGBA 保存 → 白背景で再合成）
  - Teacher checkpoint: `teacher/checkpoints/lego/point_cloud/iteration_30000/point_cloud.ply`
  - Gaussian 特徴集合（`gaussian_sh_opacity_logscale`）の内訳:
    - **SH**: 4 バンド (DC + AC) の全係数（$4^2 \times 3 = 48$ 次元）
    - **Opacity**: $\alpha$（1 次元）
    - **Scale**: log-scale $(\log s_x, \log s_y, \log s_z)$（3 次元）
  - 合計 52 次元（内部で projector によって自動的に整合）

> **Note:** `feature_pipeline.teacher_mode` を `gaussian_sh_opacity_logscale` に固定すると、教師特徴は SH + opacity + log_scale の 52 次元パックとなり、位置・回転・共分散などは含まれません。

---

## 3. 実行環境

- OS: WSL2 上の Ubuntu（Windows10/11 ホスト）
- Conda env: `kilogs`
- 乱数 seed: `2025`
- GPU/NVML ログ: 有効（peak VRAM / 平均電力を記録）
- コマンド例:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n kilogs \
  python distill/lego_response_distill.py \
  --config configs/lego_feature_teacher_full_rehab_masked_white.yaml
```

---

## 4. 学習設定

- **ステップ**: 50,000
- **Optimizer**: Adam
- **初期学習率**: 5e-4
- **スケジュール**: Cosine decay, 40k step 時点で 0.5、以降 Cosine
- **Gradient Clip**: 1.0
- **EMA**: 0.999（学生 & projector 共通）
- **train_ray_chunk**: 1,024

---

## 5. 学生モデル（KiloNeRF 系 Multi-MLP）

- `type`: `kilo_uniform_mlp`
- **セル分割**: 6×6×6（216 Sub-MLP）
- **MLP**: 各セル `hidden_dim=128`, `num_layers=4`, ReLU
time
- **Boundary blend**: 有効
  - `boundary_blend_margin = 0.08`
  - `boundary_mask_threshold = 0.60`
  - `boundary_mask_soft_transition = 0.10`

---

## 6. Feature Pipeline（教師→学生蒸留）

- Enabled (`feature_pipeline.enabled = true`)
- **Teacher mode**: `gaussian_sh_opacity_logscale`（上記 52 次元）
- **Student projector**:
  - 入力: penultimate features (128 次元)
  - 構造: Linear 128→128（LayerNorm あり, activation=identity）
  - projector_output_dim は教師次元に自動上書き → 実際には 52 次元に射影
- **比較空間**: `teacher`（教師そのものと同次元で比較）
- **Boundary mask**: `threshold=0.60`, `soft_transition=0.10`

> `allow_dim_mismatch=false` のため、教師の 52 次元に projector 出力が強制的に揃えられます。

---

## 7. 損失設定

| 損失 | 種類 | 重み | 備考 |
| --- | --- | --- | --- |
| Color | L2 | 1.0 | ─ |
| Depth | Smooth L1 | 0.3 | `alpha_threshold=0.6` |
| Opacity | L1 | 0.25 | `target=0.05`, `target_weight=0.35`, `background_threshold=0.05` |
| Feature L2 | L2 | 0.05 | Warmup 4k steps |
| Feature Cos | Cosine | 0.01 | Warmup 4k steps |

---

## 8. レンダ & 評価

- **frames**: 200（テストセット全フレーム）
- **render_chunk**: 8192
- **指標**: PSNR / SSIM / LPIPS(alex)
- **背景**: 白背景再合成を主指標（黒背景は診断用途）
- **コマンド**（例）:

```bash
PYTHONUNBUFFERED=1 conda run --no-capture-output -n kilogs \
  python distill/render_student.py \
  --config configs/lego_feature_teacher_full_rehab_masked_white.yaml \
  --checkpoint results/lego/feat_t_full/runs/teacher_full_rehab_masked_white/checkpoints/step_050000.pth \
  --render-dir results/lego/feat_t_full/renders/teacher_full_rehab_masked_white/step_050000 \
  --store-rgba --enable-nvml

PYTHONUNBUFFERED=1 conda run --no-capture-output -n kilogs \
  python tools/run_eval_pipeline.py \
  --render-root results/lego/feat_t_full/renders/teacher_full_rehab_masked_white/step_050000 \
  --teacher-root teacher/outputs/lego/test_white/ours_30000 \
  --background white \
  --frames 200 \
  --lpips alex \
  --method-name FeatureDistill_Full_Rehab_Masked_White
```

---

## 9. ログ & 再現性記録

記録必須メタ情報：

- commit SHA（学習・レンダ実行時点）
- `configs/lego_feature_teacher_full_rehab_masked_white.yaml` のスナップショット
- seed (`2025`)
- `train_ray_chunk=1024`, `render_chunk=8192`
- `frames=200`, `LPIPS=alex`, 背景=white
- 取得指標（PSNR/SSIM/LPIPS/FPS/GPU peak/Power）
- `logs/lego/feat_t_full/runs/teacher_full_rehab_masked_white/training_metrics.csv`（feature mask fraction, feature dims, depth loss の推移チェック）

---

## 10. よくある落とし穴

1. **Teacher アセット欠損**: `teacher/outputs/...` と `teacher/checkpoints/...` が揃っているか実行前に確認。
  - `python tools/check_teacher_assets.py --verbose` で 200 枚分の RGBA/gt/depth, `render_stats.json`, `transforms_test_white.json`, `point_cloud.ply` が揃っているか自動検証する。
  - 最新確認: 2025-10-14 14:15 JST 時点で `[PASS] Teacher assets are complete and consistent.` を取得済み。
2. **Feature mask collapse**: `feature_mask_fraction` が 0 に落ちないか 1–2k step で要チェック。
3. **CSV 列ずれ**: 再開時にログ列が崩れたら `append_metrics()` 周辺の改変有無を確認し、破損ファイルは再生成。
4. **TensorBoard**: `--host 127.0.0.1` で起動しないとブラウザが繋がらない。

---

## 11. カスタマイズ指針

- **教師特徴を限定したい場合**: `feature_pipeline.teacher_components` に `("sh", "opacity", "log_scale")` 等を明示（52 次元を基準に改変）。
- **Projector 次元を変えたい場合**: `allow_dim_mismatch=true` を使えば教師 52 次元とは別に比較空間 K を固定できる。
- **α 正則化調整**: `loss.opacity.target`, `loss.opacity.target_weight`, `loss.opacity.background_threshold` をセットでスイープ。

このドキュメントに沿って実行すれば、PSNR 22.5 dB ラインを揺らぎなく再現できます。変更を加える際は差分を本ファイルに追記し、再現条件を最新化してください。