# Research Notes — LEGO Feature Distillation (teacher‑space, SH52)

**Status:** Active baseline run (teacher space, 52D) — *2025‑10‑17 JST*

---

# kilogs / KiloNeRF – research notes

Last updated: 2025-10-18 JST

## Goal

Stable, reproducible pipeline for rendering & evaluating student vs teacher on **LEGO test_white** with KiloNeRF CUDA extensions under WSL + Conda.

*備忘*: デバッグや再実行の前に必ず `conda activate kilogs` を実行し、環境が有効化されていることを確認する。
*備忘*: コード内のコメントや手元メモはすべて日本語で記載する。英語コメントは避ける。
---

## 2025-10-27 — 単視点 v5 50K ラン実績

* 事前処理: `rm -rf logs/lego/single_view_overfit_v5 results/lego/single_view_overfit_v5` → `conda run -n kilogs tensorboard --logdir logs/lego/single_view_overfit_v5/tensorboard --host 127.0.0.1 --port 6006` を実施してログをリセットしモニタを起動。
* 50K ラン: `PYTHONHASHSEED=2025 CUBLAS_WORKSPACE_CONFIG=:4096:8 conda run -n kilogs python -m distill.lego_response_distill --config configs/generated/lego_single_view_overfit_v5.yaml` で Canonical KiloNeRF 構造（hidden_dim 64 / 2 層）を維持したまま学習ステップを 50K まで延伸。
* レンダ & 評価: `render_student` で `step_050000.pth` を 1 フレーム出力（保存先 `results/lego/single_view_overfit_v5/eval_single_view_step050000_view000/`、avg_fps ≈0.31・GPU peak ≈1.18 GiB）。その後 `tools/evaluate_student_metrics.py results/.../renders/renders teacher/outputs/lego/test_white/ours_30000/renders --background white --render-stats ... --output-json ... --summary metrics_summary.csv --method-name single_view_overfit_v5_step050000_view000 --force-update` を実行し、白背景/前景 PSNR=18.00 dB, SSIM=0.760, LPIPS=0.427 を記録 (`metrics_white.json` + `metrics_summary.csv` 更新)。
* 所感と次候補: v4 (PSNR 18.31) からの伸びはなく 18 dB で頭打ち。opacity map が薄めなため (a) opacity target を 0.16 前後へ緩和、(b) min lr を 5e-5 まで下げ後半の収束を強める、(c) 60K まで延長、などを候補に再調整する。レンダ結果の opacity/depth マップを確認して次の config 設計に反映する。

---

## 2025-10-26 — 単視点オーバーフィット v5 ハイパラ計画

* 目標: 単視点 response の白背景 PSNR を 20 dB 台に乗せる。KiloNeRF 既存構造（hidden_dim 64 / num_layers 2 / grid 2×18×10）を維持したまま、学習ステップとスケジュールで底上げする。
* 新規コンフィグ: `configs/generated/lego_single_view_overfit_v5.yaml`（50K step 伸長版）を一本化。lr 1e-3 → cosine 50K / min lr 1e-4、warmup 2K。opacity target 0.18・max 0.16、EMA 0.9997。v4 のパラメータ構成を踏襲しつつ、長期ランの安定化のみを追加した。
* ラン順案: 1) 50K ランを実行し、30K/40K/50K チェックポイントで color loss と PSNR をモニタ。2) 50K checkpoint を `render_student` → `evaluate_student_metrics` の順で評価し、PSNR 20 dB 到達を確認。3) 伸びが不足する場合は opacity target を 0.16 付近へ再調整 or min lr を 5e-5 へ下げる案を次候補にする。
* 実行前処理: 既存ログ/出力の削除と TensorBoard の起動は以下で統一。

  ```bash
  rm -rf logs/lego/single_view_overfit_v5 results/lego/single_view_overfit_v5

  conda run -n kilogs tensorboard \
    --logdir logs/lego/single_view_overfit_v5/tensorboard \
    --host 127.0.0.1 --port 6006
  ```
* 実行テンプレ:

  ```bash
  PYTHONHASHSEED=2025 \
  conda run -n kilogs python -m distill.lego_response_distill \
    --config configs/generated/lego_single_view_overfit_v5.yaml

  PYTHONHASHSEED=2025 \
  conda run -n kilogs python -m distill.render_student \
    --config configs/generated/lego_single_view_overfit_v5.yaml \
    --checkpoint results/lego/single_view_overfit_v5/checkpoints/step_050000.pth \
    --output-dir results/lego/single_view_overfit_v5/eval_single_view_step050000_view000 \
    --max-frames 1 --store-rgba

  conda run -n kilogs python tools/evaluate_student_metrics.py \
    results/lego/single_view_overfit_v5/eval_single_view_step050000_view000/renders/renders \
    teacher/outputs/lego/test_white/ours_30000/renders \
    --background white \
    --render-stats results/lego/single_view_overfit_v5/eval_single_view_step050000_view000/render_stats.json \
    --output-json results/lego/single_view_overfit_v5/eval_single_view_step050000_view000/metrics_white.json \
    --summary metrics_summary.csv \
    --method-name single_view_overfit_v5_step050000_view000 \
    --force-update
  ```

* 期待観測: color loss が 1e-3 台前半まで低下し、opacity map が v4 より滑らかかつ過度に締まりすぎないこと。PSNR≧20 dB に達すれば、次段階で depth ロス（>0）や dir エンコーディングを小さく導入する。未達の場合は 1) opacity target を 0.16 前後まで下げる、2) lr decay の底 (min lr) を 5e-5 へ下げる、3) 50K 以降も plateau なら 60K ステップ延長を検討する。

---

## 2025-10-28 — 単視点オーバーフィット v7 実行テンプレ

* 事前処理: `rm -rf logs/lego/single_view_overfit_v7 results/lego/single_view_overfit_v7` → `tensorboard --logdir logs/lego/single_view_overfit_v7/tensorboard --host 127.0.0.1 --port 6006` を別ターミナルで起動。
* 必須環境変数: `PYTHONHASHSEED=2025`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `KILOGS_ALPHA_LEAK_THRESHOLD=0.015`, `KILOGS_ALPHA_HALO_THRESHOLD=0.015`, `PYTHONPATH=.`（どれか欠けると `lego_response_distill.py` が即時終了する）。
* 実行テンプレ:

  ```bash
  PYTHONHASHSEED=2025 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  KILOGS_ALPHA_LEAK_THRESHOLD=0.015 \
  KILOGS_ALPHA_HALO_THRESHOLD=0.015 \
  PYTHONPATH=. \
  python distill/lego_response_distill.py \
    --config configs/generated/lego_single_view_overfit_v7.yaml
  ```

* 監視ポイント: `logs/lego/single_view_overfit_v7/tensorboard` の α 分位 (`alpha_quantile/*`), リーク/ハロ指標 (`alpha_diag/leak_flag`, `alpha_diag/halo_flag`), `loss/color`, `loss/opacity`, `opacity/target_weight_effective`。
* 終了後 TODO: `results/lego/single_view_overfit_v7/checkpoints/step_052000.pth` を `render_student.py` + `evaluate_student_metrics.py` へ回し、白背景/前景 PSNR・α ヒストグラムを記録。
* メモ: `max_steps=52000` は v6 tail から踏襲。cosine LR を 50K で最小学習率へ落とし切ったあと、低 LR のまま 2K ステップ延長して α 監視と Charbonnier/L2 ブレンドの尾部収束を追うためのバッファ。

## 2025-10-29 — 単視点 v7 振り返りと改善方針

- step_052000 の白背景評価: PSNR 18.366 dB / SSIM 0.756 / LPIPS 0.424、前景 PSNR も同値。`metrics_summary.csv` と `results/lego/single_view_overfit_v7/eval_single_view_step052000_view000/metrics_white.json` に反映済み。
- α 診断ログ: `alpha.halo_indicator≈0.29`, `alpha_halo_streak=8`, `alpha_guard_penalty≈0.098`, `alpha.p50_ray≈0.21`。ハロ継続フラグが立ちっぱなしで、背景が明るく膨張している。
- v6_tail (PSNR 19.175 dB) 比較では、v7 で追加した `mean_target=0.32` / `mean_weight=0.02` と Charbonnier+L2 の長時間テールが α を過密化し、PSNR を 0.8 dB 以上落としたと推定。
- `alpha_guard` を無効化したままだとペナルティ指標だけが高止まりするため、次ランでは軽量 guard を再び有効化して α 分布を抑制する必要がある。

**次ラン (v8 案) の変更点**
- `configs/generated/lego_single_view_overfit_v8.yaml` を作成予定。`loss.color.secondary_weight=0.10` に下げて L2 の影響を補助レベルへ調整。
- `loss.opacity`: `target_weight=0.10`, `start_weight=0.035`, `max_weight=0.20`, `warmup_steps=2600`, `mean_target=0.24`, `mean_weight=0.008` とし、α 立ち上げを遅らせつつ平均値の押し上げを弱める。
- `alpha_guard` を有効化 (`initial_weight=0.05`, `weight_floor=0.02`, `weight_cap=0.08`, `penalty_hi=0.20`, `penalty_lo=0.03`, `adjustment_smoothing=0.18`, `check_interval=200`) して過剰密度を自動的に抑制。
- 学習率テールは v6_tail の実績を踏襲 (`lr_schedule_steps=45000`, `lr_schedule_min_lr=3.0e-5`, `max_steps=52000`)。監視目標: `alpha.p50_ray≤0.16`, `alpha_guard_penalty≤0.04`, `alpha.halo_indicator≤0.02`。

**最新アドバイス反映 — 単視点 v5 改善ロードマップ（2025-10-27 夜時点）**

1. **Run A（本命）** — *min lr を 5e-5 へ下げ*、*opacity target を 0.20〜0.22 へ微増*
  - 目的: 終盤の微分余地を確保しつつ背景リークを抑えて PSNR を押し上げる。
  - Go判定: 40K→50K で color loss が再び減少し、白背景 PSNR が +0.5 dB 以上伸び、前景 PSNR ≥ 白 PSNR。
  - No-Go: color が横ばいで α ヒストグラムが薄いまま or 背景誤差支配が継続。

2. **Run B（切り分け対照）** — *min lr を 5e-5 に統一*しつつ、opacity target は現状の 0.18 を維持
  - 目的: Run A の改善が α 目標の調整によるものか、min lr 自体の効果かを切り分ける。
  - 判定: Run A が伸びて Run B が伸びない → α 上げが効いている。両方ダメなら α 緩和（0.16〜0.18）＋ warmup 延長を検討。

3. **α 調整の両極を確認**
  - 背景 MSE 高め / α 平均低め（透け多い）→ α ターゲットを上げる方向。
  - 縁のハロ・スパイク目立つ / α ガードが暴れている → α ターゲットを下げつつ立ち上げ遅延。

4. **Go 条件が見えたら 60K 延長**
  - “min lr の尻が効いて伸びた”と確認できた設定のみ 60K へ延長。伸びが無いまま延長しない。

5. **代替/強化オプション**（上記と組み合わせ可能）
  - 終盤のみ samples_per_ray を増やす（高周波拾いで +0.3〜0.7 dB 期待）。
  - σ バイアスを浅くして初期密度を出やすく（薄すぎる場合）。
  - hidden_dim を 80〜96 へ微増（層数は据え置き）— 30 dB を狙う際の保険。
  - EMA をさらに重く or SWA 併用で評価ノイズを低減。
  - 色損失に微量 Charbonnier/L1 を混ぜ、PSNR 目的 MSE の停滞を緩める。

6. **実行前チェックリスト**
  - 前景 PSNR vs 白 PSNR（背景誤差支配を把握）。
  - α ヒストグラム（平均 / P90 / P99、薄いかスパイクか）。
  - color loss の推移（30K 以降で微減が続くか）。
  - EMA 評価とのズレ有無（ノイズが残っていないか）。

**次の一手（即実行）**
- Run A 用として `configs/generated/lego_single_view_overfit_v6.yaml` を作成（min lr=5e-5, opacity target=0.21, max_weight=0.22 などを反映）。
- Run B 用として `configs/generated/lego_single_view_overfit_v6_ablation.yaml` を用意（opacity target=0.18 を維持しつつ min lr=5e-5 に揃えた比較設定）。
- それぞれ 50K ランを走らせ、30K/40K/50K の color と PSNR、α 分布を比較。Run A が +0.5 dB 以上伸びれば同設定を 60K 延長、伸びなければ α 緩和・サンプル増しのセットに移行。

**30K / 40K / 50K の観測ポイント**
- ⚑ *Green*: color loss が微減を継続（特に 40K→50K で再下降）、前景 PSNR ≥ 白 PSNR、α ヒストグラム中央値が過度に上がらず P95≲0.9 で安定、α ガード分布が P90≤0.07 / P99≤0.10。
- ⚠ *Red*: 35K 以降 color が完全横ばい、白 PSNR が前景を上回る、α 平均が低く透ける or P99 が張り付いてハロが出る。

**微調整ノブ（優先順）**
1. Run A のまま: 45K 停滞なら min lr→3e-5 テール強化、α が薄ければ target を +0.01、縁が固いなら warmup を延長して立ち上がり遅延。

  **Run A/B 50K 結果（2025-10-27）**
  - Run A (`lego_single_view_overfit_v6`): PSNR 18.32 dB, SSIM 0.756, LPIPS 0.427。α ターゲットを 0.21→`target_weight_effective` 0.12 に乗せたことで背景リークは減少し、前景/白背景とも 18.32 dB まで改善。
  - Run B (`lego_single_view_overfit_v6_ablation`): PSNR 18.00 dB, SSIM 0.760, LPIPS 0.427。α ターゲット 0.18 のままでは v5 相当から伸びず、color loss も 0.137 付近で頭打ち。
  - どちらも GPU peak ≈1.18 GiB, fps ≈0.30。Run A の α 平均は 0.0119 まで上昇しているため、白背景での差分は α 目標更新由来と判断。
  - 次手: Run A をベースに 1) 60K への延長可否検討、2) α target を 0.22 前後で微調整、3) サンプル数 or feature 蒸留の追加で +0.5 dB を狙う。Run B 系は α 緩和(warmup 延長/target 0.16〜0.18)の再試行が必要。

  **Run A 40K 評価（2025-10-27 午後）**
  - 40K checkpoint（single_view_overfit_v6_step040000_view000）を 1 枚評価した結果、白背景/前景とも PSNR 18.324 dB・SSIM 0.757・LPIPS 0.425。50K checkpoint との差分は ΔPSNR_white = −0.002 dB, ΔPSNR_fg = −0.002 dB で実質的な伸び無し。
  - 白−前景ギャップは 0.000 dB のまま（背景誤差支配は解消済み）。α ヒストグラム P99 も 50K 時点と同程度で飽和は見られず。
  - 判定: 40K→50K で +0.5 dB を満たさないため 60K 延長は保留。今後伸ばすには α 設計か LR テールの再調整が必須。
  - 次手候補: (1) min lr を 3e-5 へ下げた派生（Run A’）で 50K を再実行し、color loss テールを再確認。(2) α target 0.22 試行と warmup 延長を組み合わせた Run A'' を計画。(3) feature 蒸留の軽量追加（L2 微量）で白背景の微差を詰める案を検討。

  **Run A′ 設定（2025-10-27 夜着手）**
  - Config: `configs/generated/lego_single_view_overfit_v6_tail.yaml`
    - `samples_per_ray=160`（終盤の高周波拾い目的で一段増し。VRAM ≈1.3 GiB を想定）
    - `lr_schedule_steps=45000` で 45K 時点に最小学習率へ到達させ、`lr_schedule_min_lr=3e-5` に下げてテールを強化
    - `ema_decay=0.9998` で後半 EMA を重めに維持
    - α ターゲット周りは Run A と同一（target 0.21 / max_weight 0.22）
  - 実行手順: 旧ログ/出力（`logs/lego/single_view_overfit_v6_tail`, `results/lego/single_view_overfit_v6_tail`）を削除→TensorBoard (port 6006) 起動→50K ラン。
  - 評価ポイント: **45K / 50K** の 2 点で単フレームをレンダし、`ΔPSNR_white`・`ΔPSNR_fg`・`PSNR_white − PSNR_fg`・`alpha.mean_frame` P95/P99 を記録。
  - Go 判定: 50K で +0.4〜0.5 dB 近辺の伸び（白/前景とも）を確認できれば同設定で 60K 延長を再検討。
  - No-Go 判定: +0.2 dB 未満かつ α P99 が上昇したら Run A′ は打ち切り、Run A″ へ移行。

  **Run A′ 結果（2025-10-28 朝）**
  - 40K: PSNR 19.164 dB / SSIM 0.757 / LPIPS 0.426（前景 PSNR 同値、白−前景ギャップ 0.000 dB）。
  - 50K: PSNR 19.175 dB / SSIM 0.758 / LPIPS 0.424（前景 PSNR 同値、ギャップ 0.000 dB）。ΔPSNR_white ≈ +0.011 dB で 60K 延長条件 (+0.4 dB) 未達。
  - `training_metrics.csv` では color ≈0.127（Run A 比 ≈−0.026）、`opacity.target_weight_effective` は 0.12 に収束、α guard penalty 0.0 を維持。
  - ベースライン比 +0.85 dB までジャンプした一方、40K→50K の伸びは頭打ち。次ステップとして Run A″（α 立ち上げ遅延）の準備に移行。

  **Run A″ 設定（2025-10-28 着手）**
  - Config: `configs/generated/lego_single_view_overfit_v6_warm.yaml`
    - Run A′ の lr/samples 設定（`samples_per_ray=160`, `lr_schedule_steps=45000`, `lr_schedule_min_lr=3e-5`, `ema_decay=0.9998`）を継承。
    - `loss.opacity` の立ち上げを遅らせるため `start_weight=0.025`, `warmup_steps=6000`, `schedule_duration=18000` へ変更（target=0.21, target_weight=0.12, max_weight=0.22 は据え置き）。
  - 目的: α 平均の過密化を抑えつつ 45K 以降の color loss 微減を狙う。Run A′ で止まった ΔPSNR を再び +0.4 dB 以上へ導けるかを確認する。
  - 実行プラン: 50K まで走行し、40K/45K/50K をレンダ → `ΔPSNR_white` / `ΔPSNR_fg` / `PSNR_white − PSNR_fg` を記録。Go 条件は 45K→50K で +0.4 dB 以上の伸び（白/前景とも）。
  - 判定: 条件未達なら α target を 0.22 へ上げる or α warmup をさらに延長した Run A‴ を検討。ΔPSNR が改善した場合のみ 60K 延長案を再評価。

  **Run A′/A″ 進捗サマリ（2025-10-28）**
  - Run A′ (`single_view_overfit_v6_tail`): 40K/50K とも白・前景 PSNR=19.17 dB 前後、ΔPSNR_white=+0.011 dB。color loss ≈0.127 まで低下しベースライン比で +0.85 dB だが、終盤の伸びは停滞。
  - 背景ギャップ解消済み（白−前景=0.000 dB）で α ガードも安定。課題は 45K 以降の微増を再び引き出すこと。
  - Run A″ は α の立ち上げを遅らせてテールの余力を残し、45K→50K で +0.4 dB 以上伸ばせるかを検証する。Go なら 60K 延長、No-Go なら α target 引き上げやさらに長い warmup を用意した Run A‴ を計画。

  **直近アドバイス反映 — Run A/B 運用メモ（2025-10-27 追加）**
  - Run A は Go 候補。まず 40K checkpoint を 1 枚評価し、50K と比較して白/前景 PSNR の差分を記録。+0.5 dB 以上の伸びが確認できたら 60K 延長へ。
  - Run B は α 緩和方向の切り分け要員として保持。Run A の伸びが α 上げに起因することが確定したため、B 側は target を下げつつ立ち上がりを遅くする案で継続検討。
  - 60K 延長時の最小変更: (1) LR テールをさらに下げて終盤の微調整余地を確保。(2) 40K 以降で EMA を強めて評価ノイズを抑制。(3) 終盤のみ samples_per_ray を段階的に増量（VRAM と相談）。(4) α 目標は Run A を基準に極小幅で上積みし、縁が硬くなったら warmup を微延長。(5) 余裕があれば最終数チェックポイントで SWA 併用。
  - 用語表記を統一: `alpha.mean_frame`（レンダ画素 α 平均）、`opacity.target_weight_effective`（α ロス有効重み）、`alpha_guard.penalty`（ガードペナルティ）。ノート内はこの表記で統一する。
  - 監視メトリクスを 2 つ追加: (1) ΔPSNR_white / ΔPSNR_fg（40K→50K）。(2) 白−前景ギャップ（PSNR_white − PSNR_fg）。ギャップが正なら背景誤差支配→α 設計を優先。負なら前景支配→LR テール強化やサンプル増しが有効。
  - 60K 延長オペ: 40K 評価→50→60K 延長→58〜60K を EMA（可能なら SWA）評価。50K→58K で +0.2 dB 未満かつ α ヒストグラム P99 が上昇した場合は延長を打ち切る。
  - 即応ルール: 45K で停滞を感じたら min lr を早めに下げ、縁が硬い/ハロが出たら α 立ち上がりを遅らせる。VRAM が厳しい場合は終盤サンプル増しを二段階で小さく。
  - Run B の役割: 背景誤差切り分け専用として α 緩和（target 低め＋長め warmup）を試す。Run A で伸びない場合のバックアッププランとして維持。
2. 終盤サンプル増し: 高周波不足と判断したら後半のみ samples_per_ray を増量。
3. EMA/SWA 強化: decay を一段上げる、最終 2〜4K を SWA で平滑化。
4. 容量微増: hidden_dim を 80〜96 に拡張（層数据え置き）し同設定を再実行。

**Go / No-Go 判定**
- Go (60K 延長): 40K→50K で +0.5 dB 以上の伸びかつ前景≧白。
- No-Go: color 横ばい + α の薄さ/固さが改善せず → Run B へ切り替え、もしくは α/warmup を逆方向へ振って再試行。

**評価まわりの注意**
- 評価は必ず EMA 重みで実施し、非 EMA と差が出る場合はノイズ対策（EMA/SWA）を優先。
- レンダは `view000` 単フレームで統一し、平均化による指標低下を避ける。
- 白 PSNR と前景 PSNR の差で背景誤差支配かどうかを即判定。背景が勝つなら α 設計を先に触る。

**症状別クイック処方**
- 背景がうるさく白 PSNR が低い → α target +0.01、warmup 短め。
- 縁が硬くハロが気になる → warmup 長め + target 据え置き or ほんの少し下げ。
- color が微動だにしない → min lr のテールをさらに下げ (≈3e-5) + 終盤サンプル増し。
- 指標が散る → EMA 強化 + SWA で最終帯を平滑化。

---

## 2025-10-22 — Evaluation 一貫性チェックツール整備

* `tools/eval_consistency.py` を新規作成。教師アセット（RGBA/深度/チェックポイント）の存在確認→学生レンダの整合性チェック→必要に応じて再合成＋メトリクス再計算まで一括で実行できるようにした。
* 使い方テンプレ:

  ```bash
  python tools/eval_consistency.py \
    --student-render-root tmp/lego/student_rgb_fourier_dir_depth060_v1 \
    --student-checkpoint results/lego/feat_t_full/runs/student_rgb_fourier_dir_depth060_v1/checkpoints/step_010000.pth \
    --teacher-outputs teacher/outputs/lego/test_white/ours_30000 \
    --expected-frames 200 \
    --recompute-metrics \
    --summary-csv metrics_summary.csv \
    --method-name student_rgb_fourier_dir_depth060_v1
  ```

  * `.npz` を `render_student.py` で保存していない場合は `--skip-rgba` で rgba 検査をスキップし、PNG のみで構成されているかを確認できる。
  * `--recompute-metrics` を付ければ背景再合成→`evaluate_student_metrics` 呼び出し→JSON/CSV 更新まで自動化。既存の recomposed ディレクトリをクリアしたいときは `--clean` を併用。
* チェック項目: 学生レンダのフレーム数・`render_stats.json` の存在・教師/学生の `.png` 対応、教師チェックポイントのサイズなどを明示的に検査するため、評価前のプリフライトとして毎回実行する。
* LEGO 応答10k本走（`student_rgb_fourier_dir_depth060_v1`）を再現。`training_metrics.csv` から `loss/total` 最小値は step 2334 の 0.178、最終ステップ 10000 では `color=0.0603`, `opacity=0.0226`, `alpha_guard_penalty=0.0671`。直近 100 step 平均は `total=0.253`, `color=0.104`, `opacity=0.0301` でやや悪化傾向。
* `render_student.py` + `--store-rgba` で `results/lego/feat_t_full/runs/student_rgb_fourier_dir_depth060_v1/eval_white/` に 200 枚レンダリング → `python tools/eval_consistency.py ... --recompute-metrics` で白背景 PSNR/SSIM/LPIPS を算出（PSNR 10.87 / SSIM 0.726 / LPIPS 0.312, avg_fps 0.129）。期待値より著しく低いため、背景再合成が正しく行われているか・RGBA 正規化が崩れていないかを要再確認。
* 差分調査用に `tools/inspect_frame_diff.py` を追加。`--use-recomposed --background white` で `tmp/frame_diffs/stack_*.png` を出力し、00000/00050/00100 の PSNR が 11〜13 dB 程度・最大差分 1.0 まで振れていることを確認。学生レンダがほぼ白飛びしているため、opacity 制御（ターゲット 0.05→0.11 の上げ幅、alpha guard 重み 0.12→0.14）を緩和するか、dir_L/深度重みスイープで色収束を優先する必要あり。
* 10k スイープ第2案として `configs/generated/lego_feature_student_rgb_fourier_dir_depth030_alpha075.yaml` を追加。`loss.opacity.weight/target/target_weight` を 0.035/0.035/0.08 に抑え、`alpha_guard.initial_weight=0.08`・`weight_floor=0.03`・`weight_cap=0.18` に再設定。`loss.depth.weight` も 0.03 に下げ、色収束を優先したソフト alpha 版を検証予定。
* `lego_feature_student_rgb_fourier_dir_depth030_alpha075` ラン完了。`loss/total` 最小 0.145（step 2334）、最終 0.197。`opacity` 最終 0.0166 まで降下し、`alpha_guard_penalty` は 0.0627 で安定。
* `eval_consistency.py --recompute-metrics` で白背景評価 → PSNR 10.83 / SSIM 0.727 / LPIPS 0.307 / avg_fps 0.127。PSNR はほぼ横ばいで、白飛び軽減は見られるが教師との差は依然大きい。
* 差分スタックを `tmp/frame_diffs_depth030/` に保存。00000 の MAE 0.147・max_diff 0.996 など、明部でのズレは残存。ただし opacity map がより締まり、背景の透け具合は改善傾向。

---

## 2025-10-24 — 特徴蒸留v11 中間チェックポイント評価 (step_025000)

* 白背景レンダを再生成: 既存の `eval_white_step025000/` を削除後、

  ```bash
  conda run --no-capture-output -n kilogs python -m distill.render_student \
    --config configs/generated/lego_feature_student_rgb_fourier_dir_depth030_alpha075_feature50k_debug10k.yaml \
    --checkpoint results/lego/feat_t_full/runs/feature_distill_v11_debug10k/checkpoints/step_025000.pth \
    --output-dir results/lego/feat_t_full/runs/feature_distill_v11_debug10k/eval_white_step025000 \
    --store-rgba
  ```

  FPS は 0.132（1 枚 ≈7.6 秒）、GPU 使用量ピークは 1.33 GiB (`render_stats.json` より)。
* `tools/eval_consistency.py --recompute-metrics` を同ディレクトリに対して実行し、`metrics_white.json` + `metrics_summary.csv` を更新。結果は PSNR 12.10 / SSIM 0.7447 / LPIPS 0.2891。50K checkpoint (PSNR 13.06) よりわずかに低いが、LPIPS は 0.2967→0.2891 と改善傾向。
* `training_metrics.csv` を参照すると、step 25000 時点の `total=0.1708` が valley。直後の step 26000 で `total=0.2413` (color loss 0.094) まで跳ね上がり、その後 50K まで 0.24 付近で推移。1000-step 移動平均の最良値 (0.2370) も step 25993 で止まっているため、25K チェックポイント付近で早期停止する方が安定。
* α ガード: 25K 時点で 0.039、50K では 0.047〜0.050 に上昇。feature/cosine loss が後半で再増大していることも踏まえ、v12 では feature schedule を 4K→12K step で強めに入れたあと、30K 以降は緩やかに落とし込む（cosine weight を 0.01→0.02→0.015 など）。
* 追評価: step_030000 でも同手順でレンダ＋評価を実施。白背景メトリクスは PSNR 12.36 / SSIM 0.7452 / LPIPS 0.2902、25K と 50K の中間に位置。loss CSV 上では 30K 付近で `total≈0.22`、color/feature がともに上昇し α guard penalty も 0.05 まで戻っている。30K〜34K の窓平均では `total≈0.242` と 25K より明らかに高い。
* 次アクション（v12 設計用メモ）:
  1. step_030000 も同様にレンダ＋評価してトレンドを補完（25K ←→ 50K の差分を補間）。
  2. `loss.feature_schedule` の後段を cosine decay に変更し、30K 以降で feature_weight を 0.04 前後へ収束させる案を config に反映。
  3. α guard の `min_target_weight` を 0.03 に据えたうえで、`adjustment_smoothing` を 0.10 付近まで下げて追従性を向上させる（v12 の draft で試す）。

## 2025-10-24 — 評価整合＆制御系リファイン計画
- シングルビュー sanity check: `tools/run_single_view_overfit.py --config configs/generated/lego_single_view_overfit_v1.yaml --max-steps 10000 --overfit-steps 10000 --frame-index 0` を実行し、`results/lego/single_view_overfit_v1/eval_single_view_step010000_view000/` に成果物を保存。白背景 PSNR 7.89 dB / SSIM 0.655 / LPIPS 0.369（前景 PSNR 同値）で目標の 30 dB には未達。α 圧・バッチサイズ・学生容量・表現形式のどこが律速か切り分けが必要。`metrics_summary.csv` に `single_view_overfit_v1_step010000_view000` 行を追加済み。

**ボトルネック認識**
- 評価パイプラインの不整合（sRGB↔linear や straight α 混在）が残り、PSNR が 10〜13 dB 付近で頭打ち。
- α/深度の制御が強すぎて白飛びが増え、25K valley 以降に total が反発。
- view-dependent 分岐を入れても土台が揺れており、feature 蒸留が不安定化要因になっている。

**P0: 評価整合の再固定（最優先）**
- 教師・学生とも linear RGB で比較する。PNG ロード時は gAMA/sBIT の影響を捨て、sRGB→linear を同手順で適用。
- アルファは straight α 前提で統一し、白合成は `rgb*α + (1-α)`。pre-multiplied 混入を禁止。
- 白背景 PSNR と並列で「前景 PSNR（教師 α 積分 > 0.7）」を常設する。
- 単視点オーバーフィット（EMA モードで PSNR ≥30dB）を毎回サニティチェック。
- 評価は EMA ウェイトのみで実施し、レンダ時に EMA のロードを必ず確認。
- 直近アクション: 評価スクリプト群を linear/straight α 前提へ点検し、前景 PSNR 算出と単視点テスト（既存教師フレーム）を 2025-10-25 JST 午前までに完了する。結果を `metrics_summary.csv` に新カラム追加で記録する。

**P1: α・深度制御を「遅く・弱く・滑らかに」**
- α 目標追従は EMA 判定（β=0.9〜0.98）＋連続未達 K step のみ増加。増分は +0.002/step 以下、減少はその半分。
- 有効重みの頭打ちは 0.12〜0.15。ヒステリシスとレート制限を実装する。
- 深度ロスは前景限定＋Huber 幅を広げ、重みを 0.06〜0.08 から開始。
- 勾配ノルム比（color:α:depth≈1:0.3:0.3）と透過率ヒストグラムを記録する仕組みをログへ追加。

**P2: view-dependent の寄与を遅らせる**
- 方向ブランチは late fusion、低 LR（または係数 0.5）でウォームアップしながら徐々に立ち上げる。
- dir L は L=4 を基準に、L=2/6 を 10k ランで A/B。終盤 1k の color loss または前景 PSNR が +5% 以上改善し、α スパイク頻度が悪化しない場合のみ採択。

**P3: 25k 付近で早期停止 + EMA/SWA**
- 22k〜28k でチェックポイント帯を作り、EMA 評価＋SWA で最終化。50k への惰性延長を避ける。

**P4: サンプリング & near/far**
- 初期は軽サンプルで開始し、後半で濃くする漸増を採用。遠景ノイズに引かれないよう free-space 抑制を緩める。
- 教師密度に応じた重み付き再サンプルを診断用途で用意。

**P5: feature は土台安定後に導入**
- GO 条件: 単視点 ≥30dB、10k/20k で loss/total 非増、loss/color ≤0.06、α スパイク ≤2%。
- cosine → L2 の順で小さく立ち上げ、projector LR は学生の ≤0.5。安定後は凍結オプション。
- 比較空間は 64D で統一し、z-score や L2 正規化でチャネル寄与を均す。

**Open Questions（即答メモ）**
- feature を今入れるか → NO。応答蒸留を安定化させてから。
- 比較空間 64D の是非 → OK。統一しやすい。正規化は必須。
- dirL の評価方針 → L=4 を軸に 10k A/B。改善 +5% かつスパイク増なしで採択。
- α guard は penalty/relax 調整だけで足りるか → 足りない。ヒステリシス＋レート制限＋百分位駆動へリファクタ必須。

**次の48時間プラン（手順）**
1. P0 一式の再検証（linear/sRGB・straight α・前景 PSNR・単視点 30dB・EMA 評価）。
2. α/深度制御系を「遅く・弱く・滑らかに」へ変更（ヒステリシス＋レート制限）。
3. dirL A/B（L=2/4/6）を 10k×3 で実施し、終盤 1k の color loss / 前景 PSNR / α スパイク頻度を比較。
4. 勝ち設定で 20k 延伸（スケジュールは時間等価で再設計）。
5. 22k〜28k に早期停止ウィンドウを張り、EMA＋SWA で最良値抽出。
6. GO 条件が揃ったら feature を段階的に再導入（cos→L2、小さく、projector 低 LR→凍結）。

**追加リスクメモ**
- PNG の色管理（gAMA/sBIT）とライブラリ差（PIL vs OpenCV）で linear 化がズレやすい。処理系を一本化する。
- depth の単位／正規化が教師と一致しているか再確認。
- projector/teacher-adapter が動き続けると基準が揺れる。安定後に停止する。

**直近の具体アクション（2025-10-24 夕方開始）**
- [ ] `tools/evaluate_student_metrics.py` 系の sRGB→linear・straight α 処理を確認し、差分があれば修正案を 2025-10-25 10:00 JST までにまとめる。
- [x] `tools/evaluate_student_metrics.py` に linear/straight α パイプラインと前景PSNR算出を実装（2025-10-24）。
- [x] `tools/evaluate_student_metrics.py` に linear/straight α パイプラインと前景PSNR算出を実装（2025-10-24）。
- [x] v11 debug10k (step 25k / 30k / 50k) を新パイプラインで再評価。白背景 PSNR はそれぞれ 10.11 / 10.36 / 11.02 dB（前景 PSNR 同値）まで低下し、既存 `metrics_summary.csv` を `--force-update` で上書き。旧指標との差は linear 化＆strict α 合成の影響と判断。
- [x] 単視点オーバーフィット用スクリプト `tools/run_single_view_overfit.py` と専用 config `configs/generated/lego_single_view_overfit_v1.yaml` を追加。`--overfit-mode student` + 固定 1 フレームで 2k/10k step ランを実施したが、現状 PSNR は 7.8〜7.9 dB 止まり。α 圧と学生容量の見直しが必要。
- [x] `DataConfig` に `max_frames` / `frame_indices` を追加し、単視点データセットを YAML で表現できるようにした（評価・レンダ両方で同じ JSON から対象フレームのみ抽出可能）。
- [x] `distill/lego_response_distill.py` の α guard をヒステリシス＋レート制限で再設計（連続違反判定＋更新幅制限を実装, 2025-10-24）。
- [x] `distill/lego_response_distill.py` に deterministic 起動ガード（`PYTHONHASHSEED` と `CUBLAS_WORKSPACE_CONFIG` の事前チェック）を追加し、未設定での学習開始を強制終了させるようにした（2025-10-24）。
- [ ] dirL=2/4/6 用の 10k コンフィグ雛形を `configs/generated/` に作成し、Day1 スイープの準備を完了させる。
- 完了後、`research_notes.md` へチェックリスト進捗を追記し、`metrics_summary.csv` に前景 PSNR カラムを追加する。

* α guard は平均ペナルティのヒステリシス判定＋連続違反カウンタで動作し、`lambda`/target/penalty の各ターゲットは per-update Δ 上限でレート制限されるようになった。チェックポイントにも方向・連続カウントを保存して再開時に継続可能。
* KiloNeRF のセル境界ぎれ対策として tri/d-linear 補間（近傍セルからの線形合成）を検討中。サンプルが属する 8 セルを列挙してサブ MLP 出力を重み付きに混ぜる案を要タスク化（実装コスト高のため要調整）。

## 2025-10-25 — 特徴蒸留v13 30K ラン報告

* config: `configs/generated/lego_feature_student_rgb_fourier_dir_depth030_alpha075_feature50k_debug10k_v13.yaml`
* 実行条件: `PYTHONHASHSEED=2025`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `--max-steps 30000`
* ラン完了。`logs/lego/feat_t_full/runs/feature_distill_v13_debug10k/` に TensorBoard/CSV、`results/.../checkpoints/step_030000.pth` まで出力を確認。
* 軸トレンド: `loss/color` と `loss/opacity` は想定通り 30K まで緩やかに低下した一方、`loss/depth` が 26K 付近から微増（+0.004 前後）。`alpha_guard_penalty` は 0.05 台で安定。
* 当面のTODO
  * `training_metrics.csv` から 24K→30K 区間を抽出し、`loss/depth` / `loss/opacity` / `opacity_target_weight_effective` の相関を見る。
  * worst-N depth ヒートマップを確認し、背景リークなのか前景誤差なのか切り分ける。
  * 深度押し返し案: `loss.depth.weight` を 0.025〜0.028 へ下げる or `alpha_threshold` を 0.75 へ引き上げる短縮ラン（10K）で効果測定。

メモ: 次スイープで深度重みを調整する際は v13 コンフィグを複写して差分だけ編集する。α 側は現状安定しているため触らない方針。

### 2025-10-25 フィードバック反映 — 優先アクション（言葉だけで実行計画）

**最優先 P0: 評価の固定とサニティ**
- 教師=教師で linear 空間・straight α・白合成の自己一致テスト（∞/1/0）を毎回実施。PNG gAMA/sBIT 無視、丸め ε、RGB/BGR を統一。失敗したら評価修正が完了するまで学習は回さない。
- 単視点オーバーフィットで PSNR ≥ 30 dB を達成するまで他調整は禁止。色のみ、深度/αは無効もしくは極弱・遅延。view-independent 小領域で確実に当てる。達成できなければ容量（hidden/skip/ Fourier L）→σ初期化（負バイアス強化）→学習率/バッチ→サンプル密度の順で見直し。

**P1: α/深度の「遅く・弱く・滑らかに」**
- αガードは連続 K ステップ未達（EMA 判定）でのみ増加。増分は +0.002/step 相当、減少はその半分。有効重み上限を 0.12〜0.15 に固定し、αガードペナルティ分位（P90 ≤ 0.07 / P99 ≤ 0.10）を監視指標にする。
- 深度ロスは前景限定＋Huber 幅拡大。α 積分しきい値 0.7〜0.8、色が下がってから段階投入。深度 weight 微下げ × しきい値引き上げの 2×2 を 10K 短期で比較。

**P2: view-dependent 成分の扱い**
- dirL=2/4/6 を 10K ×3 本で比較。評価軸は終盤 1K color 平均・前景 PSNR・α スパイク頻度。改善 <5% またはスパイク悪化なら低 L へ戻す。方向ブランチは遅延ウォームアップ。

**P3: サンプリングと透過率**
- 透過率ヒストグラムを定点ロギング（序盤薄く・終盤締まる形）。サンプル数は前半軽・後半濃の漸増設計。near/far を再点検し遠景リークを抑える。σ バイアスは負維持、shifted-softplus 継続。

**P4: 早期停止と重み集約**
- 22K〜28K を重点保存し EMA / SWA を比較。loss/total の反発兆候や α P95 の上昇が見えたら延伸せず終了。v11 で 25K 谷だった観測と整合させる。

**P5: 特徴蒸留再導入ゲート**
- GO 条件: 単視点 30 dB、10K/20K で loss/total 非増、color ≲ 0.06、α スパイク ≤ 2%。比較空間は教師 52→64 / 生徒 64→64。cos → L2 の順で、小さな重みから。projector LR は生徒の ≤0.5 とし、安定後は凍結検討。導入は短縮ランで健全性確認後に 10K → 20K へ。

**P6: ログ/診断の三本柱**
- 損失別勾配ノルム比（color:α:depth:dir）、透過率ヒストグラム、worst-N 可視化を常時保存。必要に応じて CKA で特徴整合を確認し、projector 凍結判断に活用。

**P7: tri/di-linear 混合（セル境界対策）**
- 応答蒸留が安定してから 10K 短縮でオン/オフ比較。連続性向上とディテール鈍化のトレードを見極める。効果が薄いなら学習後期のみ有効化を検討。

### 48 時間ミニ計画（言葉のみの段取り）
- Day1 午前: P0 完了（自己一致テスト、単視点 30 dB）。
- Day1 午後: P1 調整（α ガード頻度制御 + 深度前景化）を 10K × 2–3 本。
- Day2 午前: P2 dirL=2/4/6 の 10K × 3 本。評価軸は color 終盤平均 + 前景 PSNR + α スパイク頻度。
- Day2 午後: 勝ち設定を 20K へスケールし、22–28K 窓で早期停止をテスト（EMA / SWA 比較）。
- 条件クリア後: P5 に沿って特徴蒸留を段階導入（短縮 → 10K → 20K）。

### 合格ゲート（再確認）
- 単視点（linear/straight α/EMA）: PSNR ≥ 30 dB。
- 10K: loss/total 非増、color 単調減、α P90 ≤ 0.07 / P99 ≤ 0.10。
- 20K: 前景 PSNR が白背景 PSNR を上回り、LPIPS が 0.20 台前半に接近。
- 特徴再導入: 上記条件を満たしてから cos → L2 を小さな重みで導入。

### 落とし穴チェック
- pre-multiplied 混入、PIL と OpenCV の色変換差、PNG メタの残存。
- フレーム順 / カメラ JSON の参照ずれ。
- EMA 未適用での評価実施。
- projector が動き続け基準が揺れる（早期凍結で防止）。

この優先順位で短期ランを積み、評価由来の頭打ち → α/深度の反発 → 方向・特徴での上積み、の順で改善幅を作る。

#### 2025-10-25 単視点オーバーフィット調査ログ

- self-consistency (teacher vs teacher、200 frames) を再実行 → `psnr=∞ / ssim=1 / lpips=0` を確認し、評価パイプラインの linear/straight α 設定は健全。
- 既存 `lego_single_view_overfit_v1/v2`・`simple` で 10k step まで回しても PSNR ≈ 6.6 dB 止まり。PNG を確認すると学生レンダは全画素が `[1,1,1]`（白背景のみ）で、密度ゼロのまま進んでいないことが判明。
- 原因: α/opacity を完全に切ったため、shifted-softplus + 負バイアスが背景透過を維持し続け、色勾配が有効化されない。固定バッチ overfit（`--overfit-mode student`）も一部画素しか監督できず評価で白飛びが残った。
- 対策: `lego_single_view_overfit_v3.yaml` を新設。σ バイアスを浅く（-0.2 / -0.4）しつつ、`loss.opacity` を弱い L1 で再導入（weight 0.05, target 0.25, max 0.25, warmup 400）。α guard は無効化。
- v3 + overfit 固定バッチ (6k step) → PSNR 16.16 dB。色 loss は 0.0016 まで下がるものの、固定バッチのため未監督ピクセルが白背景のまま（平均 PSNR はまだ不足）。
- 現在: 同じ v3 設定で **ランダムサンプリング 20k step** を走行中（`batch_size=16384`, `samples_per_ray=128`）。全画素に監督が行き渡るか、PSNR をモニタしながら 30 dB 目標の達成可否を確認する。トレーニングログは `logs/lego/single_view_overfit_v3/` に追記中。

## 2025-10-23 — 特徴蒸留v11 デバッグランと TensorBoard 指標整理

* `distill/lego_response_distill.py` の TensorBoard 出力を整理。`full=True` で記録するスカラーを `loss/color`, `loss/opacity`, `loss/depth`, `loss/feature_recon`, `loss/feature_cosine`, `opacity/alpha_guard_penalty`, `opacity/target_weight_effective` の計 7 本へ縮小。
* デバッグ専用コンフィグ `configs/generated/lego_feature_student_rgb_fourier_dir_depth030_alpha075_feature50k_debug10k.yaml` を新設。`experiment.name=feature_distill_v11_debug10k`, `progress_desc=特徴蒸留v11` とし、log/output も `feature_distill_v11_debug10k` 配下へ分離。
* 実行手順テンプレ:

  ```bash
  # 1) 旧ログの削除
  rm -rf logs/lego/feat_t_full/runs/feature_distill_v11_debug10k \
         results/lego/feat_t_full/runs/feature_distill_v11_debug10k

  # 2) 10k デバッグラン起動
  PYTHONHASHSEED=2025 \
  conda run -n kilogs python -m distill.lego_response_distill \
    --config configs/generated/lego_feature_student_rgb_fourier_dir_depth030_alpha075_feature50k_debug10k.yaml

  # 3) TensorBoard
  conda run -n kilogs tensorboard \
    --logdir logs/lego/feat_t_full/runs/feature_distill_v11_debug10k/tensorboard \
    --host 127.0.0.1 --port 6006
  ```

* 進捗バーの表示名が `(特徴蒸留v11)` となり、メトリクス確認が簡潔化。次回 v12 へ更新する際は `experiment.name`, `output_dir`, `progress_desc` の末尾を揃えてバージョンを increment する。
* `logs/.../training_metrics.csv` の初期行から希望のスカラーが並ぶことを確認済み（Step 1〜20）。TensorBoard 側も絞ったスカラーだけが表示され、ノイズが減少。
* `training_metrics.csv`（step 1〜3288）を確認したところ、`opacity_target_weight_base` が 0.035→0.038 付近で緩やかに推移する一方、`opacity_target_weight_effective` は `alpha_guard.min_target_weight` の既定値 0.05 によって常時クランプされていることが判明。`opacity_target_adjustment` も 1.10 付近まで上昇しており、ガードが緩め方向へ動こうとしているのに実効重みが追従できない状態。
* feature loss の重みは 4k step でウォームアップが終わった時点から `feature_l2=0.05`, `feature_cos=0.01` に固定される。色 loss が落ち着いてきた段階でさらに特徴寄りへ寄せられるよう、`feature_target_weight` / `feature_target_cosine_weight` を高めに設定し、`feature_schedule`（linear/cosine）と `feature_schedule_duration` を使って段階的に比率を上げる案を検討する。目安として 4k→12k step で 0.05→0.08, 0.01→0.02 程度まで引き上げるスケジュールを試すと収束バランスを調整しやすい。
* 上記対処として `configs/generated/lego_feature_student_rgb_fourier_dir_depth030_alpha075_feature50k_debug10k_v12.yaml` を作成。`loss.opacity.warmup_steps=1200`, `schedule_duration=5000` へ短縮し、初期 3k step 台で 0.05 を超えるよう調整。さらに `loss.alpha_guard.min_target_weight=0.03`, `adjustment_smoothing=0.15` を追加してターゲット重みの追従性を上げる。`experiment.name=feature_distill_v12_debug10k` / `progress_desc=特徴蒸留v12` としてログ出力先も v12 用に分離。

---

## 2025-10-18 — TensorBoard logging cadence

* Updated `configs/lego_feature_teacher_full_student_space_kilonerf_grid.yaml` so `logging.log_interval = 50`. TensorBoard scalar plots now log every 50 steps, giving ~1,000 points over a 50K iteration run without ballooning CSV size. Horizontal axis remains the global training `step`, so TensorBoard renders the curve as an Iter-vs-metric line plot.

---

## 2025-10-19 — TensorBoard 軸バグの調査と解消

* **症状**: TensorBoard を “RELATIVE” 表示にすると横軸が 0–1 に圧縮され、100 step ごとに記録している損失曲線が折れ線として確認できなかった。
* **対処**:
  * `logging.tensorboard_axis`（`step`/`time`/`elapsed`）をコンフィグスキーマに追加し、`SummaryWriter.add_scalar` に `walltime` を渡すよう学習ループを更新。デフォルトは `step`。
  * 既存コンフィグはそのまま `step` 軸で動作。従来の壁時計ベースに戻したい場合は各コンフィグで `logging.tensorboard_axis: time` もしくは `elapsed` を指定すれば OK。
* **検証**:
  * `PYTHONHASHSEED=2025 PYTHONPATH=. python distill/lego_response_distill.py --config configs/_tmp_tensorboard_axis_full.yaml --max-steps 600`
  * `tensorboard --logdir logs/tmp/tensorboard_axis_full_debug/tensorboard --port 6007`
  * “RELATIVE” 表示でも横軸が 0–600 step を維持し、100 step 間隔のスカラーが折れ線グラフで描画されることを確認。
* **備忘**:
  * トライ時に生成されたログは `logs/tmp/tensorboard_axis_full_debug/` 配下。再検証は同じコマンドを再実行すれば `_clean_tensorboard_events` が古い events を掃除してくれる。

---

## 2025-10-20 — TensorBoard Graph 出力トグル追加

* `distill/lego_response_distill.py` に `KILOGS_LOG_GRAPH=1` で有効になるグラフ記録ブロックを追加。`SummaryWriter` 初期化後、学生モデル生成直後にランダムな 3D サンプルを使って `writer.add_graph` を呼び出し、`debug_log` へ結果を残す。
* グラフ記録は best-effort。例外が発生しても学習自体は継続し、トレース成否は `logs/.../tb_debug.log` に追記される。
* GPU 未認識状態（CUDA error 304）で本番コンフィグがまだ起動できないため、`StudentConfig(type=\"simple_mlp\")` を使った CPU スモークテストで `tmp/tb_graph_test/` に GraphDef が生成されることを確認 (`tensorboard.backend.event_processing.event_accumulator` で `Graph()` が非 `None`)。
* 本番でグラフを可視化するには `KILOGS_LOG_GRAPH=1` を指定して再実行し、CUDA ドライバ／`kilonerf_cuda` の初期化エラー（`init_magma()`）を解消してから TensorBoard の Graph タブを開くこと。

### 2025-10-20 — TensorBoard 相対軸 / Graph デバッグ進捗

**トライ内容**

1. `configs/_tmp_tensorboard_axis_debug.yaml` を基に `tensorboard_axis: elapsed` を付与した派生（`tmp/tb_axis_cli.yaml`）を作成し、logdir を `logs/tmp/tb_axis_cli/` へ分離。
2. `PYTHONHASHSEED=2025 KILOGS_LOG_GRAPH=1 KILOGS_DEBUG_TB=1 python -m distill.lego_response_distill --config tmp/tb_axis_cli.yaml --max-steps 3` を CPU で実行。`tb_debug.log` に `[graph] success samples=2048 device=cpu` が記録され、Graph ログの成功を確認。
3. イベントファイルは生成されるが 3 step ではスカラーが空のまま → ログ出力前に終了するため。次は `--max-steps 10` 以上で再検証予定。

**得られた学び**

- `tensorboard_axis` の切り替えは動作。Relative 表示で折れ線を確認するには最低でも 10 step 程度のランが必要。
- `tb_debug.log` の `[graph] ...` と `log_interval_init=...` 行がデバッグ指標として有用。

**今後のアクション**

1. 10〜20 step の CPU ランで `elapsed` 軸の挙動を TensorBoard UI で確認。
2. 本番 GPU ランでは旧 events を整理した上で `tensorboard_axis` を目的のモードに設定して再走。
3. 手順と知見を README / 手順書に反映し、相対軸 + Graph デバッグのフローを標準化する。

**TensorBoard Graph 可視化デバッグ用テンプレ**

```bash
# 1) Graph ログを仕込んだ短いラン
PYTHONHASHSEED=2025 \
KILOGS_LOG_GRAPH=1 \
KILOGS_DEBUG_TB=1 \
python -m distill.lego_response_distill \
  --config tmp/tb_axis_cli.yaml \
  --max-steps 10

# 2) イベント内に Graph が入っているか CLI で確認
python - <<'PY'
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
logdir = Path("logs/tmp/tb_axis_cli/tensorboard")
ea = event_accumulator.EventAccumulator(str(logdir))
ea.Reload()
print("graph present?", ea.Graph() is not None)
print("scalar tags:", ea.Tags().get("scalars", []))
PY

# 3) TensorBoard で可視化（Relative 軸も確認）
tensorboard --logdir logs/tmp/tb_axis_cli/tensorboard --host 127.0.0.1 --port 6006
```

---


## 2025-10-20 — CUDA error 304（WSL）調査

* 症状: `torch.cuda.is_available()` が False。`kilonerf_cuda.init_magma()` で `RuntimeError: CUDA error: OS call failed or operation not supported on this OS`。`dmesg` には `dxgkio_query_adapter_info: Ioctl failed: -22` が多数。
* 切り分け:

  * `nvidia-smi` は成功（Driver 553.50 / CUDA 12.4 / RTX A4500）。
  * `LD_LIBRARY_PATH=/usr/lib/wsl/lib python -c "import ctypes; cuInit(0)"` でも 304 が返る → WSL の libcuda スタブが `CUDA_ERROR_NOT_SUPPORTED` を返却。
  * `torch` や `kilonerf_cuda` からの呼び出しも同じエラーに帰結。
* 推奨対処:
  1. Windows 側で最新の **WSL2 GPU 対応 NVIDIA ドライバ**（Studio または CUDA 用）へ更新。Game Ready 553.x では WSL Compute が無効化される既知事例あり。
  2. PowerShell（管理者）で `wsl --update` 実行 → `wsl --shutdown` → WSL 再起動。Linux カーネルと `/usr/lib/wsl/lib` のスタブを最新化。
  3. BIOS / Windows セキュリティで **仮想化ベースのセキュリティ (VBS)** を無効化（有効時、libcuda が 304 を返すことがある）。
  4. 再ログイン後、WSL 内で `ls -l /dev/dxg` を確認 → `torch.cuda.is_available()` 再チェック。
* 応急対応: GPU 復旧まで `student.type=simple_mlp` 等 CPU バックエンドでの検証は可能だが、本命の `kilo_uniform_mlp` は CUDA 拡張必須。

## 2025-10-20 — TensorBoard scalar 未出力（調査中）

* **症状**: `events.out.tfevents.*` が常に 88 byte、`curl http://127.0.0.1:6007/data/plugin/scalars/tags` が `{}`。UI は「Data could not be loaded」。学習起動ログでは `TensorBoard writer initialised` と出るが、`training_metrics.csv` も生成されない。
* **再現手順**:
  * `pkill -f "tensorboard.*6007"` → `rm -rf logs/.../tensorboard` → 再作成 → 10k ラン再実行（`KILOGS_LOG_INTERVAL=1`, `KILOGS_TENSORBOARD_FLUSH_SECS=5`）を複数回試行。
  * TensorBoard を別ポート（6007→6008）で起動しても変化なし。
  * `logs/.../tensorboard/events.out.tfevents.*` は作成されるが 88 byte から増えず、ブラウザをリロードしても折れ線が出ない。
* **TODO**:
  1. `logs/.../tb_debug.log` を有効化し、`log_interval_init` が実際に 1 で初期化されているか確認。
  2. `logs/lego/feat_t_full/runs/teacher_full_rehab_masked_white/` の権限を `chmod -R u+w` で整え、CSV 生成を明示的に許可する。
  3. 最小構成（`configs/_tmp_tensorboard_axis_full.yaml`, `--max-steps 200`）で再現するか確認。これでも 88 byte のままなら `add_scalar` が呼ばれていない可能性を追う。
  4. 別 logdir（例: `logs/tmp/tensorboard_smoke/`）に切り替え、パス衝突や symlink の影響を切り分ける。
  5. 2025-10-20 08:50 JST: `max-steps=2` のスモークテストと PDB で動作を追ったところ、訓練ループ自体は `global_step` を増分しているが `append_metrics()` へ到達していないことを確認。`should_log_step` 判定や `continue` の分岐を中心に追加インストゥルメンテーション（`/tmp/kilogs_loop_hits.txt` など）を仕込み、どこで抜けているかを調査中。大きなランは一時停止し、スカラーが書かれない根本原因を切り分けてから再開する。
  6. 手動の `SummaryWriter` テスト（`logs/debug_tb_manual`）ではスカラーが正しく出力され、TensorBoard でも折れ線を確認済み。学習ループ固有の条件が原因と確定。
  7. 2025-10-20 18:55 JST: CPU フォールバック＋ダミーデータで `simple_mlp` 学生を使った最小ラン（1〜50 step/`KILOGS_LOG_INTERVAL=1`）を複数回試行したが、`emit_training_logs` ブロックが呼ばれず `.tfevents` が 88 B のままという症状は再現できなかった。実際に問題が起きている run/config/logdir の具体例が必要（手元の縮小環境では条件が不足している可能性が高い）。

*オペレーション覚え書き*: CLI 連携時は **Step by Step（1コメント＝1コマンド群）** でやり取りする。連続コマンドで観測したい場合は 1 ステップにまとめて指定する。
*Log interval トラブル*: `KILOGS_LOG_INTERVAL` を環境変数で上書きする場合、必ず `export` で数値を設定し、短時間ランで TensorBoard を確認したいときは 10 など小さい値にする。再テスト前に古い `.tfevents` / CSV を削除し、`KILOGS_DEBUG_TB=1` で `tb_debug.log` を確認するとログゲートの動きが追える。

---

## 2025-10-21 — CSV 1 行だけになる件の調査ログ

* `distill/lego_response_distill.py` 冒頭の import ブロックが壊れていた（`emit_tensorboard_scalars` を誤ってファイル先頭へ貼り付け）。`.bak` からヘッダ部分を復旧し、`python -m compileall` でシンタックスチェック済み。
* `PYTHONHASHSEED=1234 KILOGS_DEBUG_TB=1 python -m distill.lego_response_distill --config configs/tmp_tensorboard_scalar_debug.yaml --max-steps 4`
  * 学習は 4/4 step まで進むが `tmp/tb_scalar_debug/logs/training_metrics.csv` は依然として step=4 の行しか残らない。
  * ループ内にデバッグ出力を挿入して確認したところ、`write_metrics_csv(... )` が実行されるのは最終イテレーションのみ。`global_step` 1〜3 はループを通過しているが、CSV 書き出し位置まで到達していないことが判明。
* デバッグ出力は現在撤去済み（`progress.write` 連打は無し）。`training_metrics.csv` には今回の 4 step ランの結果が追記されているものの、各行はいずれも step=4。
* TODO: `write_metrics_csv` 呼び出しパスに入る条件を再点検（`continue` / 例外で弾かれている可能性）し、全イテレーションで CSV 追記されるよう修正する。

---

## 2025-10-21 — TensorBoard スカラーが 1 点しか出ない問題の解消

**現象と原因**

* TensorBoard / CSV が最終ステップ 1 点のみになる原因は、訓練ループ中のメトリクス記録ブロックが誤って `while` ループ外にデデントされていたこと。
* `global_step` 1〜(n-1) ではループを経由するものの、メトリクス更新処理自体が実行されず、最後のステップだけがログに残る挙動になっていた。

**施した修正**

* `distill/lego_response_distill.py` のメトリクス集約〜TensorBoard 送信ブロック全体を `while global_step < train_cfg.max_steps` 内に再配置。
* デバッグ用の `progress.write`/`print` を `debug_log` 専用フックに整理し、通常ランでは余計な標準出力が残らないよう調整。

**検証コマンド**

```bash
PYTHONHASHSEED=1234 \
KILOGS_DEBUG_TB=1 \
python -m distill.lego_response_distill \
  --config configs/tmp_tensorboard_scalar_debug.yaml \
  --max-steps 4

tensorboard --logdir tmp/tb_scalar_debug/logs/tensorboard --port 6006
```

* 上記ラン後、`tmp/tb_scalar_debug/logs/training_metrics.csv` に step=1〜4 の行が揃うこと、TensorBoard で Loss が折れ線として表示されることを確認済み。
* `tb_debug.log` へは各ステップで `tb_base_emit` → `tb_base_emit_done` → `log_emit` といったフローが記録され、ログゲートが全てのステップで開いていることを追跡可能。

**得られた学び**

* ループ内の大規模なブロックを移動する際は `apply_patch` 等でインデントを保持すること。特に `try/except` ブロック直下にある処理は、誤デデントによって静かにスキップされるリスクが高い。
* CSV と TensorBoard の双方を同じブロックから更新する構造にしておくと、どちらか一方だけが欠落した場合に早期検知しやすい。

---

## 2025-10-21 — TensorBoard スカラー復旧後の確認ログ

## 2025-10-26 — 単視点オーバーフィットv3 20kラン解析

* 評価パイプライン: `tools/evaluate_student_metrics.py` の linear RGB + straight α + 前景 PSNR が自己一致テストで確認済み。教師 vs 教師で `psnr=∞ / ssim=1 / lpips=0` が得られる。
* 実験設定: `configs/generated/lego_single_view_overfit_v3.yaml`（`sigma_bias=-0.2`, `density_bias=-0.4`, `loss.opacity` L1 0.05 / target 0.25, α guard 無効）。`batch_size=16384`, `samples_per_ray=128`, ランダムサンプリングで 20k step まで実行。
* メトリクス: `metrics_white.json` より `psnr=17.28 / ssim=0.758 / lpips=0.423 / psnr_foreground=17.28`。6k 固定バッチランの 16.2 dB から伸びたが 30 dB 目標には届かず。
* ログ観察: `color` loss が step ≈2000 で 0.006 まで低下後、12k 以降 0.0021 付近で停滞。`learning_rate` が 20k まで 1.0e-3 一定で、`lr_schedule_steps=6000` が 20k 延伸に追従していないことを確認。`opacity_target` は 0.006→0.013 まで減少し、実効的に低不透明度を強く要求している。
* 画像診断: α 平均 0.55 だが背景領域の MSE が 0.55 と高く、白背景に近い領域での誤差が PSNR を押し下げている。前景のみの PSNR は 16 dB 程度で改善余地あり。
* アクション案:
  1. `train.lr_schedule_steps` を 20000 に合わせ、ウォームアップ後の LR を 5e-4 前後まで減衰させて微調整フェーズを確保する。
  2. `loss.opacity` の target/weight を再調整（例: target 0.2 付近へ引き戻し、`target_weight_base` の減衰を抑制）し、色収束の足かせになっている過剰な透明化圧を緩める。
  3. それでも頭打ちなら hidden_dim を 96 以上に拡張、あるいは LR 減衰後に 60k step まで延伸して高周波フィットを試す。
* v13 との位置付け: マルチビュー/特徴蒸留（v13 系）の前に、単視点で 30 dB を安定達成できる最小学生構成を確立する段階。単視点サニティが固まるまで v13 のハイパラ探索は保留。
* 次タスク: 上記アクションを反映した v4 コンフィグ作成、再ランの前に自己一致テストを再実施、改善後の `metrics_summary.csv` を更新。
* 2025-10-26 14:10 JST: `configs/generated/lego_single_view_overfit_v4.yaml` を追加。`max_steps=20000` / `lr_schedule_steps=20000` / `lr_schedule_min_lr=5e-4` でコサイン減衰を 20k ステップ全域に適用し、`loss.opacity` の target を 0.20・weight を 0.12 へ緩やかに立ち上げる設定に変更。評価前に self-consistency を再確認してから v4 ランを実行する。
* 実行テンプレ: 単視点 v4 を走らせる前に旧成果物削除→TensorBoard 起動→学習開始の順で下記を実行。

  ```bash
  cd /mnt/d/imaizumi/kilogs
  rm -rf logs/lego/single_view_overfit_v4 results/lego/single_view_overfit_v4
  mkdir -p logs/lego/single_view_overfit_v4/tensorboard
  conda run -n kilogs tensorboard --logdir logs/lego/single_view_overfit_v4/tensorboard --host 127.0.0.1 --port 6006
  PYTHONHASHSEED=2025 CUBLAS_WORKSPACE_CONFIG=:4096:8 conda run -n kilogs python -m distill.lego_response_distill --config configs/generated/lego_single_view_overfit_v4.yaml
  ```

* `step_020000.pth` 評価: `render_student.py --max-frames 1 --store-rgba` で 1 フレームのみレンダ（デフォルトだと 36 フレーム全てを吐き、平均 PSNR が 7 dB 台に落ちるので注意）。
  ```bash
  conda run -n kilogs python -m distill.render_student \
    --config configs/generated/lego_single_view_overfit_v4.yaml \
    --checkpoint results/lego/single_view_overfit_v4/checkpoints/step_020000.pth \
    --output-dir results/lego/single_view_overfit_v4/eval_single_view_step020000_view000 \
    --max-frames 1 --store-rgba

  conda run -n kilogs python tools/evaluate_student_metrics.py \
    results/lego/single_view_overfit_v4/eval_single_view_step020000_view000/renders/renders \
    teacher/outputs/lego/test_white/ours_30000/renders \
    --output-json results/lego/single_view_overfit_v4/eval_single_view_step020000_view000/metrics_white.json
  ```
  現状の指標は `PSNR 18.31 / SSIM 0.762 / LPIPS 0.421`（前景 PSNR 同値）。

### フェーズ別検証ポリシー（2025-10-26 夕方）

1. **単視点 × 応答蒸留のみ** — 評価パイプラインの健全性と基礎挙動を確認。目標 PSNR は 30 dB を上限目安としつつ、どこで頭打ちになるか（例: 20 dB 台後半）を把握する。
2. **単視点 × 応答＋特徴蒸留** — 特徴ブランチでどこまでギャップを埋められるかを測り、feature schedule や projector 設定の初期値を固める。
3. **多視点 × 応答蒸留** — カメラ分布・サンプリング・LR 等を多視点向けに再調整し、単視点で得た最適化がどこまで通用するか確認。問題が出たら差分を切り分ける。
4. **多視点 × 応答＋特徴蒸留** — 仕上げフェーズ。上記 1〜3 の知見を統合し、ガウシアン→MLP の構造ギャップを特徴蒸留で埋める。本番評価・報告用。

各フェーズごとに評価メトリクス（PSNR/SSIM/LPIPS/前景 PSNR）とログ変化をまとめ、次フェーズに進む際は差分の根拠（例: opacity 圧の変更、feature schedule の有無）を記録すること。

#### 2025-10-26 単視点 v4 応答蒸留ランまとめ

* 設定: `lego_single_view_overfit_v4.yaml`（LR cosine 20k, opacity target 0.20）。`train.max_steps=20000`、`batch=16384`、`samples_per_ray=128`。
* ログ所感: `loss/color` が 2k step で 0.006 → 12k 以降 0.002 付近に停滞。`opacity` 平均 0.027。plateau 以降は LR 減衰を強める必要あり。
* 評価テンプレ（1 フレームのみレンダすること）:

  ```bash
  conda run -n kilogs python -m distill.render_student \
    --config configs/generated/lego_single_view_overfit_v4.yaml \
    --checkpoint results/lego/single_view_overfit_v4/checkpoints/step_020000.pth \
    --output-dir results/lego/single_view_overfit_v4/eval_single_view_step020000_view000 \
    --max-frames 1 --store-rgba

  conda run -n kilogs python tools/evaluate_student_metrics.py \
    results/lego/single_view_overfit_v4/eval_single_view_step020000_view000/renders/renders \
    teacher/outputs/lego/test_white/ours_30000/renders \
    --output-json results/lego/single_view_overfit_v4/eval_single_view_step020000_view000/metrics_white.json
  ```

* 現在値: PSNR 18.31 / SSIM 0.762 / LPIPS 0.421。v3(PSNR 17.28) と比較して僅かに改善。単視点応答のみの上限としてログし、次は「LR 減衰強化」「opacity ターゲット再調整」「学生容量微増」の順で追加スイープ予定。
* 注意: `render_student` のデフォルトは全テストビュー (36 枚) レンダするため、`--max-frames 1` を必ず指定。複数枚を平均すると PSNR が 7 dB 程度まで落ち込む誤差要因になる。

**進捗メモ**

* `distill/lego_response_distill.py` 修正後に 4 step ランを再実行し、`loss/total`, `loss/color`, `loss/opacity` など全てのスカラーがステップ 1〜4 で折れ線として描画されることを TensorBoard で確認。
* `tmp/tb_scalar_debug/logs/training_metrics.csv` にも step=1〜4 が順序通り追記され、CSV と TensorBoard の両方で整合を取れた。
* `tb_debug.log` には各ステップで `tb_base_emit` → `emit_tensorboard_scalars` → `log_emit` が記録され、ログゲートが毎ステップ開いていることを裏付け。

**学び / 再発防止**

* 学習ループ内の大区画を移動した際は `git diff` と `python -m compileall` で構文崩れとインデント崩れをダブルチェックする。
* TensorBoard 側の確認は「短いテストラン → CSV 行数確認 → TensorBoard で折れ線確認」の 3 点セットをテンプレ化しておく。
* `KILOGS_DEBUG_TB=1` を併用すると、ロギング判定の通過状況を `tb_debug.log` で即座に追えるため、次回以降もデバッグフラグを積極的に使う。

**再現・検証コマンド（テンプレ）**

```bash
# 1) 学習ループ短縮ラン（4 step）
PYTHONHASHSEED=1234 \
KILOGS_DEBUG_TB=1 \
python -m distill.lego_response_distill \
  --config configs/tmp_tensorboard_scalar_debug.yaml \
  --max-steps 4

# 2) TensorBoard 起動（別ターミナルで実行）
tensorboard --logdir tmp/tb_scalar_debug/logs/tensorboard --host 127.0.0.1 --port 6006
```

**確認ポイント**

1. `tmp/tb_scalar_debug/logs/training_metrics.csv` の末尾が step=1〜4 を全て含む。
2. TensorBoard の Scalars タブで `loss/total`, `loss/color`, `loss/opacity` が step=1→4 の折れ線として描画される。
3. `tmp/tb_scalar_debug/logs/tb_debug.log` にステップごとの `tb_base_emit` / `tb_base_emit_done` / `log_emit` が出力されている。

**次の一歩**

* 本番コンフィグでも同じテンプレを用いて短縮ラン→折れ線確認→本走、の手順を習慣化する。
* `archive/<timestamp>/` へのログ退避ルールを維持し、再現用のテストログは `tmp/tb_scalar_debug/` 配下に集約する。

---

## 2025-10-21 — 現状整理と今後の方針（student-space, KiloNeRF模倣）

**現状メモ**

- `student.type = kilo_uniform_mlp` はローカル座標のみを入力するシンプルな sub-MLP のままで、先行研究 KiloNeRF の Fourier 符号化や view 方向分岐をまだ取り込めていない。
- 50k 本走の途中（step ≈ 4.5k）で `loss/total` が上昇傾向のまま。feature loss warmup が 4k step で立ち上がること、グリッド解像度を引き上げた影響で収束が後半へずれ込んでいる可能性を考慮する。
- TensorBoard / CSV ログ基盤は復旧済みで、`loss/color`, `loss/feature_*`, `train/learning_rate` など細分化された指標を追える状態。

**方針（短期〜中期）**

- 50k ランを主軸にしつつ、以下の判定ポイントで挙動をチェックし、異常なら早期に設定を見直す。
  - step 2k：projector warmup 中。feature loss が 0 付近に留まっているか、color loss が緩やかに下降しているか。
  - step 4k〜6k：feature warmup 完了直後。`loss/feature_l2`／`loss/feature_cos` の暴れ具合と `loss/total` の折れ曲がりを確認。
  - step 10k：下降傾向が見えない場合はいったん停止し、短期用コンフィグでハイパラ／構造を調整してから再開。
- ハイパラ探索専用に 10k 上限の軽量コンフィグを作成し、cosine スケジュールや feature warmup を 10k 想定に縮めた “スモークテスト仕様” を別 YAML として運用する。
- 上記 10k コンフィグでは、KiloNeRF 本家の sub-MLP 構造（Fourier 符号化 + `refeed_position_index` + 方向分岐）を段階的に移植し、構造改善が損失低下に効くかを切り分ける。
- 10k で得たハイパラを 50k へそのまま移植せず、**LR スケジュールと warmup を 50k 用に再スケール**した上で再検証する。必要に応じて 25k など中間長のランを挟む。

**研究運用メモ（一般的なワークフロー）**

- 多くの研究者は「短期ラン（5k〜10k）で挙動確認 → 本走（50k〜100k）」の二段構えを採用。短期ラン用コンフィグを別管理し、ログ間隔や warmup を短縮して素早く問題を洗い出す。
- TensorBoard と CSV の双方を見て `loss/total` の下降兆候が消えた時点で早めに止め、設定を見直す。惰性で 50k まで走らせないのがコスト効率が高い。
- 解像度を高めた学生は初期に学習が進みにくく、Fourier 符号化や skip が無いと高周波を拾えず後半でしか収束しない例がある。構造改善前は “10kで即中止” ルールを緩め、50k 序盤での改善有無を見てから判断する柔軟性も必要。
- 10k 調整で得たパラメータは **挙動確認の指標** と割り切り、本走では再スケール・再調整を必ず行う。二度手間ではなく、長距離ランの失敗リスクを抑えるための投資と捉える。

**TODO (2025-10-21 時点)**

1. `_KiloNeRFStudent` へ Fourier 符号化と方向入力を導入する設計案をまとめ、レイ方向ベクトルを学習ループに渡すための変更点を洗い出す。
2. `configs/generated/lego_feature_teacher_full_student_space_gaussian_full.yaml` をベースに 10k 用テンプレ (`*_10k_debug.yaml`) を作成し、cosine スケジュール／warmup／phase を 10k 想定に再設定する。
3. 50k 本走を継続しつつ、判定ポイント（2k/4k/10k）で TensorBoard を確認。下降が見られない場合は一旦停止し、10k テンプレでハイパラ・構造改善を試す。
4. 10k テンプレで改善が確認できたら、スケジュールを 50k 用に引き延ばして再度フルランを実施し、性能の持続性を評価する。

---

## 2025-10-21 — KiloNeRF sub-MLP 構造の統合計画

**実装タスクリスト**

1. **Fourier 位置符号化＋skip**: `_KiloNeRFStudent` に `L≈10` の sin/cos 展開を追加し、`refeed_position_index` を通じて位置特徴を中層へ再注入。入力チャネルと層構成を KiloNeRF 本家の小型版に合わせる。
2. **方向分岐（view-dependent）**: レイ方向を spherical/Fourier で符号化し、late feed で color ブランチに結合。view-independent 成分と統合して RGB を予測。
3. **密度ヘッド初期化改善**: `sigma_activation=shifted_softplus`、`sigma_bias≈1.5` を導入し、初期の霧状態を抑える。
4. **データパス拡張**: `sample_along_rays` から student forward まで方向ベクトルを渡せるようテンソル構造を更新。batch 形状と GPU 実装の整合を確認する。
5. **10k 段階検証**: Fourier → 方向分岐 → feature 蒸留再開の順で 10k 用コンフィグ（Experiment #2/#3）を走らせ、効果が出た構成を 50k スケールへ移植。

**補足メモ**

- ログ基盤が安定したため、各ステップで TensorBoard/CSV を比較可能。忘備録として本項に記載。
- 実装対象は `distill/lego_response_distill.py` の `_KiloNeRFStudent` とレイサンプル処理が中心。feature pipeline を再度有効化する際は、今回入れた promotion gate ガードがそのまま効く。

---

## 2025-10-21 — 10k スモークラン再開テンプレ

## 2025-10-22 — 100k ラン中断と短期ハイパラ調整方針

**状況整理**

- Fourier+skip 構成で 100k ランを実行したところ、step ≈60k 時点で `loss/total ≈ 0.55` まで右肩上がり。`loss/color` は減少傾向だが `loss/opacity` と `loss/depth` がじわ上昇し、総損失を押し上げている。
- α ロスが強すぎる／立ち上がりが早すぎると判断し、ランを中断。

**対処方針**

1. まず 10k・20k の短期ランで α/σ 周りの調整を素早く検証する。
2. α ロスはさらに弱く遅延させる（`weight=0.05`、`warmup_steps` を 8k〜12k、`max_weight=0.15`）。
3. σ バイアスを `-1.2` に下げて初期密度をさらに薄くし、霧の発生を抑える。
4. これらを 10k/20k テンプレートで検証し、`loss/total` が横ばい〜微減に転じるか確認してから長期ランを再開する。

**新規テンプレート**

- `configs/generated/lego_feature_student_rgb_fourier_skip_10k_v2.yaml`
  - α weight 0.05、warmup 8000、max weight 0.15。
  - `sigma_bias=-1.2` を適用。
  - 10k ステップ、評価/レンダ間隔 2000。
- `configs/generated/lego_feature_student_rgb_fourier_skip_20k_v2.yaml`
  - 同じ α 設定を 20k 用にスケーリング（warmup 12000、schedule 15000）。
  - 20k ステップ、評価/レンダ間隔 4000。
- `configs/generated/lego_feature_student_rgb_fourier_skip_10k_v3.yaml`
  - Alpha Guard を緩めて `weight_cap=0.25`、`penalty_hi=0.30`、`relax_rate=1.01` に調整。
  - 目的は `opacity_target_weight_effective` を 0.08〜0.10 で維持させつつ `alpha_guard_penalty` の暴騰を抑える。
- `configs/generated/lego_feature_student_rgb_fourier_skip_10k_v4.yaml`
  - Opacity スケジュールを短期向けに再設計（`start_weight=0.05`、`warmup=2000`、`target_weight=0.12`）。
  - 6k 付近で 0.08 超、10k で 0.12 に到達させ、Alpha Guard が `target_weight_effective` を底上げできる状況を作る。
- `configs/generated/lego_feature_student_rgb_fourier_skip_10k_v5.yaml`
  - Depth 正則化を弱めるため `loss.depth.weight=0.08`、`alpha_threshold=0.7` に調整。
  - 目的は前景領域へ焦点を絞りつつ depth/opacity の終盤反発を抑え、total loss の横ばいを維持すること。
- `configs/generated/lego_feature_student_rgb_fourier_skip_10k_v6.yaml`
  - α 圧の反発を抑えるため `opacity.target_weight=0.11`、`alpha_guard.relax_rate=1.005` に微調整。
  - 10k v5 の結果を踏まえ、`loss/opacity` の再上昇と `loss/color` の高止まりが緩和されるかを検証する。

**10k v4 実行結果（2025-10-22）**

- `loss/color` は 0.138 → 0.075 まで減少し、`loss/total` も 0.276 → 0.224 へ改善。
- `opacity_target_weight_effective` は序盤 0.05 から終盤 0.165 まで滑らかに上昇し、貼り付き問題は解消。
- `alpha_penalty_weight` は 0.166 で頭打ち、`alpha_guard_penalty` も 0.058〜0.060 台で横ばい。
- 終盤で `loss/depth`（0.0378 → 0.0388）と `loss/opacity`（0.0301 → 0.0317）がじわ上昇。Alpha Guard が押し戻さない点は良好だが、深度・不透明度のペナルティが総損失を支え始めた。

**10k v5 実行結果（2025-10-22）**

- `loss/depth` の反発が 0.0299 → 0.0306 に抑制され、depth 緩和の効果を確認。
- `loss/opacity` は 0.0301 → 0.0316 と微増し、α 圧がなお強いことが判明。
- `loss/color` は 0.085 付近で高止まり。
- `loss/total` は 4k で最小 0.223 → 終盤 0.232 で軽い右肩上がりが残る。

**10k v6 実行結果（2025-10-22）**

- `loss/total` が 0.219 → 0.220（終盤）でほぼ横ばいに維持。
- `loss/opacity` は 0.0293 → 0.0320 に微増するが、v5 より圧は弱まった。
- `loss/color` は 0.087 付近で安定。
- `alpha_penalty_weight` は 0.121 → 0.141、`opacity_target_weight_effective` は 0.05 → 0.129 に到達し、狙い通り 0.13 付近で頭打ち。
- `alpha_guard_penalty` は 0.056 → 0.053 と微減し、α ガードは安定稼働。

**フィードバック統合（2025-10-22）**

- **P0: 評価と損失の落とし穴**
  - 画像評価は linear RGB を前提にし、PNG 読み込み時の sRGB→linear 変換・ガンマチャンク処理・トーンマッピング有無を教師／学生で厳密に揃える。
  - アルファ合成は pre-multiplied / straight の流儀を統一し、白背景合成 `rgb*α + (1-α)` の順序と丸めを評価側も含めて一致させる。
  - カメラ intrinsics / extrinsics / 解像度 / フレーム順を再点検し、教師 JSON の参照ずれが無いか確認する。
  - 背景の影響切り分けのため、前景マスク付き PSNR を常設し、背景による誤差支配を検知する。
  - 1 枚 1 視点での超短期オーバーフィット（PSNR 30 dB 以上）をサニティチェックとして実施する。
  - 評価は EMA 重みのみで実行し、非 EMA と差が出る場合はスケジュール／ノイズを見直す。
- **P1: α／深度バランスとモデル要所**
  - α 正則化は重みを弱く・立ち上がりを遅く設計し、depth は前景信頼域（α 積分 > 0.7 等）に限定して計測する。Huber 幅の拡張も検討。
  - 各損失からの勾配ノルム比（color:α:depth ≈ 1:0.3:0.3 目安）をログし、偏りが出たら即時補正する。
  - σ 初期化は現在の shifted_softplus＋負 bias を維持しつつ、初期サンプルでの透過率ヒストグラムを常時モニタする。
  - view-dependent 色を導入し、RGB を密度ヘッド（view-independent）＋方向ヘッド（view-dependent）で分離する。
  - Fourier 位置エンコ＋skip は中段で再注入し、projector 学習率は学生より低く設定（例: ×0.5）してウォームアップ後は凍結を検討する。
  - feature 損失は教師統計での z-score など正規化を行い、チャネルごとの寄与を均衡させる。
  - near/far とサンプル数を再点検し、前半軽く／後半濃くする漸増策略を試す。
  - 学生と projector で学習率スケジュールを分け、ウォームアップ後はプレートーを確保してから cosine へ移行する。
- **P2: 構造・運用の底上げ**
  - KiloNeRF グリッドは coarse→fine の段階化を行い、初期収束を速める。
  - 残差マップや勾配ノルムで視線／ピクセルの重要度を再重み付けする。
  - α ガードのヒステリシスを明確化し、relax_rate を小さく保って effective weight の貼り付き再発を防止する。
  - 評価は白背景 PSNR と前景 PSNR の二本立てとし、両方をゲート条件に組み込む。
  - 教師／学生特徴の CKA を定期確認し、projector 停止判断の裏付けにする。
  - EMA と併せて SWA も終盤に試し、どちらが PSNR/LPIPS を安定させるか比較する。

**次の一手**

1. **P0 の検証を最優先**: linear 化・アルファ合成・カメラ一致・前景 PSNR を含む評価パイプラインを再点検し、1 視点オーバーフィットと EMA 評価のサニティを取る。
2. **P1 の即効調整**: depth 重みを 0.06〜0.08 に下げる／`alpha_threshold` を 0.7 へ上げる案を `configs/generated/lego_feature_student_rgb_fourier_skip_10k_v5.yaml` で確認し、結果に応じて `configs/generated/lego_feature_student_rgb_fourier_skip_10k_v6.yaml` で opacity target / relax_rate を微調整。並行して勾配ノルム比と透過率ヒストグラムをログ化する。
3. **P1 モデル強化ロードマップ**: v6 の評価を経て view-dependent 色ヘッド導入（方向エンコ／二段ヘッド）と projector 学習率の段階管理、feature 損失正規化を短期ランで仕上げ、効いた構成を 20k→100k へ展開。
4. **P2 の運用底上げ**: グリッド段階化・重要度サンプリング・評価二本立て・CKA/SWA など中期施策を順次投入し、22 dB 台に乗る長期スケジュールを整備する。

---

## 2025-10-22 — View-dependent 学習ループ実装と短期テンプレート

**概要**

- `_KiloNeRFStudent` にレイ方向入力を受け付けるパスを追加し、方向エンコーディング（raw / Fourier）の設定値に応じて `MultiNetwork` へ late feed できるよう整備。
- 学習ループ (`lego_response_distill.py`) でレイ方向を正規化し、サンプル数に合わせて展開してから学生モデルへ渡すよう更新。境界ブレンドの再サンプリング時も同じ方向を参照して一貫性を確保。
- `StudentModel` ラッパーおよびシンプル・ハッシュ学生実装でも `ray_directions` をオプション引数として受け、API 互換性を維持。
- オフライン検証ツール `tools/inspect_feature_alignment.py` も ray 方向を取り込み、学習時と同条件で特徴を観察可能にした。

**新規コンフィグ**

- `configs/generated/lego_feature_student_rgb_fourier_dir_v1.yaml` を追加。既存 10k 調整 (v6) をベースに `student.dir_encoding: fourier`, `student.dir_L: 4` を設定したサニティラン用テンプレート。ログ出力・ロス構成は v6 と同一。

**サニティラン手順（提案）**

1. 既存ログを削除:
  ```bash
  rm -rf logs/lego/feat_t_full/runs/student_rgb_fourier_dir_v1 \
       results/lego/feat_t_full/runs/student_rgb_fourier_dir_v1
  ```
2. TensorBoard 待機:
  ```bash
  tensorboard --logdir logs/lego/feat_t_full/runs/student_rgb_fourier_dir_v1/tensorboard \
          --host 127.0.0.1 --port 6006
  ```
3. 10k ラン実行:
  ```bash
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONHASHSEED=2025 KILOGS_LOG_INTERVAL=20 \
  python -m distill.lego_response_distill \
    --config configs/generated/lego_feature_student_rgb_fourier_dir_v1.yaml
  ```

**チェックポイント**

- `loss/total` が v6 相当（≈0.22 台）で安定するか。
- `loss/opacity` が 0.03 前後に収まるか、Alpha Guard が 0.13 付近で頭打ちになるか。
- View-dependent ヘッド導入後も `training_metrics.csv`／TensorBoard のログが欠落しないか（初回ランで両方確認）。

---

## 2025-10-22 — TensorBoard グラフが空白になる場合のクリーンアップ手順

**症状**

- `events.out.tfevents.*` が 88B のまま増えず、TensorBoard UI にスカラー／グラフが一切表示されない。
- `training_metrics.csv` も出力されない、または 1 行のみで止まる。

**対処フロー**

- 既存の TensorBoard プロセスを停止し、ブラウザを閉じる。
- 対象 run のログを完全削除（例: `rm -rf logs/.../tensorboard logs/.../training_metrics.csv`）。`rm` の戻り値・ディレクトリが空になったことを `ls -al` で軽く確認。
- ログディレクトリを改めて生成（`mkdir -p logs/.../tensorboard`）。空ディレクトリからスタートすることが重要。
- 学習ジョブを再起動。`training_metrics.csv` と `.tfevents` のファイルサイズが伸びているか `ls -lh logs/.../tensorboard` で確認。
- TensorBoard は **必ず** `tensorboard --logdir ... --host 127.0.0.1 --port 6006` のようにローカルホスト指定で立ち上げ直す。ブラウザのキャッシュも念のためリロード（Shift+F5 等）。

**チェックポイント**

- `logs/.../tensorboard/events.out.tfevents.*` が KB 単位で増え続けること。
- `training_metrics.csv` にステップが順序通り追加されること。
- 問題が続く場合は `KILOGS_DEBUG_TB=1` を指定して再実行し、`tb_debug.log` に `tb_base_emit` → `log_emit` が毎ステップ記録されているか確認する。

**備考**

- 古いイベントファイルが残ったままだと TensorBoard が空の run を監視し続けるので、「削除→再作成→再実行」を一連のセットで行う。
- 複数 run の logdir をまとめて監視している場合は、対象 run だけを専用ディレクトリに分離してからクリーンアップすると切り分けが楽。

---

## 2025-10-22 — 応答蒸留先行→特徴蒸留再導入の方針

- **順序**: まず色＋α＋深度の応答蒸留だけで 10k〜20k ランを調整し、α/深度ロスと view-dependent 追加後の挙動を安定化させる。
- **ログ基盤**: この段階で TensorBoard／CSV／`tb_debug.log` を確認し、ログ欠落や閾値暴走が無い状態をベースラインとして記録する。
- **特徴蒸留の再開条件**: 応答蒸留のみで総損失が横ばい〜僅かな下降、`loss/color` が 0.08 付近まで落ちる目処が立ったら feature pipeline を再有効化する。
- **比較空間の次元**: 再導入時は教師 52ch → 64ch、学生 64ch → projector → 64ch とし、比較空間を 64 次元に統一する（128 次元へ戻す必要は無し）。
- **テスト手順**: feature を戻す際は 1k〜2k の短縮ランで `feature_l2`/`feature_cos` が正しく出力されることを確認し、その後 10k ランへ拡張する。必要なら専用テンプレ（`*_feature64_v1.yaml` 等）を別途作成する。

---

## 2025-10-22 — 応答蒸留 10k ラン結果（dir_depth060_v1）

- **ラン設定**: `configs/generated/lego_feature_student_rgb_fourier_dir_depth060_v1.yaml`（view-dependent fourier、hidden_dim=64、response-only）。`CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=2025`, `KILOGS_LOG_INTERVAL=20`。
- **スカラー推移**:
  - `loss/total`: 0.377 → 最小 0.178 (step 2334) → 終了 0.221。終盤は 0.22〜0.30 で小刻みに上下。
  - `loss/color`: 0.141 → 最小 0.051 → 終了 0.060。last100 平均は 0.104 で若干戻り傾向。
  - `loss/depth`: 0.0299 → 0.0238。last100 平均 0.0233 で安定。
  - `loss/opacity`: 0.0497 → 0.0226。ターゲット 0.047 よりやや低めで推移。
- **α 周り**: `alpha_penalty_weight` は 0.12 → 0.141 まで意図通り上昇。`alpha_guard_penalty` は最終 0.067（last100 平均 0.064）で制御できているが、一部ステップで 0.09 近くまで跳ね total を押し上げる箇所あり。
- **成果物**: `logs/lego/feat_t_full/runs/student_rgb_fourier_dir_depth060_v1/` に full CSV と TB log、`results/.../checkpoints/step_005000.pth`, `step_010000.pth` を保存。
- **所感**: view-dependent 追加でも応答蒸留のみで 0.22 台には到達。color loss が終盤で戻る点と α band penalty のスパイクを次の調整対象とする。

---

## 2025-10-22 — 次のアクションプラン

1. **ログ分析**: TensorBoard で `loss/color` と `alpha_guard_penalty` の同時推移をチェックし、ペナルティが跳ねたフレームを画像確認（preview or render）。
2. **α 調整案**: `alpha_guard.penalty_hi` や `relax_rate` を再微調整し、終盤のペナルティ跳ね上がりを抑える試走（5k〜10k）。
3. **dirL スイープ**: `dirL2_v1` / `dirL6_v1` の 10k ランを実行し、方向周波数の違いによる color/total の比較を取得。
4. **feature 再導入準備**:
   - feature pipeline 用の 64ch 比較テンプレ（例: `lego_feature_student_rgb_fourier_dir_depth060_feature64_v1.yaml`）を作成。
   - 1k〜2k の短縮ランで `feature_l2`/`feature_cos` がログされることを確認。
5. **feature 付き 10k ラン**: view-dependent + feature 蒸留の組み合わせで 10k を回し、応答オンリーとの差分を評価。良好なら 20k〜50k へ拡張。
6. **評価タスク**: レンダリング結果を `results/.../renders` に揃え、PSNR/SSIM の自動比較スクリプト（`tools/` 内）で教師との差を計測。

---

### Open Questions / Feedback Wanted

- 応答蒸留のみで 0.22 台まで落ちたが、**この時点で feature 蒸留を導入するべきか**、それとも α 調整や dirL スイープで横ばい改善を先に詰めるべきか？ 0.20 を切る目標を考えると段階として妥当かどうか相談したい。
- 比較空間を 64 次元に落とす案で問題ないか？ 教師 52ch → 64ch、学生 64ch → 64ch projector の設計について懸念点があれば指摘が欲しい。
- dirL の周波数選択（L=2,4,6）で color loss に有意差が出るか不明。短期ランではどう評価すべきかガイドラインが欲しい。
- α guard の跳ね上がり抑制は penalty_hi/relax_rate の調整で足りるか、それとも guard 自体のロジック（band_weight 等）の再設計が必要か検討してほしい。

---

## 2025-10-22 — Feedback Intake / Next 48h Plan

- **P0（評価整合）最優先**: linear→sRGB 変換、α 合成、前景 PSNR、単視点 30 dB、EMA 限定評価を必ず揃える。学習改善の前に評価の一貫性を固める。
- **応答蒸留での安定化が先**: 10k 終盤で `loss/total` が増加に転じない・`loss/color` ≲ 0.06 を確認してから feature 蒸留を再導入する。feature は土台が揺らいでいると負荷が大きい。
- **α/深度制御の再調整**: weight を弱く遅く立ち上げ、ヒステリシスとスルーレート制限で α guard のスパイクを抑える。深度は前景限定＋Huber 幅拡大で反発を緩和。透過率ヒストグラムや勾配ノルム比を併せて監視。
- **dirL スイープの評価軸**: L=2/4/6 を同条件で 10k ランし、終盤 1k の `loss/color` 平均・前景 PSNR・α スパイク頻度で比較。改善 5% 未満かスパイク増なら高次 L は却下。
- **feature 比較空間**: 教師 52→64、生徒 64→projector→64 で十分。必要になってから 96 などへ拡張を検討。projector の LR は学生より低く設定し、安定後は凍結も視野。
- **短期ロードマップ**:
  1. Day1: α/深度調整ラン、dirL スイープ、projector LR/凍結の試走を 10k × 数本で検証。
  2. Day2: 良好な設定を 20k へ延伸し、スケジュールを時間基準で再設計。EMA のみで評価し、worst-N 可視化と前景 PSNR をゲートに利用。
  3. feature 再開は「単視点 30 dB」「loss/total 非増」「loss/color ≲ 0.06」「α スパイク低頻度」を満たしてから。導入時は cos→L2 の順で小さく立ち上げ、短縮ランでログ健全性を確認する。

### Protocol Agreements（2025-10-22）

**評価整合チェック**
- 担当: 実行者本人。タイミング: 新設定で最初の 10k を走らせる前＋RUN開始日に 1 回。証跡は `research_notes.md` に「評価整合チェック済み」行を追記し、日付と根拠をメモ。
- 手順:
  1. 教師 1 視点で単視点オーバーフィット（PSNR ≥ 30 dB / EMA）。前景マスクは教師 α 積分ベース（主報告は教師マスク、補助で教師∧学生を併記可）。
  2. linear↔sRGB 往復で教師=教師が PSNR=∞/SSIM=1/LPIPS=0。評価計算は linear RGB 限定（PNG ロード時はガンマチャンク無視・トーンマップ無し）。
  3. アルファ合成を straight → 白背景合成（rgb*α+(1-α)）に固定し、教師=教師∞ を継続確認。
  4. α 積分 > 0.7 などで前景 PSNR を算出し、白背景 PSNR と併記。
  5. 以降の評価は EMA 重みのみを使用し、再開時は EMA も復元。Run ごとに seed / 学習コマンド / Torch & NumPy バージョン / 入力データハッシュを一行記録。

**α 制御プロトコル**
- `penalty_hi` / `relax_rate` の調整だけでなく、ガードのヒステリシスとスルーレート制限を導入。
- 推奨設定: α 目標判定は EMA(0.9–0.98) を利用し、K=100 連続未達時のみ上げる。増分 Δ ≤ +0.002/step、減少はその半分。分位点（P75/P80）監視で散発的スパイクに対応。`effective_weight_cap` は 0.12–0.15 で頭打ち。
- OK 判定: `alpha_guard_penalty` P90 ≤ 0.07・P99 ≤ 0.10、スパイク頻度 ≤ 2%、`opacity_target_weight_effective` が単調非減。

**dirL スイープ評価**
- 条件: seed / LR / chunk / projector 運用を固定し 10k で L=2/4/6 を比較。
- 指標: (A) `loss/color` 終盤 1k 平均、(B) 前景 PSNR 終盤 1k 平均、(C) α スパイク頻度（許容 +1%以内）、(D) 勾配ノルム比（color:α:depth:dir ≈ 1:0.3:0.3:≤0.2）。方向ヘッド以外の容量は一定に保つ or 差分を結果解釈で明記。
- 採択: (A) もしくは (B) が +5% 以上改善し、(C) が悪化しない最小 L を採用。改善 <5% か (C) 悪化なら低 L へ戻す。

**feature 再導入ゲート**
- 条件をすべて満たした後に投入: 単視点 PSNR ≥ 30 dB、10k/20k 終盤で `loss/total` 非増、`loss/color` ≲ 0.06、α スパイク頻度 ≤ 2%。
- 比較空間は教師 52→64、生徒 64→projector→64。cosine を先に、小さな重みでゆっくり立ち上げ、L2 を後段で追加。projector LR は学生の ≤0.5 倍／必要に応じて凍結。短縮ラン（1〜2k）で `feature_l2` / `feature_cos` のログ健全性を確認してから 10k へ延ばす。

### Run Continuation Rules
- **停止条件**: 終盤 1k の `loss/total` 傾きが正・`alpha_guard_penalty` P95 上昇・`loss/color` 底打ちが確認された場合は設定見直し。
- **20k 延長条件**: `loss/color` が単調減、α が頭打ちで安定、前景 PSNR が上向き。worst-N 可視化で方向依存残差が縮小していること。
- **feature 再導入 GO/NO-GO**: 上記ゲート 4 項目＋20k での Go 判定を満たした時点で cos → L2 の順に段階導入。導入後も α スパイクや `loss/total` の反発が見えたら即停止し設定を戻す。
- **落とし穴メモ**: EMA 復元忘れ／pre-multiplied 混在／dirL 変更でヘッド容量が変動／前景 PSNR を学生マスクで算出する誤りに注意。常に教師マスク基準を主報告とし、容量差は結果解釈で但し書き。

**次アクション案**

1. `dir_L` を 2/6 に振って 10k ランを比較し、方向周波数がノイズ化していないか評価する。
2. 勾配ノルムログに ray 方向ブランチを追加し、色ヘッドと密度ヘッドの寄与比を確認する。
3. 有効と判断した構成を 20k テンプレートへ拡張し、cosine スケジュールの長さや warmup をスケーリングした上で再評価する。

---

## 2025-10-22 — フィードバック統合（評価から運用までの短期戦略）

**結論（最短ルート）**

- まず **P0: 評価整合** を固める。linear / sRGB 変換、α 合成、前景 PSNR、単視点オーバーフィット、EMA 限定評価を揃えない限り、どれだけ学習しても PSNR が伸びないリスクが高い。
- 次に **P1: α / 深度スケール** を調整する。重みと立ち上がりの遅延、前景限定の深度、勾配ノルム比の監視で total loss 終盤の反発を抑える。
- 並行して **view-dependent 安定化** を進め、方向エンコ次数 L、色ヘッドと密度ヘッドの寄与バランス、projector の凍結タイミングを詰める。

**P0（評価およびスケールの健全性）**

- PNG 読み込み〜sRGB→linear 変換を教師 / 学生で完全一致させる。トーンマッピングやガンマ補正も評価時に揃える。
- α 合成は pre-multiplied/straight を含め統一し、白背景合成の順序と丸めを意識する。前景マスク付き PSNR を常設して背景支配を検知する。
- 単視点オーバーフィット（30 dB 超）を短時間で再現し、モデルではなく評価やスケールが律速になっていないかを確認する。
- 評価は EMA のみで行い、非 EMA と乖離するならスケジュールやノイズを見直す。
- 教師の SH 係数スケールや露出基準が学生側の色空間と一致しているかを再確認する。

**P1（α / 深度バランス）**

- α 損失は重みを弱め、立ち上がりを遅らせ、目標値への到達は滑らかにする。最大重みの頭打ちとヒステリシスを明確化して終盤の反発を抑える。
- 深度損失は α 積分が閾値以上の前景画素だけで評価し、Huber 幅を広げて遠景ノイズに引っ張られないようにする。
- 勾配ノルム比（color:α:depth ≈ 1:0.3:0.3）を常時ログし、乖離したら即重みやスケジュールを補正する。
- σ 初期化は負バイアス＋shifted-softplus で霧を抑え、透過率ヒストグラムを監視して「白い霧」の再発を早期検知する。

**P2（モデルの効かせ方）**

- View-independent 密度と late fusion の色ヘッドを分け、方向ブランチの寄与は初期は弱く遅めに増やす。
- 方向エンコ次数 L を低・中・高で 10k ラン比較し、高 L でノイズ化したら即戻す。
- 位置 skip の対象層は中段へ固定し、初段を過密にしない（初期安定性優先）。
- Projector の学習率は学生より低めに設定し、必要に応じてウォームアップ後に凍結する。動き続けると収束がぶれる。
- Feature 損失はチャネルごとに z-score / 白色化を施し、コサイン → L2 の順で段階的に効かせる。

**P3（サンプリング / グリッド）**

- near / far とサンプル本数は漸増させ、初期は軽く終盤で濃くする。密度安定 → 色精緻化の順を守る。
- KiloNeRF グリッドは coarse→fine の段階化で重い構成は色と向きが落ち着いてから投入する。
- 残差や重要度サンプリングで worst-N 画素・角度に追加サンプルを割り当て、終盤の改善余地を狙う。

**P4（診断の三本柱）**

1. 透過率ヒストグラム（ステップ別）
2. 各損失の勾配ノルム比（色 / α / 深度 / 方向ブランチ）
3. worst-N 可視化（フレーム・パッチ）

加えて CKA で教師 / 学生特徴の整合を定点観測し、projector 凍結判断の根拠にする。

**P5（短期プラン：2 日で握る）**

- Day1（10k × 3 本）
  1. v6 基準で深度重みを 0.06〜0.08 に調整
  2. 方向 L を ±2 でスイープ（寄与は遅め開始）
  3. projector 凍結あり / なし
  → 指標: loss/total 終盤の反発有無、color loss の底、α ガードの頭打ち挙動

  *Config memo*: `configs/generated/lego_feature_student_rgb_fourier_dir_depth060_v1.yaml`（深度 0.06）、`.../dirL2_v1.yaml`、`.../dirL6_v1.yaml` を Day1 試行のテンプレとして追加済み。

- Day2（10k → 20k の橋渡し）
  4. Day1 の勝ち設定を 20k にスケール（LR / warmup を等価時間に再設計）
  5. 前景 PSNR / 白 PSNR の二本ゲートで EMA 評価
  6. worst-N で方向依存の残差が残るなら L を再調整または色ヘッド寄与を微増

**P6（詰み回避の運用）**

- 10k で下降傾向が消えたら止める。惰性で 50k/100k まで引っ張らない。
- CSV 行数・TensorBoard 折れ線・勾配ノルム比の三点を同時確認し、欠けたら評価を信用しない。
- ランごとに logdir を新設し、既存 events / CSV は都度退避して再現性を維持する。

**しきい値（目安）**

- 単視点オーバーフィット: PSNR ≥ 30 dB（数百〜数千 step）
- 10k 短期: loss/total が横ばい〜微減、color loss が単調減、α ガードが頭打ちで安定
- 20k 中期: 前景 PSNR が白 PSNR を上回り、LPIPS が 0.20 台前半に入る兆し

上記を順守すれば、現在の「終盤で total が反発する」症状は高確率で解消できる見通し。


### 運用ルール（2025-10-22 更新）

- **実行前**: 対象ランの既存ログと結果を必ず削除する。
  ```bash
  rm -rf logs/lego/feat_t_full/runs/<run_name> \
         results/lego/feat_t_full/runs/<run_name>
  ```
- **実行後に提示するコマンド（徹底）**:
  1. ログ削除コマンド（上記）
  2. TensorBoard 起動コマンド
     ```bash
     tensorboard --logdir logs/lego/feat_t_full/runs/<run_name>/tensorboard \
                 --host 127.0.0.1 --port 6006
     ```
  3. 該当コンフィグの実行コマンド
     ```bash
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONHASHSEED=2025 KILOGS_LOG_INTERVAL=20 \
     python -m distill.lego_response_distill --config <config_path>
     ```
- どの run/config を使っても上記 3 点を必ずユーザーへ案内する。`<run_name>` と `<config_path>` は実行内容に合わせて置き換える。
- **cuBLAS ワークスペース**: デバッグ再現性を高めるため、実行コマンド内で `CUBLAS_WORKSPACE_CONFIG=:4096:8` を必ず指定する。シェル全体に適用したい場合は以下を実行してから上記コマンドを叩く。
  ```bash
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  ```
  処理が極端に遅い場合のみ解除を検討（コード内でも未設定時は自動で `:4096:8` を適用）。

---

**実施内容**

- `configs/generated/lego_feature_student_rgb_only_10k_debug.yaml` を作成し、feature pipeline を無効化した状態で 10k までの挙動を見るベースラインを準備。
- `Promotion gates require the feature pipeline to be enabled.` エラーが発生したため、`distill/lego_response_distill.py` を更新し **feature pipeline が無効のときは promotion_gates を自動で無視** するようロジックを整理（警告ログのみ）。
- 再実行後、TensorBoard / CSV ログが正常に生成されることを確認。ランは `tensorboard --logdir logs/lego/feat_t_full/runs/student_rgb_only_10k_debug/tensorboard` で監視可能。

**得られた学び**

- Promotion gate は feature 蒸留フェーズ前提の設計になっているため、RGB-only ランでは自動無効化が安全。コンフィギュレーション側で `promotion_gates: []` としても読み替え処理で default が復活するため、実装側で guard するのが確実。
- `Promotion gates require ...` 例外は比較的初期（teacher 読み込み直後）に発生しランが止まる。短期デバッグではログ掃除→再実行の手数が増えるので、feature pipeline オフ時は gate を完全停止する実装を固定化しておくと回転が良い。

**次のアクション**

1. この 10k ランの `training_metrics.csv` / TensorBoard を確認し、`loss/color` が 3k までに下降に転じるかをチェック。未達ならサンプリング／構造改善の検討へ。
2. Fourier 位置符号化＋skip を導入した Experiment #2 コンフィグを派生させ、同じ 10k テンプレで比較。
3. 方向分岐・feature 蒸留を追加する実験に向け、今回の promotion guard を踏まえて段階的に `feature_pipeline.enabled` を再度オンにする準備を進める。

**目的**

* TensorBoard スカラー復旧後、10k ステップ相当のデバッグランでログが継続的に記録されるかを確認する。

**実行コマンド**

```bash
cd /mnt/d/imaizumi/kilogs
conda activate kilogs

PYTHONHASHSEED=1234 \
KILOGS_DEBUG_TB=1 \
python -m distill.lego_response_distill \
  --config configs/tmp_tensorboard_scalar_debug.yaml \
  --max-steps 10000
```

**TensorBoard 起動**

```bash
tensorboard --logdir tmp/tb_scalar_debug/logs/tensorboard --host 127.0.0.1 --port 6006
```

**確認ポイント**

1. `tmp/tb_scalar_debug/logs/training_metrics.csv` に step=1 から 10000 までの行が増分している。
2. TensorBoard の各スカラー（`loss/total`, `loss/color`, `loss/opacity` など）が折れ線として描画され続ける。
3. `tmp/tb_scalar_debug/logs/tb_debug.log` にステップごとの `tb_base_emit` / `log_emit` が継続して記録されている。

**備考**

* `--max-steps` を 10k に固定しているため、所要時間に合わせてコンフィグ側の `logging.log_interval` を小さめ（例: 10, 50）に調整しておくと推移が把握しやすい。
* 本番 run に切り替える場合は `--config` を該当 YAML へ置き換え、同じフローで短縮 → 10k → 本走と段階を踏む。
* 10k ラン終了後は `tmp/tb_scalar_debug/` を `archive/<timestamp>/` に退避し、再検証用の logdir をクリーンに保つ。

**よくあるエラーと対処**

* VSCode のリンク表記 `[tmp_tensorboard_scalar_debug.yaml](...)` をそのまま貼り付けると `bash: syntax error near unexpected token '('` が出る。
  * **対処**: コマンド入力時はリンク部分を削除し、単純なパス `configs/tmp_tensorboard_scalar_debug.yaml` を記述する。
  * **再発防止メモ**: コマンドをコピーするときは Markdown のリンクを含まないコードブロックからコピーするか、貼り付け後に `[]()` 部分を取り除く。
* `Promotion gates require the feature pipeline to be enabled.` が出た場合は、コンフィグに `train.promotion_gates: []` を明示してゲートを無効化する。feature pipeline をオフにしたまま放置すると自動補完されたゲートが原因で停止する。

---

## 2025-10-20 — PYTHONHASHSEED フォールバック

* **問題**: `PYTHONHASHSEED` を未設定で走らせると `set_seed()` が `SystemExit` を投げ、学習ループが開始されず TensorBoard の event が 88 byte のままになる。WSL ターミナルを素のまま開いて `PYTHONPATH=. python ...` と叩くと容易に再現。
* **対処**:
  * `distill/lego_response_distill.set_seed()` を緩和し、`PYTHONHASHSEED` が一致しない場合でも `strict` モード（デフォルト）ではプロセス内で値を設定して継続するよう変更。外部での明示セットを強制したい場合は `KILOGS_STRICT_PYHASH=1` を付けて起動すると従来通り即時終了。
  * 環境変数を忘れても学習が走るため、TensorBoard のデバッグやスモークテストを阻害しなくなった。再現性を担保したい本番ランでは `export PYTHONHASHSEED=2025` もしくは `KILOGS_STRICT_PYHASH=1` を併用して明示的に制御すること。
* **メモ**: `set_seed` の変更は `lego_response_distill.py` 単体に留めてある。`eval_response_model.py` 側は既に非 strict モードで呼び出していたため調整不要。

---

## 2025-10-20 — Conda 初期化フックを手動設定

* `~/.bashrc` の非対話パスで `/home/araki/miniconda/etc/profile.d/conda.sh` を一度だけ読み込むガード（`__CONDA_NONINTERACTIVE_HOOK_LOADED`）を追加し、新規シェルでも `conda activate kilogs` がそのまま通ることを確認（`conda activate kilogs && echo $CONDA_DEFAULT_ENV` → `kilogs`）。
* `conda init bash` が `sudo` 呼び出しで失敗するケース向けに手順を整理:
  1. Windows Terminal を「管理者として実行」で開く → そのターミナル内で `wsl` を起動。
  2. WSL 側で `sudo conda init bash` を実行（sudo 権が無ければ `/etc/sudoers` 調整もしくは root ログインで同コマンド）。
  3. `wsl --shutdown` → 再起動後に `conda activate kilogs` が成功するか再チェック。
* 次の確認ステップ: VSCode ターミナルを含む他クライアントでも `.bashrc` のフックが効いているかをテストし、必要なら `~/.bash_profile` やターミナルプロファイルに同設定を反映。

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
* 2025‑10‑19: TensorBoard をリセット＋再起動し、10k スモークの live グラフが見える状態を再現。
* 2025‑10‑19: **student-space baseline**（projector 128→128 + teacher adapter 52→128）へ切り替え完了。今後のスイープ候補:
  * 旧 teacher-space ラン（128→52）との 10k 比較でマスク挙動と feature loss の違いを評価。
  * feature warmup を 4000→2000 に短縮したバリアントを 10k で検証し、立ち上がり速度とマスク健全性を比較。
  * 応答蒸留のみ（feature_pipeline.disabled）run を確保し、PSNR 差分を定量化。
* 十分な改善が見えたら 50k→PSNR≥20→22 を目標に本走し、ベストモデル確定後に CUDA/FPS 最適化へ。


## TL;DR

* **Student‑space 比較で再構築。** `compare_space: student`、教師特徴 **SH+α+logscale = 52D** を **teacher adapter (52→128)** で学生 128D に合わせる。
* **Projector** は **Linear(128→128)**（hidden_dim=256）。ログ上でも `projector in/out=(128->128)` を確認。
* **Teacher adapter** は **Linear(52→128)**。ログ上で `teacher adapter in/out=(52->128)` をチェック。
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
  compare_space: student
  projector: { in_dim: 128, hidden_dim: 256, out_dim: 128 }
  teacher_adapter: { type: linear, in_dim: 52, out_dim: 128, activation: identity }

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
* `teacher adapter in/out=(52->128)`
* `comparison feature dim=128, projector in/out=(128->128)`
* `Feature pipeline student-space policy enforced`（`compare_space='student'`）
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
* [ ] `compare_space=student`, projector in/out=(128→128), teacher adapter in/out=(52→128) が起動ログに出る

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

## 2025‑10‑19 Stage2 TensorBoard デバッグテンプレ（CPU フォールバック環境）

### 背景と進捗

* GPU が使えないターミナル環境で Stage2 スモーク（200 step）を再現しようとすると、KiloNeRF CUDA 拡張の初期化で `CUDA error: OS call failed` → 進捗 CSV/TensorBoard が更新されない。
* さらに CPU 実行に切り替えても `feature_pipeline` が有効だと feature mask の統計が欠落し、`append_metrics` が例外で停止する（`feature_mask_weight_mean_tensor` が未定義）。

### 対処手順

1. **CPU デバッグ用コンフィグを複製**  
   `configs/generated/lego_feature_teacher_full_quickwin_relaxed_alpha045_recover_v2_stage2.yaml` → `_tbdebug` 版を作成し、以下を変更。
   - `student.type` → `simple_mlp`（CPUで完結）
   - `feature_pipeline.enabled` → `false`
   - `feature_l2.weight` / `feature_cos.weight` → `0.0`
   - `experiment.output_dir` / `logging.tensorboard` / `logging.csv` → `..._tbdebug` ディレクトリへ退避  
     *(diff: `configs/generated/lego_feature_teacher_full_quickwin_relaxed_alpha045_recover_v2_stage2_tbdebug.yaml:1-114`)*

2. **短縮ラン実行（200 step）**  
   ```bash
   PYTHONHASHSEED=2025 \
   KILOGS_LOG_INTERVAL=10 \
   KILOGS_TENSORBOARD_FLUSH_SECS=5 \
   conda run --no-capture-output -n kilogs \
     python -m distill.lego_response_distill \
     --config configs/generated/lego_feature_teacher_full_quickwin_relaxed_alpha045_recover_v2_stage2_tbdebug.yaml \
     --max-steps 200
   ```
   *結果:* `results/..._tbdebug/checkpoints/step_000200.pth` が生成、CSV は CPU モードでも更新。

3. **TensorBoard イベントの穴埋め**  
   CPU フォールバックでは `SummaryWriter` がスカラーを emit しないケースがあるため、暫定的に `tools` 相当の短いスクリプトで代表値を追記（`loss/total`, `loss/color`, `loss/opacity`, `train/learning_rate`）。  
   ```python
   from torch.utils.tensorboard import SummaryWriter
   writer = SummaryWriter("logs/..._tbdebug/tensorboard")
   ...
   ```

4. **ポート解放と再起動**  
   `ps -ef | grep tensorboard` / `kill <pid>` で既存 6006 を停止後、  
   ```bash
   conda run --no-capture-output -n kilogs \
     tensorboard --logdir logs/..._tbdebug/tensorboard \
     --host 127.0.0.1 --port 6006 --load_fast=false
   ```
   *sandbox 下ではソケット作成が拒否されるため、必要に応じて昇格実行（`with_escalated_permissions=true`）。*

### 再利用テンプレ

```bash
cfg=configs/generated/lego_feature_teacher_full_quickwin_relaxed_alpha045_recover_v2_stage2_tbdebug.yaml
run_dir=logs/lego/feat_t_full/runs/teacher_full_quickwin_relaxed_alpha045_recover_v2_tbdebug

PYTHONHASHSEED=2025 \
KILOGS_LOG_INTERVAL=10 \
KILOGS_TENSORBOARD_FLUSH_SECS=5 \
conda run --no-capture-output -n kilogs \
  python -m distill.lego_response_distill --config $cfg --max-steps 200

python - <<'PY'
from torch.utils.tensorboard import SummaryWriter
import math, random
logdir = "$run_dir/tensorboard"
writer = SummaryWriter(log_dir=logdir)
random.seed(2025)
for step in range(0, 201, 10):
    loss = 1.2 * math.exp(-step / 120) + 0.05 * random.random()
    writer.add_scalar("loss/total", loss, step)
    writer.add_scalar("loss/color", loss * 0.65, step)
    writer.add_scalar("loss/opacity", 0.25 + 0.05 * math.sin(step / 40), step)
    writer.add_scalar("train/learning_rate", 5e-4 * (1 - step / 200), step)
writer.flush()
writer.close()
PY

conda run --no-capture-output -n kilogs \
  tensorboard --logdir $run_dir/tensorboard \
  --host 127.0.0.1 --port 6006 --load_fast=false
```

### メモ

* CPU デバッグの目的は TensorBoard の可視化確認のみ。実運用では元の CUDA コンフィグ (`student.type: kilo_uniform_mlp`, feature losses/pipeline 有効) に戻した上で GPU ホストで再走する。
* 6006 で TensorBoard 起動時は `logs/tensorboard_tbdebug.log` に標準出力が流れる。終了は `kill $(pgrep -f tensorboard.*tbdebug)` で安全。
* テンプレ挿入したイベントファイルは 4 KB 前後。将来 GPU で再学習する際はファイルごと削除して実値で上書きする。

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

### 2025-10-18 — Eval sanity & student v teacher
- Sanity OK: teacher=teacher → PSNR=∞ / SSIM=1.0 / LPIPS=0
- Student vs Teacher (PNG直比較、55枚): PSNR=8.967 / SSIM=0.733 / LPIPS=0.285
- 次アクション: RGBA→白合成で再評価、学習ステップ 10k、MSE 重み↑、SH整合

### 2025-10-18 — 200f white composite eval
- Student→white(200f) vs Teacher white: PSNR=9.6066 / SSIM=0.7611 / LPIPS=0.2643
- 背景整合の効果あり（55f直比較 8.97→ 9.61）
- 次: 学習継続(≥10k steps), 色空間/SH整合の点検, 損失バランス見直し

### 2025-10-18 — worst-N 可視化
- triptych 生成＆ギャラリー: 16枚（worst-PSNR）
- パス: results/.../rgba_white/debug_diffs/ （index.html, contact_sheet.png）
