# Kilo-GS Knowledge Distillation Workspace

このディレクトリは 3D Gaussian Splatting (3D-GS) を教師、KiloNeRF を生徒として蒸留し、高速・高品質・軽量な "Kilo-GS" モデルを構築するための作業場所です。まず LEGO シーンを対象にパイプラインを整備し、その後 Matrix City など大規模シーンへ拡張する方針です。

## 💡 全体方針
- **教師 (3D-GS)**: `d:\imaizumi\3dgs` で学習済みの LEGO モデルとレンダリング結果を使用。
- **生徒 (KiloNeRF)**: `d:\imaizumi\kilogs\student` 以下に軽量化した MLP シャーディング構造を実装。
- **蒸留パラダイム**: まずは応答蒸留 (teacher RGB/opacity → student) を LEGO シーンで確立し、PVD-AL 由来の特徴蒸留へ拡張。

## 📁 ディレクトリ構成
```
kilogs/
  README.md
  environment.yml          # `kilogs` 用 Conda 環境定義
  configs/
    lego_response.yaml     # LEGO 応答蒸留設定
  distill/
    lego_response_distill.py  # 応答蒸留ドライバ
  student/                 # 学生モデル (KiloNeRF ベース) 実装置き場
  teacher/                 # 教師側成果物/シンボリックリンク置き場
  data/                    # LEGO 等のデータセット参照 (必要に応じてリンク)
  logs/                    # 蒸留学習ログ
  results/                 # 評価出力
```

## 🧪 LEGO 応答蒸留手順
1. **環境準備**
   ```bash
   conda create --name kilogs --clone 3dgs
   conda activate kilogs
   cd /mnt/d/imaizumi/kilogs
   ```
   既存の `3dgs` 環境が整っていればクローンでほぼ準備完了です。独立環境が必要な場合は `environment.yml` から作成してください。

2. **教師アセットの配置**
   - `teacher/checkpoints/lego/` に 3D-GS の LEGO 学習済みチェックポイント (例: `point_cloud/iteration_30000/`) をコピーまたはシンボリックリンク。
   - `teacher/outputs/lego/` に白背景合成済みレンダリング (`eval_outputs/lego_white/` など) を配置。

3. **学生モデルの初期化**
   - `student/` に KiloNeRF 実装を配置 (後で自動化予定)。
   - `configs/lego_response.yaml` の `student` セクションでハイパーパラメータを調整。

4. **蒸留実行**
   ```bash
   conda activate kilogs
   python distill/lego_response_distill.py --config configs/lego_response.yaml
   ```
   デフォルトでは教師レンダリングと一致するカメラポーズをサンプリングし、RGB + 透過率の L2 損失で学生を更新します。
   - TensorBoard のログはステップ軸で記録されます。実時間ベースで見たい場合は各設定ファイルの `logging.tensorboard_axis` を `time` に変更してください（`step` と `elapsed` も選択可）。

5. **評価**
   - `results/lego/` に保存されたチェックポイントを用いて、KiloNeRF 推論スクリプトでレンダリングし、`3dgs/tools/export_metrics_csv.py` を再利用して PSNR/SSIM/LPIPS を記録。
   - `teacher` と同じ評価パイプラインで比較し、速度・品質を表形式にまとめます。

## 🔁 今後の TODO
- KiloNeRF 実装を `student/` に配置し、LEGO 推論パイプラインを整備。
- 蒸留ループ内で生徒ネットワークのパラレル GPU 実行を最適化。
- PVD-AL 風の特徴蒸留 (ガウス表現 → 局所 MLP 特徴) を追加。
- Matrix City などの屋外大型シーンへ拡張、バッチ実験の自動化。

---
最初の一歩として LEGO シーンで応答蒸留をしっかり成功させ、ベースライン指標を確立してから Matrix City 等へ進みましょう。
