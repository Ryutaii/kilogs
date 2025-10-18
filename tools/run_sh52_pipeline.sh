#!/usr/bin/env bash
# =====================================================================================
# All-in-One Playbook (Automation Suite): SH52 smoke -> 50k -> render/eval -> auto diagnostics
# =====================================================================================
# Usage:
#   chmod +x tools/run_sh52_pipeline.sh && tools/run_sh52_pipeline.sh
#
# Tune the variables below to match your environment (paths, configs, thresholds, notification hooks). Set RUN_SMOKE, RUN_50K, RUN_RENDER, and RUN_DIAG to 0 to skip stages.
# =====================================================================================

set -Eeuo pipefail

# ---------------------------
# 0) Environment & paths (edit as needed)
# ---------------------------
PROJ_ROOT="/mnt/d/imaizumi/kilogs"
CONDA_ENV="kilogs"

export PYTHONPATH="${PROJ_ROOT}:${PYTHONPATH:-}"

CFG_SMOKE="configs/lego_feature_teacher_full_rehab_masked_white_smoke_s1_sh52.yaml"
CFG_MAIN="configs/lego_feature_teacher_full_rehab_masked_white.yaml"

RUN_DIR_SMOKE="logs/lego/feat_t_sh52/runs/teacher_full_rehab_masked_white_smoke_s1_sh52"
TB_DIR_SMOKE="${RUN_DIR_SMOKE}/tensorboard"
RUN_DIR_MAIN="logs/lego/feat_t_full/runs/teacher_full_rehab_masked_white"
RUN_DIR_MAIN_RESULTS="results/lego/feat_t_full/runs/teacher_full_rehab_masked_white"
TB_DIR_MAIN="${RUN_DIR_MAIN}/tensorboard"

CKPT_10K="${RUN_DIR_SMOKE}/checkpoints/step_010000.pth"
TRAIN_METRICS_SMOKE="${RUN_DIR_SMOKE}/training_metrics.csv"
TRAIN_METRICS_MAIN="${RUN_DIR_MAIN}/training_metrics.csv"
METHOD_NAME_50K="FeatureDistill_SH52_50k"
RENDER_CHUNK=8192
CKPT_50K="${RUN_DIR_MAIN_RESULTS}/checkpoints/step_050000.pth"
CKPT_10K_PATTERN='**/teacher_full_rehab_masked_white_smoke_s1*/checkpoints/step_010000.pth'
CKPT_50K_PATTERN='**/teacher_full_rehab_masked_white*/checkpoints/step_050000.pth'
RENDER_OUTPUT_ROOT="results/lego/feat_t_full/renders/teacher_full_rehab_masked_white"
EVAL_SUMMARY_CSV="metrics_summary.csv"

ALPHA_MEAN_MIN=0.35
ALPHA_MEAN_MAX=0.45
ALPHA_FRAC95_MAX=0.15

MASK_EMERGENCY_MAX=0
WEIGHT_DROP_TOLERANCE=1e-4

SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
NOTIFY_SHELL_CMD="${NOTIFY_SHELL_CMD:-}"

RESULTS_GLOB='results/**/metrics_white.json'

RUN_SMOKE=${RUN_SMOKE:-1}
RUN_50K=${RUN_50K:-1}
RUN_RENDER=${RUN_RENDER:-1}
RUN_DIAG=${RUN_DIAG:-1}

# ---------------------------
# Helper utilities
# ---------------------------
run_in_conda() {
  PYTHONUNBUFFERED=1 conda run --no-capture-output -n "${CONDA_ENV}" "$@"
}

ensure_dirs() {
  mkdir -p "${PROJ_ROOT}/${TB_DIR_SMOKE}" "${PROJ_ROOT}/${TB_DIR_MAIN}"
}

py() {
  run_in_conda python - "$@"
}

find_latest() {
  local pattern="$1"
  py "${pattern}" <<'PY'
import glob
import os
import sys

pattern = sys.argv[1]
paths = glob.glob(pattern, recursive=True)
if not paths:
    print("")
else:
    paths.sort(key=os.path.getmtime, reverse=True)
    print(paths[0])
PY
}

find_checkpoint_rel() {
    local pattern="$1"
    py "${PROJ_ROOT}" "${pattern}" <<'PY'
import glob
import os
import sys

root, pattern = sys.argv[1:3]
search_pattern = os.path.join(root, pattern)
matches = [p for p in glob.glob(search_pattern, recursive=True) if os.path.isfile(p)]
if not matches:
        print("")
else:
        matches.sort(key=os.path.getmtime, reverse=True)
        print(os.path.relpath(matches[0], root))
PY
}

abs_path() {
    local target="$1"
    if [[ -z "${target}" ]]; then
        return 1
    fi
    if [[ "${target}" == /* ]]; then
        realpath -m "${target}"
    else
        realpath -m "${PROJ_ROOT}/${target}"
    fi
}

derive_render_dir() {
    local ckpt_path="$1"
    local step_tag
    step_tag=$(basename "${ckpt_path}" .pth)
    if [[ -z "${step_tag}" || "${step_tag}" == "${ckpt_path}" ]]; then
        step_tag="step_050000"
    fi
    printf '%s/%s\n' "${RENDER_OUTPUT_ROOT}" "${step_tag}"
}

send_notification() {
  local title="$1"
  local body="$2"
  if [[ -n "${SLACK_WEBHOOK_URL}" ]]; then
    local payload
    payload=$(py "${title}" "${body}" <<'PY'
import json
import sys

title = sys.argv[1]
body = sys.argv[2]
print(json.dumps({"text": f"*{title}*\n{body}"}))
PY
)
    curl -s -X POST -H 'Content-type: application/json' --data "${payload}" "${SLACK_WEBHOOK_URL}" >/dev/null || true
  fi
  if [[ -n "${NOTIFY_SHELL_CMD}" ]]; then
    printf '%s\n%s\n' "${title}" "${body}" | bash -lc "${NOTIFY_SHELL_CMD}" || true
  fi
}

summarise_weights() {
  local csv_path="$1"
  local column="$2"
  py "${csv_path}" "${column}" "${WEIGHT_DROP_TOLERANCE}" <<'PY'
import csv
import math
import sys

csv_path, column, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])
try:
    with open(csv_path, newline="") as handle:
        rows = list(csv.reader(handle))
except FileNotFoundError:
    print(f"[weights] CSV missing: {csv_path}")
    sys.exit(12)

if not rows:
    print("[weights] CSV empty")
    sys.exit(12)

header = rows[0]
if column not in header:
    print(f"[weights] Column '{column}' missing")
    sys.exit(12)

idx = header.index(column)
values = []
for row in rows[1:]:
    if len(row) != len(header):
        continue
    cell = row[idx].strip()
    if not cell:
        continue
    try:
        value = float(cell)
    except ValueError:
        continue
    if math.isfinite(value):
        values.append(value)

if not values:
    print(f"[weights] No numeric entries for {column}")
    sys.exit(12)

started = False
prev = None
violations = []
effective = []

for pos, value in enumerate(values):
    if not started:
        if abs(value) <= tol:
            continue
        started = True
        prev = value
        effective.append(value)
        continue
    effective.append(value)
    if value + tol < prev:
        violations.append((pos, prev, value))
    if value > prev:
        prev = value

if not started:
    print(f"[weights] {column} monotonic OK (only warmup zeros observed)")
    sys.exit(0)

if violations:
    print(f"[weights] Monotonic check failed for {column}: {len(violations)} drops detected")
    for pos, before, after in violations[:5]:
        print(f"  idx={pos} before={before:.6f} after={after:.6f}")
    sys.exit(12)

print(f"[weights] {column} monotonic OK ({len(effective)} samples)")
PY
}

check_mask_emergency() {
  local csv_path="$1"
  local limit="$2"
  py "${csv_path}" "${limit}" <<'PY'
import csv
import sys

csv_path, limit = sys.argv[1], float(sys.argv[2])
try:
    with open(csv_path, newline="") as handle:
        rows = list(csv.reader(handle))
except FileNotFoundError:
    print(f"[mask] CSV missing: {csv_path}")
    sys.exit(12)

if not rows:
    print("[mask] CSV empty")
    sys.exit(12)

header = rows[0]
count_candidates = ["mask_emergency_count", "mask_emergency_total", "mask_emergency"]
streak_candidates = ["mask_low_fraction_streak", "mask_low_fraction_run", "mask_low_fraction_streak_v2"]

def resolve(candidates):
    for name in candidates:
        if name in header:
            return header.index(name), name
    return None, None

idx_count, count_name = resolve(count_candidates)
idx_streak, streak_name = resolve(streak_candidates)

missing = []
if idx_count is None:
    missing.append("mask_emergency_count")
if idx_streak is None:
    missing.append("mask_low_fraction_streak")

if missing:
    print(f"[mask] Missing columns: {missing}")
    sys.exit(12)

count_values = []
for row in rows[1:]:
    if len(row) != len(header):
        continue
    cell = row[idx_count].strip()
    if not cell:
        count_values.append(0.0)
        continue
    try:
        count_values.append(float(cell))
    except ValueError:
        count_values.append(0.0)

final_count = count_values[-1] if count_values else 0.0
last_row = rows[-1] if len(rows) > 1 else rows[0]
streak_cell = last_row[idx_streak].strip()
try:
    streak_value = float(streak_cell) if streak_cell else 0.0
except ValueError:
    streak_value = 0.0

print(f"[mask] {count_name} final={final_count}")
print(f"[mask] {streak_name} final={streak_value}")

if final_count > limit:
    sys.exit(12)
PY
}

check_alpha_metrics() {
  local csv_path="$1"
  local mean_min="$2"
  local mean_max="$3"
  local frac95_max="$4"
  py "${csv_path}" "${mean_min}" "${mean_max}" "${frac95_max}" <<'PY'
import csv
import math
import sys

csv_path, mean_min, mean_max, frac95_max = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
try:
    with open(csv_path, newline="") as handle:
        rows = list(csv.reader(handle))
except FileNotFoundError:
    print(f"[alpha] CSV missing: {csv_path}")
    sys.exit(12)

if len(rows) < 2:
    print("[alpha] CSV empty")
    sys.exit(12)

header = rows[0]
required = ["alpha_mean", "alpha_fraction_ge95"]
missing = [name for name in required if name not in header]
if missing:
    print(f"[alpha] Missing columns: {missing}")
    sys.exit(12)

idx_mean = header.index("alpha_mean")
idx_frac = header.index("alpha_fraction_ge95")
last = rows[-1]

def parse(idx):
    cell = last[idx].strip()
    try:
        return float(cell)
    except ValueError:
        return float("nan")

alpha_mean = parse(idx_mean)
alpha_frac = parse(idx_frac)

print(f"[alpha] alpha_mean={alpha_mean}")
print(f"[alpha] alpha_fraction_ge95={alpha_frac}")

ok = True
if not math.isfinite(alpha_mean) or not math.isfinite(alpha_frac):
    print("[alpha] Non-finite values detected")
    ok = False
if ok and not (mean_min <= alpha_mean <= mean_max):
    print(f"[alpha] mean {alpha_mean} outside [{mean_min}, {mean_max}]")
    ok = False
if ok and alpha_frac > frac95_max:
    print(f"[alpha] fraction>=0.95 {alpha_frac} exceeds {frac95_max}")
    ok = False

if not ok:
    sys.exit(12)
PY
}

verify_training_csv() {
  local csv_path="$1"
  py "${csv_path}" <<'PY'
import csv
import sys

csv_path = sys.argv[1]
required = ["metrics_schema_version", "_eor_checksum"]

try:
    with open(csv_path, newline="") as handle:
        rows = list(csv.reader(handle))
except FileNotFoundError:
    print(f"[csv] missing: {csv_path}")
    sys.exit(12)

if not rows:
    print("[csv] empty file")
    sys.exit(12)

header = rows[0]
missing = [name for name in required if name not in header]
if missing:
    print(f"[csv] Missing required columns: {missing}")
    sys.exit(12)

idx_schema = header.index("metrics_schema_version")
idx_checksum = header.index("_eor_checksum")
tail = rows[-1]
schema_val = tail[idx_schema].strip()
checksum_val = tail[idx_checksum].strip()

print(f"[csv] metrics_schema_version={schema_val}")
if not schema_val:
    print("[csv] metrics_schema_version missing in tail row")
    sys.exit(12)
if not checksum_val:
    print("[csv] _eor_checksum missing in tail row")
    sys.exit(12)
PY
}

run_diagnostics() {
  local csv_path="$1"
  printf '[INFO] Diagnostics on %s\n' "${csv_path}"
  verify_training_csv "${csv_path}"
  check_alpha_metrics "${csv_path}" "${ALPHA_MEAN_MIN}" "${ALPHA_MEAN_MAX}" "${ALPHA_FRAC95_MAX}"
  check_mask_emergency "${csv_path}" "${MASK_EMERGENCY_MAX}"
  summarise_weights "${csv_path}" "opacity_target_weight_effective"
  summarise_weights "${csv_path}" "feature_weight_effective"
  printf '[INFO] Diagnostics OK\n'
}

TEACHER_RENDER_DIR=$(py "${PROJ_ROOT}" "${CFG_MAIN}" <<'PY'
import os
import sys
import yaml

proj_root, cfg_path = sys.argv[1:3]
with open(cfg_path, "r", encoding="utf-8") as fp:
    cfg = yaml.safe_load(fp)

candidates = []
data = cfg.get("data", {})
teacher_outputs = data.get("teacher_outputs")
if teacher_outputs:
    candidates.append(teacher_outputs)
teacher_section = cfg.get("teacher", {})
render_stats = teacher_section.get("render_stats")
if render_stats:
    candidates.append(os.path.dirname(render_stats))

found = ""
for candidate in candidates:
    if not candidate:
        continue
    if not os.path.isabs(candidate):
        candidate = os.path.join(proj_root, candidate)
    candidate = os.path.abspath(candidate)
    render_dir = os.path.join(candidate, "renders")
    if os.path.isdir(render_dir):
        found = render_dir
        break

print(found)
PY
)

if [[ -z "${TEACHER_RENDER_DIR}" ]]; then
  TEACHER_RENDER_DIR="${PROJ_ROOT}/teacher/outputs/lego/test_white/ours_30000/renders"
fi

TEACHER_RENDER_DIR=$(abs_path "${TEACHER_RENDER_DIR}")
EVAL_SUMMARY_PATH=$(abs_path "${EVAL_SUMMARY_CSV}")

# ---------------------------
# 0-1) Monitoring hint (TensorBoard)
# ---------------------------
cat <<'EOS'
=== 監視専用 TensorBoard（別セッション推奨） ===
wsl.exe -e bash -lc 'conda run --no-capture-output -n kilogs tensorboard \
  --logdir logs/lego/feat_t_sh52/runs/teacher_full_rehab_masked_white_smoke_s1_sh52/tensorboard \
  --host 127.0.0.1 --port 6006'
EOS

printf '\n'

# ---------------------------
# 1) SH52 smoke S1 (10k)
# ---------------------------
if [[ "${RUN_SMOKE}" == "1" ]]; then
    cd "${PROJ_ROOT}"
    ensure_dirs
    export PYTHONHASHSEED=2025
    printf '[INFO] Start SH52 smoke 10k: %s\n' "${CFG_SMOKE}"
    run_in_conda python -u distill/lego_response_distill.py --config "${CFG_SMOKE}"
    printf '[OK] 10k smoke 完了。ゲート失敗なら *.bailout を確認して再投入してください。\n'
    send_notification "SH52 smoke 10k finished" "config=${CFG_SMOKE}\nrun_dir=${RUN_DIR_SMOKE}"
            resolved_ckpt=$(find_checkpoint_rel "${CKPT_10K_PATTERN}")
    if [[ -n "${resolved_ckpt}" ]]; then
        CKPT_10K="${resolved_ckpt}"
        printf '[INFO] 10k checkpoint detected at %s\n' "${CKPT_10K}"
    else
        printf '[WARN] 10k checkpoint could not be located automatically; expected %s\n' "${CKPT_10K}"
    fi
else
    printf '[SKIP] 10k smoke ステージを実行しません (RUN_SMOKE=%s)\n' "${RUN_SMOKE}"
fi

printf '\n=== 10k 直後チェックメモ ===\n'
cat <<EOS
- ${TRAIN_METRICS_SMOKE}: feature_mask_fraction >=0.25, feature_src_dim=128, projector_out_dim=52
- feature/opacity effective weights should be monotonic (CSV columns)
- 起動ログで teacher_mode=gaussian_sh_opacity_logscale / projector 128->52 を確認
- 必要なら 200frame render (white) で PSNR ≈10 dB を目視
EOS

# ---------------------------
# 2) Promote to 50k (--resume)
# ---------------------------
if [[ "${RUN_50K}" == "1" ]]; then
    cd "${PROJ_ROOT}"
    export PYTHONHASHSEED=2025
    if [[ ! -f "${CKPT_10K}" ]]; then
        fallback_ckpt=$(find_checkpoint_rel "${CKPT_10K_PATTERN}")
        if [[ -n "${fallback_ckpt}" && -f "${fallback_ckpt}" ]]; then
            CKPT_10K="${fallback_ckpt}"
            printf '[INFO] Resolved 10k checkpoint fallback: %s\n' "${CKPT_10K}"
        fi
    fi
    if [[ ! -f "${CKPT_10K}" ]]; then
        printf '[ERROR] 10k checkpoint not found: %s\n' "${CKPT_10K}"
        exit 12
    fi
    printf '[INFO] Resume source checkpoint: %s\n' "${CKPT_10K}"
    printf '[INFO] Resume 10k -> 50k: %s (ckpt: %s)\n' "${CFG_MAIN}" "${CKPT_10K}"
    run_in_conda python -u distill/lego_response_distill.py --config "${CFG_MAIN}" --resume "${CKPT_10K}"
    printf '[OK] 50k 学習完了 (10k/20k/50k gate logs available)\n'
    send_notification "SH52 training 50k finished" "config=${CFG_MAIN}\nrun_dir=${RUN_DIR_MAIN}"
else
    printf '[SKIP] 50k 昇格ステージを実行しません (RUN_50K=%s)\n' "${RUN_50K}"
fi

# ---------------------------
# 3) Render + Evaluate + Quicklook
# ---------------------------
if [[ "${RUN_RENDER}" == "1" ]]; then
  cd "${PROJ_ROOT}"
    local config_path checkpoint_path render_output metrics_json render_stats_json teacher_dir
    config_path=$(abs_path "${CFG_MAIN}")
    checkpoint_path=$(abs_path "${CKPT_50K}")
    if [[ ! -f "${checkpoint_path}" ]]; then
        local fallback_ckpt
        fallback_ckpt=$(find_checkpoint_rel "${CKPT_50K_PATTERN}")
        if [[ -n "${fallback_ckpt}" ]]; then
            checkpoint_path=$(abs_path "${fallback_ckpt}")
            CKPT_50K="${fallback_ckpt}"
            printf '[INFO] Resolved 50k checkpoint fallback: %s\n' "${checkpoint_path}"
        fi
    fi
    if [[ ! -f "${checkpoint_path}" ]]; then
        printf '[ERROR] 50k checkpoint not found: %s\n' "${checkpoint_path}"
        exit 12
    fi

    render_output=$(abs_path "$(derive_render_dir "${checkpoint_path}")")
    metrics_json="${render_output}/metrics_white.json"
    render_stats_json="${render_output}/render_stats.json"
    teacher_dir="${TEACHER_RENDER_DIR}"

    if [[ ! -d "${teacher_dir}" ]]; then
        printf '[ERROR] Teacher renders directory not found: %s\n' "${teacher_dir}"
        exit 12
    fi

    mkdir -p "${render_output}"

    printf '[INFO] Render (RGBA+NVML) -> %s\n' "${render_output}"
    run_in_conda python -u distill/render_student.py \
        --config "${config_path}" \
        --checkpoint "${checkpoint_path}" \
        --output-dir "${render_output}" \
        --store-rgba \
        --chunk "${RENDER_CHUNK}" \
        --enable-nvml

    printf '[INFO] Evaluate (white background)\n'
    run_in_conda python -u tools/run_eval_pipeline.py \
        "${render_output}" \
        --teacher-renders "${teacher_dir}" \
        --background white \
        --method-name "${METHOD_NAME_50K}" \
        --summary "${EVAL_SUMMARY_PATH}" \
        --render-stats "${render_stats_json}" \
        --output-json "${metrics_json}" \
        --clean
  printf '[INFO] quicklook png\n'
  run_in_conda python -u tools/quicklook_metrics.py --csv "${TRAIN_METRICS_MAIN}"

  latest_metrics=$(find_latest "${RESULTS_GLOB}")
  if [[ -n "${latest_metrics}" ]]; then
    summary=$(py "${latest_metrics}" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    data = json.load(handle)

psnr = data.get("psnr")
ssim = data.get("ssim")
lpips = data.get("lpips")
print(f"PSNR={psnr} dB, SSIM={ssim}, LPIPS={lpips}")
PY
)
        send_notification "SH52 render/eval finished" "${summary}\nmetrics=${latest_metrics}"
    fi
else
    printf '[SKIP] render/eval ステージを実行しません (RUN_RENDER=%s)\n' "${RUN_RENDER}"
fi

# ---------------------------
# 4) Diagnostics suite
# ---------------------------
if [[ "${RUN_DIAG}" == "1" ]]; then
  cd "${PROJ_ROOT}"
  run_diagnostics "${TRAIN_METRICS_MAIN}"
else
    printf '[SKIP] 診断ステージを実行しません (RUN_DIAG=%s)\n' "${RUN_DIAG}"
fi

cat <<'EOS'
[合格目安（白）]
- PSNR >= 22.5 dB / SSIM >= 0.82 / LPIPS <= 0.19
- alpha mean 0.35-0.45 / frac(alpha>=0.95) <= 0.15
- CSV: metrics_schema_version=4 / _eor_checksum present
EOS
