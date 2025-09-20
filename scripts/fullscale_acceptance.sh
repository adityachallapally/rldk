#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="${ROOT_DIR}/artifacts/fullscale"
LOG_FILE="${ARTIFACT_DIR}/cli_logs.txt"
VENV_DIR="${ARTIFACT_DIR}/.venv"

rm -rf "${ARTIFACT_DIR}"
mkdir -p "${ARTIFACT_DIR}"
: >"${LOG_FILE}"

python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip >/dev/null
python -m pip install --no-cache-dir -e "${ROOT_DIR}" >/dev/null

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE="${ARTIFACT_DIR}/hf_cache"

RUN_JSON="${ARTIFACT_DIR}/run.jsonl"
BASELINE_JSON="${ARTIFACT_DIR}/baseline.jsonl"
rm -f "${RUN_JSON}" "${BASELINE_JSON}"

TRAIN_COMMON=("python" "${ROOT_DIR}/scripts/fullscale_train_rl.py" "--max-steps" "40" "--batch-size" "4" "--max-new-tokens" "32" "--learning-rate" "8e-5" "--kl-coeff" "0.08" "--aux-weight" "0.35" "--noise-weight" "0.2" "--temperature" "0.95" "--max-grad-norm" "2.5" "--outdir" "${ARTIFACT_DIR}" "--print-interval" "10")

(
  echo "=== TRAINING: primary run ==="
  "${TRAIN_COMMON[@]}" "--seed" "31415" "--run-id" "run"
) | tee -a "${LOG_FILE}"

(
  echo "=== TRAINING: baseline (no updates) ==="
  "${TRAIN_COMMON[@]}" "--seed" "31415" "--run-id" "baseline" "--disable-updates"
) | tee -a "${LOG_FILE}"

MONITOR_ALERTS_JSON="${ARTIFACT_DIR}/monitor_alerts.jsonl"
MONITOR_ALERTS_TXT="${ARTIFACT_DIR}/monitor_alerts.txt"
MONITOR_REPORT="${ARTIFACT_DIR}/monitor_report.json"
python -m rldk.cli monitor --once "${RUN_JSON}" --rules "${ROOT_DIR}/configs/rules/fullscale_rules.yaml" --alerts "${MONITOR_ALERTS_JSON}" --alerts-txt "${MONITOR_ALERTS_TXT}" --report "${MONITOR_REPORT}" | tee -a "${LOG_FILE}"

INGEST_OUTPUT="${ARTIFACT_DIR}/training_metrics.jsonl"
python -m rldk.cli ingest "${RUN_JSON}" --output "${INGEST_OUTPUT}" | tee -a "${LOG_FILE}"

REWARD_DIR="${ARTIFACT_DIR}/reward_health"
python -m rldk.cli reward-health --run "${RUN_JSON}" --output-dir "${REWARD_DIR}" --reward-col "reward_mean" --step-col "step" | tee -a "${LOG_FILE}"

DIFF_DIR="${ARTIFACT_DIR}/diff"
python -m rldk.cli diff --a "${RUN_JSON}" --b "${BASELINE_JSON}" --signals "reward_mean,reward_aux,kl,loss,grad_norm" --output-dir "${DIFF_DIR}" | tee -a "${LOG_FILE}"

DETERMINISM_DIR="${ARTIFACT_DIR}/determinism_report"
mkdir -p "${DETERMINISM_DIR}"
DETERMINISM_CMD="python ${ROOT_DIR}/scripts/fullscale_train_rl.py --seed 2718 --max-steps 12 --batch-size 4 --max-new-tokens 24 --learning-rate 8e-5 --kl-coeff 0.08 --aux-weight 0.35 --noise-weight 0.2 --temperature 0.95 --max-grad-norm 2.5 --outdir ${ARTIFACT_DIR}/determinism --run-id det"
python -m rldk.cli check-determinism --cmd "${DETERMINISM_CMD}" --compare "reward_mean,kl,loss" --runs 2 --stride 1 --output-dir "${DETERMINISM_DIR}" --tolerance 0.5 | tee -a "${LOG_FILE}"

CARD_DIR="${ARTIFACT_DIR}/cards"
python -m rldk.cli card reward "${RUN_JSON}" --output-dir "${CARD_DIR}" | tee -a "${LOG_FILE}"

python - <<PY
from pathlib import Path

from rldk.acceptance.summary import summarize_from_artifacts

artifact_root = Path("${ARTIFACT_DIR}")
result = summarize_from_artifacts(artifact_root)
(artifact_root / "ACCEPTANCE_SUMMARY.md").write_text("\n".join(result.lines))
if not result.ok:
    raise SystemExit(1)
PY

