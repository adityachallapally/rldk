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
python -m rldk.cli monitor --once "${RUN_JSON}" --rules "${ROOT_DIR}/rules/fullscale_rules.yaml" --alerts "${MONITOR_ALERTS_JSON}" --alerts-txt "${MONITOR_ALERTS_TXT}" --report "${MONITOR_REPORT}" | tee -a "${LOG_FILE}"

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
import json
import sys
from pathlib import Path

root = Path("${ARTIFACT_DIR}")
run_path = root / "run.jsonl"
baseline_path = root / "baseline.jsonl"
alerts_path = root / "monitor_alerts.jsonl"
monitor_report = root / "monitor_report.json"
reward_summary = root / "reward_health" / "reward_health_summary.json"
diff_report = root / "diff" / "diff_report.json"
determinism_card = root / "determinism_report" / "determinism_card.json"
card_json = root / "cards" / "reward_card.json"

summary_lines = ["# Fullscale Acceptance Summary", ""]
status_ok = True

def _load_json_lines(path: Path) -> list[dict]:
    records = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

run_events = _load_json_lines(run_path)
baseline_events = _load_json_lines(baseline_path)
summary_lines.append(f"- Training events: {len(run_events)} entries in run.jsonl")
summary_lines.append(f"- Baseline events: {len(baseline_events)} entries in baseline.jsonl")
if len(run_events) <= 1000:
    status_ok = False
    summary_lines.append("  - ❌ Expected >1000 events in run.jsonl")
if len(baseline_events) <= 1000:
    status_ok = False
    summary_lines.append("  - ❌ Expected >1000 events in baseline.jsonl")

reward_series = [event["value"] for event in run_events if event.get("name") == "reward_mean"]
if reward_series:
    delta = reward_series[-1] - reward_series[0]
    summary_lines.append(f"- Reward mean delta: {delta:.3f}")
    if abs(delta) < 0.02:
        status_ok = False
        summary_lines.append("  - ❌ Reward mean did not change enough over run")
else:
    status_ok = False
    summary_lines.append("- ❌ reward_mean series missing from run events")

alert_records = []
if alerts_path.exists():
    alert_records = _load_json_lines(alerts_path)

summary_lines.append(f"- Monitor alerts fired: {len(alert_records)}")
if not alert_records:
    summary_lines.append("  - ℹ️ No monitor alerts were emitted during the run")

if monitor_report.exists():
    report = json.loads(monitor_report.read_text())
    summary_lines.append(f"- Monitor report status: {report.get('status', 'unknown')}")
else:
    status_ok = False
    summary_lines.append("- ❌ Monitor report missing")

if reward_summary.exists():
    reward_payload = json.loads(reward_summary.read_text())
    summary_lines.append(f"- Reward health verdict: {reward_payload.get('overall_status', 'unknown')}")
    if not reward_payload.get("passed", False):
        summary_lines.append("  - ⚠️ Reward health reported issues")
else:
    status_ok = False
    summary_lines.append("- ❌ Reward health summary missing")

if diff_report.exists():
    diff_payload = json.loads(diff_report.read_text())
    summary_lines.append(f"- Diff verdict: {diff_payload.get('summary', {}).get('verdict', 'unknown')}")
else:
    status_ok = False
    summary_lines.append("- ❌ Diff report missing")

if determinism_card.exists():
    det_payload = json.loads(determinism_card.read_text())
    summary_lines.append(f"- Determinism check passed: {det_payload.get('passed', False)}")
    if not det_payload.get('passed', False):
        status_ok = False
        summary_lines.append("  - ❌ Determinism check failed")
else:
    status_ok = False
    summary_lines.append("- ❌ Determinism card missing")

if card_json.exists():
    card_payload = json.loads(card_json.read_text())
    summary_lines.append(f"- Reward card status: {'HEALTHY' if card_payload.get('passed') else 'ISSUES'}")
else:
    status_ok = False
    summary_lines.append("- ❌ Reward card missing")

summary_lines.append("")
summary_lines.append("## Overall Result")
summary_lines.append("PASS" if status_ok else "FAIL")

(root / "ACCEPTANCE_SUMMARY.md").write_text("\n".join(summary_lines))
if not status_ok:
    sys.exit(1)
PY

