#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

DATE_STAMP="$(date +%Y%m%d)"
BASE_DIR="${REPO_ROOT}/rldk_reports/stability_micro/${DATE_STAMP}"
mkdir -p "${BASE_DIR}"

declare -a ALGORITHMS=("ppo" "grpo")
declare -a MODELS=("sshleifer/tiny-gpt2" "hf-internal-testing/tiny-random-gpt2")
declare -a TASKS=("gsm8k_mini" "humaneval_lite")
declare -a SEEDS=(0 1 2)

echo "📊 Writing benchmark artifacts to ${BASE_DIR}" >&2

target_for_algo() {
    local algo="$1"
    if [[ "${algo}" == "ppo" ]]; then
        echo "ppo_train.py"
    else
        echo "grpo_train.py"
    fi
}

monitor_rules_for_algo() {
    local algo="$1"
    if [[ "${algo}" == "ppo" ]]; then
        echo "ppo_safe"
    else
        echo "grpo_safe"
    fi
}

for algo in "${ALGORITHMS[@]}"; do
    train_script="$(target_for_algo "${algo}")"
    for model in "${MODELS[@]}"; do
        for task in "${TASKS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                run_dir="${BASE_DIR}/${algo}/${task}/$(echo "${model}" | tr '/ ' '__')/seed_${seed}"
                mkdir -p "${run_dir}"
                echo "🚀 Training ${algo} model=${model} task=${task} seed=${seed}" >&2
                python3 "${SCRIPT_DIR}/${train_script}" \
                    --model "${model}" \
                    --task "${task}" \
                    --seed "${seed}" \
                    --output-dir "${run_dir}"

                ruleset="$(monitor_rules_for_algo "${algo}")"
                echo "🛡️  Running monitor (${ruleset}) for ${run_dir}" >&2
                python3 -m rldk.cli monitor \
                    --once "${run_dir}/run.jsonl" \
                    --rules "${ruleset}" \
                    --alerts "${run_dir}/alerts.jsonl" \
                    --report "${run_dir}/report.json" \
                    >"${run_dir}/monitor.log" 2>&1 || true
            done
        done
    done
done

python3 "${SCRIPT_DIR}/make_report.py" --base-dir "${BASE_DIR}"

echo "✅ Stability micro-benchmark complete." >&2
