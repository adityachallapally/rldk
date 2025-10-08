#!/usr/bin/env bash
set -euo pipefail

# 1) Build fixtures offline
python3 tests/_make_fixtures.py

# 2) Determinism card
rldk env-audit test_artifacts/logs_clean

# 3) PPO forensics on doctored logs
rldk log-scan test_artifacts/logs_doctored_kl_spike

# 4) Checkpoint diffs
rldk diff-ckpt test_artifacts/ckpt_identical/a.pt test_artifacts/ckpt_identical/b.pt
rldk diff-ckpt test_artifacts/ckpt_value_head_edit/a.pt test_artifacts/ckpt_value_head_edit/b.pt

# 5) Reward drift
rldk reward-drift test_artifacts/reward_drift_demo/rmA test_artifacts/reward_drift_demo/rmB \
  --prompts test_artifacts/reward_drift_demo/prompts.jsonl

echo
echo "Artifacts written to ./rldk_reports"
ls -1 rldk_reports || true