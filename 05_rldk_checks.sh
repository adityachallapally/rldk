#!/bin/bash
# 05_rldk_checks.sh - Run RLDK validation against all training artifacts

set -e  # Exit on any error

echo "=========================================="
echo "Running RLDK validation checks"
echo "=========================================="

# Ensure we're in the virtual environment
source venv/bin/activate

# Create reports directory
mkdir -p ./rldk_demos/reports

echo "Checking RLDK CLI availability..."
which rldk || { echo "RLDK CLI not found!"; exit 1; }

echo "Running RLDK checks against training artifacts..."

# 1. Determinism check across A and B
echo ""
echo "1. Running determinism check (A vs B)..."
echo "   Expected: Should detect nondeterminism due to tokenizer padding side change"
rldk determinism check \
  --runs ./rldk_demos/ppo_a ./rldk_demos/ppo_b \
  --expect_same yes \
  --report ./rldk_demos/reports/determinism_a_b.json \
  --verbose

if [ $? -eq 0 ]; then
    echo "   ✓ Determinism check completed"
else
    echo "   ⚠ Determinism check had issues (this may be expected)"
fi

# 2. Reward drift check across A and C
echo ""
echo "2. Running reward drift check (A vs C)..."
echo "   Expected: Should detect reward drift due to different seed and data shuffle"
rldk reward-drift \
  --runs ./rldk_demos/ppo_a ./rldk_demos/ppo_c \
  --probes ./rldk_demos/probes.jsonl \
  --report ./rldk_demos/reports/reward_drift_a_c.json \
  --verbose

if [ $? -eq 0 ]; then
    echo "   ✓ Reward drift check completed"
else
    echo "   ⚠ Reward drift check had issues (this may be expected)"
fi

# 3. Reward health check on D
echo ""
echo "3. Running reward health check (D)..."
echo "   Expected: Should detect saturation/hacking due to reward clamping"
rldk reward-health \
  --run ./rldk_demos/ppo_d \
  --rm_path ./rldk_demos/rm_a \
  --report ./rldk_demos/reports/reward_health_d.json \
  --verbose

if [ $? -eq 0 ]; then
    echo "   ✓ Reward health check completed"
else
    echo "   ⚠ Reward health check had issues (this may be expected)"
fi

# 4. Calibration check on reward model
echo ""
echo "4. Running calibration check on reward model..."
echo "   Expected: Should provide ECE and reliability metrics"
rldk calibration check \
  --rm_path ./rldk_demos/rm_a \
  --eval_pairs ./rldk_demos/rm_eval_pairs.jsonl \
  --report ./rldk_demos/reports/calibration_rm_a.json \
  --verbose

if [ $? -eq 0 ]; then
    echo "   ✓ Calibration check completed"
else
    echo "   ⚠ Calibration check had issues (this may be expected)"
fi

# 5. Additional checks for comprehensive validation

# KL divergence check
echo ""
echo "5. Running KL divergence check..."
rldk kl-divergence \
  --runs ./rldk_demos/ppo_a ./rldk_demos/ppo_b \
  --report ./rldk_demos/reports/kl_divergence_a_b.json \
  --verbose

if [ $? -eq 0 ]; then
    echo "   ✓ KL divergence check completed"
else
    echo "   ⚠ KL divergence check had issues"
fi

# Training stability check
echo ""
echo "6. Running training stability check..."
rldk training-stability \
  --run ./rldk_demos/ppo_a \
  --report ./rldk_demos/reports/training_stability_a.json \
  --verbose

if [ $? -eq 0 ]; then
    echo "   ✓ Training stability check completed"
else
    echo "   ⚠ Training stability check had issues"
fi

# Model integrity check
echo ""
echo "7. Running model integrity check..."
rldk model-integrity \
  --run ./rldk_demos/ppo_a \
  --report ./rldk_demos/reports/model_integrity_a.json \
  --verbose

if [ $? -eq 0 ]; then
    echo "   ✓ Model integrity check completed"
else
    echo "   ⚠ Model integrity check had issues"
fi

# Summary of all reports
echo ""
echo "=========================================="
echo "RLDK validation summary"
echo "=========================================="

echo "Reports generated:"
ls -la ./rldk_demos/reports/

echo ""
echo "Report contents preview:"
for report in ./rldk_demos/reports/*.json; do
    if [ -f "$report" ]; then
        echo ""
        echo "--- $(basename "$report") ---"
        head -20 "$report" | jq . 2>/dev/null || head -20 "$report"
    fi
done

echo ""
echo "=========================================="
echo "RLDK validation completed!"
echo "=========================================="
echo "All reports saved to: ./rldk_demos/reports/"
echo ""
echo "Expected results:"
echo "- Determinism check should FAIL (detect tokenizer differences)"
echo "- Reward drift check should FAIL (detect distribution shift)"
echo "- Reward health check should FAIL (detect saturation/hacking)"
echo "- Calibration check should provide metrics (may pass or fail based on thresholds)"
echo ""
echo "Next step: Run 06_make_report.py to generate comprehensive analysis"