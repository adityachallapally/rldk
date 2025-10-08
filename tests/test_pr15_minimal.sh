#!/usr/bin/env bash
set -euo pipefail

echo "=== Minimal PR #15 write_drift_card test ==="

# Quick setup
python3 -m venv .venv_minimal
source .venv_minimal/bin/activate
pip install -e ".[dev]"

# Generate fixtures
python tests/_make_fixtures.py

# Test the core PR #15 functionality
echo "Testing write_drift_card import and functionality..."
python - << 'PY'
from pathlib import Path
from rldk.io import write_drift_card

# Test import works
print("✓ write_drift_card imports successfully")

# Test function call
out = Path("rldk_reports/drift_card_test")
write_drift_card(
    {
        "diverged": True,
        "first_step": 847,
        "tripped_signals": ["kl_spike", "controller_stuck"],
        "signals_monitored": ["kl", "kl_coef", "grad_ratio"],
        "tolerance": 3.0,
        "k_consecutive": 5,
        "window_size": 50,
        "output_path": str(out)
    },
    out
)

# Verify both files are created
assert (out / "drift_card.json").exists(), "drift_card.json missing"
assert (out / "drift_card.md").exists(), "drift_card.md missing"
print("✓ Both drift_card.json and drift_card.md created")

# Verify JSON content
import json
drift_data = json.load(open(out / "drift_card.json"))
assert drift_data["diverged"] == True
assert "kl_spike" in drift_data["tripped_signals"]
print("✓ drift_card.json has correct content")

# Verify markdown content
md_content = (out / "drift_card.md").read_text()
assert "Drift Detection Card" in md_content
assert "🚨 Drift Detected" in md_content
assert "kl_spike" in md_content
assert "controller_stuck" in md_content
print("✓ drift_card.md has correct content")

print("PR #15 write_drift_card functionality verified successfully!")
PY

echo "=== Minimal test completed ==="