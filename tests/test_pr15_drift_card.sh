#!/usr/bin/env bash
set -euo pipefail

echo "=== Testing PR #15 write_drift_card functionality ==="

# 0) Tooling and clean env
echo "0) Checking tooling and environment..."
python3 -V
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip build wheel

# 1) Install rldk with dev extras
echo "1) Installing rldk with dev extras..."
pip install -e ".[dev]"

# 2) Generate offline fixtures
echo "2) Generating offline fixtures..."
python tests/_make_fixtures.py

# 3) Lint, typecheck, tests
echo "3) Running linting, typechecking, and tests..."
ruff check
black --check .
mypy src
pytest -q

# 4) End-to-end smoke via the provided script if present
echo "4) Running end-to-end smoke test..."
if [ -f scripts/phase_a_smoke.sh ]; then
  bash scripts/phase_a_smoke.sh
fi

# 5) Exercise the CLIs individually on fixtures (Phase A surface)
echo "5) Exercising CLIs individually..."
rldk env-audit test_artifacts/logs_clean
rldk log-scan test_artifacts/logs_doctored_kl_spike

# Only run checkpoint tests if PyTorch is available
if python3 -c "import torch" 2>/dev/null; then
    echo "PyTorch available, running checkpoint tests..."
    rldk diff-ckpt test_artifacts/ckpt_identical/a.pt test_artifacts/ckpt_identical/b.pt
    rldk diff-ckpt test_artifacts/ckpt_value_head_edit/a.pt test_artifacts/ckpt_value_head_edit/b.pt
    rldk reward-drift test_artifacts/reward_drift_demo/rmA test_artifacts/reward_drift_demo/rmB \
      --prompts test_artifacts/reward_drift_demo/prompts.jsonl
else
    echo "PyTorch not available, skipping checkpoint tests..."
fi

# 6) Sanity-check JSON artifacts exist and look right (no jq needed)
echo "6) Sanity-checking JSON artifacts..."
python - << 'PY'
import json, sys, pathlib
p = pathlib.Path("rldk_reports")
need = ["determinism_card.json","ppo_scan.json","ckpt_diff.json","reward_drift.json"]
missing = [n for n in need if not (p/n).exists()]
assert not missing, f"Missing artifacts: {missing}"
# quick content checks
det = json.load(open(p/"determinism_card.json"))
assert "flags" in det and "rng" in det
ppo = json.load(open(p/"ppo_scan.json"))
assert "rules_fired" in ppo and isinstance(ppo["rules_fired"], list)
cd = json.load(open(p/"ckpt_diff.json"))
assert "summary" in cd and "top_movers" in cd
rd = json.load(open(p/"reward_drift.json"))
assert "pearson" in rd and "spearman" in rd
print("Artifact content sanity OK")
PY

# 7) Explicitly test the new write_drift_card import path and output
echo "7) Testing write_drift_card import and output..."
python - << 'PY'
from pathlib import Path
from rldk.io import write_drift_card  # must import cleanly
out = Path("rldk_reports/drift_card_test")
write_drift_card(
  {
    "diverged": True,
    "first_step": 847,
    "tripped_signals": ["kl_spike","controller_stuck"],
    "signals_monitored": ["kl","kl_coef","grad_ratio"],
    "tolerance": 3.0,
    "k_consecutive": 5,
    "window_size": 50,
    "output_path": str(out)
  },
  out
)
assert (out/"drift_card.json").exists(), "drift_card.json missing"
assert (out/"drift_card.md").exists(), "drift_card.md missing"
print("write_drift_card OK")
PY

# 8) Negative check: clean logs should not fire spike/controller
echo "8) Negative check: clean logs should not fire spike/controller..."
python - << 'PY'
import json
ppo = json.load(open("rldk_reports/ppo_scan.json"))
rules = set(ppo.get("rules_fired", []))
# accept other benign rules, but ensure at least one spike rule on doctored and none on clean rescan
print("Doctored rules:", rules)
PY
rldk log-scan test_artifacts/logs_clean
python - << 'PY'
import json
ppo = json.load(open("rldk_reports/ppo_scan.json"))
rules = set(ppo.get("rules_fired", []))
assert not any("spike" in r or "controller" in r for r in rules), f"Unexpected anomaly on clean logs: {rules}"
print("Clean logs show no spike/controller anomalies")
PY

# 9) Packaging check: build sdist/wheel and install into a fresh venv, ensure console entry works
echo "9) Packaging check..."
python -m build
deactivate
python -m venv .venv_pkg
source .venv_pkg/bin/activate
pip install dist/*.whl
rldk --help >/dev/null
python -c "import rldk, importlib; importlib.import_module('rldk.io'); print('import OK')"
echo "Packaging sanity OK"

# 10) Additional assertions for PR #15 specific functionality
echo "10) Additional PR #15 assertions..."
python - << 'PY'
import json
import pathlib

# Check that drift_card.json has the expected structure
drift_json = pathlib.Path("rldk_reports/drift_card_test/drift_card.json")
assert drift_json.exists(), "drift_card.json not found"
drift_data = json.load(open(drift_json))
assert "diverged" in drift_data, "drift_card.json missing 'diverged' field"
assert "tripped_signals" in drift_data, "drift_card.json missing 'tripped_signals' field"
assert drift_data["diverged"] == True, "drift_card.json diverged should be True"
assert "kl_spike" in drift_data["tripped_signals"], "drift_card.json missing kl_spike in tripped_signals"

# Check that drift_card.md has human-readable content
drift_md = pathlib.Path("rldk_reports/drift_card_test/drift_card.md")
assert drift_md.exists(), "drift_card.md not found"
md_content = drift_md.read_text()
assert "Drift Detection Card" in md_content, "drift_card.md missing title"
assert "🚨 Drift Detected" in md_content, "drift_card.md missing drift detected message"
assert "kl_spike" in md_content, "drift_card.md missing kl_spike mention"
assert "controller_stuck" in md_content, "drift_card.md missing controller_stuck mention"

print("PR #15 drift card assertions passed")
PY

# 11) Verify identical checkpoint diff has high cosine similarity
echo "11) Verifying identical checkpoint diff cosine similarity..."
python - << 'PY'
import json
ckpt_diff = json.load(open("rldk_reports/ckpt_diff.json"))
assert "avg_cosine" in ckpt_diff, "ckpt_diff.json missing avg_cosine"
assert ckpt_diff["avg_cosine"] >= 0.9999, f"Identical checkpoints should have avg_cosine >= 0.9999, got {ckpt_diff['avg_cosine']}"
print(f"Identical checkpoint avg_cosine: {ckpt_diff['avg_cosine']} ✓")
PY

# 12) Verify doctored logs contain spike and controller rules
echo "12) Verifying doctored logs contain spike and controller rules..."
python - << 'PY'
import json
ppo = json.load(open("rldk_reports/ppo_scan.json"))
rules = ppo.get("rules_fired", [])
spike_rules = [r for r in rules if "spike" in r.lower()]
controller_rules = [r for r in rules if "controller" in r.lower()]
print(f"Spike rules found: {spike_rules}")
print(f"Controller rules found: {controller_rules}")
assert len(spike_rules) > 0, "No spike rules found in doctored logs"
assert len(controller_rules) > 0, "No controller rules found in doctored logs"
print("Doctored logs correctly contain spike and controller rules ✓")
PY

# 13) Final artifact verification
echo "13) Final artifact verification..."
python - << 'PY'
import pathlib
p = pathlib.Path("rldk_reports")
expected_files = [
    "determinism_card.json",
    "ppo_scan.json", 
    "ppo_kl.png",
    "ppo_grad_ratio.png",
    "ckpt_diff.json",
    "ckpt_top_movers.png",
    "reward_drift.json",
    "reward_drift_scatter.png",
    "drift_card_test/drift_card.json",
    "drift_card_test/drift_card.md"
]
missing = [f for f in expected_files if not (p/f).exists()]
if missing:
    print(f"Missing expected files: {missing}")
    print("Available files:")
    for f in p.rglob("*"):
        if f.is_file():
            print(f"  {f.relative_to(p)}")
    raise AssertionError(f"Missing expected files: {missing}")
print("All expected artifacts present ✓")
PY

echo
echo "=== PR #15 write_drift_card test completed successfully! ==="
echo "All artifacts written to ./rldk_reports"
ls -la rldk_reports/