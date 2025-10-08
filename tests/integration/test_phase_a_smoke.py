import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "rldk_reports"


def sh(args):
    # args is a list, we capture output for debugging
    r = subprocess.run(
        args, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert r.returncode == 0, f"Command failed: {' '.join(args)}\n{r.stdout}"
    return r.stdout


def load_json(p):
    with open(p) as f:
        return json.load(f)


def has_pytorch():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def test_phase_a_end_to_end(tmp_path):
    # Clean any old reports
    if REPORTS.exists():
        for p in REPORTS.glob("*"):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                import shutil

                shutil.rmtree(p)

    # 1) Fixtures
    sh([sys.executable, "tests/_make_fixtures.py"])

    # 2) Env audit
    sh(["rldk", "env-audit", "test_artifacts/logs_clean"])
    det = load_json(REPORTS / "determinism_card.json")
    assert "version" in det and "flags" in det and "rng" in det

    # 3) PPO forensics on doctored logs
    sh(["rldk", "log-scan", "test_artifacts/logs_doctored_kl_spike"])
    ppo = load_json(REPORTS / "ppo_scan.json")
    rules = [r.get("rule", "") for r in ppo.get("rules_fired", [])]
    assert (
        "kl_controller_stuck" in rules
    ), f"Expected kl_controller_stuck in rules_fired, got {rules}"
    # tolerate naming variants for the controller rule
    assert any("controller" in r or "coef" in r for r in rules)

    # 4) Checkpoint diffs
    # Only run checkpoint tests if PyTorch is available
    if not has_pytorch():
        pytest.skip("PyTorch not available, skipping checkpoint diff tests")

    sh(
        [
            "rldk",
            "diff-ckpt",
            "test_artifacts/ckpt_identical/a.pt",
            "test_artifacts/ckpt_identical/b.pt",
        ]
    )
    diff_ident = load_json(REPORTS / "ckpt_diff.json")
    avg_cos = diff_ident.get("summary", {}).get("avg_cosine", 0.0)
    assert avg_cos >= 0.9999

    sh(
        [
            "rldk",
            "diff-ckpt",
            "test_artifacts/ckpt_value_head_edit/a.pt",
            "test_artifacts/ckpt_value_head_edit/b.pt",
        ]
    )
    diff_edit = load_json(REPORTS / "ckpt_diff.json")
    movers = [m.get("name", "").lower() for m in diff_edit.get("top_movers", [])]
    assert any(("value" in n) or ("v_head" in n) for n in movers)

    # 5) Reward drift
    sh(
        [
            "rldk",
            "reward-drift",
            "test_artifacts/reward_drift_demo/rmA",
            "test_artifacts/reward_drift_demo/rmB",
            "--prompts",
            "test_artifacts/reward_drift_demo/prompts.jsonl",
        ]
    )
    drift = load_json(REPORTS / "reward_drift.json")
    # prove it is not identical and that sign flips exist or slice deltas present
    assert drift.get("pearson", 1.0) < 0.5
    flips = drift.get("sign_flip_rate", 0.0)
    slices = drift.get("slice_deltas", {})
    assert flips > 0.0 or any(
        abs(v.get("delta_mean", 0.0)) > 0 for v in slices.values()
    )
