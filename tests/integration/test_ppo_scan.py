"""Test PPO scan functionality."""

from rldk.forensics.ppo_scan import scan_ppo_events
from rldk.io.schemas import PPOScanReportV1, validate


def test_ppo_scan_clean_logs():
    """Test PPO scan on clean logs (should have no anomalies)."""
    # Generate clean events
    events = []
    for step in range(100):
        event = {
            "step": step,
            "kl": 0.05 + 0.01 * (step % 10) / 10,  # Steady KL
            "kl_coef": 0.1,
            "entropy": 2.0,
            "advantage_mean": 0.0,
            "advantage_std": 1.0,
            "grad_norm_policy": 0.5,
            "grad_norm_value": 0.3,
        }
        events.append(event)

    # Run scan
    report = scan_ppo_events(iter(events))

    # Validate report
    validate(PPOScanReportV1, report)

    # Should have no rules fired for clean logs
    assert len(report["rules_fired"]) == 0
    assert report["earliest_step"] == 0


def test_ppo_scan_kl_spike():
    """Test PPO scan detects KL spike."""
    # Generate events with KL spike
    events = []
    for step in range(100):
        if step < 50:
            kl = 0.05  # Normal KL
        else:
            kl = 0.5  # KL spike (10x normal)

        event = {
            "step": step,
            "kl": kl,
            "kl_coef": 0.1,
            "entropy": 2.0,
            "advantage_mean": 0.0,
            "advantage_std": 1.0,
            "grad_norm_policy": 0.5,
            "grad_norm_value": 0.3,
        }
        events.append(event)

    # Run scan
    report = scan_ppo_events(iter(events))

    # Should detect KL spike
    kl_spike_rules = [r for r in report["rules_fired"] if r["rule"] == "kl_spike"]
    assert len(kl_spike_rules) > 0

    # Check step range
    for rule in kl_spike_rules:
        assert "step_range" in rule
        assert rule["step_range"][0] >= 50  # Should start around step 50


def test_ppo_scan_gradient_ratio():
    """Test PPO scan detects gradient ratio anomalies."""
    # Generate events with extreme gradient ratio
    events = []
    for step in range(100):
        if step < 50:
            policy_norm = 0.5
            value_norm = 0.3
        else:
            policy_norm = 5.0  # Much higher policy gradient
            value_norm = 0.3

        event = {
            "step": step,
            "kl": 0.05,
            "kl_coef": 0.1,
            "entropy": 2.0,
            "advantage_mean": 0.0,
            "advantage_std": 1.0,
            "grad_norm_policy": policy_norm,
            "grad_norm_value": value_norm,
        }
        events.append(event)

    # Run scan
    report = scan_ppo_events(iter(events))

    # Should detect gradient ratio anomaly
    grad_rules = [r for r in report["rules_fired"] if "grad_ratio" in r["rule"]]
    assert len(grad_rules) > 0


def test_ppo_scan_empty_events():
    """Test PPO scan with empty events."""
    report = scan_ppo_events(iter([]))

    # Should return valid report with no rules
    validate(PPOScanReportV1, report)
    assert len(report["rules_fired"]) == 0
    assert report["earliest_step"] is None
