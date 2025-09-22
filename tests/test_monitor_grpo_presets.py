"""Tests for GRPO monitor presets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import pytest

from rldk.monitor import Alert, ActionExecutor, Event, MonitorEngine, load_rules
from rldk.monitor.presets import get_rule_preset

_FIXTURE_PATH = Path("test_artifacts/logs_grpo/seed_1/run.jsonl")


class _NoopExecutor(ActionExecutor):
    """Capture alerts without side effects for testing."""

    def execute(self, action, rule, event, window):  # type: ignore[override]
        return Alert(
            rule_id=rule.id,
            action=action.kind,
            event=event,
            window_size=rule.window_size,
            window_kind=rule.window_kind,
            message=None,
            status="success",
        )


def _iter_fixture_events(metric_names: Sequence[str]) -> Iterable[Event]:
    with _FIXTURE_PATH.open() as handle:
        for line in handle:
            payload = json.loads(line)
            step = int(payload["step"])
            timestamp = "2024-01-01T00:00:00Z"
            for metric in metric_names:
                value = payload.get(metric)
                if value is None:
                    continue
                yield Event(
                    time=timestamp,
                    step=step,
                    name=metric,
                    value=float(value),
                    run_id="seed_1",
                )


def _drain_events(engine: MonitorEngine, events: Iterable[Event]) -> set[str]:
    fired: set[str] = set()
    for event in events:
        for alert in engine.process_event(event):
            fired.add(alert.rule_id)
    return fired


def test_grpo_safe_preset_triggers_on_fixture_anomalies() -> None:
    preset = get_rule_preset("grpo_safe")
    assert preset is not None
    rules = load_rules(preset)
    engine = MonitorEngine(rules, action_executor=_NoopExecutor())

    metrics = ["kl", "kl_coef", "reward_mean"]
    fired = _drain_events(engine, _iter_fixture_events(metrics))

    assert "grpo_safe_kl_spike" in fired
    assert "grpo_safe_kl_coef_stall" in fired
    assert "grpo_safe_reward_saturation" in fired


@pytest.mark.parametrize(
    "metric, values, expected_rule",
    [
        (
            "entropy",
            [1.7] * 8,
            "grpo_strict_entropy_floor",
        ),
        (
            "advantage_std",
            [0.3] * 6,
            "grpo_strict_advantage_collapse",
        ),
        (
            "acceptance_rate",
            [0.5, 0.55, 0.58, 0.6, 0.62, 0.1, 0.85, 0.12, 0.8, 0.15],
            "grpo_strict_acceptance_swings",
        ),
    ],
)
def test_grpo_strict_preset_flags_strict_failures(
    metric: str, values: Sequence[float], expected_rule: str
) -> None:
    preset = get_rule_preset("grpo_strict")
    assert preset is not None
    rules = load_rules(preset)
    engine = MonitorEngine(rules, action_executor=_NoopExecutor())

    baseline_metrics = ["kl", "kl_coef", "reward_mean", "entropy", "advantage_std"]
    fired = _drain_events(engine, _iter_fixture_events(baseline_metrics))

    for offset, value in enumerate(values, start=1):
        event = Event(
            time="2024-01-02T00:00:00Z",
            step=1500 + offset,
            name=metric,
            value=float(value),
            run_id="seed_1",
        )
        for alert in engine.process_event(event):
            fired.add(alert.rule_id)

    assert "grpo_strict_kl_spike" in fired
    assert "grpo_strict_kl_coef_stall" in fired
    assert "grpo_strict_reward_saturation" in fired
    assert expected_rule in fired


def test_grpo_safe_kl_spike_triggers_after_step_reset() -> None:
    preset = get_rule_preset("grpo_safe")
    assert preset is not None
    rules = load_rules(preset)
    engine = MonitorEngine(rules, action_executor=_NoopExecutor())

    def make_event(step: int, value: float) -> Event:
        return Event(
            time="2024-01-03T00:00:00Z",
            step=step,
            name="kl",
            value=value,
            run_id="step_reset",
        )

    fired_steps: list[int] = []

    # Prime the rule to satisfy the configured grace period.
    for step in range(1, 13):
        engine.process_event(make_event(step, 0.1))

    # First spike with monotonically increasing steps.
    for step in (13, 14):
        alerts = engine.process_event(make_event(step, 0.45))
        fired_steps.extend(
            alert.event.step
            for alert in alerts
            if alert.rule_id == "grpo_safe_kl_spike"
        )

    assert 14 in fired_steps

    # Step counter resets; cooldown state should be cleared and allow a new alert.
    for step, value in ((4, 0.1), (5, 0.5), (6, 0.55)):
        alerts = engine.process_event(make_event(step, value))
        fired_steps.extend(
            alert.event.step
            for alert in alerts
            if alert.rule_id == "grpo_safe_kl_spike"
        )

    assert 5 not in fired_steps
    assert 6 in fired_steps
