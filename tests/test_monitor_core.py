import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rldk.cli import app
from rldk.emit import EventWriter
from rldk.monitor import Event, MonitorEngine, load_rules, read_stream


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def test_event_writer_writes_canonical_event(tmp_path: Path) -> None:
    output = tmp_path / "events.jsonl"
    writer = EventWriter(output)
    event = writer.log(step=5, name="kl", value=0.42, run_id="run-1", tags={"env": "test"})
    writer.close()

    lines = output.read_text().strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["name"] == "kl"
    assert payload["step"] == 5
    assert payload["value"] == pytest.approx(0.42)
    assert payload["run_id"] == "run-1"
    assert payload["tags"] == {"env": "test"}
    assert "time" in payload
    assert event["name"] == "kl"


def test_read_stream_handles_partial_lines(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"

    def writer() -> None:
        with path.open("w", encoding="utf-8") as handle:
            event1 = {"time": _now_iso(), "step": 1, "name": "kl", "value": 0.1}
            text1 = json.dumps(event1)
            handle.write(text1[: len(text1) // 2])
            handle.flush()
            time.sleep(0.05)
            handle.write(text1[len(text1) // 2 :] + "\n")
            handle.flush()
            time.sleep(0.05)
            event2 = {"time": _now_iso(), "step": 2, "name": "kl", "value": 0.2}
            handle.write(json.dumps(event2) + "\n")
            handle.flush()

    writer_thread = threading.Thread(target=writer)
    writer_thread.start()
    gen = read_stream(path, poll_interval=0.01)
    events = []
    try:
        for event in gen:
            events.append(event)
            if len(events) == 2:
                break
    finally:
        gen.close()
    writer_thread.join()

    assert [evt.step for evt in events] == [1, 2]
    assert events[0].value == pytest.approx(0.1)
    assert events[1].value == pytest.approx(0.2)


def test_read_stream_handles_rotation(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.touch()
    gen = read_stream(path, poll_interval=0.01)
    try:
        first = {"time": _now_iso(), "step": 1, "name": "kl", "value": 0.3}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(first) + "\n")
            handle.flush()
        event1 = next(gen)
        assert event1.step == 1

        rotated = path.with_suffix(".rot")
        path.rename(rotated)
        second = {"time": _now_iso(), "step": 2, "name": "kl", "value": 0.4}
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(second) + "\n")
            handle.flush()
        time.sleep(0.05)
        event2 = next(gen)
        assert event2.step == 2
    finally:
        gen.close()


def test_monitor_engine_warn_action(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        """
rules:
  - id: high_kl
    where: name == "kl"
    condition: value > 0.5
    window:
      size: 2
    grace_steps: 2
    cooldown_steps: 1
    actions:
      - warn:
          msg: "KL {value:.2f} at step {step}"
"""
    )
    rules = load_rules(rules_path)
    engine = MonitorEngine(rules)
    events = [
        Event(time=_now_iso(), step=1, name="kl", value=0.4),
        Event(time=_now_iso(), step=2, name="kl", value=0.6),
        Event(time=_now_iso(), step=3, name="kl", value=0.7),
        Event(time=_now_iso(), step=6, name="kl", value=0.8),
    ]
    messages = []
    for event in events:
        for alert in engine.process_event(event):
            messages.append(alert.message)
    assert messages == ["KL 0.60 at step 2", "KL 0.80 at step 6"]

    report = engine.generate_report().to_dict()
    rule_summary = report["rules"]["high_kl"]
    assert rule_summary["activations"] == 2
    assert rule_summary["first_activation"]["step"] == 2
    assert rule_summary["last_activation"]["step"] == 6


def test_monitor_cli_once_writes_report(tmp_path: Path, runner: CliRunner) -> None:
    events_path = tmp_path / "events.jsonl"
    events = [
        {"time": _now_iso(), "step": 1, "name": "kl", "value": 0.4},
        {"time": _now_iso(), "step": 2, "name": "kl", "value": 0.6},
        {"time": _now_iso(), "step": 3, "name": "kl", "value": 0.7},
    ]
    events_path.write_text("\n".join(json.dumps(evt) for evt in events) + "\n")
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        """
rules:
  - id: stop_kl
    where: name == "kl"
    condition: value > 0.5
    actions:
      - warn:
          msg: "{name} {value:.2f}"
"""
    )
    report_path = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "monitor",
            "--once",
            str(events_path),
            "--rules",
            str(rules_path),
            "--report",
            str(report_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    report = json.loads(report_path.read_text())
    assert report["rules"]["stop_kl"]["activations"] == 2
    assert "stop_kl" in report["rules"]


def test_emit_cli_writes_event(tmp_path: Path, runner: CliRunner) -> None:
    output = tmp_path / "events.jsonl"
    result = runner.invoke(
        app,
        [
            "emit",
            "--to",
            str(output),
            "--name",
            "reward",
            "--value",
            "1.5",
            "--step",
            "10",
            "--tags",
            '{"policy": "test"}',
        ],
    )
    assert result.exit_code == 0, result.stdout
    payloads = [json.loads(line) for line in output.read_text().splitlines()]
    assert payloads[0]["name"] == "reward"
    assert payloads[0]["step"] == 10
    assert payloads[0]["tags"] == {"policy": "test"}
    assert "reward" in result.stdout
