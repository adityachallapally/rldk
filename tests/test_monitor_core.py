import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import List, Optional

import pytest
from typer.testing import CliRunner

from rldk.cli import app
from rldk.emit import EventWriter
from rldk.monitor import (
    ActionDispatcher,
    Alert,
    Event,
    MonitorEngine,
    load_rules,
    read_events_once,
    read_stream,
)


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


def test_read_events_once_regex_trl(tmp_path: Path) -> None:
    log_path = tmp_path / "stdout.log"
    log_path.write_text(
        "\n".join(
            [
                "step=1 kl=0.31 reward=-0.10",
                "step=2 kl=0.62 reward=-0.05",
            ]
        )
        + "\n"
    )

    events = read_events_once(log_path, regex="trl")
    sequence = [(event.step, event.name) for event in events]
    assert sequence == [(1, "kl"), (1, "reward"), (2, "kl"), (2, "reward")]
    kl_values = [event.value for event in events if event.name == "kl"]
    assert kl_values == pytest.approx([0.31, 0.62])


def test_read_stream_regex_pattern(tmp_path: Path) -> None:
    path = tmp_path / "metrics.log"
    pattern = r"step=(?P<step>\d+)\s+metric=(?P<name>\w+)\s+value=(?P<value>[-+0-9.]+)"

    def writer() -> None:
        with path.open("w", encoding="utf-8") as handle:
            for step, value in enumerate([0.2, 0.4, 0.6], start=1):
                handle.write(f"step={step} metric=kl value={value}\n")
                handle.flush()
                time.sleep(0.01)

    thread = threading.Thread(target=writer)
    thread.start()
    gen = read_stream(path, regex=pattern, poll_interval=0.01)
    try:
        events = [next(gen), next(gen), next(gen)]
    finally:
        gen.close()
    thread.join()

    assert [event.step for event in events] == [1, 2, 3]
    assert [event.value for event in events] == pytest.approx([0.2, 0.4, 0.6])


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
    assert messages == ["KL 0.70 at step 3", "KL 0.80 at step 6"]

    report = engine.generate_report().to_dict()
    rule_summary = report["rules"]["high_kl"]
    assert rule_summary["activations"] == 2
    assert rule_summary["first_activation"]["step"] == 3
    assert rule_summary["last_activation"]["step"] == 6


def test_count_aggregator_respects_predicate(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        """
rules:
  - id: count_high_kl
    where: name == "kl"
    condition: count(value > 0.5) >= 3
    window:
      size: 4
    actions:
      - warn: {}
"""
    )
    rules = load_rules(rules_path)
    engine = MonitorEngine(rules)

    values = [0.6, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    alerts: List[Alert] = []
    for idx, val in enumerate(values, start=1):
        event = Event(time=_now_iso(), step=idx, name="kl", value=val)
        alerts.extend(engine.process_event(event))

    assert [alert.event.step for alert in alerts] == [7]


def test_monitor_engine_supports_rolling_window(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        """
rules:
  - id: rolling_mean
    where: name == "kl"
    condition: mean(value) > 0.6
    window:
      size: 3
      kind: rolling
    actions:
      - warn: {}
"""
    )
    engine = MonitorEngine(load_rules(rules_path))
    triggered: List[int] = []
    for idx, value in enumerate([0.4, 0.8, 0.9, 0.2], start=1):
        event = Event(time=_now_iso(), step=idx, name="kl", value=value)
        for alert in engine.process_event(event):
            triggered.append(alert.event.step)
    assert triggered == [3, 4]


def test_consecutive_window_requires_back_to_back_hits(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        """
rules:
  - id: consecutive_high_kl
    where: name == "kl"
    condition: value > 0.5
    window:
      size: 5
    actions:
      - warn: {}
"""
    )
    engine = MonitorEngine(load_rules(rules_path))
    values = [0.9, 0.85, 0.2, 0.92, 0.95, 0.97, 0.99, 1.01]
    activations: List[int] = []
    for step, value in enumerate(values, start=1):
        event = Event(time=_now_iso(), step=step, name="kl", value=value)
        for alert in engine.process_event(event):
            activations.append(alert.event.step)

    assert activations == [8]


def test_rolling_allows_single_spike_but_consecutive_does_not(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        """
rules:
  - id: consecutive_high_kl
    where: name == "kl"
    condition: value > 0.8
    window:
      size: 3
    actions:
      - warn: {}
  - id: rolling_spike
    where: name == "kl"
    condition: any(value > 0.8)
    window:
      size: 3
      kind: rolling
    actions:
      - warn: {}
"""
    )

    engine = MonitorEngine(load_rules(rules_path))
    values = [0.2, 0.95, 0.3, 0.1]
    consecutive_hits: List[int] = []
    rolling_hits: List[int] = []
    for step, value in enumerate(values, start=1):
        event = Event(time=_now_iso(), step=step, name="kl", value=value)
        for alert in engine.process_event(event):
            if alert.rule_id == "consecutive_high_kl":
                consecutive_hits.append(alert.event.step)
            elif alert.rule_id == "rolling_spike":
                rolling_hits.append(alert.event.step)

    assert consecutive_hits == []
    assert rolling_hits == [3, 4]


def test_load_rules_builtin_preset() -> None:
    rules = load_rules("ppo_safe")
    ids = {rule.id for rule in rules}
    assert "ppo_high_kl_guard" in ids


def test_stop_action_terminates_process(tmp_path: Path) -> None:
    script = (
        "import signal, time, sys\n"
        "signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))\n"
        "print('ready', flush=True)\n"
        "while True:\n"
        "    time.sleep(0.1)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    alerts: List[Alert] = []
    exit_code: Optional[int] = None
    try:
        assert proc.stdout is not None
        proc.stdout.readline()
        rules_path = tmp_path / "rules.yaml"
        rules_path.write_text(
            """
rules:
  - id: stop_loop
    where: name == "kl"
    condition: value > 0.5
    actions:
      - stop: {}
"""
        )
        rules = load_rules(rules_path)
        engine = MonitorEngine(
            rules, action_executor=ActionDispatcher(pid=proc.pid, kill_timeout_sec=1.0)
        )
        alerts = engine.process_event(
            Event(time=_now_iso(), step=1, name="kl", value=0.7)
        )
        exit_code = proc.wait(timeout=2)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)
    assert exit_code == 0
    assert alerts and alerts[0].action == "stop"
    assert alerts[0].status == "success"
    assert alerts[0].details.get("terminated") is True


def test_read_stream_directory_handles_new_files(tmp_path: Path) -> None:
    directory = tmp_path / "logs"
    directory.mkdir()

    def writer() -> None:
        time.sleep(0.05)
        first_path = directory / "first.jsonl"
        with first_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": _now_iso(), "step": 1, "name": "kl", "value": 0.2}) + "\n")
            handle.flush()
            time.sleep(0.05)
            handle.write(json.dumps({"time": _now_iso(), "step": 2, "name": "kl", "value": 0.3}) + "\n")
            handle.flush()
        time.sleep(0.05)
        second_path = directory / "second.jsonl"
        with second_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": _now_iso(), "step": 3, "name": "kl", "value": 0.4}) + "\n")
            handle.flush()

    writer_thread = threading.Thread(target=writer)
    writer_thread.start()

    gen = read_stream(directory, poll_interval=0.01)
    try:
        events = [next(gen), next(gen), next(gen)]
    finally:
        gen.close()
    writer_thread.join()

    assert [event.step for event in events] == [1, 2, 3]


def test_sentinel_action_writes_file(tmp_path: Path) -> None:
    sentinel = tmp_path / "stop.flag"
    path_json = json.dumps(str(sentinel))
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        f"""
rules:
  - id: sentinel_flag
    where: name == "reward"
    condition: value < 0
    actions:
      - sentinel:
          path: {path_json}
          contents: "{{name}}:{{value:.2f}}"
"""
    )
    engine = MonitorEngine(load_rules(rules_path), action_executor=ActionDispatcher())
    alerts = engine.process_event(
        Event(time=_now_iso(), step=5, name="reward", value=-1.25)
    )
    assert sentinel.exists()
    assert sentinel.read_text().strip() == "reward:-1.25"
    assert alerts[0].action == "sentinel"
    assert alerts[0].status == "success"


def test_shell_action_captures_output(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.yaml"
    python_path = json.dumps(sys.executable)
    rules_path.write_text(
        f"""
rules:
  - id: shell_success
    where: name == "kl"
    condition: value > 0.1
    actions:
      - shell:
          cmd: [{python_path}, "-c", "print('hello')"]
"""
    )
    dispatcher = ActionDispatcher(http_timeout_sec=1.0)
    engine = MonitorEngine(load_rules(rules_path), action_executor=dispatcher)
    alerts = engine.process_event(
        Event(time=_now_iso(), step=1, name="kl", value=0.2)
    )
    assert alerts[0].action == "shell"
    assert alerts[0].status == "success"
    assert "hello" in alerts[0].details.get("stdout", "")


def test_shell_action_failure_logs_exit_code(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.yaml"
    python_path = json.dumps(sys.executable)
    rules_path.write_text(
        f"""
rules:
  - id: shell_fail
    where: name == "kl"
    condition: value > 0.1
    actions:
      - shell:
          cmd: [{python_path}, "-c", "import sys; sys.exit(3)"]
"""
    )
    dispatcher = ActionDispatcher(http_timeout_sec=1.0, retries=0)
    engine = MonitorEngine(load_rules(rules_path), action_executor=dispatcher)
    alerts = engine.process_event(
        Event(time=_now_iso(), step=1, name="kl", value=0.2)
    )
    assert alerts[0].action == "shell"
    assert alerts[0].status == "error"
    assert alerts[0].details.get("returncode") == 3


def test_http_action_records_status_code(tmp_path: Path) -> None:
    class _Handler(BaseHTTPRequestHandler):
        status_code = 500
        body = b"error"

        def do_POST(self) -> None:  # pragma: no cover - network side effect
            length = int(self.headers.get("Content-Length", "0"))
            if length:
                self.rfile.read(length)
            self.send_response(self.status_code)
            self.end_headers()
            self.wfile.write(self.body)

        def log_message(self, format: str, *args: object) -> None:  # pragma: no cover
            return

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    try:
        rules_path = tmp_path / "rules.yaml"
        url_json = json.dumps(url)
        rules_path.write_text(
            f"""
rules:
  - id: http_fail
    where: name == "kl"
    condition: value > 0.1
    actions:
      - http:
          url: {url_json}
          method: "POST"
          payload:
            value: "{{value}}"
"""
        )
        dispatcher = ActionDispatcher(http_timeout_sec=1.0, retries=0)
        engine = MonitorEngine(load_rules(rules_path), action_executor=dispatcher)
        alerts = engine.process_event(
            Event(time=_now_iso(), step=1, name="kl", value=0.2)
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
    assert alerts[0].action == "http"
    assert alerts[0].status == "error"
    assert alerts[0].details.get("status_code") == 500


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
    alerts_path = tmp_path / "alerts.jsonl"
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
            "--alerts",
            str(alerts_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    report = json.loads(report_path.read_text())
    assert report["rules"]["stop_kl"]["activations"] == 2
    assert "stop_kl" in report["rules"]
    assert alerts_path.exists()
    alerts_lines = alerts_path.read_text().strip().splitlines()
    assert len(alerts_lines) == 2
    alerts_text = alerts_path.with_suffix(".txt")
    assert alerts_text.exists()
    assert len(alerts_text.read_text().strip().splitlines()) == 2


def test_fullscale_rules_include_remediation_hints(tmp_path: Path, runner: CliRunner) -> None:
    events_path = tmp_path / "events.jsonl"
    with events_path.open("w", encoding="utf-8") as handle:
        for step in range(1, 6):
            payload = {"time": _now_iso(), "step": step, "name": "kl", "value": 0.5}
            handle.write(json.dumps(payload) + "\n")

    rules_path = Path(__file__).resolve().parents[1] / "rules" / "fullscale_rules.yaml"
    alerts_path = tmp_path / "alerts.jsonl"
    result = runner.invoke(
        app,
        [
            "monitor",
            "--once",
            str(events_path),
            "--rules",
            str(rules_path),
            "--alerts",
            str(alerts_path),
        ],
    )
    assert result.exit_code == 0, result.stdout

    alerts = [json.loads(line) for line in alerts_path.read_text().splitlines() if line.strip()]
    assert alerts, "expected at least one alert from KL spike guard"
    messages = "\n".join(alert.get("message", "") for alert in alerts)
    assert "--temperature" in messages
    assert "--learning-rate" in messages

    text_summary = alerts_path.with_suffix(".txt").read_text()
    assert "--temperature" in text_summary
    assert "--learning-rate" in text_summary


def test_monitor_cli_field_map_preset(tmp_path: Path, runner: CliRunner) -> None:
    log_path = tmp_path / "metrics.jsonl"
    values = [0.2, 0.36, 0.38, 0.4, 0.42, 0.44]
    with log_path.open("w", encoding="utf-8") as handle:
        for step, value in enumerate(values, start=1):
            payload = {
                "timestamp": _now_iso(),
                "global_step": step,
                "metric": "kl",
                "value": value,
            }
            handle.write(json.dumps(payload) + "\n")
    report_path = tmp_path / "report.json"
    alerts_path = tmp_path / "alerts.jsonl"
    result = runner.invoke(
        app,
        [
            "monitor",
            "--once",
            str(log_path),
            "--rules",
            "ppo_safe",
            "--preset",
            "trl",
            "--report",
            str(report_path),
            "--alerts",
            str(alerts_path),
        ],
    )
    assert result.exit_code == 0
    report = json.loads(report_path.read_text())
    assert report["rules"]["ppo_high_kl_guard"]["activations"] >= 1
    assert any(alert["rule_id"] == "ppo_high_kl_guard" for alert in report["alerts"])


def test_monitor_cli_regex_preset(tmp_path: Path, runner: CliRunner) -> None:
    log_path = tmp_path / "stdout.log"
    log_path.write_text(
        "\n".join(
            [
                "step=1 kl=0.30 reward=0.10",
                "step=2 kl=0.60 reward=0.15",
            ]
        )
        + "\n"
    )
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        """
rules:
  - id: regex_kl_stop
    where: name == "kl"
    condition: value > 0.5
    actions:
      - warn: {}
"""
    )
    report_path = tmp_path / "report.json"
    alerts_path = tmp_path / "alerts.jsonl"

    result = runner.invoke(
        app,
        [
            "monitor",
            "--once",
            str(log_path),
            "--rules",
            str(rules_path),
            "--regex",
            "trl",
            "--report",
            str(report_path),
            "--alerts",
            str(alerts_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    alerts = [json.loads(line) for line in alerts_path.read_text().splitlines()]
    assert len(alerts) == 1
    assert alerts[0]["rule_id"] == "regex_kl_stop"


def test_monitor_cli_from_wandb(monkeypatch, tmp_path: Path, runner: CliRunner) -> None:
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        """
rules:
  - id: kl_gate
    where: name == "kl"
    condition: value > 0.5
    actions:
      - warn: {}
"""
    )
    events = [
        Event(time=_now_iso(), step=1, name="kl", value=0.4),
        Event(time=_now_iso(), step=2, name="kl", value=0.7),
    ]

    def fake_stream(target: str):
        for event in events:
            yield event

    monkeypatch.setattr("rldk.cli.stream_from_wandb", fake_stream)

    alerts_path = tmp_path / "alerts.jsonl"
    report_path = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "monitor",
            "--from-wandb",
            "entity/project/run",
            "--rules",
            str(rules_path),
            "--alerts",
            str(alerts_path),
            "--report",
            str(report_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    payloads = [json.loads(line) for line in alerts_path.read_text().splitlines()]
    assert any(alert["step"] == 2 for alert in payloads)


def test_monitor_cli_from_mlflow(monkeypatch, tmp_path: Path, runner: CliRunner) -> None:
    rules_path = tmp_path / "rules.yaml"
    rules_path.write_text(
        """
rules:
  - id: reward_warn
    where: name == "reward"
    condition: value < 0.0
    actions:
      - warn: {}
"""
    )
    events = [
        Event(time=_now_iso(), step=1, name="reward", value=0.2),
        Event(time=_now_iso(), step=2, name="reward", value=-0.3),
    ]

    def fake_mlflow(run_id: str):
        for event in events:
            yield event

    monkeypatch.setattr("rldk.cli.stream_from_mlflow", fake_mlflow)

    alerts_path = tmp_path / "alerts.jsonl"
    result = runner.invoke(
        app,
        [
            "monitor",
            "--from-mlflow",
            "fake-run-id",
            "--rules",
            str(rules_path),
            "--alerts",
            str(alerts_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    alerts = [json.loads(line) for line in alerts_path.read_text().splitlines()]
    assert any(alert["step"] == 2 and alert["name"] == "reward" for alert in alerts)

def test_monitor_cli_defaults_to_stdin(tmp_path: Path, runner: CliRunner) -> None:
    events = []
    for step in range(1, 7):
        payload = {"time": _now_iso(), "step": step, "name": "kl", "value": 0.4}
        events.append(json.dumps(payload))
    report_path = tmp_path / "report.json"
    alerts_path = tmp_path / "alerts.jsonl"
    result = runner.invoke(
        app,
        [
            "monitor",
            "--rules",
            "ppo_safe",
            "--preset",
            "trl",
            "--report",
            str(report_path),
            "--alerts",
            str(alerts_path),
        ],
        input="\n".join(events) + "\n",
    )
    assert result.exit_code == 0
    report = json.loads(report_path.read_text())
    assert report["rules"]["ppo_high_kl_guard"]["activations"] >= 1
    assert alerts_path.exists()


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
