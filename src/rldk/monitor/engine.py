"""Core monitoring engine for streaming JSONL events."""

import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, TextIO, Union

import yaml

from .rules import Rule, RuleEngine


def read_stream(
    path_or_stdin: Union[str, Path, None],
    field_map: Optional[Dict[str, str]] = None,
    follow: bool = False
) -> Generator[Dict[str, Any], None, None]:
    """Read and canonicalize events from JSONL stream.

    Args:
        path_or_stdin: Path to file, "-" for stdin, or None for stdin
        field_map: Optional mapping from source fields to canonical schema
        follow: Whether to follow file like tail -F (streaming mode)

    Yields:
        Canonicalized event dictionaries
    """
    if path_or_stdin is None or path_or_stdin == "-":
        yield from _read_file_stream(sys.stdin, field_map)
    else:
        path = Path(path_or_stdin)

        if path.suffix == ".gz":
            if follow:
                raise ValueError("Cannot follow gzip files in streaming mode")
            with gzip.open(path, "rt", encoding="utf-8") as f:
                yield from _read_file_stream(f, field_map)
        else:
            if follow:
                yield from _follow_file(path, field_map)
            else:
                with open(path, encoding="utf-8") as f:
                    yield from _read_file_stream(f, field_map)


def _read_file_stream(
    file_handle: TextIO,
    field_map: Optional[Dict[str, str]] = None
) -> Generator[Dict[str, Any], None, None]:
    """Read events from file handle."""
    partial_line = ""

    for line in file_handle:
        line = partial_line + line

        if not line.endswith("\n"):
            partial_line = line
            continue
        else:
            partial_line = ""

        line = line.strip()
        if not line:
            continue

        try:
            raw_event = json.loads(line)
            canonical_event = _canonicalize_event(raw_event, field_map)
            if canonical_event:
                yield canonical_event
        except json.JSONDecodeError:
            continue


def _follow_file(
    path: Path,
    field_map: Optional[Dict[str, str]] = None
) -> Generator[Dict[str, Any], None, None]:
    """Follow file like tail -F with rotation and truncation handling."""
    last_size = 0
    last_inode = None

    while True:
        try:
            stat = path.stat()
            current_inode = stat.st_ino
            current_size = stat.st_size

            if last_inode is not None and current_inode != last_inode:
                last_size = 0

            if current_size < last_size:
                last_size = 0

            if current_size > last_size:
                with open(path, encoding="utf-8") as f:
                    f.seek(last_size)
                    yield from _read_file_stream(f, field_map)
                    last_size = f.tell()

            last_inode = current_inode
            time.sleep(0.1)

        except (OSError, FileNotFoundError):
            time.sleep(0.1)
            continue


def _canonicalize_event(
    raw_event: Dict[str, Any],
    field_map: Optional[Dict[str, str]] = None
) -> Optional[Dict[str, Any]]:
    """Convert raw event to canonical schema.

    Args:
        raw_event: Raw event dictionary
        field_map: Optional field mapping {"source_field": "canonical_field"}

    Returns:
        Canonicalized event dictionary or None if invalid
    """
    if field_map is None:
        field_map = {}

    default_map = {
        "time": "time",
        "step": "step",
        "name": "name",
        "value": "value",
        "run_id": "run_id",
        "tags": "tags",
        "meta": "meta",
    }

    effective_map = {**default_map, **field_map}

    canonical = {}

    for canonical_field, source_field in effective_map.items():
        if source_field in raw_event:
            canonical[canonical_field] = raw_event[source_field]

    required_fields = ["time", "step", "name", "value"]
    for field in required_fields:
        if field not in canonical:
            return None

    try:
        canonical["step"] = int(canonical["step"])
        canonical["value"] = float(canonical["value"])
        canonical["name"] = str(canonical["name"])
        canonical["time"] = str(canonical["time"])
    except (ValueError, TypeError):
        return None

    return canonical


class MonitorEngine:
    """Main monitoring engine for processing JSONL events."""

    def __init__(
        self,
        rules: List[Rule],
        alerts_path: Optional[Union[str, Path]] = None,
        report_path: Optional[Union[str, Path]] = None
    ):
        """Initialize monitoring engine.

        Args:
            rules: List of rules to evaluate
            alerts_path: Path to write alerts.jsonl (default: artifacts/alerts.jsonl)
            report_path: Path to write report.json (optional)
        """
        self.rule_engine = RuleEngine(rules)

        if alerts_path is None:
            alerts_path = Path("artifacts/alerts.jsonl")
        self.alerts_path = Path(alerts_path)
        self.alerts_path.parent.mkdir(parents=True, exist_ok=True)

        self.report_path = Path(report_path) if report_path else None

        self.alerts: List[Dict[str, Any]] = []
        self.last_seen_metrics: Dict[str, Any] = {}
        self.rule_activation_counts: Dict[str, int] = {}
        self.first_activations: Dict[str, Dict[str, Any]] = {}
        self.last_activations: Dict[str, Dict[str, Any]] = {}

    def process_stream(
        self,
        path_or_stdin: Union[str, Path, None],
        field_map: Optional[Dict[str, str]] = None,
        follow: bool = False
    ) -> None:
        """Process event stream and generate alerts.

        Args:
            path_or_stdin: Path to JSONL file, "-" for stdin, or None for stdin
            field_map: Optional field mapping for non-canonical schemas
            follow: Whether to follow file in streaming mode
        """
        for event in read_stream(path_or_stdin, field_map, follow):
            self._process_event(event)

        if self.report_path:
            self._write_report()

    def _process_event(self, event: Dict[str, Any]) -> None:
        """Process a single event."""
        metric_key = event["name"]
        if event.get("tags"):
            tag_str = "&".join(f"{k}={v}" for k, v in sorted(event["tags"].items()))
            metric_key = f"{metric_key}#{tag_str}"
        self.last_seen_metrics[metric_key] = event

        alerts = self.rule_engine.evaluate_event(event)

        for alert in alerts:
            self._handle_alert(alert)

    def _handle_alert(self, alert: Dict[str, Any]) -> None:
        """Handle a triggered alert."""
        rule_id = alert["rule_id"]

        self.rule_activation_counts[rule_id] = self.rule_activation_counts.get(rule_id, 0) + 1

        if rule_id not in self.first_activations:
            self.first_activations[rule_id] = {
                "step": alert["step"],
                "timestamp": alert["ts"]
            }
        self.last_activations[rule_id] = {
            "step": alert["step"],
            "timestamp": alert["ts"]
        }

        for rule in self.rule_engine.rules:
            if rule.id == rule_id:
                self._execute_actions(alert, rule.actions)
                break

        self.alerts.append(alert)

        self._write_alert(alert)

    def _execute_actions(self, alert: Dict[str, Any], actions: List[Dict[str, Any]]) -> None:
        """Execute actions for an alert."""
        for action in actions:
            if "warn" in action:
                self._execute_warn_action(alert, action["warn"])

    def _execute_warn_action(self, alert: Dict[str, Any], warn_config: Dict[str, Any]) -> None:
        """Execute warn action."""
        message = warn_config.get("msg", "Alert triggered")
        templated_message = self._template_message(message, alert)

        print(f"WARNING: {templated_message}", file=sys.stderr)

        alert["action"] = "warn"
        alert["message"] = templated_message

    def _template_message(self, template: str, alert: Dict[str, Any]) -> str:
        """Template a message with alert data."""
        try:
            context = {
                "name": alert["metric"],
                "value": alert["value"],
                "step": alert["step"],
                "rule_id": alert["rule_id"],
                "ts": alert["ts"],
                "metric": alert["metric"],
            }
            if alert.get("run_id"):
                context["run_id"] = alert["run_id"]
            if alert.get("tags"):
                context["tags"] = alert["tags"]
            if alert.get("meta"):
                context["meta"] = alert["meta"]

            return template.format(**context)
        except (KeyError, ValueError):
            return template

    def _write_alert(self, alert: Dict[str, Any]) -> None:
        """Write alert to alerts.jsonl file."""
        with open(self.alerts_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert) + "\n")
            f.flush()

    def _write_report(self) -> None:
        """Write summary report to JSON file."""
        if not self.report_path:
            return

        report = {
            "rules_summary": {
                "total_rules": len(self.rule_engine.rules),
                "activated_rules": len(self.rule_activation_counts),
                "activation_counts": self.rule_activation_counts,
                "first_activations": self.first_activations,
                "last_activations": self.last_activations,
            },
            "last_seen_metrics": self.last_seen_metrics,
            "total_alerts": len(self.alerts),
        }

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


def load_rules_from_yaml(yaml_path: Union[str, Path]) -> List[Rule]:
    """Load rules from YAML file.

    Args:
        yaml_path: Path to YAML rules file

    Returns:
        List of Rule objects
    """
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rules = []
    for rule_data in data.get("rules", []):
        rules.append(Rule.from_dict(rule_data))

    return rules
