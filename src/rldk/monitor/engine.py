"""Core monitoring engine for streaming JSONL training events."""
from __future__ import annotations

import ast
import json
import logging
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import yaml
import pandas as pd

from .presets import FIELD_MAP_PRESETS, RULE_PRESETS, get_field_map_preset, get_rule_preset
from .regexes import create_line_parser

logger = logging.getLogger(__name__)

CANONICAL_FIELDS = {"time", "step", "name", "value", "run_id", "tags", "meta"}
DEFAULT_WINDOW_KIND = "consecutive"
AGGREGATOR_FUNCTIONS = {"mean", "max", "min", "any", "all", "sum", "count"}

FieldMapSpec = Optional[Mapping[str, str] | str]

DEFAULT_ACTION_MESSAGES: Dict[str, str] = {
    "warn": "{name} {value:.4f} at step {step} (rule {rule_id})",
    "stop": "Sent stop signal to PID {pid} for {name} at step {step}",
    "sentinel": "Wrote sentinel file at {path}",
    "shell": "Executed shell command '{cmd}' (exit={returncode})",
    "http": "HTTP {method} {url} -> {status_code}",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sanitize_for_meta(value: Any) -> Any:
    """Convert analysis outputs into JSON-serializable values."""

    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, pd.Series):
        return [_sanitize_for_meta(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return {
            key: [_sanitize_for_meta(item) for item in series]
            for key, series in value.to_dict(orient="list").items()
        }
    if isinstance(value, dict):
        return {key: _sanitize_for_meta(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_meta(item) for item in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


class SafeExpression:
    """Compile and evaluate a restricted Python expression."""

    def __init__(self, expression: str, allowed_functions: Optional[Sequence[str]] = None) -> None:
        if not expression:
            raise ValueError("Expression must be a non-empty string")
        self.expression = expression
        self._allowed_functions = set(allowed_functions or [])
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid expression '{expression}': {exc}") from exc
        _SafeExpressionValidator(self._allowed_functions).visit(tree)
        self._code = compile(tree, "<rule>", "eval")

    def evaluate(self, context: Dict[str, Any], functions: Optional[Dict[str, Callable[..., Any]]] = None) -> Any:
        env: Dict[str, Any] = dict(context)
        if functions:
            env.update(functions)
        try:
            return eval(self._code, {"__builtins__": {}}, env)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Failed to evaluate expression '{self.expression}': {exc}") from exc


class _SafeExpressionValidator(ast.NodeVisitor):
    """Validate that an AST only contains safe nodes."""

    _ALLOWED_NODES = (
        ast.Expression,
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Call,
        ast.Attribute,
        ast.Subscript,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.Set,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
        ast.USub,
        ast.UAdd,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
        ast.Is,
        ast.IsNot,
        ast.IfExp,
    )

    def __init__(self, allowed_functions: Sequence[str]) -> None:
        super().__init__()
        self._allowed_functions = set(allowed_functions)

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, self._ALLOWED_NODES):
            raise ValueError(f"Unsupported expression node: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed")
        if node.func.id not in self._allowed_functions:
            raise ValueError(f"Function '{node.func.id}' is not allowed in expressions")
        super().generic_visit(node)


@dataclass
class Event:
    """Canonical representation of a monitoring event."""

    time: str
    step: int
    name: str
    value: float
    run_id: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "time": self.time,
            "step": self.step,
            "name": self.name,
            "value": self.value,
        }
        if self.run_id is not None:
            payload["run_id"] = self.run_id
        if self.tags:
            payload["tags"] = self.tags
        if self.meta:
            payload["meta"] = self.meta
        return payload


class WindowExpression:
    """Base class for lazily evaluated expressions over a window of events."""

    def latest(self) -> Any:
        raise NotImplementedError

    def resolve_series(self, length: Optional[int] = None) -> Sequence[Any]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __bool__(self) -> bool:
        return bool(self.latest())

    def __float__(self) -> float:
        return float(self.latest())

    def __int__(self) -> int:
        return int(self.latest())


class ConstantExpression(WindowExpression):
    """A constant value used in window operations."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def latest(self) -> Any:
        return self._value

    def resolve_series(self, length: Optional[int] = None) -> Sequence[Any]:
        if length is None or length <= 1:
            return [self._value]
        return [self._value] * length

    def __len__(self) -> int:
        return 1


class WindowAccessor(WindowExpression):
    """Provides access to an attribute over a window of events."""

    def __init__(self, attribute: str, window: Sequence[Event]):
        self._attribute = attribute
        self._window = window

    def _value_for_event(self, event: Event) -> Any:
        return getattr(event, self._attribute)

    def latest(self) -> Any:
        return self._value_for_event(self._window[-1])

    def resolve_series(self, length: Optional[int] = None) -> Sequence[Any]:
        return [self._value_for_event(evt) for evt in self._window]

    def __len__(self) -> int:
        return len(self._window)

    def _wrap(self, other: Any, op: Callable[[Any, Any], Any]) -> "WindowComparison":
        other_expr = other if isinstance(other, WindowExpression) else ConstantExpression(other)
        return WindowComparison(self, other_expr, op)

    def __eq__(self, other: Any) -> "WindowComparison":  # type: ignore[override]
        return self._wrap(other, lambda left, right: left == right)

    def __ne__(self, other: Any) -> "WindowComparison":  # type: ignore[override]
        return self._wrap(other, lambda left, right: left != right)

    def __lt__(self, other: Any) -> "WindowComparison":
        return self._wrap(other, lambda left, right: left < right)

    def __le__(self, other: Any) -> "WindowComparison":
        return self._wrap(other, lambda left, right: left <= right)

    def __gt__(self, other: Any) -> "WindowComparison":
        return self._wrap(other, lambda left, right: left > right)

    def __ge__(self, other: Any) -> "WindowComparison":
        return self._wrap(other, lambda left, right: left >= right)


class WindowComparison(WindowExpression):
    """Comparison between two window expressions."""

    def __init__(
        self,
        left: WindowExpression,
        right: WindowExpression,
        comparator: Callable[[Any, Any], Any],
    ) -> None:
        self._left = left
        self._right = right
        self._comparator = comparator

    def __len__(self) -> int:
        return len(self._left)

    def latest(self) -> Any:
        return self._comparator(self._left.latest(), self._right.latest())

    def resolve_series(self, length: Optional[int] = None) -> Sequence[Any]:
        series_length = len(self) if length is None else length
        left_series = list(self._left.resolve_series(series_length))
        right_series = list(self._right.resolve_series(series_length))
        if len(right_series) == 1 and series_length > 1:
            right_series = list(right_series) * series_length
        if len(left_series) != len(right_series):
            raise ValueError("Mismatched comparison series lengths")
        return [self._comparator(l, r) for l, r in zip(left_series, right_series)]


@dataclass
class RuleDefinition:
    """Rule configuration for monitoring."""

    id: str
    condition: SafeExpression
    window_size: int = 1
    window_kind: str = DEFAULT_WINDOW_KIND
    where: Optional[SafeExpression] = None
    grace_steps: int = 0
    cooldown_steps: int = 0
    actions: List["ActionConfig"] = field(default_factory=list)


@dataclass
class ActionConfig:
    """Runtime configuration for an action."""

    kind: str
    message_template: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert emitted by the monitoring engine."""

    rule_id: str
    action: str
    event: Event
    window_size: int
    window_kind: str
    message: Optional[str] = None
    status: str = "success"
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "ts": self.timestamp,
            "rule_id": self.rule_id,
            "action": self.action,
            "step": self.event.step,
            "time": self.event.time,
            "name": self.event.name,
            "value": self.event.value,
            "metric": self.event.name,
            "window": {"size": self.window_size, "kind": self.window_kind},
            "status": self.status,
        }
        if self.message:
            payload["message"] = self.message
        if self.details:
            payload["details"] = self.details
        if self.event.run_id is not None:
            payload["run_id"] = self.event.run_id
        if self.event.tags:
            payload["tags"] = self.event.tags
        if self.event.meta:
            payload["meta"] = self.event.meta
        return payload

    def summary(self) -> str:
        base = self.message
        if not base:
            base = (
                f"[{self.rule_id}] {self.event.name} {self.event.value:.4f} "
                f"at step {self.event.step} ({self.action})"
            )
        if self.status == "error":
            return f"ERROR: {base}"
        if self.status not in {"success", "ok"}:
            return f"{self.status.upper()}: {base}"
        return base


@dataclass
class RuleStats:
    """Track runtime statistics for a rule."""

    activations: int = 0
    first_activation: Optional[Dict[str, Any]] = None
    last_activation: Optional[Dict[str, Any]] = None


@dataclass
class MonitorReport:
    """Structured report produced from a monitoring session."""

    rules: Dict[str, Dict[str, Any]]
    alerts: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        ordered_rules = dict(sorted(self.rules.items(), key=lambda item: item[0]))
        ordered_alerts = sorted(self.alerts, key=lambda item: (item.get("step", 0), item.get("rule_id", "")))
        return {"rules": ordered_rules, "alerts": ordered_alerts}


def _ensure_dict(value: Any, field: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise ValueError(f"Expected '{field}' to be a mapping, received {type(value).__name__}")


def _canonical_time(value: Any) -> str:
    if value is None:
        return _now_iso()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid ISO8601 timestamp: {value}") from exc
        if value.endswith("+00:00"):
            return value.replace("+00:00", "Z")
        return value
    raise ValueError(f"Unsupported time value type: {type(value).__name__}")


def _resolve_field_map(field_map: FieldMapSpec) -> Optional[Dict[str, str]]:
    if field_map is None:
        return None
    if isinstance(field_map, dict):
        return field_map
    if isinstance(field_map, str):
        preset = get_field_map_preset(field_map)
        if preset is None:
            available = ", ".join(sorted(FIELD_MAP_PRESETS))
            raise ValueError(
                f"Unknown field map preset '{field_map}'. Available presets: {available}"
            )
        return preset
    if isinstance(field_map, Mapping):
        return dict(field_map)
    raise TypeError("field_map must be a mapping or preset name")


def _apply_field_map(event: Dict[str, Any], field_map: Optional[Mapping[str, str]]) -> Dict[str, Any]:
    if not field_map:
        return dict(event)
    mapped: Dict[str, Any] = {}
    for key, value in event.items():
        canonical_key = field_map.get(key, key)
        mapped[canonical_key] = value
    return mapped


def canonicalize_event(payload: Dict[str, Any], field_map: FieldMapSpec = None) -> Event:
    resolved_field_map = _resolve_field_map(field_map)
    mapped = _apply_field_map(payload, resolved_field_map)
    missing = [field for field in ("time", "step", "name", "value") if field not in mapped]
    if missing:
        raise ValueError(f"Event missing required fields: {', '.join(missing)}")
    time_value = _canonical_time(mapped.get("time"))
    try:
        step_value = int(mapped.get("step"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid step value: {mapped.get('step')}") from exc
    try:
        numeric_value = float(mapped.get("value"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value: {mapped.get('value')}") from exc
    name_value = str(mapped.get("name"))
    run_id = mapped.get("run_id")
    if run_id is not None:
        run_id = str(run_id)
    tags = _ensure_dict(mapped.get("tags"), "tags")
    meta = _ensure_dict(mapped.get("meta"), "meta")
    extras = {key: value for key, value in mapped.items() if key not in CANONICAL_FIELDS}
    return Event(
        time=time_value,
        step=step_value,
        name=name_value,
        value=numeric_value,
        run_id=run_id,
        tags=tags,
        meta=meta,
        raw={**extras, **{key: mapped[key] for key in CANONICAL_FIELDS if key in mapped}},
    )


def read_events_once(
    path: str | os.PathLike[str],
    field_map: FieldMapSpec = None,
    regex: Optional[str] = None,
) -> List[Event]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Event log '{path}' does not exist")
    opener: Callable[..., Iterable[str]]
    if path_obj.suffix == ".gz":
        import gzip

        def _open_gzip(p: Path) -> Iterable[str]:
            with gzip.open(p, "rt", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    yield line

        opener = _open_gzip
    else:
        def _open_text(p: Path) -> Iterable[str]:
            with p.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    yield line

        opener = _open_text
    parser = create_line_parser(regex) if regex else None
    resolved_field_map = _resolve_field_map(field_map)
    events: List[Event] = []
    auto_step = 0

    def _next_auto_step() -> int:
        nonlocal auto_step
        auto_step += 1
        return auto_step

    for line in opener(path_obj):
        line = line.strip()
        if not line:
            continue
        if parser is None:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse JSON event: %s", exc)
                continue
            try:
                events.append(canonicalize_event(payload, resolved_field_map))
            except ValueError as exc:
                logger.warning("Skipping invalid event: %s", exc)
        else:
            try:
                parsed_events = list(parser.parse(line))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Regex parser failed for line: %s", exc)
                continue
            for payload in parsed_events:
                data = dict(payload)
                data.setdefault("time", _now_iso())
                data.setdefault("step", _next_auto_step())
                try:
                    events.append(canonicalize_event(data, resolved_field_map))
                except ValueError as exc:
                    logger.warning("Skipping invalid parsed event: %s", exc)
    return events


def read_stream(
    path_or_stdin: str | os.PathLike[str],
    field_map: FieldMapSpec = None,
    regex: Optional[str] = None,
    poll_interval: float = 0.5,
) -> Iterator[Event]:
    """Yield canonical events from a stream, following file rotations."""

    parser = create_line_parser(regex) if regex else None
    resolved_field_map = _resolve_field_map(field_map)
    auto_step = 0

    def _next_auto_step() -> int:
        nonlocal auto_step
        auto_step += 1
        return auto_step

    def _line_events(line: str) -> Iterable[Event]:
        nonlocal auto_step
        if parser is None:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse JSON event: %s", exc)
                return []
            try:
                event = canonicalize_event(payload, resolved_field_map)
                auto_step = max(auto_step, event.step)
                return [event]
            except ValueError as exc:
                logger.warning("Skipping invalid event: %s", exc)
                return []
        try:
            parsed_payloads = list(parser.parse(line)) if parser else []
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Regex parser failed for line: %s", exc)
            return []
        events: List[Event] = []
        for payload in parsed_payloads:
            data = dict(payload)
            data.setdefault("time", _now_iso())
            data.setdefault("step", _next_auto_step())
            try:
                event = canonicalize_event(data, resolved_field_map)
                auto_step = max(auto_step, event.step)
                events.append(event)
            except ValueError as exc:
                logger.warning("Skipping invalid parsed event: %s", exc)
        return events

    if str(path_or_stdin) == "-":
        import sys

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            for event in _line_events(line):
                yield event
        return

    raw_target = str(path_or_stdin)
    base_path = Path(raw_target)
    buffer = ""
    handle = None
    current_inode: Optional[int] = None
    treat_as_directory = base_path.is_dir() or raw_target.endswith(os.sep)
    watch_directory: Optional[Path] = base_path if treat_as_directory else None
    active_path: Optional[Path] = None if treat_as_directory else base_path

    def _open_file(target: Path) -> Optional[Any]:
        nonlocal handle, current_inode
        try:
            handle = target.open("r", encoding="utf-8", errors="replace")
            current_inode = os.fstat(handle.fileno()).st_ino
            return handle
        except FileNotFoundError:
            return None

    def _select_latest_jsonl(directory: Path) -> Optional[Path]:
        try:
            if not directory.exists():
                return None
        except OSError:
            return None
        newest: Optional[Path] = None
        newest_mtime = -1.0
        for candidate in directory.glob("*.jsonl"):
            try:
                stat_result = candidate.stat()
            except FileNotFoundError:
                continue
            mtime = stat_result.st_mtime
            if (
                newest is None
                or mtime > newest_mtime
                or (mtime == newest_mtime and candidate.name > newest.name)
            ):
                newest = candidate
                newest_mtime = mtime
        return newest

    try:
        while True:
            if not treat_as_directory and base_path.is_dir():
                treat_as_directory = True
                watch_directory = base_path
                active_path = None
            if treat_as_directory:
                directory = watch_directory or base_path
                latest_file = _select_latest_jsonl(directory)
                if latest_file is None:
                    if handle is not None:
                        handle.close()
                        handle = None
                    buffer = ""
                    current_inode = None
                    time.sleep(poll_interval)
                    continue
                if active_path is None or latest_file != active_path:
                    if handle is not None:
                        handle.close()
                        handle = None
                    buffer = ""
                    current_inode = None
                    active_path = latest_file
            if active_path is None:
                time.sleep(poll_interval)
                continue
            if handle is None:
                handle = _open_file(active_path)
                if handle is None:
                    time.sleep(poll_interval)
                    continue
            chunk = handle.read()
            if chunk:
                buffer += chunk
                while True:
                    newline_index = buffer.find("\n")
                    if newline_index == -1:
                        break
                    line = buffer[:newline_index]
                    buffer = buffer[newline_index + 1 :]
                    line = line.strip()
                    if not line:
                        continue
                    for event in _line_events(line):
                        yield event
            else:
                if active_path is None:
                    time.sleep(poll_interval)
                    continue
                try:
                    stat = active_path.stat()
                except FileNotFoundError:
                    if handle is not None:
                        handle.close()
                        handle = None
                    buffer = ""
                    current_inode = None
                    time.sleep(poll_interval)
                    continue
                if current_inode is not None and stat.st_ino != current_inode:
                    handle.close()
                    handle = None
                    buffer = ""
                    current_inode = None
                    continue
                if handle.tell() > stat.st_size:
                    handle.seek(0)
                    buffer = ""
                    continue
                time.sleep(poll_interval)
    finally:
        if handle is not None:
            handle.close()


MetricKey = Tuple[str, Optional[str], Tuple[Tuple[str, str], ...]]


def _normalize_tags(tags: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    if not tags:
        return ()
    try:
        items = tuple(sorted((key, json.dumps(value, sort_keys=True)) for key, value in tags.items()))
    except TypeError:
        normalized = []
        for key, value in tags.items():
            try:
                normalized.append((key, json.dumps(value, sort_keys=True)))
            except TypeError:
                normalized.append((key, repr(value)))
        items = tuple(sorted(normalized))
    return items


def _metric_key(event: Event) -> MetricKey:
    return (event.name, event.run_id, _normalize_tags(event.tags))


class MonitorEngine:
    """Evaluate rules against incoming events."""

    def __init__(
        self,
        rules: Sequence[RuleDefinition],
        action_executor: Optional["ActionExecutor"] = None,
        *,
        reward_health_window: Optional[int] = None,
    ):
        if not rules:
            raise ValueError("MonitorEngine requires at least one rule")
        self._rules = list(rules)
        self._buffers: Dict[str, Dict[MetricKey, Deque[Event]]] = defaultdict(dict)
        self._counts: Dict[str, Dict[MetricKey, int]] = defaultdict(lambda: defaultdict(int))
        self._cooldowns: Dict[str, Dict[MetricKey, int]] = defaultdict(dict)
        self._stats: Dict[str, RuleStats] = defaultdict(RuleStats)
        self._alerts: List[Alert] = []
        self._executor: ActionExecutor = action_executor or SimpleActionExecutor()
        self._reward_health_monitor = None
        if reward_health_window and reward_health_window > 0:
            self._reward_health_monitor = _RewardHealthMonitor(reward_health_window)

    def process_event(
        self,
        event: Event,
        *,
        executor: Optional["ActionExecutor"] = None,
        _allow_reward_health: bool = True,
    ) -> List[Alert]:
        fired_alerts: List[Alert] = []
        action_executor = executor or self._executor
        for rule in self._rules:
            if rule.where:
                context = _event_context(event)
                if not bool(rule.where.evaluate(context, {})):
                    continue
            key = _metric_key(event)
            buffer_map = self._buffers[rule.id]
            buffer = buffer_map.get(key)
            counts_map = self._counts[rule.id]
            cooldown_map = self._cooldowns[rule.id]
            last_step = cooldown_map.get(key)
            if last_step is not None and event.step < last_step:
                cooldown_map.pop(key, None)
                counts_map.pop(key, None)
                buffer = deque()
                buffer_map[key] = buffer
            if buffer is None:
                buffer = deque()
                buffer_map[key] = buffer
            buffer.append(event)
            if rule.window_kind == "rolling":
                while len(buffer) > rule.window_size:
                    buffer.popleft()
            else:
                while len(buffer) > rule.window_size:
                    buffer.popleft()
            counts_map[key] += 1
            if len(buffer) < rule.window_size:
                continue
            if counts_map[key] < max(rule.grace_steps, 0):
                continue
            last_step = cooldown_map.get(key)
            if last_step is not None:
                if event.step <= last_step + max(rule.cooldown_steps, 0):
                    continue
            window_snapshot = tuple(buffer)
            try:
                evaluation = rule.condition.evaluate(
                    _window_context(window_snapshot),
                    _aggregator_functions(window_snapshot),
                )
                if rule.window_kind == "consecutive":
                    series = _extract_series(evaluation, window_snapshot)
                    condition_met = bool(series) and all(bool(item) for item in series)
                else:
                    condition_met = bool(evaluation)
            except ValueError as exc:
                logger.warning("Rule '%s' evaluation failed: %s", rule.id, exc)
                continue
            if not condition_met:
                continue
            self._cooldowns[rule.id][key] = event.step
            action_configs = rule.actions or [ActionConfig(kind="warn")]
            alerts_for_rule: List[Alert] = []
            for action in action_configs:
                try:
                    alert = action_executor.execute(action, rule, event, window_snapshot)
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.exception(
                        "Action '%s' for rule '%s' failed", action.kind, rule.id
                    )
                    alert = _create_alert(
                        rule,
                        event,
                        action.kind,
                        message=f"Action {action.kind} failed: {exc}",
                        status="error",
                        details={"error": str(exc)},
                    )
                alerts_for_rule.append(alert)
                _log_alert(alert)
            if alerts_for_rule:
                self._record_stats(rule.id, event, alerts_for_rule[0])
                for alert in alerts_for_rule:
                    fired_alerts.append(alert)
                    self._alerts.append(alert)
        if _allow_reward_health and self._reward_health_monitor is not None:
            synthetic_events = self._reward_health_monitor.update(event)
            for synthetic_event in synthetic_events:
                fired_alerts.extend(
                    self.process_event(
                        synthetic_event,
                        executor=action_executor,
                        _allow_reward_health=False,
                    )
                )
        return fired_alerts

    def _record_stats(self, rule_id: str, event: Event, alert: Optional[Alert]) -> None:
        stats = self._stats[rule_id]
        stats.activations += 1
        activation_info = {
            "step": event.step,
            "time": event.time,
            "value": event.value,
            "name": event.name,
        }
        if alert and alert.message:
            activation_info["message"] = alert.message
        if alert and alert.status:
            activation_info["status"] = alert.status
        if event.run_id is not None:
            activation_info["run_id"] = event.run_id
        if stats.first_activation is None:
            stats.first_activation = activation_info
        stats.last_activation = activation_info

    def generate_report(self) -> MonitorReport:
        rules_summary: Dict[str, Dict[str, Any]] = {}
        for rule in self._rules:
            stats = self._stats.get(rule.id, RuleStats())
            summary: Dict[str, Any] = {
                "condition": rule.condition.expression,
                "window": {"size": rule.window_size, "kind": rule.window_kind},
                "grace_steps": rule.grace_steps,
                "cooldown_steps": rule.cooldown_steps,
                "activations": stats.activations,
            }
            if rule.where:
                summary["where"] = rule.where.expression
            if stats.first_activation:
                summary["first_activation"] = stats.first_activation
            if stats.last_activation:
                summary["last_activation"] = stats.last_activation
            rules_summary[rule.id] = summary
        alert_payloads = [alert.to_dict() for alert in self._alerts]
        return MonitorReport(rules=rules_summary, alerts=alert_payloads)


class AlertWriter:
    """Append alerts to JSONL and text summaries."""

    def __init__(
        self,
        jsonl_path: Optional[Path] = None,
        text_path: Optional[Path] = None,
    ) -> None:
        self._jsonl_path = Path(jsonl_path) if jsonl_path else None
        self._text_path = Path(text_path) if text_path else None
        for path in (self._jsonl_path, self._text_path):
            if path and path.parent and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, alert: Alert) -> None:
        if self._jsonl_path is not None:
            with self._jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(alert.to_dict(), sort_keys=True) + "\n")
        if self._text_path is not None:
            line = f"{alert.timestamp} {alert.summary()}"
            with self._text_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

class _RewardHealthMonitor:
    """Aggregate recent events and synthesize reward health alerts."""

    _REWARD_METRICS = {
        "reward_mean",
        "reward",
        "group_reward_mean",
        "normalized_reward_mean",
    }

    def __init__(self, window_size: int) -> None:
        self._window_size = max(int(window_size), 1)
        self._buffers: Dict[str, "OrderedDict[int, Dict[str, Any]]"] = defaultdict(OrderedDict)
        self._last_step_evaluated: Dict[str, int] = defaultdict(lambda: -1)

    def update(self, event: Event) -> List[Event]:
        key = event.run_id or "__default__"
        buffer = self._buffers[key]
        row = buffer.get(event.step)
        if row is None:
            row = {"step": event.step}
            buffer[event.step] = row
        row[event.name] = event.value
        row.setdefault("time", event.time)

        # Re-establish ordering by step to handle out-of-order updates gracefully.
        self._buffers[key] = OrderedDict(sorted(buffer.items()))
        buffer = self._buffers[key]

        while len(buffer) > self._window_size:
            buffer.popitem(last=False)

        if event.name not in self._REWARD_METRICS:
            return []

        if event.step <= self._last_step_evaluated[key]:
            return []

        if len(buffer) < min(self._window_size, 8):
            return []

        df = pd.DataFrame(buffer.values()).copy()
        if df.empty or "step" not in df.columns:
            return []

        df = df.sort_values("step").dropna(subset=["step"])
        if df.empty:
            return []

        df["step"] = df["step"].astype(int)

        reward_col = "reward_mean"
        if reward_col not in df.columns:
            fallback = next(
                (
                    col
                    for col in ("group_reward_mean", "normalized_reward_mean", "reward")
                    if col in df.columns
                ),
                None,
            )
            if fallback:
                df = df.rename(columns={fallback: reward_col})

        if reward_col not in df.columns:
            return []

        df = df.dropna(subset=[reward_col])
        if len(df) < min(self._window_size, 8):
            return []

        self._last_step_evaluated[key] = event.step

        try:
            from rldk.reward import health as reward_health_function
        except ImportError as exc:  # pragma: no cover - defensive guard
            logger.debug("Reward health analysis unavailable: %s", exc)
            return []

        try:
            report = reward_health_function(df, reward_col=reward_col, step_col="step")
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Reward health analysis failed: %s", exc)
            return []

        base_meta = {
            "analysis_window": len(df),
            "latest_step": int(df["step"].iloc[-1]),
            "reward_column": reward_col,
        }

        synthetic_events: List[Event] = []

        if report.drift_detected:
            meta = dict(base_meta)
            if not report.drift_metrics.empty:
                meta["drift_metrics"] = _sanitize_for_meta(report.drift_metrics)
            synthetic_events.append(
                self._build_event(event, "reward_health.drift_flag", 1.0, meta)
            )

        if report.saturation_issues:
            meta = dict(base_meta)
            meta["issues"] = list(report.saturation_issues)
            if report.saturation_analysis:
                meta["analysis"] = _sanitize_for_meta(report.saturation_analysis)
            synthetic_events.append(
                self._build_event(
                    event,
                    "reward_health.saturation_flag",
                    float(len(report.saturation_issues)),
                    meta,
                )
            )

        if report.shortcut_signals:
            meta = dict(base_meta)
            meta["signals"] = list(report.shortcut_signals)
            if report.shortcut_analysis:
                meta["analysis"] = _sanitize_for_meta(report.shortcut_analysis)
            synthetic_events.append(
                self._build_event(
                    event,
                    "reward_health.shortcut_flag",
                    float(len(report.shortcut_signals)),
                    meta,
                )
            )

        if report.label_leakage_risk >= 0.3:
            meta = dict(base_meta)
            meta["risk"] = float(report.label_leakage_risk)
            synthetic_events.append(
                self._build_event(
                    event,
                    "reward_health.label_leakage_risk",
                    float(report.label_leakage_risk),
                    meta,
                )
            )

        overopt = getattr(report, "overoptimization", None)
        if overopt and overopt.flagged:
            meta = dict(base_meta)
            meta["details"] = _sanitize_for_meta(overopt.to_dict())
            synthetic_events.append(
                self._build_event(
                    event,
                    "reward_health.overoptimization_flag",
                    float(overopt.delta),
                    meta,
                )
            )

        return synthetic_events

    @staticmethod
    def _build_event(source: Event, name: str, value: float, meta: Mapping[str, Any]) -> Event:
        return Event(
            time=source.time,
            step=source.step,
            name=name,
            value=float(value),
            run_id=source.run_id,
            meta={"reward_health": _sanitize_for_meta(dict(meta))},
        )


class _AttrDict(dict):
    """Dictionary with attribute-style access used for rule evaluation."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc


def _wrap_attr(value: Any) -> Any:
    if isinstance(value, dict):
        return _AttrDict({key: _wrap_attr(val) for key, val in value.items()})
    if isinstance(value, list):
        return [_wrap_attr(item) for item in value]
    return value


def _event_context(event: Event) -> Dict[str, Any]:
    return {
        "time": event.time,
        "step": event.step,
        "name": event.name,
        "value": event.value,
        "run_id": event.run_id,
        "tags": _wrap_attr(event.tags),
        "meta": _wrap_attr(event.meta),
    }


def _window_context(window: Sequence[Event]) -> Dict[str, Any]:
    latest = window[-1]
    context: Dict[str, Any] = {
        "value": WindowAccessor("value", window),
        "step": WindowAccessor("step", window),
        "time": WindowAccessor("time", window),
        "run_id": WindowAccessor("run_id", window),
        "name": latest.name,
        "tags": _wrap_attr(latest.tags),
        "meta": _wrap_attr(latest.meta),
    }
    return context


def _extract_series(expression: Any, window: Sequence[Event]) -> List[Any]:
    if isinstance(expression, WindowExpression):
        return list(expression.resolve_series(len(window)))
    if isinstance(expression, (list, tuple)):
        return list(expression)
    return [expression]


def _aggregator_functions(window: Sequence[Event]) -> Dict[str, Callable[..., Any]]:
    def mean_fn(arg: Any) -> float:
        data = _extract_series(arg, window)
        if not data:
            return 0.0
        return float(sum(data) / len(data))

    def max_fn(arg: Any) -> Any:
        data = _extract_series(arg, window)
        if not data:
            return float("-inf")
        return max(data)

    def min_fn(arg: Any) -> Any:
        data = _extract_series(arg, window)
        if not data:
            return float("inf")
        return min(data)

    def sum_fn(arg: Any) -> Any:
        data = _extract_series(arg, window)
        return sum(data)

    def any_fn(arg: Any) -> bool:
        data = _extract_series(arg, window)
        return any(bool(item) for item in data)

    def all_fn(arg: Any) -> bool:
        data = _extract_series(arg, window)
        return all(bool(item) for item in data)

    def count_fn(arg: Any = None) -> int:
        if arg is None:
            return len(window)
        data = _extract_series(arg, window)
        return sum(1 for item in data if bool(item))

    return {
        "mean": mean_fn,
        "max": max_fn,
        "min": min_fn,
        "sum": sum_fn,
        "any": any_fn,
        "all": all_fn,
        "count": count_fn,
    }


def _template_context(
    rule: RuleDefinition, event: Event, extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "name": event.name,
        "value": event.value,
        "step": event.step,
        "time": event.time,
        "rule_id": rule.id,
        "run_id": event.run_id,
        "tags": event.tags,
        "meta": event.meta,
        "window_size": rule.window_size,
        "window_kind": rule.window_kind,
        "action": extra.get("action") if extra else None,
    }
    if extra:
        for key, value in extra.items():
            context[key] = value
    return context


def _format_action_value(value: Any, context: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        try:
            return value.format(**context)
        except Exception as exc:  # pragma: no cover - formatting guard
            logger.warning("Failed to format template '%s': %s", value, exc)
            return value
    if isinstance(value, dict):
        return {key: _format_action_value(val, context) for key, val in value.items()}
    if isinstance(value, list):
        return [_format_action_value(item, context) for item in value]
    if isinstance(value, tuple):
        return tuple(_format_action_value(item, context) for item in value)
    return value


def _render_action_message(
    action: ActionConfig,
    rule: RuleDefinition,
    event: Event,
    extra_context: Optional[Dict[str, Any]] = None,
) -> str:
    default_template = DEFAULT_ACTION_MESSAGES.get(action.kind, DEFAULT_ACTION_MESSAGES["warn"])
    template = action.message_template or default_template
    merged_context = dict(extra_context or {})
    merged_context.setdefault("action", action.kind)
    context = _template_context(rule, event, merged_context)
    try:
        return template.format(**context)
    except Exception as exc:  # pragma: no cover - formatting guard
        logger.warning("Failed to format message for rule '%s': %s", rule.id, exc)
        return template


def _create_alert(
    rule: RuleDefinition,
    event: Event,
    action_kind: str,
    message: Optional[str],
    status: str,
    details: Optional[Dict[str, Any]] = None,
) -> Alert:
    return Alert(
        rule_id=rule.id,
        action=action_kind,
        event=event,
        window_size=rule.window_size,
        window_kind=rule.window_kind,
        message=message,
        status=status,
        details=dict(details or {}),
    )


def _log_alert(alert: Alert) -> None:
    message = alert.message or (
        f"{alert.event.name} {alert.event.value:.4f} at step {alert.event.step}"
    )
    if alert.status == "error":
        logger.error("[%s] %s", alert.rule_id, message)
    elif alert.action == "warn":
        logger.warning("[%s] %s", alert.rule_id, message)
    else:
        logger.info("[%s] %s", alert.rule_id, message)


def _tail(text: Any, limit: int = 4096) -> str:
    value = "" if text is None else str(text)
    if len(value) <= limit:
        return value
    return value[-limit:]


class ActionExecutor:
    """Execute an action defined in a rule."""

    def execute(
        self,
        action: ActionConfig,
        rule: RuleDefinition,
        event: Event,
        window: Sequence[Event],
    ) -> Alert:
        raise NotImplementedError


class SimpleActionExecutor(ActionExecutor):
    """Executor that only supports warn actions."""

    def execute(
        self,
        action: ActionConfig,
        rule: RuleDefinition,
        event: Event,
        window: Sequence[Event],
    ) -> Alert:
        if action.kind != "warn":
            raise ValueError(f"Unsupported action '{action.kind}' for simple executor")
        message = _render_action_message(action, rule, event, {})
        return _create_alert(rule, event, "warn", message=message, status="success", details={})


class ActionDispatcher(ActionExecutor):
    """Dispatch actions with side effects such as stopping processes or HTTP calls."""

    def __init__(
        self,
        *,
        pid: Optional[int] = None,
        kill_timeout_sec: float = 5.0,
        http_timeout_sec: float = 5.0,
        retries: int = 0,
    ) -> None:
        self._pid = pid
        self._kill_timeout_sec = max(float(kill_timeout_sec), 0.0)
        self._http_timeout_sec = max(float(http_timeout_sec), 0.0)
        self._retries = max(int(retries), 0)
        self._warn_executor = SimpleActionExecutor()

    def execute(
        self,
        action: ActionConfig,
        rule: RuleDefinition,
        event: Event,
        window: Sequence[Event],
    ) -> Alert:
        handler = getattr(self, f"_handle_{action.kind}", None)
        if handler is not None:
            return handler(action, rule, event, window)
        if action.kind == "warn":
            return self._warn_executor.execute(action, rule, event, window)
        raise ValueError(f"Unsupported action '{action.kind}'")

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_warn(
        self,
        action: ActionConfig,
        rule: RuleDefinition,
        event: Event,
        window: Sequence[Event],
    ) -> Alert:
        return self._warn_executor.execute(action, rule, event, window)

    def _handle_stop(
        self,
        action: ActionConfig,
        rule: RuleDefinition,
        event: Event,
        window: Sequence[Event],
    ) -> Alert:
        pid_value = action.options.get("pid", self._pid)
        details: Dict[str, Any] = {}
        if pid_value is None:
            message = _render_action_message(action, rule, event, {"pid": None})
            details["error"] = "pid not provided"
            return _create_alert(rule, event, "stop", message, "error", details)
        try:
            pid = int(pid_value)
        except (TypeError, ValueError):
            message = _render_action_message(action, rule, event, {"pid": pid_value})
            details["error"] = "invalid pid"
            return _create_alert(rule, event, "stop", message, "error", details)
        timeout_option = action.options.get("timeout") or action.options.get("kill_timeout_sec")
        try:
            kill_timeout = (
                float(timeout_option)
                if timeout_option is not None
                else self._kill_timeout_sec
            )
        except (TypeError, ValueError):
            kill_timeout = self._kill_timeout_sec
        details["pid"] = pid
        details["timeout"] = kill_timeout
        signals: List[str] = []
        terminated = False
        try:
            os.kill(pid, signal.SIGTERM)
            signals.append("SIGTERM")
        except ProcessLookupError:
            terminated = True
            signals.append("SIGTERM")
        except PermissionError as exc:
            details["error"] = str(exc)
            message = _render_action_message(
                action, rule, event, {"pid": pid, "error": str(exc)}
            )
            return _create_alert(rule, event, "stop", message, "error", details)
        if not terminated:
            terminated = self._wait_for_exit(pid, kill_timeout)
        if not terminated:
            try:
                os.kill(pid, signal.SIGKILL)
                signals.append("SIGKILL")
            except ProcessLookupError:
                terminated = True
            except PermissionError as exc:
                details["error"] = str(exc)
                message = _render_action_message(
                    action, rule, event, {"pid": pid, "error": str(exc)}
                )
                return _create_alert(rule, event, "stop", message, "error", details)
            else:
                terminated = self._wait_for_exit(pid, max(kill_timeout * 0.2, 0.5))
        if not terminated:
            terminated = self._wait_for_exit(pid, 0.5)
        if not terminated and self._is_zombie(pid):
            terminated = True
            details["zombie"] = True
        terminated = terminated or (not self._pid_exists(pid))
        details["signals"] = signals
        details["terminated"] = terminated
        status = "success" if terminated else "error"
        if not terminated:
            details.setdefault("error", "process did not terminate")
        message = _render_action_message(
            action, rule, event, {"pid": pid, "terminated": terminated}
        )
        return _create_alert(rule, event, "stop", message, status, details)

    def _wait_for_exit(self, pid: int, timeout: float) -> bool:
        if timeout <= 0:
            return not self._pid_exists(pid)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._pid_exists(pid):
                return True
            time.sleep(min(0.2, max(timeout / 10.0, 0.05)))
        return not self._pid_exists(pid)

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    def _is_zombie(pid: int) -> bool:
        status_path = Path("/proc") / str(pid) / "status"
        try:
            with status_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("State:"):
                        return "Z" in line
        except FileNotFoundError:
            return False
        except OSError:  # pragma: no cover - defensive
            return False
        return False

    def _handle_sentinel(
        self,
        action: ActionConfig,
        rule: RuleDefinition,
        event: Event,
        window: Sequence[Event],
    ) -> Alert:
        context = _template_context(rule, event, {"action": action.kind})
        raw_path = action.options.get("path")
        path_value = _format_action_value(raw_path, context)
        path = Path(str(path_value))
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        contents = action.options.get("contents")
        if contents is not None:
            rendered_contents = _format_action_value(contents, context)
            payload = _tail(rendered_contents, limit=8192)
        else:
            payload = _render_action_message(action, rule, event, {"path": str(path)})
        append = bool(action.options.get("append", False))
        try:
            if append:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(str(payload) + "\n")
            else:
                path.write_text(str(payload), encoding="utf-8")
        except OSError as exc:
            details = {"path": str(path), "error": str(exc)}
            message = _render_action_message(
                action, rule, event, {"path": str(path), "error": str(exc)}
            )
            return _create_alert(rule, event, "sentinel", message, "error", details)
        details = {"path": str(path), "append": append}
        message = _render_action_message(action, rule, event, {"path": str(path)})
        return _create_alert(rule, event, "sentinel", message, "success", details)

    def _handle_shell(
        self,
        action: ActionConfig,
        rule: RuleDefinition,
        event: Event,
        window: Sequence[Event],
    ) -> Alert:
        context = _template_context(rule, event, {"action": action.kind})
        raw_cmd = action.options.get("cmd")
        formatted_cmd = _format_action_value(raw_cmd, context)
        if isinstance(formatted_cmd, str):
            command = formatted_cmd
            display_cmd = formatted_cmd
            shell_flag = True
        elif isinstance(formatted_cmd, (list, tuple)):
            command = [str(item) for item in formatted_cmd]
            display_cmd = " ".join(command)
            shell_flag = False
        else:
            message = _render_action_message(action, rule, event, {"cmd": str(formatted_cmd)})
            details = {"cmd": str(formatted_cmd), "error": "invalid command"}
            return _create_alert(rule, event, "shell", message, "error", details)
        timeout_option = (
            action.options.get("timeout")
            or action.options.get("timeout_sec")
            or self._http_timeout_sec
        )
        try:
            timeout = float(timeout_option) if timeout_option is not None else None
        except (TypeError, ValueError):
            timeout = self._http_timeout_sec
        retries = int(action.options.get("retries", self._retries))
        attempt = 0
        details: Dict[str, Any] = {
            "cmd": display_cmd,
            "shell": shell_flag,
            "timeout": timeout,
        }
        while attempt <= retries:
            details.pop("error", None)
            try:
                completed = subprocess.run(
                    command,
                    shell=shell_flag,
                    timeout=timeout if timeout and timeout > 0 else None,
                    capture_output=True,
                    text=True,
                )
                details.update(
                    returncode=completed.returncode,
                    stdout=_tail(completed.stdout),
                    stderr=_tail(completed.stderr),
                )
                status = "success" if completed.returncode == 0 else "error"
                message = _render_action_message(
                    action,
                    rule,
                    event,
                    {"cmd": display_cmd, "returncode": completed.returncode},
                )
                if status == "success" or attempt == retries:
                    return _create_alert(rule, event, "shell", message, status, details)
            except Exception as exc:
                details["error"] = str(exc)
                if attempt == retries:
                    message = _render_action_message(
                        action,
                        rule,
                        event,
                        {"cmd": display_cmd, "error": str(exc)},
                    )
                    return _create_alert(rule, event, "shell", message, "error", details)
            attempt += 1
            time.sleep(min(0.5 * (2 ** attempt), 2.0))
        message = _render_action_message(
            action, rule, event, {"cmd": display_cmd, "returncode": details.get("returncode")}
        )
        return _create_alert(rule, event, "shell", message, "error", details)

    def _handle_http(
        self,
        action: ActionConfig,
        rule: RuleDefinition,
        event: Event,
        window: Sequence[Event],
    ) -> Alert:
        base_context = _template_context(rule, event, {"action": action.kind})
        raw_url = action.options.get("url")
        url_value = str(_format_action_value(raw_url, base_context))
        method = str(action.options.get("method", "POST"))
        method_upper = method.upper()
        payload = action.options.get("payload")
        formatted_payload = (
            _format_action_value(payload, base_context) if payload is not None else None
        )
        headers = action.options.get("headers") or {}
        formatted_headers = _format_action_value(headers, base_context)
        if not isinstance(formatted_headers, dict):
            raise ValueError("HTTP action headers must be a mapping")
        timeout_option = (
            action.options.get("timeout")
            or action.options.get("timeout_sec")
            or self._http_timeout_sec
        )
        try:
            timeout = float(timeout_option) if timeout_option is not None else None
        except (TypeError, ValueError):
            timeout = self._http_timeout_sec
        retries = int(action.options.get("retries", self._retries))
        attempt = 0
        details: Dict[str, Any] = {"url": url_value, "method": method_upper}
        while attempt <= retries:
            details.pop("error", None)
            data_bytes: Optional[bytes] = None
            headers_copy = {str(k): str(v) for k, v in formatted_headers.items()}
            if formatted_payload is not None:
                if isinstance(formatted_payload, (bytes, bytearray)):
                    data_bytes = bytes(formatted_payload)
                elif isinstance(formatted_payload, str):
                    data_bytes = formatted_payload.encode("utf-8")
                else:
                    data_bytes = json.dumps(formatted_payload).encode("utf-8")
                    headers_copy.setdefault("Content-Type", "application/json")
            request = urllib.request.Request(url_value, data=data_bytes, method=method_upper)
            for key, value in headers_copy.items():
                request.add_header(key, value)
            try:
                with urllib.request.urlopen(
                    request, timeout=timeout if timeout and timeout > 0 else None
                ) as response:
                    status_code = getattr(response, "status", response.getcode())
                    body = response.read()
                body_text = _tail(body.decode("utf-8", errors="replace"))
                details.update(status_code=status_code, response=body_text)
                status = "success" if 200 <= status_code < 400 else "error"
                message = _render_action_message(
                    action,
                    rule,
                    event,
                    {"url": url_value, "method": method_upper, "status_code": status_code},
                )
                if status == "success" or attempt == retries:
                    return _create_alert(rule, event, "http", message, status, details)
            except Exception as exc:
                details["error"] = str(exc)
                status_code = getattr(exc, "code", None)
                if status_code is not None:
                    details["status_code"] = status_code
                if attempt == retries:
                    message = _render_action_message(
                        action,
                        rule,
                        event,
                        {
                            "url": url_value,
                            "method": method_upper,
                            "status_code": details.get("status_code", "error"),
                            "error": str(exc),
                        },
                    )
                    return _create_alert(rule, event, "http", message, "error", details)
            attempt += 1
            time.sleep(min(0.5 * (2 ** attempt), 2.0))
        message = _render_action_message(
            action,
            rule,
            event,
            {
                "url": url_value,
                "method": method_upper,
                "status_code": details.get("status_code", "error"),
            },
        )
        return _create_alert(rule, event, "http", message, "error", details)


def load_rules(source: str | os.PathLike[str] | Mapping[str, Any]) -> List[RuleDefinition]:
    if isinstance(source, Mapping):
        data = source
    else:
        source_str = str(source)
        preset = get_rule_preset(source_str)
        if preset is not None:
            data = preset
        else:
            path_obj = Path(source_str)
            if not path_obj.exists():
                available = ", ".join(sorted(RULE_PRESETS))
                raise FileNotFoundError(
                    f"Rules source '{source_str}' does not exist. "
                    f"Available presets: {available}"
                )
            data = yaml.safe_load(path_obj.read_text())
    if not data or "rules" not in data:
        raise ValueError("Rules file must contain a 'rules' list")
    rules: List[RuleDefinition] = []
    for index, rule_data in enumerate(data.get("rules", [])):
        if not isinstance(rule_data, dict):
            raise ValueError(f"Rule at index {index} must be a mapping")
        rule_id = rule_data.get("id")
        if not rule_id:
            raise ValueError(f"Rule at index {index} is missing an 'id'")
        condition_expr = rule_data.get("condition")
        if not condition_expr:
            raise ValueError(f"Rule '{rule_id}' missing required 'condition'")
        window_cfg = rule_data.get("window", {}) or {}
        if not isinstance(window_cfg, dict):
            raise ValueError(f"Rule '{rule_id}' has invalid window configuration")
        window_size = int(window_cfg.get("size", 1))
        if window_size <= 0:
            raise ValueError(f"Rule '{rule_id}' window size must be positive")
        window_kind_raw = window_cfg.get("kind", DEFAULT_WINDOW_KIND) or DEFAULT_WINDOW_KIND
        window_kind = str(window_kind_raw).lower()
        if window_kind not in {DEFAULT_WINDOW_KIND, "rolling"}:
            raise ValueError(f"Rule '{rule_id}' uses unsupported window kind '{window_kind_raw}'")
        grace_steps = int(rule_data.get("grace_steps", 0))
        if grace_steps < 0:
            raise ValueError(f"Rule '{rule_id}' grace_steps must be non-negative")
        cooldown_steps = int(rule_data.get("cooldown_steps", 0))
        if cooldown_steps < 0:
            raise ValueError(f"Rule '{rule_id}' cooldown_steps must be non-negative")
        where_expr = rule_data.get("where")
        where = SafeExpression(where_expr, []) if where_expr else None
        condition = SafeExpression(condition_expr, AGGREGATOR_FUNCTIONS)
        actions_data = rule_data.get("actions", []) or []
        actions: List[ActionConfig] = []
        for action_entry in actions_data:
            if isinstance(action_entry, str):
                action_name = action_entry
                params = {}
            elif isinstance(action_entry, dict) and len(action_entry) == 1:
                action_name, params = next(iter(action_entry.items()))
                params = params or {}
            else:
                raise ValueError(f"Rule '{rule_id}' has invalid action definition: {action_entry}")
            if not isinstance(params, dict):
                raise ValueError(
                    f"Rule '{rule_id}' action '{action_name}' must use a mapping for configuration"
                )
            action_kind = str(action_name).lower()
            options = dict(params)
            message_template = options.pop("msg", None) or options.pop("message", None)
            if message_template is not None and not isinstance(message_template, str):
                raise ValueError(
                    f"Rule '{rule_id}' action '{action_name}' message template must be a string"
                )
            if action_kind == "warn":
                actions.append(ActionConfig(kind=action_kind, message_template=message_template, options=options))
            elif action_kind == "stop":
                actions.append(ActionConfig(kind=action_kind, message_template=message_template, options=options))
            elif action_kind == "sentinel":
                if "path" not in options:
                    raise ValueError(f"Rule '{rule_id}' sentinel action requires a 'path'")
                if not isinstance(options["path"], str):
                    raise ValueError(f"Rule '{rule_id}' sentinel path must be a string")
                actions.append(ActionConfig(kind=action_kind, message_template=message_template, options=options))
            elif action_kind == "shell":
                if "cmd" not in options:
                    raise ValueError(f"Rule '{rule_id}' shell action requires a 'cmd'")
                actions.append(ActionConfig(kind=action_kind, message_template=message_template, options=options))
            elif action_kind == "http":
                if "url" not in options:
                    raise ValueError(f"Rule '{rule_id}' http action requires a 'url'")
                headers = options.get("headers")
                if headers is not None and not isinstance(headers, dict):
                    raise ValueError(f"Rule '{rule_id}' http action headers must be a mapping")
                actions.append(ActionConfig(kind=action_kind, message_template=message_template, options=options))
            else:
                raise ValueError(f"Rule '{rule_id}' references unsupported action '{action_name}'")
        rules.append(
            RuleDefinition(
                id=rule_id,
                condition=condition,
                window_size=window_size,
                window_kind=window_kind,
                where=where,
                grace_steps=grace_steps,
                cooldown_steps=cooldown_steps,
                actions=actions,
            )
        )
    return rules


__all__ = [
    "ActionConfig",
    "ActionDispatcher",
    "ActionExecutor",
    "Alert",
    "AlertWriter",
    "Event",
    "MonitorEngine",
    "MonitorReport",
    "RuleDefinition",
    "canonicalize_event",
    "load_rules",
    "read_events_once",
    "read_stream",
]
