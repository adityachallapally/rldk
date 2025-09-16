"""Core monitoring engine for streaming JSONL training events."""
from __future__ import annotations

import ast
import json
import logging
import os
import signal
import subprocess
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import yaml

logger = logging.getLogger(__name__)

CANONICAL_FIELDS = {"time", "step", "name", "value", "run_id", "tags", "meta"}
DEFAULT_WINDOW_KIND = "consecutive"
SUPPORTED_WINDOW_KINDS = {"consecutive", "rolling"}
AGGREGATOR_FUNCTIONS = {"mean", "max", "min", "any", "all", "sum", "count"}


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
        _SafeExpressionValidator(list(self._allowed_functions)).visit(tree)
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
class WarnAction:
    """Warn action definition."""

    message_template: Optional[str] = None


@dataclass
class StopAction:
    """Stop action definition."""

    pid: Optional[int] = None
    kill_timeout_sec: int = 5


@dataclass
class SentinelAction:
    """Sentinel file action definition."""

    path: str = "stop.sentinel"


@dataclass
class ShellAction:
    """Shell command action definition."""

    command: str
    timeout_sec: int = 30
    retries: int = 0


@dataclass
class HttpAction:
    """HTTP request action definition."""

    url: str
    method: str = "POST"
    payload: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, Any]] = None
    timeout_sec: int = 10
    retries: int = 0


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
    actions: List[Union[WarnAction, StopAction, SentinelAction, ShellAction, HttpAction]] = field(default_factory=list)


@dataclass
class Alert:
    """Alert emitted by the monitoring engine."""

    rule_id: str
    action: str
    event: Event
    window_size: int
    window_kind: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "rule_id": self.rule_id,
            "action": self.action,
            "step": self.event.step,
            "time": self.event.time,
            "name": self.event.name,
            "value": self.event.value,
            "window": {"size": self.window_size, "kind": self.window_kind},
        }
        if self.message:
            payload["message"] = self.message
        if self.event.run_id is not None:
            payload["run_id"] = self.event.run_id
        if self.event.tags:
            payload["tags"] = self.event.tags
        if self.event.meta:
            payload["meta"] = self.event.meta
        return payload


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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def _apply_field_map(event: Dict[str, Any], field_map: Optional[Dict[str, str]]) -> Dict[str, Any]:
    if not field_map:
        return dict(event)
    mapped: Dict[str, Any] = {}
    for key, value in event.items():
        canonical_key = field_map.get(key, key)
        mapped[canonical_key] = value
    return mapped


def canonicalize_event(payload: Dict[str, Any], field_map: Optional[Dict[str, str]] = None) -> Event:
    mapped = _apply_field_map(payload, field_map)
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


def read_events_once(path: str | os.PathLike[str], field_map: Optional[Dict[str, str]] = None) -> List[Event]:
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
    events: List[Event] = []
    for line in opener(path_obj):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse JSON event: %s", exc)
            continue
        try:
            events.append(canonicalize_event(payload, field_map))
        except ValueError as exc:
            logger.warning("Skipping invalid event: %s", exc)
    return events


def read_stream(
    path_or_stdin: str | os.PathLike[str],
    field_map: Optional[Dict[str, str]] = None,
    poll_interval: float = 0.5,
) -> Iterator[Event]:
    """Yield canonical events from a stream, following file rotations."""

    if str(path_or_stdin) == "-":
        import sys

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                yield canonicalize_event(payload, field_map)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse JSON event from stdin: %s", exc)
            except ValueError as exc:
                logger.warning("Skipping invalid stdin event: %s", exc)
        return

    path = Path(path_or_stdin)
    buffer = ""
    handle = None
    current_inode: Optional[int] = None

    def _open_file() -> Optional[Any]:
        nonlocal handle, current_inode
        try:
            handle = path.open("r", encoding="utf-8", errors="replace")
            current_inode = os.fstat(handle.fileno()).st_ino
            return handle
        except FileNotFoundError:
            return None

    try:
        while True:
            if handle is None:
                handle = _open_file()
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
                    try:
                        payload = json.loads(line)
                        yield canonicalize_event(payload, field_map)
                    except json.JSONDecodeError as exc:
                        logger.warning("Failed to parse JSON event: %s", exc)
                    except ValueError as exc:
                        logger.warning("Skipping invalid event: %s", exc)
            else:
                try:
                    stat = path.stat()
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

    def __init__(self, rules: Sequence[RuleDefinition], pid: Optional[int] = None):
        if not rules:
            raise ValueError("MonitorEngine requires at least one rule")
        self._rules = list(rules)
        self._buffers: Dict[str, Dict[MetricKey, Deque[Event]]] = defaultdict(dict)
        self._counts: Dict[str, Dict[MetricKey, int]] = defaultdict(lambda: defaultdict(int))
        self._cooldowns: Dict[str, Dict[MetricKey, int]] = defaultdict(dict)
        self._stats: Dict[str, RuleStats] = defaultdict(RuleStats)
        self._alerts: List[Alert] = []
        self._pid = pid

    def process_event(self, event: Event) -> List[Alert]:
        fired_alerts: List[Alert] = []
        for rule in self._rules:
            if rule.where:
                context = _event_context(event)
                if not bool(rule.where.evaluate(context, {})):
                    continue
            key = _metric_key(event)
            buffer = self._buffers[rule.id].get(key)
            if buffer is None:
                if rule.window_kind == "rolling":
                    buffer = deque(maxlen=rule.window_size)
                else:  # consecutive
                    buffer = deque(maxlen=rule.window_size)
                self._buffers[rule.id][key] = buffer
            buffer.append(event)
            self._counts[rule.id][key] += 1
            
            if rule.window_kind == "rolling":
                should_evaluate = len(buffer) >= min(rule.window_size, max(rule.grace_steps, 0) + 1)
            else:
                should_evaluate = len(buffer) >= rule.window_size
            
            if not should_evaluate:
                continue
            if self._counts[rule.id][key] < max(rule.grace_steps, 0):
                continue
            last_step = self._cooldowns[rule.id].get(key)
            if last_step is not None and event.step <= last_step + max(rule.cooldown_steps, 0):
                continue
            window_context = _window_context(buffer)
            try:
                condition_met = bool(
                    rule.condition.evaluate(window_context, _aggregator_functions(buffer))
                )
            except ValueError as exc:
                logger.warning("Rule '%s' evaluation failed: %s", rule.id, exc)
                continue
            if not condition_met:
                continue
            self._cooldowns[rule.id][key] = event.step
            actions = rule.actions or [WarnAction()]
            
            for action in actions:
                alert = None
                if isinstance(action, WarnAction):
                    message = _render_warn_message(action, rule, event)
                    alert = Alert(
                        rule_id=rule.id,
                        action="warn",
                        event=event,
                        window_size=rule.window_size,
                        window_kind=rule.window_kind,
                        message=message,
                    )
                    _log_warn(alert)
                elif isinstance(action, StopAction):
                    result = _execute_stop_action(action, rule, event, self._pid)
                    alert = Alert(
                        rule_id=rule.id,
                        action="stop",
                        event=event,
                        window_size=rule.window_size,
                        window_kind=rule.window_kind,
                        message=f"Stop action: {result}",
                    )
                elif isinstance(action, SentinelAction):
                    result = _execute_sentinel_action(action, rule, event)
                    alert = Alert(
                        rule_id=rule.id,
                        action="sentinel",
                        event=event,
                        window_size=rule.window_size,
                        window_kind=rule.window_kind,
                        message=f"Sentinel action: {result}",
                    )
                elif isinstance(action, ShellAction):
                    result = _execute_shell_action(action, rule, event)
                    alert = Alert(
                        rule_id=rule.id,
                        action="shell",
                        event=event,
                        window_size=rule.window_size,
                        window_kind=rule.window_kind,
                        message=f"Shell action: {result}",
                    )
                elif isinstance(action, HttpAction):
                    result = _execute_http_action(action, rule, event)
                    alert = Alert(
                        rule_id=rule.id,
                        action="http",
                        event=event,
                        window_size=rule.window_size,
                        window_kind=rule.window_kind,
                        message=f"HTTP action: {result}",
                    )
                
                if alert:
                    fired_alerts.append(alert)
                    self._alerts.append(alert)
                    self._record_stats(rule.id, event, alert)
        return fired_alerts

    def _record_stats(self, rule_id: str, event: Event, alert: Alert) -> None:
        stats = self._stats[rule_id]
        stats.activations += 1
        activation_info = {
            "step": event.step,
            "time": event.time,
            "value": event.value,
            "name": event.name,
        }
        if alert.message:
            activation_info["message"] = alert.message
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


def _render_warn_message(action: WarnAction, rule: RuleDefinition, event: Event) -> str:
    template = action.message_template or "{name} {value:.4f} at step {step} (rule {rule_id})"
    context = {
        "name": event.name,
        "value": event.value,
        "step": event.step,
        "time": event.time,
        "rule_id": rule.id,
        "run_id": event.run_id,
        "tags": event.tags,
        "meta": event.meta,
    }
    try:
        return template.format(**context)
    except Exception as exc:  # pragma: no cover - formatting guard
        logger.warning("Failed to format warn message for rule '%s': %s", rule.id, exc)
        return template


def _execute_stop_action(action: StopAction, rule: RuleDefinition, event: Event, pid: Optional[int] = None) -> Dict[str, Any]:
    """Execute stop action with SIGTERM -> SIGKILL fallback."""
    target_pid = action.pid or pid
    if not target_pid:
        return {"success": False, "error": "No PID specified for stop action"}
    
    try:
        os.kill(target_pid, signal.SIGTERM)
        
        for _ in range(action.kill_timeout_sec):
            try:
                os.kill(target_pid, 0)  # Check if process exists
                time.sleep(1)
            except ProcessLookupError:
                return {"success": True, "signal": "SIGTERM"}
        
        os.kill(target_pid, signal.SIGKILL)
        return {"success": True, "signal": "SIGKILL"}
        
    except ProcessLookupError:
        return {"success": False, "error": "Process not found"}
    except PermissionError:
        return {"success": False, "error": "Permission denied"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _execute_sentinel_action(action: SentinelAction, rule: RuleDefinition, event: Event) -> Dict[str, Any]:
    """Execute sentinel file action."""
    try:
        Path(action.path).touch()
        return {"success": True, "path": action.path}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _execute_shell_action(action: ShellAction, rule: RuleDefinition, event: Event) -> Dict[str, Any]:
    """Execute shell command action with templating."""
    template_context = {
        "name": event.name, "value": event.value, "step": event.step,
        "time": event.time, "rule_id": rule.id, "run_id": event.run_id,
        "tags": event.tags, "meta": event.meta
    }
    
    try:
        templated_command = action.command.format(**template_context)
    except Exception as e:
        return {"success": False, "error": f"Template error: {e}"}
    
    for attempt in range(action.retries + 1):
        try:
            result = subprocess.run(
                templated_command, shell=True, capture_output=True, 
                text=True, timeout=action.timeout_sec
            )
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "attempt": attempt + 1
            }
        except subprocess.TimeoutExpired:
            if attempt == action.retries:
                return {"success": False, "error": "Command timed out"}
        except Exception as e:
            if attempt == action.retries:
                return {"success": False, "error": str(e)}
    
    return {"success": False, "error": "All retries failed"}


def _execute_http_action(action: HttpAction, rule: RuleDefinition, event: Event) -> Dict[str, Any]:
    """Execute HTTP request action with templating."""
    try:
        import requests
    except ImportError:
        return {"success": False, "error": "requests library not available"}
    
    template_context = {
        "name": event.name, "value": event.value, "step": event.step,
        "time": event.time, "rule_id": rule.id, "run_id": event.run_id,
        "tags": event.tags, "meta": event.meta
    }
    
    try:
        if action.payload:
            payload = json.loads(json.dumps(action.payload).format(**template_context))
        else:
            payload = template_context
    except Exception as e:
        return {"success": False, "error": f"Payload template error: {e}"}
    
    for attempt in range(action.retries + 1):
        try:
            response = requests.request(
                method=action.method,
                url=action.url,
                json=payload,
                headers=action.headers,
                timeout=action.timeout_sec
            )
            return {
                "success": response.status_code < 400,
                "status_code": response.status_code,
                "response": response.text[:500],  # Truncate response
                "attempt": attempt + 1
            }
        except requests.RequestException as e:
            if attempt == action.retries:
                return {"success": False, "error": str(e)}
    
    return {"success": False, "error": "All retries failed"}


def _log_warn(alert: Alert) -> None:
    if alert.message:
        logger.warning("[%s] %s", alert.rule_id, alert.message)
    else:
        logger.warning(
            "[%s] %s %.4f at step %s",
            alert.rule_id,
            alert.event.name,
            alert.event.value,
            alert.event.step,
        )


def load_rules(path: str | os.PathLike[str]) -> List[RuleDefinition]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Rules file '{path}' does not exist")
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
        window_kind = window_cfg.get("kind", DEFAULT_WINDOW_KIND) or DEFAULT_WINDOW_KIND
        if window_kind not in SUPPORTED_WINDOW_KINDS:
            raise ValueError(f"Rule '{rule_id}' uses unsupported window kind '{window_kind}'. Supported: {SUPPORTED_WINDOW_KINDS}")
        grace_steps = int(rule_data.get("grace_steps", 0))
        if grace_steps < 0:
            raise ValueError(f"Rule '{rule_id}' grace_steps must be non-negative")
        cooldown_steps = int(rule_data.get("cooldown_steps", 0))
        if cooldown_steps < 0:
            raise ValueError(f"Rule '{rule_id}' cooldown_steps must be non-negative")
        where_expr = rule_data.get("where")
        where = SafeExpression(where_expr, []) if where_expr else None
        condition = SafeExpression(condition_expr, list(AGGREGATOR_FUNCTIONS))
        actions_data = rule_data.get("actions", []) or []
        actions: List[Union[WarnAction, StopAction, SentinelAction, ShellAction, HttpAction]] = []
        
        for action_entry in actions_data:
            if isinstance(action_entry, str):
                action_name = action_entry
                params = {}
            elif isinstance(action_entry, dict) and len(action_entry) == 1:
                action_name, params = next(iter(action_entry.items()))
                params = params or {}
            else:
                raise ValueError(f"Rule '{rule_id}' has invalid action definition: {action_entry}")
            
            if action_name == "warn":
                message_template = params.get("msg") if isinstance(params, dict) else None
                actions.append(WarnAction(message_template=message_template))
            elif action_name == "stop":
                actions.append(StopAction(
                    pid=params.get("pid"),
                    kill_timeout_sec=params.get("kill_timeout_sec", 5)
                ))
            elif action_name == "sentinel":
                actions.append(SentinelAction(path=params.get("path", "stop.sentinel")))
            elif action_name == "shell":
                if "command" not in params:
                    raise ValueError(f"Rule '{rule_id}' shell action missing 'command'")
                actions.append(ShellAction(
                    command=params["command"],
                    timeout_sec=params.get("timeout_sec", 30),
                    retries=params.get("retries", 0)
                ))
            elif action_name == "http":
                if "url" not in params:
                    raise ValueError(f"Rule '{rule_id}' http action missing 'url'")
                actions.append(HttpAction(
                    url=params["url"],
                    method=params.get("method", "POST"),
                    payload=params.get("payload"),
                    headers=params.get("headers"),
                    timeout_sec=params.get("timeout_sec", 10),
                    retries=params.get("retries", 0)
                ))
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
    "Alert",
    "Event",
    "MonitorEngine",
    "MonitorReport",
    "RuleDefinition",
    "WarnAction",
    "StopAction",
    "SentinelAction",
    "ShellAction",
    "HttpAction",
    "canonicalize_event",
    "load_rules",
    "read_events_once",
    "read_stream",
]
