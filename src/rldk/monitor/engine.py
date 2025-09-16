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
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import requests
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
class WarnAction:
    """Warn action definition."""

    message_template: Optional[str] = None


@dataclass
class StopAction:
    """Stop action definition for PID-based process termination."""

    pid: Optional[int] = None
    kill_timeout_sec: int = 5


@dataclass
class SentinelAction:
    """Sentinel file action definition."""

    path: str


@dataclass
class ShellAction:
    """Shell command action definition."""

    command: str
    timeout_sec: int = 30


@dataclass
class HttpAction:
    """HTTP request action definition."""

    url: str
    method: str = "POST"
    payload: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    timeout_sec: int = 30
    retries: int = 3


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
    actions: List[Any] = field(default_factory=list)  # List of action objects


@dataclass
class Alert:
    """Alert emitted by the monitoring engine."""

    rule_id: str
    action: str
    event: Event
    window_size: int
    window_kind: str
    message: str
    action_result: Optional[Dict[str, Any]] = None

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
        if self.action_result:
            payload["action_result"] = self.action_result
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


def _execute_stop_action(action: StopAction, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute stop action by sending signals to a process."""
    pid = action.pid
    if pid is None:
        return {"success": False, "error": "No PID provided for stop action"}
    
    try:
        # Send SIGTERM first
        os.kill(pid, signal.SIGTERM)
        logger.info("Sent SIGTERM to PID %d", pid)
        
        # Wait for graceful shutdown
        time.sleep(action.kill_timeout_sec)
        
        # Check if process is still running
        try:
            os.kill(pid, 0)  # Check if process exists
            # Process still running, send SIGKILL
            os.kill(pid, signal.SIGKILL)
            logger.warning("Process %d did not terminate gracefully, sent SIGKILL", pid)
            return {"success": True, "method": "SIGKILL", "pid": pid}
        except ProcessLookupError:
            # Process terminated gracefully
            logger.info("Process %d terminated gracefully", pid)
            return {"success": True, "method": "SIGTERM", "pid": pid}
            
    except ProcessLookupError:
        return {"success": False, "error": f"Process {pid} not found"}
    except PermissionError:
        return {"success": False, "error": f"Permission denied to signal process {pid}"}
    except Exception as exc:
        return {"success": False, "error": f"Failed to stop process {pid}: {exc}"}


def _execute_sentinel_action(action: SentinelAction, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute sentinel action by creating a file."""
    try:
        path = Path(action.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"Alert triggered at {context.get('time', _now_iso())}\n")
        logger.info("Created sentinel file: %s", path)
        return {"success": True, "path": str(path)}
    except Exception as exc:
        logger.error("Failed to create sentinel file %s: %s", action.path, exc)
        return {"success": False, "error": f"Failed to create sentinel file: {exc}"}


def _execute_shell_action(action: ShellAction, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute shell action by running a command."""
    try:
        # Template the command with context
        command = action.command.format(**context)
        
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=action.timeout_sec
        )
        
        logger.info("Shell command executed: %s (exit code: %d)", command, result.returncode)
        return {
            "success": True,
            "command": command,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        logger.error("Shell command timed out: %s", action.command)
        return {"success": False, "error": f"Command timed out after {action.timeout_sec}s"}
    except Exception as exc:
        logger.error("Shell command failed: %s", exc)
        return {"success": False, "error": f"Command execution failed: {exc}"}


def _execute_http_action(action: HttpAction, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute HTTP action by making a request."""
    try:
        # Template the URL and payload with context
        url = action.url.format(**context)
        
        payload = action.payload
        if payload:
            # Template payload values
            payload = {k: str(v).format(**context) if isinstance(v, str) else v 
                     for k, v in payload.items()}
        
        headers = action.headers or {}
        
        for attempt in range(action.retries + 1):
            try:
                response = requests.request(
                    method=action.method,
                    url=url,
                    json=payload,
                    headers=headers,
                    timeout=action.timeout_sec
                )
                
                logger.info("HTTP %s request completed: %s (status: %d)", 
                           action.method, url, response.status_code)
                return {
                    "success": True,
                    "url": url,
                    "method": action.method,
                    "status_code": response.status_code,
                    "response": response.text[:1000]  # Truncate long responses
                }
            except requests.exceptions.RequestException as exc:
                if attempt == action.retries:
                    raise exc
                logger.warning("HTTP request attempt %d failed: %s", attempt + 1, exc)
                time.sleep(1)  # Brief backoff
                
    except Exception as exc:
        logger.error("HTTP request failed: %s", exc)
        return {"success": False, "error": f"HTTP request failed: {exc}"}


def _execute_action(action: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute any action type and return result."""
    if isinstance(action, StopAction):
        return _execute_stop_action(action, context)
    elif isinstance(action, SentinelAction):
        return _execute_sentinel_action(action, context)
    elif isinstance(action, ShellAction):
        return _execute_shell_action(action, context)
    elif isinstance(action, HttpAction):
        return _execute_http_action(action, context)
    elif isinstance(action, WarnAction):
        # Warn actions are handled separately in the alert generation
        return {"success": True, "action": "warn"}
    else:
        return {"success": False, "error": f"Unknown action type: {type(action)}"}


def generate_human_summary(report: MonitorReport) -> str:
    """Generate a human-readable summary of the monitoring session."""
    lines = []
    lines.append("RLDK Monitoring Summary")
    lines.append("=" * 50)
    lines.append("")
    
    # Rules summary
    lines.append("Rules:")
    for rule_id, rule_info in report.rules.items():
        lines.append(f"  {rule_id}:")
        lines.append(f"    Condition: {rule_info['condition']}")
        lines.append(f"    Window: {rule_info['window']['size']} {rule_info['window']['kind']}")
        lines.append(f"    Activations: {rule_info['activations']}")
        if rule_info.get('first_activation'):
            fa = rule_info['first_activation']
            lines.append(f"    First: step {fa['step']}, {fa['name']}={fa['value']:.4f}")
        if rule_info.get('last_activation'):
            la = rule_info['last_activation']
            lines.append(f"    Last: step {la['step']}, {la['name']}={la['value']:.4f}")
        lines.append("")
    
    # Alerts summary
    lines.append("Alerts:")
    if not report.alerts:
        lines.append("  No alerts triggered")
    else:
        for alert in report.alerts:
            lines.append(f"  [{alert['rule_id']}] {alert['action']} - {alert['name']}={alert['value']:.4f} at step {alert['step']}")
            if alert.get('message'):
                lines.append(f"    Message: {alert['message']}")
            if alert.get('action_result'):
                result = alert['action_result']
                if result.get('success'):
                    if alert['action'] == 'stop':
                        lines.append(f"    Process {result.get('pid')} terminated via {result.get('method')}")
                    elif alert['action'] == 'sentinel':
                        lines.append(f"    Sentinel file created: {result.get('path')}")
                    elif alert['action'] == 'shell':
                        lines.append(f"    Command executed (exit code: {result.get('exit_code')})")
                    elif alert['action'] == 'http':
                        lines.append(f"    HTTP {result.get('method')} to {result.get('url')} (status: {result.get('status_code')})")
                else:
                    lines.append(f"    Action failed: {result.get('error')}")
            lines.append("")
    
    return "\n".join(lines)


class MonitorEngine:
    """Evaluate rules against incoming events."""

    def __init__(self, rules: Sequence[RuleDefinition]):
        if not rules:
            raise ValueError("MonitorEngine requires at least one rule")
        self._rules = list(rules)
        self._buffers: Dict[str, Dict[MetricKey, Deque[Event]]] = defaultdict(dict)
        self._counts: Dict[str, Dict[MetricKey, int]] = defaultdict(lambda: defaultdict(int))
        self._cooldowns: Dict[str, Dict[MetricKey, int]] = defaultdict(dict)
        self._stats: Dict[str, RuleStats] = defaultdict(RuleStats)
        self._alerts: List[Alert] = []

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
            
            # For rolling windows, we can evaluate as soon as we have the window size
            # For consecutive windows, we need exactly the window size
            if rule.window_kind == "rolling":
                if len(buffer) < rule.window_size:
                    continue
            else:  # consecutive
                if len(buffer) < rule.window_size:
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
            
            # Execute actions
            actions = rule.actions or [WarnAction()]
            for action in actions:
                action_type = type(action).__name__.lower().replace("action", "")
                message = _render_action_message(action, rule, event)
                
                # Execute the action
                action_result = None
                if not isinstance(action, WarnAction):
                    context = _event_context(event)
                    context.update({
                        "rule_id": rule.id,
                        "window_size": rule.window_size,
                        "window_kind": rule.window_kind
                    })
                    action_result = _execute_action(action, context)
                
                alert = Alert(
                    rule_id=rule.id,
                    action=action_type,
                    event=event,
                    window_size=rule.window_size,
                    window_kind=rule.window_kind,
                    message=message,
                    action_result=action_result,
                )
                fired_alerts.append(alert)
                self._alerts.append(alert)
                _log_alert(alert)
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


def _render_action_message(action: Any, rule: RuleDefinition, event: Event) -> str:
    """Render message for any action type."""
    if isinstance(action, WarnAction):
        template = action.message_template or "{name} {value:.4f} at step {step} (rule {rule_id})"
    elif isinstance(action, StopAction):
        template = "Stopping process {pid} - {name} {value:.4f} at step {step}"
    elif isinstance(action, SentinelAction):
        template = "Creating sentinel file {path} - {name} {value:.4f} at step {step}"
    elif isinstance(action, ShellAction):
        template = "Executing shell command - {name} {value:.4f} at step {step}"
    elif isinstance(action, HttpAction):
        template = "Making HTTP {method} request to {url} - {name} {value:.4f} at step {step}"
    else:
        template = "{name} {value:.4f} at step {step} (rule {rule_id})"
    
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
    
    # Add action-specific context
    if isinstance(action, StopAction) and action.pid:
        context["pid"] = action.pid
    elif isinstance(action, SentinelAction):
        context["path"] = action.path
    elif isinstance(action, ShellAction):
        context["command"] = action.command
    elif isinstance(action, HttpAction):
        context["url"] = action.url
        context["method"] = action.method
    
    try:
        return template.format(**context)
    except Exception as exc:  # pragma: no cover - formatting guard
        logger.warning("Failed to format message for rule '%s': %s", rule.id, exc)
        return template


def _log_alert(alert: Alert) -> None:
    """Log alert with appropriate level based on action type."""
    if alert.action == "warn":
        level = logging.WARNING
    elif alert.action == "stop":
        level = logging.ERROR
    else:
        level = logging.INFO
    
    if alert.message:
        logger.log(level, "[%s] %s", alert.rule_id, alert.message)
    else:
        logger.log(
            level,
            "[%s] %s %.4f at step %s",
            alert.rule_id,
            alert.event.name,
            alert.event.value,
            alert.event.step,
        )
    
    # Log action result if available
    if alert.action_result and not alert.action_result.get("success", True):
        logger.error("Action failed: %s", alert.action_result.get("error", "Unknown error"))


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
        condition = SafeExpression(condition_expr, AGGREGATOR_FUNCTIONS)
        actions_data = rule_data.get("actions", []) or []
        actions: List[Any] = []
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
                pid = params.get("pid") if isinstance(params, dict) else None
                kill_timeout = int(params.get("kill_timeout_sec", 5)) if isinstance(params, dict) else 5
                actions.append(StopAction(pid=pid, kill_timeout_sec=kill_timeout))
            elif action_name == "sentinel":
                path = params.get("path") if isinstance(params, dict) else None
                if not path:
                    raise ValueError(f"Rule '{rule_id}' sentinel action missing required 'path'")
                actions.append(SentinelAction(path=path))
            elif action_name == "shell":
                command = params.get("command") if isinstance(params, dict) else None
                if not command:
                    raise ValueError(f"Rule '{rule_id}' shell action missing required 'command'")
                timeout = int(params.get("timeout_sec", 30)) if isinstance(params, dict) else 30
                actions.append(ShellAction(command=command, timeout_sec=timeout))
            elif action_name == "http":
                url = params.get("url") if isinstance(params, dict) else None
                if not url:
                    raise ValueError(f"Rule '{rule_id}' http action missing required 'url'")
                method = params.get("method", "POST") if isinstance(params, dict) else "POST"
                payload = params.get("payload") if isinstance(params, dict) else None
                headers = params.get("headers") if isinstance(params, dict) else None
                timeout = int(params.get("timeout_sec", 30)) if isinstance(params, dict) else 30
                retries = int(params.get("retries", 3)) if isinstance(params, dict) else 3
                actions.append(HttpAction(
                    url=url, method=method, payload=payload, headers=headers,
                    timeout_sec=timeout, retries=retries
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
    "generate_human_summary",
    "load_rules",
    "read_events_once",
    "read_stream",
]
