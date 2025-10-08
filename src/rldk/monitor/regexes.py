from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


_NAME_ALIASES: Sequence[str] = (
    "name",
    "metric",
    "metric_name",
    "key",
)
_VALUE_ALIASES: Sequence[str] = (
    "value",
    "v",
    "scalar",
    "metric_value",
)
_STEP_ALIASES: Sequence[str] = (
    "step",
    "steps",
    "global_step",
    "s",
    "iteration",
    "iter",
    "t",
)
_TIME_ALIASES: Sequence[str] = (
    "time",
    "timestamp",
    "wall_time",
    "walltime",
    "ts",
)
_RUN_ALIASES: Sequence[str] = (
    "run_id",
    "run",
    "runid",
)


class LineParser:
    """Parse raw log lines into canonical event payloads."""

    def __init__(self) -> None:
        self._auto_step = 0

    def parse(self, line: str) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError

    def _next_step(self) -> int:
        self._auto_step += 1
        return self._auto_step

    def _coerce_step(self, raw: Optional[str]) -> int:
        if raw is None:
            return self._next_step()
        value = raw.strip()
        if not value:
            return self._next_step()
        try:
            step = int(float(value))
        except (TypeError, ValueError):
            return self._next_step()
        if step > self._auto_step:
            self._auto_step = step
        return step

    def _coerce_time(self, raw: Optional[str]) -> str:
        if raw is None:
            return _now_iso()
        value = raw.strip()
        if not value:
            return _now_iso()
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return _now_iso()
            return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        if timestamp > 1e12:
            timestamp = timestamp / 1000.0
        try:
            parsed = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (OSError, OverflowError):
            return _now_iso()
        return parsed.isoformat().replace("+00:00", "Z")


def _first(groups: Dict[str, Any], aliases: Sequence[str]) -> Optional[str]:
    for key in aliases:
        value = groups.get(key)
        if value not in (None, ""):
            return str(value)
    return None


class RegexLineParser(LineParser):
    """Parse metrics using a user-provided regular expression."""

    def __init__(self, pattern: re.Pattern[str]) -> None:
        super().__init__()
        self._pattern = pattern

    def parse(self, line: str) -> Iterable[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        current_step: Optional[int] = None
        current_time: Optional[str] = None
        current_run: Optional[str] = None
        for match in self._pattern.finditer(line):
            groups = {key: value for key, value in match.groupdict().items() if value is not None}
            if not groups:
                continue
            name_value = _first(groups, _NAME_ALIASES)
            value_value = _first(groups, _VALUE_ALIASES)
            if name_value is None or value_value is None:
                continue
            step_token = _first(groups, _STEP_ALIASES)
            if step_token is not None:
                step_value = self._coerce_step(step_token)
                current_step = step_value
            elif current_step is not None:
                step_value = current_step
            else:
                step_value = self._next_step()
                current_step = step_value
            time_token = _first(groups, _TIME_ALIASES)
            if time_token is not None:
                time_value = self._coerce_time(time_token)
                current_time = time_value
            else:
                time_value = current_time or _now_iso()
                current_time = time_value
            run_token = _first(groups, _RUN_ALIASES)
            if run_token is not None:
                current_run = run_token.strip()
            event: Dict[str, Any] = {
                "name": name_value.strip(),
                "value": value_value.strip(),
                "step": step_value,
                "time": time_value,
            }
            if current_run:
                event["run_id"] = current_run
            events.append(event)
        return events


class TRLLineParser(LineParser):
    """Parse TRL-style key/value metrics from stdout/stderr."""

    _PAIR = re.compile(
        r"(?P<key>[A-Za-z0-9_./-]+)\s*(?:=|:)\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?=[\s,;]|$)"
    )
    _STEP_KEYS = {"step", "steps", "global_step", "iteration", "iter"}
    _TIME_KEYS = {"time", "timestamp", "wall_time", "walltime"}
    _RUN_KEYS = {"run", "run_id"}

    def parse(self, line: str) -> Iterable[Dict[str, Any]]:
        matches = list(self._PAIR.finditer(line))
        if not matches:
            return []
        metrics: List[Dict[str, Any]] = []
        step_value: Optional[int] = None
        time_value: Optional[str] = None
        run_value: Optional[str] = None
        for match in matches:
            key = match.group("key")
            raw_value = match.group("value")
            if key is None or raw_value is None:
                continue
            normalized_key = key.strip()
            lower = normalized_key.lower()
            if lower in self._STEP_KEYS:
                step_value = self._coerce_step(raw_value)
                continue
            if lower in self._TIME_KEYS:
                time_value = self._coerce_time(raw_value)
                continue
            if lower in self._RUN_KEYS:
                run_value = raw_value.strip()
                continue
            metrics.append({"name": normalized_key, "value": raw_value})
        if not metrics:
            return []
        if step_value is None:
            step_value = self._next_step()
        time_iso = time_value or _now_iso()
        events: List[Dict[str, Any]] = []
        for metric in metrics:
            event = {
                "name": metric["name"],
                "value": metric["value"],
                "step": step_value,
                "time": time_iso,
            }
            if run_value:
                event["run_id"] = run_value
            events.append(event)
        return events


def create_line_parser(spec: str) -> LineParser:
    """Create a line parser from a preset name or regular expression."""

    if spec is None:
        raise ValueError("Regex specification must not be None")
    pattern = spec.strip()
    if not pattern:
        raise ValueError("Regex specification must be a non-empty string")
    preset = pattern.lower()
    if preset == "trl":
        return TRLLineParser()
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid regex pattern '{spec}': {exc}") from exc
    return RegexLineParser(compiled)


__all__ = ["LineParser", "create_line_parser"]
