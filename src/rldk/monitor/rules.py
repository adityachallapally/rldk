"""Rule engine for monitoring JSONL events."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List


class WindowKind(Enum):
    """Window types for rule evaluation."""
    CONSECUTIVE = "consecutive"
    ROLLING = "rolling"


@dataclass
class Window:
    """Window configuration for rule evaluation."""
    size: int
    kind: WindowKind = WindowKind.CONSECUTIVE

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Window":
        """Create window from dictionary."""
        return cls(
            size=data["size"],
            kind=WindowKind(data.get("kind", "consecutive"))
        )


@dataclass
class Rule:
    """Rule definition for monitoring events."""
    id: str
    where: str  # Python-like boolean filter
    condition: str  # Expression on event window
    window: Window
    cooldown_steps: int = 0
    grace_steps: int = 0
    actions: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.actions is None:
            self.actions = []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rule":
        """Create rule from dictionary."""
        window_data = data.get("window", {"size": 1, "kind": "consecutive"})
        return cls(
            id=data["id"],
            where=data["where"],
            condition=data["condition"],
            window=Window.from_dict(window_data),
            cooldown_steps=data.get("cooldown_steps", 0),
            grace_steps=data.get("grace_steps", 0),
            actions=data.get("actions", [])
        )


class RuleEngine:
    """Engine for evaluating rules against event streams."""

    def __init__(self, rules: List[Rule]):
        """Initialize rule engine.

        Args:
            rules: List of rules to evaluate
        """
        self.rules = rules
        self._metric_windows: Dict[str, List[Any]] = {}
        self._rule_cooldowns: Dict[str, int] = {}
        self._rule_grace_counters: Dict[str, int] = {}

    def evaluate_event(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate a single event against all rules.

        Args:
            event: Event dictionary with canonical schema

        Returns:
            List of alert dictionaries for triggered rules
        """
        alerts = []

        for rule in self.rules:
            if not self._matches_where_clause(event, rule.where):
                continue

            metric_key = self._get_metric_key(event, rule)

            self._update_window(metric_key, event, rule.window)

            if not self._check_grace_period(rule, metric_key):
                continue

            if self._is_in_cooldown(rule, event["step"]):
                continue

            if self._evaluate_condition(metric_key, rule):
                alert = self._create_alert(event, rule, metric_key)
                alerts.append(alert)

                self._rule_cooldowns[rule.id] = event["step"] + rule.cooldown_steps

        return alerts

    def _matches_where_clause(self, event: Dict[str, Any], where_clause: str) -> bool:
        """Check if event matches the where clause."""
        try:
            expr = where_clause

            expr = expr.replace('name', f'"{event.get("name", "")}"')
            expr = expr.replace('step', str(event.get("step", 0)))
            expr = expr.replace('value', str(event.get("value", 0)))

            if "tags." in expr and event.get("tags"):
                for key, value in event["tags"].items():
                    tag_ref = f"tags.{key}"
                    if isinstance(value, str):
                        expr = expr.replace(tag_ref, f'"{value}"')
                    else:
                        expr = expr.replace(tag_ref, str(value))

            return eval(expr)
        except Exception:
            return False

    def _get_metric_key(self, event: Dict[str, Any], rule: Rule) -> str:
        """Get unique key for metric windowing."""
        key = event["name"]
        if event.get("tags"):
            tag_str = "&".join(f"{k}={v}" for k, v in sorted(event["tags"].items()))
            key = f"{key}#{tag_str}"
        return key

    def _update_window(self, metric_key: str, event: Dict[str, Any], window: Window) -> None:
        """Update the window for a metric."""
        if metric_key not in self._metric_windows:
            self._metric_windows[metric_key] = []

        window_data = self._metric_windows[metric_key]
        window_data.append(event)

        if len(window_data) > window.size:
            if window.kind == WindowKind.CONSECUTIVE:
                self._metric_windows[metric_key] = window_data[-window.size:]
            elif window.kind == WindowKind.ROLLING:
                pass

    def _check_grace_period(self, rule: Rule, metric_key: str) -> bool:
        """Check if grace period has been satisfied."""
        if rule.grace_steps == 0:
            return True

        if metric_key not in self._rule_grace_counters:
            self._rule_grace_counters[metric_key] = 0

        self._rule_grace_counters[metric_key] += 1
        return self._rule_grace_counters[metric_key] >= rule.grace_steps

    def _is_in_cooldown(self, rule: Rule, current_step: int) -> bool:
        """Check if rule is in cooldown period."""
        if rule.id not in self._rule_cooldowns:
            return False
        return current_step <= self._rule_cooldowns[rule.id]

    def _evaluate_condition(self, metric_key: str, rule: Rule) -> bool:
        """Evaluate the condition on the current window."""
        if metric_key not in self._metric_windows:
            return False

        window_data = self._metric_windows[metric_key]
        if not window_data:
            return False

        if rule.window.kind == WindowKind.CONSECUTIVE:
            eval_window = window_data
        else:  # rolling
            eval_window = window_data[-rule.window.size:] if len(window_data) >= rule.window.size else window_data

        if len(eval_window) < rule.window.size:
            return False

        try:
            values = [event["value"] for event in eval_window]

            condition = rule.condition

            if eval_window:
                condition = condition.replace("value", str(eval_window[-1]["value"]))

            condition = condition.replace("mean(value)", str(sum(values) / len(values)))
            condition = condition.replace("max(value)", str(max(values)))
            condition = condition.replace("min(value)", str(min(values)))

            any_pattern = r"any\(value\s*([><=!]+)\s*([\d.]+)\)"
            match = re.search(any_pattern, condition)
            if match:
                op, threshold = match.groups()
                threshold = float(threshold)
                if op == ">":
                    result = any(v > threshold for v in values)
                elif op == ">=":
                    result = any(v >= threshold for v in values)
                elif op == "<":
                    result = any(v < threshold for v in values)
                elif op == "<=":
                    result = any(v <= threshold for v in values)
                elif op == "==":
                    result = any(v == threshold for v in values)
                elif op == "!=":
                    result = any(v != threshold for v in values)
                else:
                    result = False
                condition = condition.replace(match.group(0), str(result))

            return eval(condition)
        except Exception:
            return False

    def _create_alert(self, event: Dict[str, Any], rule: Rule, metric_key: str) -> Dict[str, Any]:
        """Create alert dictionary for triggered rule."""
        return {
            "ts": event["time"],
            "step": event["step"],
            "rule_id": rule.id,
            "metric": event["name"],
            "value": event["value"],
            "window": {
                "size": rule.window.size,
                "kind": rule.window.kind.value
            },
            "run_id": event.get("run_id"),
            "tags": event.get("tags"),
            "meta": event.get("meta"),
        }
