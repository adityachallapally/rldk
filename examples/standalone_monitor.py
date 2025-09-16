#!/usr/bin/env python3
"""Standalone monitor for testing the monitoring engine."""

import json
import sys
import yaml
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from typing import Any, Dict, Generator, List, Optional, Union


class Event:
    """Canonical event representation."""
    
    def __init__(self, data: Dict[str, Any]):
        self.time = data.get("time")
        self.step = int(data.get("step", 0))
        self.name = str(data.get("name", ""))
        self.value = float(data.get("value", 0.0))
        self.run_id = data.get("run_id")
        self.tags = data.get("tags", {})
        self.meta = data.get("meta", {})
        self._raw = data
    
    def __repr__(self):
        return f"Event(step={self.step}, name='{self.name}', value={self.value})"
    
    @property
    def metric_key(self) -> str:
        """Unique key for this metric (name + tags)."""
        if self.tags:
            # Create a deterministic key from sorted tags
            tag_str = ",".join(f"{k}={v}" for k, v in sorted(self.tags.items()))
            return f"{self.name}|{tag_str}"
        return self.name


class Rule:
    """Monitoring rule with condition evaluation and windowing."""
    
    def __init__(self, rule_data: Dict[str, Any]):
        self.id = rule_data["id"]
        self.where = rule_data.get("where", "True")
        self.condition = rule_data["condition"]
        self.window_size = rule_data.get("window", {}).get("size", 1)
        self.window_kind = rule_data.get("window", {}).get("kind", "consecutive")
        self.cooldown_steps = rule_data.get("cooldown_steps", 0)
        self.grace_steps = rule_data.get("grace_steps", 0)
        self.actions = rule_data.get("actions", [])
        
        # State tracking
        self.last_trigger_step = -1
        self.event_count = 0
    
    def matches_where(self, event: Event) -> bool:
        """Check if event matches the where clause."""
        try:
            # Simple evaluation of where clause
            # Replace common patterns with event attributes
            where_expr = self.where
            where_expr = where_expr.replace("name", f'"{event.name}"')
            where_expr = where_expr.replace("step", str(event.step))
            where_expr = where_expr.replace("value", str(event.value))
            
            # Handle tags access like tags.env
            if "tags." in where_expr:
                for key, val in event.tags.items():
                    where_expr = where_expr.replace(f"tags.{key}", f'"{val}"')
            
            return eval(where_expr)
        except Exception as e:
            print(f"Warning: Error evaluating where clause '{self.where}': {e}")
            return False
    
    def evaluate_condition(self, events: List[Event]) -> bool:
        """Evaluate condition on the event window."""
        if not events:
            return False
        
        try:
            # Replace condition variables
            condition = self.condition
            
            # Handle value comparisons
            if "value" in condition:
                condition = condition.replace("value", str(events[-1].value))
            
            # Handle aggregates
            if "mean(value)" in condition:
                mean_val = sum(e.value for e in events) / len(events)
                condition = condition.replace("mean(value)", str(mean_val))
            
            if "max(value)" in condition:
                max_val = max(e.value for e in events)
                condition = condition.replace("max(value)", str(max_val))
            
            if "min(value)" in condition:
                min_val = min(e.value for e in events)
                condition = condition.replace("min(value)", str(min_val))
            
            if "any(value" in condition:
                # Handle any(value > threshold) patterns
                import re
                pattern = r'any\(value\s*([><=!]+)\s*([0-9.]+)\)'
                match = re.search(pattern, condition)
                if match:
                    op, threshold = match.groups()
                    threshold = float(threshold)
                    any_result = any(eval(f"e.value {op} {threshold}") for e in events)
                    condition = re.sub(pattern, str(any_result), condition)
            
            return eval(condition)
        except Exception as e:
            print(f"Warning: Error evaluating condition '{self.condition}': {e}")
            return False
    
    def can_trigger(self, event: Event) -> bool:
        """Check if rule can trigger (cooldown and grace period)."""
        # Check grace period
        if self.event_count < self.grace_steps:
            return False
        
        # Check cooldown
        if self.last_trigger_step >= 0:
            steps_since_trigger = event.step - self.last_trigger_step
            if steps_since_trigger < self.cooldown_steps:
                return False
        
        return True


class MetricWindow:
    """Window manager for a specific metric."""
    
    def __init__(self, window_size: int, window_kind: str = "consecutive"):
        self.window_size = window_size
        self.window_kind = window_kind
        self.events = deque(maxlen=window_size)
    
    def add_event(self, event: Event):
        """Add event to window."""
        if self.window_kind == "consecutive":
            # For consecutive windows, reset if step is not sequential
            if self.events and event.step != self.events[-1].step + 1:
                self.events.clear()
        
        self.events.append(event)
    
    def get_events(self) -> List[Event]:
        """Get current window events."""
        return list(self.events)
    
    def is_full(self) -> bool:
        """Check if window is full."""
        return len(self.events) == self.window_size


class MonitorEngine:
    """Core monitoring engine for streaming and batch analysis."""
    
    def __init__(self, rules_file: Union[str, Path], alerts_file: Optional[Union[str, Path]] = None):
        self.rules_file = Path(rules_file)
        self.alerts_file = Path(alerts_file) if alerts_file else Path("artifacts/alerts.jsonl")
        
        # Load rules
        self.rules = self._load_rules()
        
        # Per-metric windows
        self.metric_windows: Dict[str, MetricWindow] = defaultdict(
            lambda: MetricWindow(1, "consecutive")
        )
        
        # Ensure alerts directory exists
        self.alerts_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_rules(self) -> List[Rule]:
        """Load rules from YAML file."""
        with open(self.rules_file, 'r') as f:
            data = yaml.safe_load(f)
        
        rules = []
        for rule_data in data.get("rules", []):
            try:
                rule = Rule(rule_data)
                rules.append(rule)
                print(f"Loaded rule: {rule.id}")
            except Exception as e:
                print(f"Failed to load rule: {e}")
        
        return rules
    
    def _update_windows(self, event: Event):
        """Update metric windows for the event."""
        metric_key = event.metric_key
        
        # Get or create window for this metric
        if metric_key not in self.metric_windows:
            # Find the rule that applies to this metric to get window config
            window_size = 1
            window_kind = "consecutive"
            
            for rule in self.rules:
                if rule.matches_where(event):
                    window_size = rule.window_size
                    window_kind = rule.window_kind
                    break
            
            self.metric_windows[metric_key] = MetricWindow(window_size, window_kind)
        
        self.metric_windows[metric_key].add_event(event)
    
    def _execute_actions(self, rule: Rule, event: Event, window_events: List[Event]):
        """Execute rule actions."""
        alert_data = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "step": event.step,
            "rule_id": rule.id,
            "metric": event.name,
            "value": event.value,
            "window": {
                "size": len(window_events),
                "kind": rule.window_kind
            },
            "run_id": event.run_id,
            "tags": event.tags,
            "meta": event.meta
        }
        
        for action in rule.actions:
            if "warn" in action:
                warn_data = action["warn"]
                message = warn_data.get("msg", "")
                
                # Template the message
                if message:
                    try:
                        message = message.format(
                            name=event.name,
                            value=event.value,
                            step=event.step,
                            rule_id=rule.id
                        )
                    except Exception as e:
                        print(f"Warning: Error templating message: {e}")
                        message = f"Alert: {rule.id} triggered"
                
                alert_data["action"] = "warn"
                alert_data["message"] = message
                
                # Write to stderr
                print(f"ALERT: {message}", file=sys.stderr)
                
                # Write to alerts.jsonl
                with open(self.alerts_file, "a") as f:
                    f.write(json.dumps(alert_data) + "\n")
                
                print(f"Alert triggered: {rule.id} - {message}")
    
    def process_event(self, event: Event):
        """Process a single event through all rules."""
        # Update windows
        self._update_windows(event)
        
        # Check each rule
        for rule in self.rules:
            if not rule.matches_where(event):
                continue
            
            # Increment event count for grace period
            rule.event_count += 1
            
            if not rule.can_trigger(event):
                continue
            
            # Get window for this metric
            metric_key = event.metric_key
            window_events = self.metric_windows[metric_key].get_events()
            
            # Check if window is full (for consecutive windows)
            if rule.window_kind == "consecutive" and not self.metric_windows[metric_key].is_full():
                continue
            
            # Evaluate condition
            if rule.evaluate_condition(window_events):
                # Execute actions
                self._execute_actions(rule, event, window_events)
                
                # Update trigger state
                rule.last_trigger_step = event.step
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate summary report."""
        report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "rules_summary": [],
            "last_seen_metrics": {},
            "total_alerts": 0
        }
        
        # Count alerts
        if self.alerts_file.exists():
            with open(self.alerts_file, "r") as f:
                report["total_alerts"] = sum(1 for _ in f)
        
        # Rule summaries
        for rule in self.rules:
            rule_summary = {
                "id": rule.id,
                "where": rule.where,
                "condition": rule.condition,
                "window_size": rule.window_size,
                "window_kind": rule.window_kind,
                "cooldown_steps": rule.cooldown_steps,
                "grace_steps": rule.grace_steps,
                "last_trigger_step": rule.last_trigger_step,
                "event_count": rule.event_count
            }
            report["rules_summary"].append(rule_summary)
        
        # Last seen metrics
        for metric_key, window in self.metric_windows.items():
            if window.events:
                last_event = window.events[-1]
                report["last_seen_metrics"][metric_key] = {
                    "step": last_event.step,
                    "value": last_event.value,
                    "time": last_event.time
                }
        
        return report


def read_stream(path_or_stdin: Union[str, Path, None], field_map: Optional[Dict[str, str]] = None) -> Generator[Event, None, None]:
    """Read events from stream (file or stdin)."""
    if path_or_stdin == "-" or path_or_stdin is None:
        # Read from stdin
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                if field_map:
                    # Apply field mapping
                    mapped_data = {}
                    for key, value in data.items():
                        mapped_key = field_map.get(key, key)
                        mapped_data[mapped_key] = value
                    data = mapped_data
                
                yield Event(data)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON line: {line[:100]}... - {e}")
                continue
    else:
        # Read from file
        path = Path(path_or_stdin)
        
        if not path.exists():
            print(f"Warning: File does not exist: {path}")
            return
        
        with open(path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    if field_map:
                        # Apply field mapping
                        mapped_data = {}
                        for key, value in data.items():
                            mapped_key = field_map.get(key, key)
                            mapped_data[mapped_key] = value
                        data = mapped_data
                    
                    yield Event(data)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON at line {line_num}: {line[:100]}... - {e}")
                    continue


def main():
    """Main function for standalone monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Standalone monitor for JSONL events")
    parser.add_argument("--stream", help="Stream path or '-' for stdin")
    parser.add_argument("--once", help="Batch mode: process file once")
    parser.add_argument("--rules", required=True, help="Rules YAML file")
    parser.add_argument("--alerts", help="Alerts output file")
    
    args = parser.parse_args()
    
    if not args.stream and not args.once:
        print("Error: Must specify either --stream or --once")
        sys.exit(1)
    
    if args.stream and args.once:
        print("Error: Cannot specify both --stream and --once")
        sys.exit(1)
    
    engine = MonitorEngine(args.rules, args.alerts)
    
    try:
        if args.stream:
            print(f"Starting stream monitoring from {args.stream}")
            for event in read_stream(args.stream):
                engine.process_event(event)
        else:
            print(f"Starting batch monitoring of {args.once}")
            for event in read_stream(args.once):
                engine.process_event(event)
            print("Batch monitoring completed")
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"Error during monitoring: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()