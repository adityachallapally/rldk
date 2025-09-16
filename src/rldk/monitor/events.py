"""Canonical event schema and writer for framework-agnostic monitoring."""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class CanonicalEvent:
    """Canonical event schema for framework-agnostic monitoring.

    Required fields:
    - time: ISO8601 timestamp
    - step: integer step number
    - name: metric name string
    - value: numeric value

    Optional fields:
    - run_id: unique run identifier
    - tags: arbitrary key-value metadata
    - meta: additional metadata
    """
    time: str
    step: int
    name: str
    value: float
    run_id: Optional[str] = None
    tags: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "time": self.time,
            "step": self.step,
            "name": self.name,
            "value": self.value,
        }
        if self.run_id is not None:
            result["run_id"] = self.run_id
        if self.tags is not None:
            result["tags"] = self.tags
        if self.meta is not None:
            result["meta"] = self.meta
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalEvent":
        """Create event from dictionary."""
        return cls(
            time=data["time"],
            step=int(data["step"]),
            name=str(data["name"]),
            value=float(data["value"]),
            run_id=data.get("run_id"),
            tags=data.get("tags"),
            meta=data.get("meta"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "CanonicalEvent":
        """Create event from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class EventWriter:
    """Simple JSONL event writer for framework-agnostic logging."""

    def __init__(self, path: Union[str, Path], run_id: Optional[str] = None):
        """Initialize event writer.

        Args:
            path: Path to JSONL file to write to
            run_id: Optional run ID to include in all events
        """
        self.path = Path(path)
        self.run_id = run_id or f"run-{int(time.time())}"

        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._file = open(self.path, "a", buffering=1, encoding="utf-8")

    def log(
        self,
        step: int,
        name: str,
        value: float,
        time_iso: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a single event.

        Args:
            step: Training step number
            name: Metric name
            value: Metric value
            time_iso: ISO8601 timestamp (defaults to current time)
            tags: Optional tags dictionary
            meta: Optional metadata dictionary
        """
        if time_iso is None:
            time_iso = datetime.utcnow().isoformat() + "Z"

        event = CanonicalEvent(
            time=time_iso,
            step=step,
            name=name,
            value=value,
            run_id=self.run_id,
            tags=tags,
            meta=meta,
        )

        self._file.write(event.to_json() + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the file handle."""
        if hasattr(self, "_file") and not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
