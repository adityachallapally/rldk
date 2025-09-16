"""Helpers for emitting canonical JSONL monitoring events."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _isoformat(value: Optional[Any]) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, str):
        return value
    raise ValueError("time must be an ISO8601 string or datetime object")


def _ensure_mapping(value: Optional[Dict[str, Any]], field: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    raise ValueError(f"{field} must be a mapping if provided")


class EventWriter:
    """Line-buffered writer that appends events to a JSONL file."""

    def __init__(self, path: str | Path, *, buffering: int = 1) -> None:
        self.path = Path(path)
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8", buffering=buffering)

    def log(
        self,
        *,
        step: int,
        name: str,
        value: float,
        time: Optional[Any] = None,
        run_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        event: Dict[str, Any] = {
            "time": _isoformat(time),
            "step": int(step),
            "name": str(name),
            "value": float(value),
        }
        if run_id is not None:
            event["run_id"] = str(run_id)
        tags_payload = _ensure_mapping(tags, "tags")
        if tags_payload:
            event["tags"] = tags_payload
        meta_payload = _ensure_mapping(meta, "meta")
        if meta_payload:
            event["meta"] = meta_payload
        for key in extra:
            if key in event:
                raise ValueError(f"Duplicate field '{key}' provided in extra arguments")
        event.update(extra)
        self._handle.write(json.dumps(event, sort_keys=True) + "\n")
        self._handle.flush()
        return event

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "EventWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["EventWriter"]
