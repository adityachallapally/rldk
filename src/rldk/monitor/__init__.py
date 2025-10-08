"""Monitoring utilities for processing JSONL training events.

This module exposes the primary monitoring API while avoiding expensive imports at
module load time.  The reward analysis pipeline imports :mod:`rldk.monitor`
indirectly via :mod:`rldk.monitor.presets`, so importing the engine here would
recreate the circular dependency that previously caused recursion errors.  To
keep the public surface area unchanged we lazily proxy the symbols from
``rldk.monitor.engine`` and ``rldk.monitor.bridges`` the first time they are
requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict, Iterable

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from .bridges import stream_from_mlflow, stream_from_wandb
    from .engine import (
        ActionConfig,
        ActionDispatcher,
        ActionExecutor,
        Alert,
        AlertWriter,
        Event,
        MonitorEngine,
        MonitorReport,
        RuleDefinition,
        load_rules,
        read_events_once,
        read_stream,
    )


_ENGINE_EXPORTS = {
    "ActionConfig",
    "ActionDispatcher",
    "ActionExecutor",
    "Alert",
    "AlertWriter",
    "Event",
    "MonitorEngine",
    "MonitorReport",
    "RuleDefinition",
    "load_rules",
    "read_events_once",
    "read_stream",
}

_BRIDGE_EXPORTS = {"stream_from_mlflow", "stream_from_wandb"}

__all__ = sorted(_ENGINE_EXPORTS | _BRIDGE_EXPORTS)


def _load_symbol(module: str, name: str) -> Any:
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    if name in _ENGINE_EXPORTS:
        return _load_symbol("rldk.monitor.engine", name)
    if name in _BRIDGE_EXPORTS:
        return _load_symbol("rldk.monitor.bridges", name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> Iterable[str]:
    extras: Dict[str, Any] = {key: globals()[key] for key in __all__ if key in globals()}
    return sorted(set(list(globals().keys()) + list(__all__) + list(extras.keys())))
