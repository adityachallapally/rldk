import importlib
import sys
from types import ModuleType
from typing import Dict, Iterable

import pytest


_PREFIXES = ("rldk.monitor", "rldk.reward")


def _clear_modules(prefixes: Iterable[str]) -> Dict[str, ModuleType]:
    removed: Dict[str, ModuleType] = {}
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            removed[name] = sys.modules.pop(name)  # type: ignore[misc]
    return removed


def _restore_modules(modules: Dict[str, ModuleType]) -> None:
    sys.modules.update(modules)


@pytest.mark.parametrize(
    "import_order",
    [
        ("rldk.monitor.engine", "rldk.reward"),
        ("rldk.reward", "rldk.monitor.engine"),
    ],
)
def test_reward_and_monitor_import_order(import_order):
    preserved = _clear_modules(_PREFIXES)
    try:
        first = importlib.import_module(import_order[0])
        second = importlib.import_module(import_order[1])
        assert isinstance(first, ModuleType)
        assert isinstance(second, ModuleType)
    finally:
        _clear_modules(_PREFIXES)
        _restore_modules(preserved)


@pytest.mark.parametrize(
    "bootstrap_module",
    ["rldk.reward", "rldk.monitor.engine"],
)
def test_monitor_reexports_after_bootstrap(bootstrap_module):
    preserved = _clear_modules(_PREFIXES)
    try:
        importlib.import_module(bootstrap_module)
        monitor_pkg = importlib.import_module("rldk.monitor")
        engine_cls = getattr(monitor_pkg, "MonitorEngine")
        assert isinstance(engine_cls, type)
        assert engine_cls.__module__ == "rldk.monitor.engine"
    finally:
        _clear_modules(_PREFIXES)
        _restore_modules(preserved)
