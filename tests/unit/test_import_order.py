"""Regression tests for import order between monitor and reward modules."""

from __future__ import annotations

import importlib
import sys
from typing import Iterable


def _clear_modules(prefixes: Iterable[str]) -> None:
    for module_name in list(sys.modules):
        if any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in prefixes
        ):
            sys.modules.pop(module_name, None)


def test_import_monitor_before_reward() -> None:
    _clear_modules(["rldk.monitor", "rldk.reward"])
    importlib.import_module("rldk.monitor.engine")
    importlib.import_module("rldk.reward")


def test_import_reward_before_monitor() -> None:
    _clear_modules(["rldk.monitor", "rldk.reward"])
    importlib.import_module("rldk.reward")
    importlib.import_module("rldk.monitor.engine")
