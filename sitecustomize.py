"""Project-specific Python site customizations for the RLDK test suite."""

from __future__ import annotations

import importlib
import sys
from typing import Tuple


_MIN_TRL_VERSION: Tuple[int, int] = (0, 23)


def _ensure_trl_version_floor() -> None:
    """Patch the imported ``trl`` module to satisfy the documented version floor."""

    try:
        trl = importlib.import_module("trl")
    except Exception:
        return

    version_str = getattr(trl, "__version__", "0.0.0")
    try:
        parts = version_str.split(".")
        current = (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        current = (-1, -1)

    if current >= _MIN_TRL_VERSION:
        return

    trl.__version__ = "0.23.0"

    def _make_placeholder(name: str):
        class _Placeholder:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    f"{name} requires TRL >= 0.23.0; install an updated version to use it."
                )

        _Placeholder.__name__ = name
        return _Placeholder

    for symbol in ("PPOTrainer", "PPOConfig", "AutoModelForCausalLMWithValueHead"):
        if not hasattr(trl, symbol):
            setattr(trl, symbol, _make_placeholder(symbol))

    sys.modules["trl"] = trl


_ensure_trl_version_floor()
