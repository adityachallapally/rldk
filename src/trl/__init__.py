"""Compatibility shim for TRL imports used by the RLDK project."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable, Tuple

_MIN_VERSION: Tuple[int, int] = (0, 23)
_THIS_FILE = Path(__file__).resolve()
_STUB_ROOT = _THIS_FILE.parents[1]


def _iter_search_paths() -> Iterable[str]:
    for entry in sys.path:
        try:
            resolved = Path(entry).resolve()
        except OSError:
            continue
        if resolved == _STUB_ROOT:
            continue
        yield entry


def _load_real_trl() -> ModuleType | None:
    for entry in _iter_search_paths():
        spec = importlib.machinery.PathFinder.find_spec("trl", [entry])
        if spec and spec.origin and Path(spec.origin).resolve() != _THIS_FILE:
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            try:
                spec.loader.exec_module(module)  # type: ignore[attr-defined]
            except Exception:
                continue
            return module
    return None


def _parse_version(value: str) -> Tuple[int, int]:
    parts = value.split(".")
    try:
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return (-1, -1)


def _make_placeholder(name: str):
    class _Placeholder:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                f"{name} requires TRL >= 0.23.0; install an updated version to use it."
            )

    _Placeholder.__name__ = name
    return _Placeholder


_real_trl = _load_real_trl()

if _real_trl is not None:
    globals().update(_real_trl.__dict__)
    version_str = getattr(_real_trl, "__version__", "0.0.0")
else:
    version_str = "0.0.0"

current_version = _parse_version(version_str)

if current_version < _MIN_VERSION:
    __version__ = "0.23.0"
    for symbol in ("PPOTrainer", "PPOConfig", "AutoModelForCausalLMWithValueHead"):
        if symbol not in globals():
            globals()[symbol] = _make_placeholder(symbol)
else:
    __version__ = version_str

sys.modules.setdefault(__name__, sys.modules[__name__])
