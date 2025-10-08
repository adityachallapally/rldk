from __future__ import annotations

import sys
from pathlib import Path
import types

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"

if "numpy" not in sys.modules:  # pragma: no cover - lightweight stub for optional dependency
    numpy_stub = types.ModuleType("numpy")
    random_stub = types.ModuleType("numpy.random")
    random_stub._state = ("stub", 0, None)

    def _seed(value: int) -> None:
        random_stub._state = ("stub", int(value), None)

    def _get_state():
        return random_stub._state

    def _set_state(state) -> None:
        random_stub._state = state

    random_stub.seed = _seed  # type: ignore[assignment]
    random_stub.get_state = _get_state  # type: ignore[assignment]
    random_stub.set_state = _set_state  # type: ignore[assignment]

    numpy_stub.random = random_stub  # type: ignore[attr-defined]
    numpy_stub.bool_ = bool  # type: ignore[attr-defined]
    numpy_stub.ndarray = object  # type: ignore[attr-defined]

    sys.modules["numpy"] = numpy_stub
    sys.modules["numpy.random"] = random_stub

if "pandas" not in sys.modules:  # pragma: no cover - lightweight stub for optional dependency
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    pandas_stub.Series = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["pandas"] = pandas_stub


def _load_config(name: str) -> dict:
    path = CONFIG_DIR / name
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    assert isinstance(payload, dict), f"Expected mapping at {path}, found {type(payload)!r}"
    return payload


def test_ppo_config_schema() -> None:
    payload = _load_config("ppo_tiny.yaml")

    assert isinstance(payload.get("model"), str)
    assert isinstance(payload.get("steps"), int) and payload["steps"] > 0
    assert isinstance(payload.get("log_path"), str)
    assert payload["log_path"].endswith(".jsonl")
    assert isinstance(payload.get("dataset_seed"), int)
    assert isinstance(payload.get("logging_interval"), int)
    assert isinstance(payload.get("ppo_kwargs"), dict)


def test_grpo_config_schema() -> None:
    payload = _load_config("grpo_tiny.yaml")

    assert isinstance(payload.get("model"), str)
    assert isinstance(payload.get("steps"), int) and payload["steps"] > 0
    assert isinstance(payload.get("log_path"), str)
    assert payload["log_path"].endswith(".jsonl")
    assert isinstance(payload.get("dataset_seed"), int)
    assert isinstance(payload.get("logging_interval"), int)
    assert isinstance(payload.get("grpo_kwargs"), dict)
