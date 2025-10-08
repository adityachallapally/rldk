"""Smoke tests for the tiny PPO/GRPO example scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import types

import pytest

# Provide minimal NumPy/Pandas/Transformers stubs so shared fixtures can import
# without the heavy ML stack being installed in the execution environment.
if "numpy" not in sys.modules:  # pragma: no cover - test bootstrap helper
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

if "pandas" not in sys.modules:  # pragma: no cover - test bootstrap helper
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    pandas_stub.Series = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["pandas"] = pandas_stub

if "transformers" not in sys.modules:  # pragma: no cover - test bootstrap helper
    transformers_stub = types.ModuleType("transformers")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_name: str):
            return types.SimpleNamespace(pad_token=None, eos_token="</s>", padding_side="right")

    class _GenerationConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    transformers_stub.AutoTokenizer = _AutoTokenizer  # type: ignore[attr-defined]
    transformers_stub.GenerationConfig = _GenerationConfig  # type: ignore[attr-defined]
    sys.modules["transformers"] = transformers_stub

from rldk.emit import EventWriter


def _install_trl_stub(monkeypatch):
    trl_stub = types.ModuleType("trl")

    class _DummyPPOConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    trl_stub.PPOConfig = _DummyPPOConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trl", trl_stub)
    return trl_stub


@pytest.mark.xfail(reason="Tiny PPO script smoke test is experimental", strict=False)
def test_run_ppo_tiny_smoke(monkeypatch, tmp_path):
    import examples.run_ppo_tiny as run_ppo_tiny

    dummy_settings = run_ppo_tiny.TinyPPORunSettings(
        model="dummy-ppo",
        dataset_seed=0,
        steps=4,
        logging_interval=1,
        log_path=Path("artifacts/ppo_tiny/run.jsonl"),
        ppo_config=types.SimpleNamespace(run_name="ppo-test"),
    )
    monkeypatch.setattr(run_ppo_tiny, "load_ppo_config", lambda path: dummy_settings)

    dummy_dataset = [{"prompt": "hi", "response": "hello"}]
    monkeypatch.setattr(run_ppo_tiny, "build_tiny_dataset", lambda: dummy_dataset)

    dummy_tokenizer = types.SimpleNamespace(pad_token="[PAD]", eos_token="[EOS]", padding_side="right")
    monkeypatch.setattr(run_ppo_tiny, "load_tokenizer", lambda model_name: dummy_tokenizer)

    def fake_tokenize(dataset, tokenizer, **kwargs):
        sanitized = []
        for record in dataset:
            sanitized_record = dict(record)
            sanitized_record.pop("prompt", None)
            sanitized_record.pop("response", None)
            sanitized_record.setdefault("input_ids", [101, 102])
            sanitized_record.setdefault("attention_mask", [1, 1])
            sanitized.append(sanitized_record)
        return sanitized

    monkeypatch.setattr(run_ppo_tiny, "tokenize_text_column", fake_tokenize)

    def fake_create_ppo_trainer(model_name, ppo_config, train_dataset, **kwargs):
        event_log_path = Path(kwargs["event_log_path"])

        assert all("input_ids" in sample for sample in train_dataset)
        assert all("attention_mask" in sample for sample in train_dataset)
        assert all("prompt" not in sample for sample in train_dataset)
        assert all("response" not in sample for sample in train_dataset)

        class DummyTrainer:
            def __init__(self, log_path: Path) -> None:
                self._log_path = log_path

            def train(self) -> None:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with EventWriter(self._log_path) as writer:
                    writer.log(step=1, name="reward", value=0.5)
                    writer.log(step=1, name="kl", value=0.05)

        return DummyTrainer(event_log_path)

    monkeypatch.setattr(run_ppo_tiny, "create_ppo_trainer", fake_create_ppo_trainer)

    log_dir = tmp_path / "ppo"
    exit_code = run_ppo_tiny.main(["--log-dir", str(log_dir)])
    assert exit_code == 0

    log_path = log_dir / "run.jsonl"
    assert log_path.is_file()

    entries = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    names = {entry["name"] for entry in entries}
    assert {"reward", "kl"} <= names


def test_load_ppo_config_resolves_repo_relative_log_path(monkeypatch):
    import examples.run_ppo_tiny as run_ppo_tiny

    _install_trl_stub(monkeypatch)

    settings = run_ppo_tiny.load_ppo_config(run_ppo_tiny.DEFAULT_CONFIG_PATH)

    expected = (
        run_ppo_tiny.DEFAULT_CONFIG_PATH.parents[1]
        / "artifacts"
        / "ppo_tiny"
        / "run.jsonl"
    ).resolve()
    assert settings.log_path == expected


def test_load_ppo_config_resolves_external_relative_log_path(monkeypatch, tmp_path):
    import examples.run_ppo_tiny as run_ppo_tiny

    _install_trl_stub(monkeypatch)

    config_path = tmp_path / "ppo_custom.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model: custom/model",
                "dataset_seed: 1",
                "steps: 2",
                "logging_interval: 1",
                "log_path: logs/run.jsonl",
                "ppo_kwargs: {}",
            ]
        ),
        encoding="utf-8",
    )

    settings = run_ppo_tiny.load_ppo_config(config_path)

    expected = (tmp_path / "logs" / "run.jsonl").resolve()
    assert settings.log_path == expected


@pytest.mark.xfail(reason="Tiny GRPO script smoke test is experimental", strict=False)
def test_run_grpo_tiny_smoke(monkeypatch, tmp_path):
    import examples.run_grpo_tiny as run_grpo_tiny

    dummy_settings = run_grpo_tiny.TinyGRPORunSettings(
        model="dummy-grpo",
        dataset_seed=0,
        steps=4,
        logging_interval=1,
        log_path=Path("artifacts/grpo_tiny/run.jsonl"),
        grpo_config=types.SimpleNamespace(run_name="grpo-test"),
    )
    monkeypatch.setattr(run_grpo_tiny, "load_grpo_config", lambda path: dummy_settings)

    dummy_dataset = [
        {"accepted": True},
        {"accepted": False},
        {"accepted": True},
    ]
    monkeypatch.setattr(run_grpo_tiny, "build_tiny_dataset", lambda: dummy_dataset)

    dummy_tokenizer = types.SimpleNamespace(pad_token="[PAD]", eos_token="[EOS]", padding_side="right")
    monkeypatch.setattr(run_grpo_tiny, "load_tokenizer", lambda model_name: dummy_tokenizer)

    def fake_build_trainer(model_name, grpo_config, dataset, tokenizer, event_log_path):
        class DummyTrainer:
            def __init__(self, log_path: Path) -> None:
                self._log_path = log_path

            def train(self) -> None:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with EventWriter(self._log_path) as writer:
                    writer.log(step=1, name="reward", value=0.7)
                    writer.log(step=1, name="kl", value=0.02)

        return DummyTrainer(Path(event_log_path))

    monkeypatch.setattr(run_grpo_tiny, "build_grpo_trainer", fake_build_trainer)

    log_dir = tmp_path / "grpo"
    exit_code = run_grpo_tiny.main(["--log-dir", str(log_dir)])
    assert exit_code == 0

    log_path = log_dir / "run.jsonl"
    assert log_path.is_file()

    entries = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    names = {entry["name"] for entry in entries}
    assert {"reward", "kl"} <= names  # acceptance_rate logged separately in real script


def test_load_grpo_config_resolves_repo_relative_log_path(monkeypatch):
    import examples.run_grpo_tiny as run_grpo_tiny

    monkeypatch.setattr(
        run_grpo_tiny,
        "create_grpo_config",
        lambda **kwargs: types.SimpleNamespace(**kwargs),
    )

    settings = run_grpo_tiny.load_grpo_config(run_grpo_tiny.DEFAULT_CONFIG_PATH)

    expected = (
        run_grpo_tiny.DEFAULT_CONFIG_PATH.parents[1]
        / "artifacts"
        / "grpo_tiny"
        / "run.jsonl"
    ).resolve()
    assert settings.log_path == expected


def test_load_grpo_config_resolves_external_relative_log_path(monkeypatch, tmp_path):
    import examples.run_grpo_tiny as run_grpo_tiny

    monkeypatch.setattr(
        run_grpo_tiny,
        "create_grpo_config",
        lambda **kwargs: types.SimpleNamespace(**kwargs),
    )

    config_path = tmp_path / "grpo_custom.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model: custom/model",
                "dataset_seed: 2",
                "steps: 3",
                "logging_interval: 1",
                "log_path: logs/run.jsonl",
                "grpo_kwargs: {}",
            ]
        ),
        encoding="utf-8",
    )

    settings = run_grpo_tiny.load_grpo_config(config_path)

    expected = (tmp_path / "logs" / "run.jsonl").resolve()
    assert settings.log_path == expected
