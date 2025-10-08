"""Integration tests for TRL policy wrapping helpers."""

import sys
import types
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

if "numpy" not in sys.modules:
    class _RandomArray(list):
        def tolist(self) -> List[float]:  # pragma: no cover - simple helper
            return list(self)

    class _RandomStub:
        def __init__(self) -> None:
            self._state: Any = (0,)

        def get_state(self) -> Any:
            return self._state

        def set_state(self, state: Any) -> None:
            self._state = state

        def seed(self, seed: Any = None) -> None:  # pragma: no cover - deterministic stub
            self._state = (seed,)

        def random(self, size: Any = None) -> Any:
            if size is None:
                return 0.0
            if isinstance(size, int):
                return _RandomArray([0.0] * size)
            if isinstance(size, tuple):
                total = 1
                for dim in size:
                    total *= dim or 1
                return _RandomArray([0.0] * total)
            return _RandomArray([0.0])

    numpy_stub = types.ModuleType("numpy")
    numpy_stub.std = lambda *args, **kwargs: 0.0  # type: ignore[assignment]
    numpy_stub.polyfit = lambda *args, **kwargs: [0.0]  # type: ignore[assignment]
    numpy_stub.mean = lambda *args, **kwargs: 0.0  # type: ignore[assignment]
    numpy_stub.random = _RandomStub()  # type: ignore[assignment]
    numpy_stub.ndarray = list  # type: ignore[assignment]
    sys.modules["numpy"] = numpy_stub

if "pandas" not in sys.modules:
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.DataFrame = lambda *args, **kwargs: None  # type: ignore[assignment]
    sys.modules["pandas"] = pandas_stub

if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")

    class _PreTrainedModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - simple stub
            pass

    class _GenerationConfig:
        def __init__(self, **kwargs: Any) -> None:  # pragma: no cover - simple stub
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _AutoCausalLM(_PreTrainedModel):
        def __init__(self) -> None:
            super().__init__()
            self.name_or_path = "stub"
            self.base_model_prefix = "transformer"
            self.transformer = object()
            self.is_gradient_checkpointing = False
            self.generation_config = None

        def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - unused
            return {}

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*_args: Any, **_kwargs: Any) -> _AutoCausalLM:
            return _AutoCausalLM()

    class _AutoTokenizer:
        def __init__(self) -> None:
            self.pad_token = "<pad>"
            self.eos_token = "<eos>"
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.bos_token_id = 0

        @classmethod
        def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> "_AutoTokenizer":
            return cls()

    transformers_stub.PreTrainedModel = _PreTrainedModel  # type: ignore[assignment]
    transformers_stub.GenerationConfig = _GenerationConfig  # type: ignore[assignment]
    transformers_stub.AutoModelForCausalLM = _AutoModelForCausalLM  # type: ignore[assignment]
    transformers_stub.AutoTokenizer = _AutoTokenizer  # type: ignore[assignment]
    sys.modules["transformers"] = transformers_stub

from src.rldk.integrations.trl import utils as trl_utils


class DummyTokenizer:
    eos_token_id = 1
    pad_token_id = 1
    bos_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"


class DummyPolicy:
    """Minimal causal LM stand-in with preserved weights."""

    def __init__(self, name: str):
        self.name_or_path = name
        self.base_model_prefix = "transformer"
        self.transformer = object()
        self.is_gradient_checkpointing = False
        self.custom_tensor = [name]
        self.generation_config = SimpleNamespace()

    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover - unused
        return {}


class DummyValueHead:
    """Simple wrapper mimicking AutoModelForCausalLMWithValueHead behaviour."""

    call_history: List[Dict[str, Any]] = []

    def __init__(self, pretrained_model: DummyPolicy, model_name_or_path: str = ""):
        self.pretrained_model = pretrained_model
        self.name_or_path = model_name_or_path or getattr(pretrained_model, "name_or_path", "wrapped")
        self.base_model_prefix = getattr(pretrained_model, "base_model_prefix", "transformer")
        if hasattr(pretrained_model, self.base_model_prefix):
            setattr(self, self.base_model_prefix, getattr(pretrained_model, self.base_model_prefix))
        self.is_gradient_checkpointing = getattr(pretrained_model, "is_gradient_checkpointing", False)
        self.generation_config = None
        self.v_head = lambda *args, **kwargs: {"values": [0.0]}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "",
        *_,
        pretrained_model: DummyPolicy = None,
        **__,
    ) -> "DummyValueHead":
        if pretrained_model is None:
            pretrained_model = DummyPolicy(model_name_or_path)
        instance = cls(pretrained_model, model_name_or_path)
        cls.call_history.append(
            {
                "model_name_or_path": model_name_or_path,
                "pretrained_model": pretrained_model,
            }
        )
        return instance

    def to(self, *args: Any, **kwargs: Any) -> "DummyValueHead":  # pragma: no cover - unused helper
        return self


class DummyTrainer:
    """Minimal PPOTrainer replacement capturing constructor arguments."""

    def __init__(
        self,
        *,
        args: Any,
        model: DummyValueHead,
        ref_model: DummyValueHead,
        processing_class: DummyTokenizer,
        train_dataset: Any,
        callbacks: List[Any],
        **kwargs: Any,
    ) -> None:
        self.args = args
        self.model = model
        self.ref_model = ref_model
        self.processing_class = processing_class
        self.train_dataset = train_dataset
        self.callbacks = callbacks
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture(autouse=True)
def _reset_call_history() -> None:
    DummyValueHead.call_history.clear()


def _prepare_trl_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure TRL utilities to use dummy implementations."""

    dummy_trl_module = types.ModuleType("trl")
    dummy_trl_module.PPOTrainer = DummyTrainer
    dummy_trl_module.AutoModelForCausalLMWithValueHead = DummyValueHead
    monkeypatch.setitem(sys.modules, "trl", dummy_trl_module)

    monkeypatch.setattr(trl_utils, "TRL_AVAILABLE", True)
    monkeypatch.setattr(trl_utils, "AutoModelForCausalLMWithValueHead", DummyValueHead)
    monkeypatch.setattr(trl_utils, "check_trl_compatibility", lambda: {"version": "0.22.0"})


def test_prepare_models_for_ppo_sets_score_shim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reward/value models expose a callable score attribute for TRL 0.23+ shims."""

    _prepare_trl_environment(monkeypatch)
    monkeypatch.setattr(trl_utils, "check_trl_compatibility", lambda: {"version": "0.23.0"})

    tokenizer = DummyTokenizer()
    _, _, reward_model, prepared_tokenizer = trl_utils.prepare_models_for_ppo(
        "base-model",
        tokenizer=tokenizer,
    )

    assert hasattr(reward_model, "score")
    assert reward_model.score is reward_model.v_head
    assert callable(reward_model.score)

    value_model = trl_utils._prepare_value_model(
        "base-model",
        prepared_tokenizer,
    )

    assert hasattr(value_model, "score")
    assert value_model.score is value_model.v_head
    assert callable(value_model.score)


def test_legacy_wrapping_preserves_custom_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Custom policies retain their tensors after legacy TRL wrapping."""

    _prepare_trl_environment(monkeypatch)

    tokenizer = DummyTokenizer()
    policy = DummyPolicy("policy-model")
    ref_policy = DummyPolicy("ref-model")
    reward_model = DummyValueHead.from_pretrained("reward-model")

    trainer = trl_utils.create_ppo_trainer(
        "base-model",
        SimpleNamespace(run_name="run"),
        train_dataset=[{"input_ids": [0]}],
        tokenizer=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        callbacks=[],
        validate_setup=True,
    )

    # Ensure policies were wrapped and original weights preserved
    assert isinstance(trainer.model, DummyValueHead)
    assert trainer.model.pretrained_model is policy
    assert trainer.model.pretrained_model.custom_tensor == policy.custom_tensor

    assert isinstance(trainer.ref_model, DummyValueHead)
    assert trainer.ref_model.pretrained_model is ref_policy

    # Verify wrapping invoked the pretrained_model shortcut rather than reloading from disk
    assert DummyValueHead.call_history[1]["pretrained_model"] is policy
    assert DummyValueHead.call_history[2]["pretrained_model"] is ref_policy


def test_validate_setup_rejects_plain_value_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even when base policies are allowed, value model must have a value head."""

    _prepare_trl_environment(monkeypatch)

    tokenizer = DummyTokenizer()
    reward_model = DummyValueHead.from_pretrained("reward-model")

    result = trl_utils.validate_ppo_setup(
        DummyPolicy("policy"),
        DummyPolicy("ref"),
        reward_model,
        DummyPolicy("value"),
        tokenizer,
        require_value_model=True,
        allow_base_models=True,
    )

    assert not result["valid"]
    assert any(
        "value_model is not an AutoModelForCausalLMWithValueHead instance" in issue
        for issue in result["issues"]
    )


def test_fix_generation_config_initializes_warning_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fix_generation_config() ensures TRL models expose a warnings registry."""

    _prepare_trl_environment(monkeypatch)

    tokenizer = DummyTokenizer()
    model = DummyValueHead.from_pretrained("policy-model")

    assert not hasattr(model, "warnings_issued")

    patched = trl_utils.fix_generation_config(model, tokenizer)

    assert hasattr(patched, "warnings_issued")
    assert patched.warnings_issued == {}


def test_fix_generation_config_errors_without_value_head(monkeypatch: pytest.MonkeyPatch) -> None:
    """fix_generation_config() raises a clear error when value-head models are unavailable."""

    monkeypatch.setattr(trl_utils, "TRL_AVAILABLE", True)
    monkeypatch.setattr(trl_utils, "AutoModelForCausalLMWithValueHead", None)

    with pytest.raises(ImportError) as exc_info:
        trl_utils.fix_generation_config(object(), DummyTokenizer())

    assert "AutoModelForCausalLMWithValueHead" in str(exc_info.value)
