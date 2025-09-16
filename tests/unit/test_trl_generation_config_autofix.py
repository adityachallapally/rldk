import warnings
from typing import Dict, List
from unittest.mock import MagicMock

import pytest

trl = pytest.importorskip("trl")

from transformers import GenerationConfig  # noqa: E402  (import after pytest skip logic)
import transformers.utils as _hf_transformers_utils  # noqa: E402

if not hasattr(_hf_transformers_utils, "is_torch_mlu_available"):  # pragma: no cover - defensive shims
    _hf_transformers_utils.is_torch_mlu_available = lambda: False

from rldk.integrations.trl.utils import create_ppo_trainer  # noqa: E402


@pytest.fixture()
def stub_trl_environment(monkeypatch) -> Dict[str, object]:
    """Provide a deterministic TRL environment with lightweight stubs."""

    from rldk.integrations.trl import utils as trl_utils

    class DummyTokenizer:
        """Minimal tokenizer stub used for testing."""

        loaded: List[str] = []

        def __init__(self, name: str):
            self.name = name
            self.pad_token = None
            self.pad_token_id = None
            self.eos_token = "<eos>"
            self.eos_token_id = 2
            self.bos_token_id = 1

        @classmethod
        def from_pretrained(cls, name: str) -> "DummyTokenizer":
            instance = cls(name)
            cls.loaded.append(name)
            return instance

    class DummyValueHead:
        """Lightweight value-head model used for auto-fix validation."""

        def __init__(self, name: str):
            self.name_or_path = name
            self.pretrained_model = MagicMock()
            self.pretrained_model.base_model_prefix = "transformer"
            self.pretrained_model.transformer = MagicMock(name=f"{name}-transformer")
            self.pretrained_model.is_gradient_checkpointing = False
            self.pretrained_model.name_or_path = name
            self.base_model_prefix = "transformer"
            self.v_head = MagicMock()
            self.generation_config = None

        @classmethod
        def from_pretrained(cls, name: str) -> "DummyValueHead":
            return cls(name)

    fix_calls: List[str] = []

    def fake_fix_generation_config(model, tokenizer, generation_config):
        fix_calls.append(model.name_or_path)
        if generation_config is None:
            generation_config = GenerationConfig(
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=getattr(tokenizer, "bos_token_id", None),
                max_length=64,
            )
        model.generation_config = generation_config
        model.base_model_prefix = getattr(model, "base_model_prefix", "transformer") or "transformer"
        base_attr = getattr(model.pretrained_model, model.base_model_prefix, MagicMock())
        setattr(model, model.base_model_prefix, base_attr)
        model.is_gradient_checkpointing = getattr(
            model.pretrained_model, "is_gradient_checkpointing", False
        )
        return model

    class DummyPPOTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model = kwargs["model"]
            self.ref_model = kwargs["ref_model"]
            self.reward_model = kwargs.get("reward_model")
            self.value_model = kwargs.get("value_model")
            self.processing_class = kwargs.get("processing_class")

    monkeypatch.setattr(trl_utils, "AutoTokenizer", DummyTokenizer)
    monkeypatch.setattr(trl_utils, "AutoModelForCausalLMWithValueHead", DummyValueHead)
    monkeypatch.setattr(trl_utils, "fix_generation_config", fake_fix_generation_config)
    monkeypatch.setattr("trl.AutoModelForCausalLMWithValueHead", DummyValueHead, raising=False)
    monkeypatch.setattr("trl.PPOTrainer", DummyPPOTrainer)
    monkeypatch.setattr(trl_utils, "TRL_AVAILABLE", True)
    monkeypatch.setattr(trl, "__version__", "0.23.1")

    # Ensure compatibility check reports an exact version for deterministic behaviour
    monkeypatch.setattr(
        trl_utils,
        "check_trl_compatibility",
        lambda: {
            "trl_available": True,
            "version": "0.23.1",
            "warnings": [],
            "recommendations": [],
        },
    )

    return {
        "tokenizer_cls": DummyTokenizer,
        "model_cls": DummyValueHead,
        "fix_calls": fix_calls,
        "ppo_cls": DummyPPOTrainer,
    }


def test_create_ppo_trainer_auto_fix_defaults(stub_trl_environment):
    """RLDK should auto-fix models when defaults are created."""

    trainer = create_ppo_trainer(
        model_name="stub-model",
        ppo_config=MagicMock(name="ppo-config"),
        train_dataset=[{"prompt": "hi", "response": "hello"}],
        callbacks=[],
    )

    assert trainer.model.generation_config is not None
    assert trainer.ref_model.generation_config is not None
    assert trainer.reward_model.generation_config is not None
    assert trainer.value_model.generation_config is not None

    tokenizer = trainer.processing_class
    assert tokenizer is not None
    assert tokenizer.pad_token == tokenizer.eos_token
    assert tokenizer.pad_token_id == tokenizer.eos_token_id


def test_create_ppo_trainer_auto_fix_with_custom_models(stub_trl_environment):
    """User-provided models without a tokenizer should be auto-fixed."""

    env = stub_trl_environment
    model_cls = env["model_cls"]

    policy = model_cls("policy-model")
    reference = model_cls("reference-model")
    reward = model_cls("reward-model")
    value = model_cls("value-model")

    with warnings.catch_warnings(record=True) as caught:
        trainer = create_ppo_trainer(
            model_name="fallback-model",
            ppo_config=MagicMock(name="ppo-config"),
            train_dataset=[{"prompt": "hi", "response": "hello"}],
            tokenizer=None,
            model=policy,
            ref_model=reference,
            reward_model=reward,
            value_model=value,
            callbacks=[],
        )

    assert not caught, "Auto-fix should avoid warnings when successful"

    assert trainer.model is policy
    assert trainer.ref_model is reference
    assert trainer.reward_model is reward
    assert trainer.value_model is value

    assert trainer.model.generation_config is not None
    assert trainer.ref_model.generation_config is not None
    assert trainer.reward_model.generation_config is not None
    assert trainer.value_model.generation_config is not None

    tokenizer = trainer.processing_class
    assert tokenizer is not None
    assert tokenizer.pad_token == tokenizer.eos_token
    assert tokenizer.pad_token_id == tokenizer.eos_token_id

    loaded_names = set(env["tokenizer_cls"].loaded)
    assert loaded_names, "Tokenizer inference should attempt at least one candidate"
    assert loaded_names & {"policy-model", "reference-model", "reward-model", "value-model"}

    fix_targets = set(env["fix_calls"])
    assert {"policy-model", "reference-model", "reward-model", "value-model"}.issubset(fix_targets)
