"""Integration tests for the TRL PPO utilities using mocked components."""

from types import SimpleNamespace
import importlib
import sys
import types

import pytest


class _DummyOutput:
    """Simple output object exposing a ``logits`` attribute."""

    def __init__(self, tag: str):
        self.logits = [tag]


class _DummyTokenizer:
    """Tokenizer stub that provides the attributes accessed by the utilities."""

    def __init__(self) -> None:
        self.pad_token = None
        self.eos_token = "<eos>"
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.bos_token_id = 1

    @classmethod
    def from_pretrained(cls, _: str) -> "_DummyTokenizer":
        return cls()


class _DummyBasePolicy:
    """Minimal causal LM implementation used to simulate base policies."""

    base_model_prefix = "transformer"

    def __init__(self, name: str) -> None:
        self.name_or_path = name
        self.generation_config = None

    def __call__(self, *args, **kwargs):  # pragma: no cover - debug hook
        return _DummyOutput(self.name_or_path)

    forward = __call__

    @classmethod
    def from_pretrained(cls, name: str, *args, **kwargs) -> "_DummyBasePolicy":
        return cls(name)


class _DummyValueHeadModel(_DummyBasePolicy):
    """Value-head wrapper that mimics TRL's policy/value containers."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        base = _DummyBasePolicy(name)
        base.transformer = object()
        base.is_gradient_checkpointing = False
        self.pretrained_model = base
        self.v_head = object()

    @classmethod
    def from_pretrained(cls, name: str, *args, **kwargs) -> "_DummyValueHeadModel":
        return cls(name)


class _DummyPPOTrainer:
    """Mock PPO trainer that records policy inputs and simulates a step."""

    def __init__(
        self,
        *,
        args,
        policy,
        ref_policy,
        reward_model,
        value_model,
        processing_class,
        train_dataset,
        callbacks=None,
        **kwargs,
    ) -> None:
        self.args = args
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.value_model = value_model
        self.processing_class = processing_class
        self.train_dataset = train_dataset
        self.callbacks = callbacks or []

    def step(self) -> dict:
        """Simulate a PPO step by invoking the policy and ref policy."""

        policy_output = self.policy()
        ref_output = self.ref_policy()
        return {"policy_output": policy_output, "ref_output": ref_output}


class _DummyPolicyAndValueWrapper:  # pragma: no cover - attribute shim
    """Placeholder matching TRL's wrapper class."""


@pytest.fixture
def _install_dummy_trl(monkeypatch):
    """Install a dummy ``trl`` module so the utilities can be exercised."""

    trl_stub = types.ModuleType("trl")
    trl_stub.__version__ = "0.25.0"
    trl_stub.AutoModelForCausalLMWithValueHead = _DummyValueHeadModel
    trl_stub.PPOTrainer = _DummyPPOTrainer

    trainer_module = types.ModuleType("trl.trainer")
    ppo_trainer_module = types.ModuleType("trl.trainer.ppo_trainer")
    ppo_trainer_module.PolicyAndValueWrapper = _DummyPolicyAndValueWrapper
    ppo_trainer_module.PPOTrainer = _DummyPPOTrainer
    trainer_module.ppo_trainer = ppo_trainer_module

    transformers_stub = types.ModuleType("transformers")

    class _StubGenerationConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    transformers_stub.AutoModelForCausalLM = _DummyBasePolicy
    transformers_stub.AutoTokenizer = _DummyTokenizer
    transformers_stub.GenerationConfig = _StubGenerationConfig
    transformers_stub.PreTrainedModel = _DummyBasePolicy

    monkeypatch.setitem(sys.modules, "trl", trl_stub)
    monkeypatch.setitem(sys.modules, "trl.trainer", trainer_module)
    monkeypatch.setitem(sys.modules, "trl.trainer.ppo_trainer", ppo_trainer_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)

    yield trl_stub


@pytest.mark.usefixtures("_install_dummy_trl")
def test_create_ppo_trainer_uses_base_models(monkeypatch):
    """Ensure the new API path wires base policies while exposing logits."""

    utils = importlib.import_module("rldk.integrations.trl.utils")
    utils = importlib.reload(utils)

    monkeypatch.setattr(utils, "AutoModelForCausalLM", _DummyBasePolicy, raising=False)
    monkeypatch.setattr(utils, "AutoTokenizer", _DummyTokenizer, raising=False)
    monkeypatch.setattr(utils, "AutoModelForCausalLMWithValueHead", _DummyValueHeadModel, raising=False)
    monkeypatch.setattr(utils, "PreTrainedModel", _DummyBasePolicy, raising=False)

    trainer = utils.create_ppo_trainer(
        model_name="dummy-model",
        ppo_config=SimpleNamespace(run_name="mock-run"),
        train_dataset=[],
    )

    assert isinstance(trainer, _DummyPPOTrainer)
    assert isinstance(trainer.policy, _DummyBasePolicy)
    assert isinstance(trainer.ref_policy, _DummyBasePolicy)

    outputs = trainer.step()
    assert "ref_output" in outputs
    assert hasattr(outputs["ref_output"], "logits")
    assert hasattr(outputs["policy_output"], "logits")
