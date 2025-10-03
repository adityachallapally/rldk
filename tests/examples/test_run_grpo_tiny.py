"""Regression tests for :mod:`examples.run_grpo_tiny`."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="global fixtures require numpy")


@pytest.fixture()
def trl_stub(monkeypatch: pytest.MonkeyPatch):
    """Provide a minimal TRL stub without ``add_model_tags`` support."""

    stub = types.ModuleType("trl")

    class DummyValueHeadModel:
        """Model lacking ``add_model_tags`` to mimic older TRL releases."""

        base_model_prefix = "transformer"

        def __init__(self) -> None:
            self.pretrained_model = types.SimpleNamespace(
                base_model_prefix="transformer",
                transformer=object(),
                is_gradient_checkpointing=False,
            )
            self.v_head = object()
            self.generation_config = None
            self.name_or_path = "dummy"

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            pretrained_model = _kwargs.get("pretrained_model")
            instance = cls()
            if pretrained_model is not None:
                instance.pretrained_model = pretrained_model
            return instance

    class DummyGRPOTrainer:
        """Trainer that verifies ``add_model_tags`` is safely available."""

        def __init__(
            self,
            *,
            args,
            model,
            reward_funcs,
            processing_class,
            train_dataset,
            callbacks,
        ) -> None:
            self.args = args
            self.model = model
            self.reward_funcs = reward_funcs
            self.processing_class = processing_class
            self.train_dataset = train_dataset
            self.callbacks = callbacks
            self._trained = False

            # Newer TRL releases invoke ``add_model_tags`` during setup. Ensure the
            # shim installed by ``build_grpo_trainer`` prevents AttributeErrors.
            model.add_model_tags(["grpo"])

        def train(self) -> None:
            self._trained = True

    stub.AutoModelForCausalLMWithValueHead = DummyValueHeadModel
    stub.GRPOTrainer = DummyGRPOTrainer

    monkeypatch.setitem(sys.modules, "trl", stub)

    # Reload helper modules so they pick up the stub instead of real TRL.
    import rldk.integrations.trl.utils as trl_utils

    importlib.reload(trl_utils)
    import rldk.integrations.trl as trl_integration

    importlib.reload(trl_integration)

    try:
        yield stub
    finally:
        for module_name in ("rldk.integrations.trl.utils", "rldk.integrations.trl"):
            sys.modules.pop(module_name, None)


def test_build_grpo_trainer_shims_missing_add_model_tags(trl_stub, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import examples.run_grpo_tiny as run_grpo_tiny

    importlib.reload(run_grpo_tiny)

    # Avoid importing heavy dependencies that the example expects in production.
    class DummyCallback:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - trivial stub
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(run_grpo_tiny, "EventWriterCallback", DummyCallback)

    tokenizer = types.SimpleNamespace(
        eos_token_id=0,
        pad_token_id=0,
        bos_token_id=1,
    )
    dataset = [{"prompt": "hi", "reference_response": "hello", "accepted": True}]
    grpo_config = types.SimpleNamespace(run_name="test-run")
    event_log_path = tmp_path / "events.jsonl"

    trainer = run_grpo_tiny.build_grpo_trainer(
        model_name="dummy",
        grpo_config=grpo_config,
        dataset=dataset,
        tokenizer=tokenizer,
        event_log_path=event_log_path,
    )

    trainer.train()

    assert getattr(trainer, "_trained", False) is True

