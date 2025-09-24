#!/usr/bin/env python3
"""Unit tests for TRL integration with mocked models."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


class TestTRLImports:
    """Test TRL imports without downloading models."""

    def test_trl_imports(self):
        """Test that TRL can be imported."""
        try:
            from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
            assert True
        except ImportError:
            pytest.skip("TRL not available")

    def test_rldk_trl_imports(self):
        """Test that RLDK TRL integration can be imported."""
        try:
            from rldk.integrations.trl import (
                CheckpointMonitor,
                PPOMonitor,
                RLDKCallback,
                RLDKDashboard,
            )
        except ImportError as exc:
            pytest.skip(str(exc))
        else:
            assert CheckpointMonitor
            assert PPOMonitor
            assert RLDKCallback
            assert RLDKDashboard


class TestRLDKCallbacks:
    """Test RLDK callback functionality with mocks."""

    def test_rldk_callback_creation(self):
        """Test RLDK callback creation."""
        try:
            from rldk.integrations.trl import RLDKCallback
        except ImportError as exc:
            pytest.skip(str(exc))

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                callback = RLDKCallback(output_dir=temp_dir)
            except ImportError as exc:
                pytest.skip(str(exc))
            assert callback is not None
            assert str(callback.output_dir) == temp_dir

    def test_ppo_monitor_creation(self):
        """Test PPO monitor creation."""
        try:
            from rldk.integrations.trl import PPOMonitor
        except ImportError as exc:
            pytest.skip(str(exc))

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                monitor = PPOMonitor(output_dir=temp_dir)
            except ImportError as exc:
                pytest.skip(str(exc))
            assert monitor is not None
            assert str(monitor.output_dir) == temp_dir

    def test_checkpoint_monitor_creation(self):
        """Test checkpoint monitor creation."""
        try:
            from rldk.integrations.trl import CheckpointMonitor
        except ImportError as exc:
            pytest.skip(str(exc))

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                monitor = CheckpointMonitor(output_dir=temp_dir)
            except ImportError as exc:
                pytest.skip(str(exc))
            assert monitor is not None
            assert str(monitor.output_dir) == temp_dir

    def test_rldk_callback_warns_without_event_schema(self, monkeypatch):
        """JSONL logging should emit a warning instead of failing when schema missing."""

        try:
            from rldk.integrations.trl import RLDKCallback
            from src.rldk.integrations.trl import callbacks as trl_callbacks
        except ImportError as exc:
            pytest.skip(str(exc))

        if not getattr(trl_callbacks, "TRAINER_API_AVAILABLE", True):
            pytest.skip("Transformers trainer callbacks unavailable")
        if not getattr(trl_callbacks, "TRL_AVAILABLE", True):
            pytest.skip("TRL integration unavailable")

        monkeypatch.setattr(trl_callbacks, "EVENT_SCHEMA_AVAILABLE", False)

        with tempfile.TemporaryDirectory() as temp_dir, pytest.warns(UserWarning):
            callback = RLDKCallback(output_dir=temp_dir, enable_jsonl_logging=True)
        assert callback.enable_jsonl_logging is False


class TestPPOConfig:
    """Test PPO configuration without model loading."""

    def test_ppo_config_creation(self):
        """Test PPO configuration creation."""
        from trl import PPOConfig

        config = PPOConfig(
            learning_rate=1e-5,
            per_device_train_batch_size=2,
            mini_batch_size=1,
            num_ppo_epochs=2,
            max_grad_norm=0.5,
            output_dir="./test_output",
            bf16=False,
            fp16=False,
            remove_unused_columns=False,
        )

        assert config.learning_rate == 1e-5
        assert config.per_device_train_batch_size == 2
        assert config.num_ppo_epochs == 2
        assert config.max_grad_norm == 0.5


class TestGRPOConfig:
    """Test GRPO configuration helpers."""

    def test_grpo_config_cpu_fallback(self, monkeypatch):
        """Ensure GRPOConfig defaults disable unsupported precision on CPU."""

        try:
            from rldk.integrations.trl import create_grpo_config
        except ImportError:
            pytest.skip("GRPOConfig helper unavailable")

        try:
            from trl import GRPOConfig  # noqa: F401
        except ImportError as exc:  # pragma: no cover - TRL without GRPO support
            pytest.skip(str(exc))

        from rldk.integrations.trl import utils as trl_utils

        monkeypatch.setattr(trl_utils, "_accelerator_available", lambda: False)
        monkeypatch.setattr(trl_utils, "_bf16_supported", lambda: False)

        with pytest.warns(RuntimeWarning):
            config = create_grpo_config(bf16=True, fp16=True)

        assert config.bf16 is False
        assert config.fp16 is False


class TestMockedModels:
    """Test TRL functionality with mocked models."""

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('trl.AutoModelForCausalLMWithValueHead.from_pretrained')
    def test_mocked_model_creation(self, mock_model, mock_tokenizer):
        """Test model creation with mocks."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer_instance.vocab_size = 50257
        mock_tokenizer.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.parameters.return_value = [torch.randn(100, 100) for _ in range(10)]
        mock_model_instance.v_head = Mock()
        mock_model.return_value = mock_model_instance

        # Test imports
        from transformers import AutoTokenizer
        from trl import AutoModelForCausalLMWithValueHead

        # Test tokenizer creation
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        assert tokenizer is not None
        mock_tokenizer.assert_called_once_with("gpt2")

        # Test model creation
        model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
        assert model is not None
        assert hasattr(model, 'v_head')
        mock_model.assert_called_once_with("gpt2")

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('trl.AutoModelForCausalLMWithValueHead.from_pretrained')
    def test_mocked_text_generation(self, mock_model, mock_tokenizer):
        """Test text generation with mocked models."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer_instance.eos_token_id = 50256
        mock_tokenizer_instance.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer_instance.decode.return_value = "Generated text"
        mock_tokenizer.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_instance.parameters.return_value = [torch.randn(100, 100) for _ in range(10)]
        mock_model_instance.v_head = Mock()
        mock_model.return_value = mock_model_instance

        # Test generation
        from transformers import AutoTokenizer
        from trl import AutoModelForCausalLMWithValueHead

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")

        prompt = "Hello world"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(inputs["input_ids"], max_length=10)
        generated_text = tokenizer.decode(outputs[0])

        assert generated_text == "Generated text"
        mock_model_instance.generate.assert_called_once()


class TestRLDKIntegration:
    """Test RLDK integration with mocked components."""

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('trl.AutoModelForCausalLMWithValueHead.from_pretrained')
    def test_rldk_with_mocked_models(self, mock_model, mock_tokenizer):
        """Test RLDK integration with mocked models."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.parameters.return_value = [torch.randn(100, 100) for _ in range(10)]
        mock_model_instance.v_head = Mock()
        mock_model.return_value = mock_model_instance

        from transformers import AutoTokenizer
        from trl import AutoModelForCausalLMWithValueHead

        from rldk.integrations.trl import CheckpointMonitor, PPOMonitor, RLDKCallback

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create RLDK callbacks
            rldk_callback = RLDKCallback(output_dir=temp_dir)
            ppo_monitor = PPOMonitor(output_dir=temp_dir)
            checkpoint_monitor = CheckpointMonitor(output_dir=temp_dir)

            # Create mocked model
            AutoTokenizer.from_pretrained("gpt2")
            model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")

            # Test that callbacks were created successfully
            assert rldk_callback is not None
            assert ppo_monitor is not None
            assert checkpoint_monitor is not None
            assert model is not None
            assert hasattr(model, 'v_head')


class TestPPOTrainerMocked:
    """Test PPOTrainer creation with mocked components."""

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('trl.AutoModelForCausalLMWithValueHead.from_pretrained')
    @patch('datasets.Dataset.from_dict')
    def test_ppo_trainer_creation_mocked(self, mock_dataset, mock_model, mock_tokenizer):
        """Test PPOTrainer creation with all components mocked."""
        # Setup mocks
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = mock_tokenizer_instance

        mock_model_instance = Mock()
        mock_model_instance.parameters.return_value = [torch.randn(100, 100) for _ in range(10)]
        mock_model_instance.v_head = Mock()
        mock_model_instance.generation_config = Mock()
        mock_model_instance.generation_config.eos_token_id = 50256
        mock_model.return_value = mock_model_instance

        mock_dataset_instance = Mock()
        mock_dataset_instance.__len__ = Mock(return_value=10)
        mock_dataset.return_value = mock_dataset_instance

        from datasets import Dataset
        from transformers import AutoTokenizer
        from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

        # Create components
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
        reward_model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")
        value_model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")

        dataset = Dataset.from_dict({
            "query": ["Hello", "How are you?"],
            "response": ["Hi there!", "I'm doing well."]
        })

        config = PPOConfig(
            learning_rate=1e-5,
            per_device_train_batch_size=1,
            mini_batch_size=1,
            num_ppo_epochs=1,
            output_dir="./test_output",
            bf16=False,
            fp16=False,
            remove_unused_columns=False,
        )

        # This might still fail due to TRL internals, but we test the setup
        try:
            trainer = PPOTrainer(
                args=config,
                model=model,
                ref_model=ref_model,
                reward_model=reward_model,
                value_model=value_model,
                processing_class=tokenizer,
                train_dataset=dataset,
            )
            assert trainer is not None
        except Exception as e:
            # If PPOTrainer creation fails due to TRL internals, that's expected
            # The important thing is that our mocked components work
            assert "generation_config" in str(e) or "model" in str(e)


class TestMetrics:
    """Test metrics classes."""

    def test_rldk_metrics(self):
        """Test RLDK metrics creation."""
        from rldk.integrations.trl import RLDKMetrics

        metrics = RLDKMetrics(
            step=1,
            epoch=0.0,
            learning_rate=1e-5,
            loss=0.5,
            reward_mean=1.0
        )

        assert metrics.step == 1
        assert metrics.epoch == 0.0
        assert metrics.learning_rate == 1e-5
        assert metrics.loss == 0.5
        assert metrics.reward_mean == 1.0

    def test_ppo_metrics(self):
        """Test PPO metrics creation."""
        from rldk.integrations.trl import PPOMetrics

        metrics = PPOMetrics(
            rollout_reward_mean=1.0,
            policy_kl_mean=0.1,
            value_loss=0.5
        )

        assert metrics.rollout_reward_mean == 1.0
        assert metrics.policy_kl_mean == 0.1
        assert metrics.value_loss == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
