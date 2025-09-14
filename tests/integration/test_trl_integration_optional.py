#!/usr/bin/env python3
"""Optional integration tests for TRL with real models.

These tests download real models and should only be run when explicitly requested.
Use: python -m pytest test_trl_integration_optional.py -m integration
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


@pytest.mark.integration
@pytest.mark.slow
class TestTRLRealModels:
    """Integration tests with real model downloads."""

    def test_gpt2_model_download_and_generation(self):
        """Test downloading GPT-2 and generating text."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Test generation
        prompt = "The future of AI is"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=20,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        assert len(generated_text) > len(prompt)
        assert "AI" in generated_text or "artificial" in generated_text.lower()

    def test_ppo_model_creation(self):
        """Test creating PPO model with value head."""
        from transformers import AutoTokenizer
        from trl import AutoModelForCausalLMWithValueHead

        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

        assert hasattr(model, 'v_head')
        assert model.v_head is not None

        # Test forward pass
        prompt = "Hello world"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        assert 'logits' in outputs
        assert outputs.logits.shape[0] == 1  # batch size 1

    def test_rldk_integration_with_real_models(self):
        """Test RLDK integration with real models."""
        from transformers import AutoTokenizer
        from trl import AutoModelForCausalLMWithValueHead

        from rldk.integrations.trl import CheckpointMonitor, PPOMonitor, RLDKCallback

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create RLDK callbacks
            rldk_callback = RLDKCallback(output_dir=temp_dir)
            ppo_monitor = PPOMonitor(output_dir=temp_dir)
            checkpoint_monitor = CheckpointMonitor(output_dir=temp_dir)

            # Create real model
            model_name = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

            # Test that everything works together
            assert rldk_callback is not None
            assert ppo_monitor is not None
            assert checkpoint_monitor is not None
            assert model is not None
            assert hasattr(model, 'v_head')

            # Test model forward pass
            prompt = "Test prompt"
            inputs = tokenizer(prompt, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            assert 'logits' in outputs


@pytest.mark.integration
@pytest.mark.slow
class TestMultipleModels:
    """Test with multiple model sizes."""

    @pytest.mark.parametrize("model_name,expected_params", [
        ("gpt2", 124_000_000),  # ~124M parameters
        ("distilgpt2", 81_000_000),  # ~82M parameters
    ])
    def test_model_sizes(self, model_name, expected_params):
        """Test different model sizes."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_name)

        param_count = sum(p.numel() for p in model.parameters())

        # Allow for some variance in parameter count
        assert abs(param_count - expected_params) < expected_params * 0.1

        # Test generation
        prompt = "AI will"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=10,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assert len(generated_text) > len(prompt)


if __name__ == "__main__":
    # Only run integration tests if explicitly requested
    import sys
    if "--integration" in sys.argv:
        pytest.main([__file__, "-m", "integration", "-v"])
    else:
        print("Integration tests not run. Use --integration flag to run them.")
        print("These tests download real models and require network access.")
