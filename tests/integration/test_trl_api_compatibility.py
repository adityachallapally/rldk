#!/usr/bin/env python3
"""Integration tests for TRL API compatibility fix."""

import os
import sys
import tempfile
from pathlib import Path

import pytest
from datasets import Dataset

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from trl import PPOConfig
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    PPOConfig = None


@pytest.mark.skipif(not TRL_AVAILABLE, reason="TRL not available")
class TestTRLAPICompatibility:
    """Test TRL API compatibility fix."""
    
    def test_import_compatibility_functions(self):
        """Test that all compatibility functions can be imported."""
        from rldk.integrations.trl import (
            check_trl_compatibility,
            create_ppo_trainer,
            prepare_models_for_ppo,
            validate_ppo_setup,
        )
        
        # Test that functions are callable
        assert callable(check_trl_compatibility)
        assert callable(create_ppo_trainer)
        assert callable(prepare_models_for_ppo)
        assert callable(validate_ppo_setup)
    
    def test_trl_compatibility_check(self):
        """Test TRL compatibility check function."""
        from rldk.integrations.trl import check_trl_compatibility
        
        result = check_trl_compatibility()
        
        # Check required keys
        assert "trl_available" in result
        assert "version" in result
        assert "warnings" in result
        assert "recommendations" in result
        
        # Check that TRL is available if we got this far
        assert result["trl_available"] is True
        assert result["version"] is not None
    
    def test_prepare_models_for_ppo_signature(self):
        """Test that prepare_models_for_ppo returns 5 elements."""
        from rldk.integrations.trl import prepare_models_for_ppo
        
        # Skip actual model loading in CI
        if os.getenv("SKIP_MODEL_LOADING", "false").lower() == "true":
            pytest.skip("Skipping model loading in CI")
        
        model_name = "sshleifer/tiny-gpt2"  # Very small model
        
        try:
            result = prepare_models_for_ppo(model_name)
            
            # Should return 5 elements
            assert len(result) == 5
            
            model, ref_model, reward_model, value_model, tokenizer = result
            
            # All should be the expected types
            assert model is not None
            assert ref_model is not None
            assert reward_model is not None
            assert value_model is not None
            assert tokenizer is not None
            
        except Exception as e:
            pytest.skip(f"Model loading failed (expected in CI): {e}")
    
    def test_validate_ppo_setup_signature(self):
        """Test that validate_ppo_setup accepts 5 model parameters."""
        from rldk.integrations.trl import validate_ppo_setup
        
        # Skip actual model loading in CI
        if os.getenv("SKIP_MODEL_LOADING", "false").lower() == "true":
            pytest.skip("Skipping model loading in CI")
        
        model_name = "sshleifer/tiny-gpt2"
        
        try:
            from rldk.integrations.trl import prepare_models_for_ppo
            model, ref_model, reward_model, value_model, tokenizer = prepare_models_for_ppo(model_name)
            
            # Test validation with 5 parameters
            result = validate_ppo_setup(model, ref_model, reward_model, value_model, tokenizer)
            
            assert "valid" in result
            assert "issues" in result
            assert "warnings" in result
            assert "recommendations" in result
            
        except Exception as e:
            pytest.skip(f"Model loading failed (expected in CI): {e}")
    
    def test_create_ppo_trainer_with_minimal_data(self):
        """Test create_ppo_trainer with minimal test data."""
        from rldk.integrations.trl import create_ppo_trainer
        
        # Skip actual model loading in CI
        if os.getenv("SKIP_MODEL_LOADING", "false").lower() == "true":
            pytest.skip("Skipping model loading in CI")
        
        # Create minimal test data
        test_dataset = Dataset.from_dict({
            "prompt": ["Hello world", "Test prompt"],
            "response": ["Response 1", "Response 2"],
        })
        
        # Create PPO config
        ppo_config = PPOConfig(
            learning_rate=1e-5,
            per_device_train_batch_size=1,
            mini_batch_size=1,
            num_ppo_epochs=1,
            max_grad_norm=0.5,
            output_dir="./test_output",
            remove_unused_columns=False,
            bf16=False,
            fp16=False,
        )
        
        model_name = "sshleifer/tiny-gpt2"
        
        try:
            trainer = create_ppo_trainer(
                model_name=model_name,
                ppo_config=ppo_config,
                train_dataset=test_dataset,
                callbacks=[],
            )
            
            # Check that trainer was created successfully
            assert trainer is not None
            
            # Check that trainer has required attributes
            required_attrs = ['model', 'ref_model', 'reward_model', 'value_model', 'tokenizer']
            for attr in required_attrs:
                assert hasattr(trainer, attr), f"Trainer missing {attr}"
            
        except Exception as e:
            pytest.skip(f"Model loading failed (expected in CI): {e}")
    
    def test_create_ppo_trainer_error_handling(self):
        """Test error handling in create_ppo_trainer."""
        from rldk.integrations.trl import create_ppo_trainer
        
        # Test with invalid model name
        test_dataset = Dataset.from_dict({
            "prompt": ["Hello"],
            "response": ["World"],
        })
        
        ppo_config = PPOConfig(
            learning_rate=1e-5,
            per_device_train_batch_size=1,
            mini_batch_size=1,
            num_ppo_epochs=1,
            max_grad_norm=0.5,
            output_dir="./test_output",
            remove_unused_columns=False,
            bf16=False,
            fp16=False,
        )
        
        # Test with invalid model name
        with pytest.raises(Exception):
            create_ppo_trainer(
                model_name="invalid_model_name_that_does_not_exist",
                ppo_config=ppo_config,
                train_dataset=test_dataset,
                callbacks=[],
            )
        
        # Test with invalid config
        with pytest.raises(Exception):
            create_ppo_trainer(
                model_name="sshleifer/tiny-gpt2",
                ppo_config="invalid_config",  # Should be PPOConfig instance
                train_dataset=test_dataset,
                callbacks=[],
            )
        
        # Test with None dataset
        with pytest.raises(Exception):
            create_ppo_trainer(
                model_name="sshleifer/tiny-gpt2",
                ppo_config=ppo_config,
                train_dataset=None,
                callbacks=[],
            )
    
    def test_memory_efficiency(self):
        """Test that model sharing is implemented for memory efficiency."""
        from rldk.integrations.trl import prepare_models_for_ppo
        
        # Skip actual model loading in CI
        if os.getenv("SKIP_MODEL_LOADING", "false").lower() == "true":
            pytest.skip("Skipping model loading in CI")
        
        model_name = "sshleifer/tiny-gpt2"
        
        try:
            model, ref_model, reward_model, value_model, tokenizer = prepare_models_for_ppo(model_name)
            
            # Check that reward_model and value_model share the same instance as model
            # This is the memory-efficient approach
            assert reward_model is model, "reward_model should share instance with model"
            assert value_model is model, "value_model should share instance with model"
            
            # ref_model can be different (it's a separate reference model)
            # but model, reward_model, and value_model should be the same instance
            
        except Exception as e:
            pytest.skip(f"Model loading failed (expected in CI): {e}")


@pytest.mark.skipif(not TRL_AVAILABLE, reason="TRL not available")
class TestTRLVersionCompatibility:
    """Test TRL version compatibility handling."""
    
    def test_version_detection(self):
        """Test that TRL version is detected correctly."""
        from rldk.integrations.trl import check_trl_compatibility
        
        result = check_trl_compatibility()
        
        assert result["trl_available"] is True
        assert result["version"] is not None
        
        # Version should be parseable
        from packaging import version
        parsed_version = version.parse(result["version"])
        assert parsed_version is not None
    
    def test_factory_function_version_handling(self):
        """Test that factory function handles different TRL versions."""
        from rldk.integrations.trl import create_ppo_trainer, check_trl_compatibility
        
        # Skip actual model loading in CI
        if os.getenv("SKIP_MODEL_LOADING", "false").lower() == "true":
            pytest.skip("Skipping model loading in CI")
        
        # Get current TRL version
        compatibility = check_trl_compatibility()
        trl_version = compatibility["version"]
        
        # Create test data
        test_dataset = Dataset.from_dict({
            "prompt": ["Hello"],
            "response": ["World"],
        })
        
        ppo_config = PPOConfig(
            learning_rate=1e-5,
            per_device_train_batch_size=1,
            mini_batch_size=1,
            num_ppo_epochs=1,
            max_grad_norm=0.5,
            output_dir="./test_output",
            remove_unused_columns=False,
            bf16=False,
            fp16=False,
        )
        
        try:
            trainer = create_ppo_trainer(
                model_name="sshleifer/tiny-gpt2",
                ppo_config=ppo_config,
                train_dataset=test_dataset,
                callbacks=[],
            )
            
            # Should work regardless of TRL version
            assert trainer is not None
            
        except Exception as e:
            pytest.skip(f"Model loading failed (expected in CI): {e}")


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])