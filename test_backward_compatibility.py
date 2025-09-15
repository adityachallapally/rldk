#!/usr/bin/env python3
"""
Test script to verify backward compatibility fix for prepare_models_for_ppo.

This tests both create_separate_value_model=True and create_separate_value_model=False
to ensure the TRL 0.23+ compatibility works in both cases.
"""

import os
import tempfile
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig

from rldk.integrations.trl import prepare_models_for_ppo

def create_simple_dataset():
    """Create a simple dataset for testing."""
    prompts = ["The future of AI is", "Technology helps by"]
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    encoded = tokenizer(prompts, padding=True, truncation=True, max_length=20, return_tensors="pt")
    
    return Dataset.from_dict({
        "input_ids": encoded["input_ids"].tolist(),
        "attention_mask": encoded["attention_mask"].tolist(),
    })

def test_separate_value_model_true():
    """Test with create_separate_value_model=True (new behavior)."""
    print("🧪 Testing create_separate_value_model=True...")
    
    policy_model, ref_model, reward_model, value_model, tokenizer = prepare_models_for_ppo(
        model_name="gpt2",
        create_separate_value_model=True,
        create_separate_reward_model=True
    )
    
    assert hasattr(value_model, 'base_model_prefix'), "Value model missing base_model_prefix"
    assert hasattr(value_model, 'score'), "Value model missing score method"
    assert hasattr(value_model, 'transformer'), "Value model missing transformer attribute"
    
    print(f"✅ Value model type: {type(value_model).__name__}")
    print(f"✅ Value model base_model_prefix: {value_model.base_model_prefix}")
    
    with torch.no_grad():
        hidden_states = torch.randn(1, 10, 768)
        score_output = value_model.score(hidden_states)
        print(f"✅ Value model score output shape: {score_output.shape}")
    
    return policy_model, ref_model, reward_model, value_model, tokenizer

def test_separate_value_model_false():
    """Test with create_separate_value_model=False (backward compatibility)."""
    print("\n🧪 Testing create_separate_value_model=False (backward compatibility)...")
    
    policy_model, ref_model, reward_model, value_model, tokenizer = prepare_models_for_ppo(
        model_name="gpt2",
        create_separate_value_model=False,
        create_separate_reward_model=True
    )
    
    assert value_model is policy_model, "Value model should be the same as policy model"
    
    assert hasattr(value_model, 'base_model_prefix'), "Policy model missing base_model_prefix"
    assert hasattr(value_model, 'score'), "Policy model missing score method"
    
    print(f"✅ Value model type: {type(value_model).__name__}")
    print(f"✅ Value model base_model_prefix: {value_model.base_model_prefix}")
    print(f"✅ Value model is policy model: {value_model is policy_model}")
    
    with torch.no_grad():
        hidden_states = torch.randn(1, 10, 768)
        score_output = value_model.score(hidden_states)
        print(f"✅ Value model score output shape: {score_output.shape}")
    
    return policy_model, ref_model, reward_model, value_model, tokenizer

def test_ppo_trainer_compatibility(models, scenario_name):
    """Test PPOTrainer compatibility with the models."""
    policy_model, ref_model, reward_model, value_model, tokenizer = models
    
    print(f"\n🎯 Testing PPOTrainer compatibility ({scenario_name})...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = create_simple_dataset()
        config = PPOConfig(
            learning_rate=1e-5,
            per_device_train_batch_size=1,
            mini_batch_size=1,
            num_ppo_epochs=1,
            output_dir=temp_dir,
            bf16=False,
            fp16=False,
        )
        
        try:
            trainer = PPOTrainer(
                args=config,
                model=policy_model,
                ref_model=ref_model,
                reward_model=reward_model,
                value_model=value_model,
                processing_class=tokenizer,
                train_dataset=dataset,
            )
            
            print(f"✅ PPOTrainer created successfully for {scenario_name}")
            print(f"  • Trainer type: {type(trainer).__name__}")
            print(f"  • Model wrapper: {type(trainer.model).__name__}")
            
            return True
            
        except Exception as e:
            print(f"❌ PPOTrainer failed for {scenario_name}: {e}")
            return False

if __name__ == "__main__":
    print("🎯 Testing Backward Compatibility Fix")
    print("=" * 80)
    
    try:
        models_separate = test_separate_value_model_true()
        success_separate = test_ppo_trainer_compatibility(models_separate, "separate value model")
        
        models_backward = test_separate_value_model_false()
        success_backward = test_ppo_trainer_compatibility(models_backward, "backward compatibility")
        
        if success_separate and success_backward:
            print("\n" + "=" * 80)
            print("🎉 BACKWARD COMPATIBILITY FIX SUCCESSFUL!")
            print("=" * 80)
            print("✅ create_separate_value_model=True: WORKING")
            print("✅ create_separate_value_model=False: WORKING")
            print("✅ TRL 0.23+ compatibility: CONFIRMED")
            print("✅ Backward compatibility: FIXED")
        else:
            print("\n❌ Some tests failed - check output above")
            exit(1)
            
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
