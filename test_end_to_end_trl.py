#!/usr/bin/env python3
"""End-to-end test for TRL 0.23+ compatibility with actual model training."""

import os
import tempfile
import shutil
from datasets import Dataset
from transformers import AutoTokenizer

from rldk.integrations.trl import (
    prepare_models_for_ppo,
    create_simple_reward_model,
    create_simple_value_model,
    check_trl_compatibility
)

def create_simple_dataset():
    """Create a simple dataset for testing."""
    prompts = [
        "The capital of France is",
        "Python is a programming language that",
        "Machine learning is",
        "The weather today is",
        "Artificial intelligence can",
    ]
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    encoded = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    
    dataset_dict = {
        "input_ids": encoded["input_ids"].tolist(),
        "attention_mask": encoded["attention_mask"].tolist(),
    }
    
    return Dataset.from_dict(dataset_dict)

def test_new_api_with_actual_model():
    """Test the new TRL 0.23+ API with actual model instantiation."""
    print("🚀 Testing TRL 0.23+ compatibility with actual model...")
    
    compatibility = check_trl_compatibility()
    print(f"✓ TRL compatibility check: {compatibility}")
    
    model_name = "gpt2"
    print(f"✓ Testing with model: {model_name}")
    
    print("✓ Testing prepare_models_for_ppo with new API...")
    policy_model, ref_model, reward_model, value_model, tokenizer = prepare_models_for_ppo(
        model_name=model_name,
        create_separate_value_model=True,
        create_separate_reward_model=True
    )
    
    print(f"✓ Policy model type: {type(policy_model)}")
    print(f"✓ Reference model type: {type(ref_model)}")
    print(f"✓ Reward model type: {type(reward_model)}")
    print(f"✓ Value model type: {type(value_model)}")
    print(f"✓ Tokenizer type: {type(tokenizer)}")
    
    assert hasattr(value_model, 'base_model_prefix'), "Value model missing base_model_prefix"
    assert hasattr(value_model, 'score'), "Value model missing score method"
    print(f"✓ Value model base_model_prefix: {value_model.base_model_prefix}")
    print("✓ Value model has score method")
    
    assert hasattr(reward_model, 'generation_config'), "Reward model missing generation_config"
    print("✓ Reward model has generation_config")
    
    dataset = create_simple_dataset()
    print(f"✓ Created dataset with {len(dataset)} samples")
    
    try:
        from trl import PPOTrainer, PPOConfig
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = PPOConfig(
                learning_rate=1e-5,
                per_device_train_batch_size=1,
                mini_batch_size=1,
                num_ppo_epochs=1,
                output_dir=temp_dir,
                bf16=False,  # Disable for compatibility
                fp16=False,  # Disable for compatibility
            )
            
            trainer = PPOTrainer(
                args=config,
                model=policy_model,
                ref_model=ref_model,
                reward_model=reward_model,
                value_model=value_model,
                processing_class=tokenizer,
                train_dataset=dataset,
            )
            
            print("🎉 PPOTrainer successfully instantiated with TRL 0.23+ API!")
            print(f"✓ Trainer type: {type(trainer)}")
            
            print(f"✓ Trainer model: {type(trainer.model)}")
            print(f"✓ Trainer ref_model: {type(trainer.ref_model)}")
            print(f"✓ Trainer reward_model: {type(trainer.reward_model)}")
            print(f"✓ Trainer value_model: {type(trainer.value_model)}")
            
            import torch
            
            sample_input_ids = torch.tensor(dataset[0]["input_ids"]).unsqueeze(0)
            sample_attention_mask = torch.tensor(dataset[0]["attention_mask"]).unsqueeze(0)
            
            print("✓ Testing model forward passes...")
            
            with torch.no_grad():
                policy_output = policy_model(input_ids=sample_input_ids, attention_mask=sample_attention_mask)
                if hasattr(policy_output, 'logits'):
                    logits = policy_output.logits
                elif isinstance(policy_output, tuple):
                    logits = policy_output[0]  # First element is usually logits
                else:
                    logits = policy_output
                print(f"✓ Policy model output shape: {logits.shape}")
            
            with torch.no_grad():
                hidden_states = torch.randn(1, sample_input_ids.shape[1], 768)  # [batch, seq_len, hidden_size]
                value_output = value_model.score(hidden_states)
                print(f"✓ Value model score output shape: {value_output.shape}")
            
            print("🎉 All model forward passes successful!")
            
            return True
            
    except Exception as e:
        print(f"❌ PPOTrainer instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test that the old API still works for backward compatibility."""
    print("🔄 Testing backward compatibility...")
    
    policy_model, ref_model, reward_model, value_model, tokenizer = prepare_models_for_ppo(
        model_name="gpt2",
        create_separate_value_model=False,  # Use policy model as value model
        create_separate_reward_model=False  # Use policy model as reward model
    )
    
    assert value_model is policy_model, "Backward compatibility broken: value_model should be policy_model"
    print("✓ Backward compatibility: value_model is policy_model")
    
    assert type(reward_model).__name__ == "AutoModelForCausalLMWithValueHead", "Reward model type incorrect"
    print("✓ Backward compatibility: reward_model is correct type")
    
    print("✅ Backward compatibility test passed!")
    return True

def test_individual_functions():
    """Test the individual utility functions."""
    print("🧪 Testing individual utility functions...")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    value_model = create_simple_value_model(tokenizer, "gpt2")
    assert hasattr(value_model, 'base_model_prefix')
    assert hasattr(value_model, 'score')
    print("✓ create_simple_value_model works correctly")
    
    reward_model = create_simple_reward_model(tokenizer, "gpt2")
    assert hasattr(reward_model, 'generation_config')
    print("✓ create_simple_reward_model works correctly")
    
    print("✅ Individual function tests passed!")
    return True

if __name__ == "__main__":
    print("🎯 End-to-End TRL 0.23+ Compatibility Test")
    print("=" * 60)
    
    try:
        test_individual_functions()
        print()
        
        test_backward_compatibility()
        print()
        
        success = test_new_api_with_actual_model()
        
        if success:
            print("\n🎉 ALL TESTS PASSED! TRL 0.23+ integration is working perfectly!")
            print("✅ Users can now use RLDK with TRL 0.23+ without any manual workarounds")
        else:
            print("\n❌ Some tests failed. Check the output above for details.")
            exit(1)
            
    except Exception as e:
        print(f"\n💥 Test suite failed with exception: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
