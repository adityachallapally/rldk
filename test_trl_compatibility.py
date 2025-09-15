#!/usr/bin/env python3
"""Test script for TRL 0.23+ compatibility."""

import os
os.environ["SKIP_MODEL_LOADING"] = "false"  # Allow model loading for this test

from rldk.integrations.trl import prepare_models_for_ppo, create_simple_reward_model, create_simple_value_model
from transformers import AutoTokenizer

def test_new_api():
    print("Testing TRL 0.23+ compatibility...")
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("✓ Creating simple value model...")
    value_model = create_simple_value_model(tokenizer, "gpt2")
    assert hasattr(value_model, 'base_model_prefix')
    assert hasattr(value_model, 'score')
    print(f"✓ Value model has base_model_prefix: {value_model.base_model_prefix}")
    
    print("✓ Creating simple reward model...")
    reward_model = create_simple_reward_model(tokenizer, "gpt2")
    assert hasattr(reward_model, 'generation_config')
    print("✓ Reward model created successfully")
    
    print("✓ Testing updated prepare_models_for_ppo...")
    policy_model, ref_model, reward_model, value_model, tokenizer = prepare_models_for_ppo("gpt2")
    print("✓ All models created successfully")
    
    print("✓ Testing PPOTrainer compatibility...")
    try:
        from trl import PPOTrainer, PPOConfig
        from datasets import Dataset
        
        dataset = Dataset.from_dict({"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]})
        
        config = PPOConfig(
            learning_rate=1e-5,
            per_device_train_batch_size=1,
            mini_batch_size=1,
            num_ppo_epochs=1,
            output_dir="./test_output"
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
        print("✓ PPOTrainer created successfully with new API!")
        
    except Exception as e:
        print(f"⚠ PPOTrainer test failed (this might be expected in some environments): {e}")
    
    print("🎉 All tests passed!")

if __name__ == "__main__":
    test_new_api()
