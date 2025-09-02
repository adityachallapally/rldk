#!/usr/bin/env python3
"""Real TRL testing with actual models and training."""

import os
import sys
import time
import torch
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_trl_imports():
    """Test that TRL can be imported and basic classes work."""
    print("🔍 Testing TRL imports...")
    
    try:
        from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✅ TRL imports successful")
        return True
    except ImportError as e:
        print(f"❌ TRL import failed: {e}")
        return False

def test_model_download():
    """Test downloading and loading a small model."""
    print("\n📥 Testing model download...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Use a small model for testing
        model_name = "gpt2"
        print(f"📦 Downloading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        print(f"✅ Model downloaded successfully")
        print(f"   - Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"   - Vocabulary size: {tokenizer.vocab_size:,}")
        
        return tokenizer, model, model_name
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return None, None, None

def test_ppo_model_creation():
    """Test creating a PPO model with value head."""
    print("\n🎯 Testing PPO model creation...")
    
    try:
        from trl import AutoModelForCausalLMWithValueHead
        from transformers import AutoTokenizer
        
        model_name = "gpt2"
        print(f"📦 Creating PPO model from {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        
        print(f"✅ PPO model created successfully")
        print(f"   - Model type: {type(model).__name__}")
        print(f"   - Has value head: {hasattr(model, 'v_head')}")
        
        return tokenizer, model
    except Exception as e:
        print(f"❌ PPO model creation failed: {e}")
        return None, None

def test_ppo_config():
    """Test PPO configuration."""
    print("\n⚙️ Testing PPO configuration...")
    
    try:
        from trl import PPOConfig
        
        config = PPOConfig(
            learning_rate=1e-5,
            per_device_train_batch_size=2,
            mini_batch_size=1,
            num_ppo_epochs=1,
            max_grad_norm=0.5,
            output_dir="./test_ppo_output",
            bf16=False,
            fp16=False,
            remove_unused_columns=False,
        )
        
        print(f"✅ PPO config created successfully")
        print(f"   - Learning rate: {config.learning_rate}")
        print(f"   - Batch size: {config.per_device_train_batch_size}")
        print(f"   - PPO epochs: {config.num_ppo_epochs}")
        
        return config
    except Exception as e:
        print(f"❌ PPO config creation failed: {e}")
        return None

def test_rldk_integration():
    """Test RLDK integration with TRL."""
    print("\n🔗 Testing RLDK-TRL integration...")
    
    try:
        from rldk.integrations.trl import RLDKCallback, PPOMonitor, CheckpointMonitor
        
        # Create RLDK callbacks
        rldk_callback = RLDKCallback(output_dir="./test_rldk_output")
        ppo_monitor = PPOMonitor(output_dir="./test_rldk_output")
        checkpoint_monitor = CheckpointMonitor(output_dir="./test_rldk_output")
        
        print(f"✅ RLDK callbacks created successfully")
        print(f"   - RLDKCallback: {type(rldk_callback).__name__}")
        print(f"   - PPOMonitor: {type(ppo_monitor).__name__}")
        print(f"   - CheckpointMonitor: {type(checkpoint_monitor).__name__}")
        
        return [rldk_callback, ppo_monitor, checkpoint_monitor]
    except Exception as e:
        print(f"❌ RLDK integration failed: {e}")
        return None

def test_simple_generation():
    """Test simple text generation with the model."""
    print("\n💬 Testing text generation...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Simple generation test
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
        
        print(f"✅ Text generation successful")
        print(f"   - Prompt: '{prompt}'")
        print(f"   - Generated: '{generated_text}'")
        
        return True
    except Exception as e:
        print(f"❌ Text generation failed: {e}")
        return False

def test_ppo_trainer_creation():
    """Test creating a PPOTrainer (without actually training)."""
    print("\n🚀 Testing PPOTrainer creation...")
    
    try:
        from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
        from transformers import AutoTokenizer
        from datasets import Dataset
        
        # Create minimal components
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        
        # Create a minimal dataset
        dataset = Dataset.from_dict({
            "query": ["Hello", "How are you?"],
            "response": ["Hi there!", "I'm doing well."]
        })
        
        # Create config
        config = PPOConfig(
            learning_rate=1e-5,
            per_device_train_batch_size=1,
            mini_batch_size=1,
            num_ppo_epochs=1,
            output_dir="./test_ppo_output",
            bf16=False,
            fp16=False,
            remove_unused_columns=False,
        )
        
        # Create dummy reward and value models (same as main model for testing)
        reward_model = model
        value_model = model
        ref_model = model
        
        # Create PPOTrainer
        trainer = PPOTrainer(
            args=config,
            model=model,
            processing_class=tokenizer,
            ref_model=ref_model,
            reward_model=reward_model,
            value_model=value_model,
            train_dataset=dataset,
        )
        
        print(f"✅ PPOTrainer created successfully")
        print(f"   - Model: {type(trainer.model).__name__}")
        print(f"   - Dataset size: {len(trainer.train_dataset)}")
        
        return trainer
    except Exception as e:
        print(f"❌ PPOTrainer creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run all tests."""
    print("🎯 TRL Real Testing Suite")
    print("=" * 50)
    
    results = {}
    
    # Test 1: Imports
    results["imports"] = test_trl_imports()
    
    # Test 2: Model download
    tokenizer, model, model_name = test_model_download()
    results["model_download"] = tokenizer is not None
    
    # Test 3: PPO model creation
    ppo_tokenizer, ppo_model = test_ppo_model_creation()
    results["ppo_model"] = ppo_model is not None
    
    # Test 4: PPO config
    config = test_ppo_config()
    results["ppo_config"] = config is not None
    
    # Test 5: RLDK integration
    callbacks = test_rldk_integration()
    results["rldk_integration"] = callbacks is not None
    
    # Test 6: Simple generation
    results["generation"] = test_simple_generation()
    
    # Test 7: PPOTrainer creation
    trainer = test_ppo_trainer_creation()
    results["ppo_trainer"] = trainer is not None
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title():<20} {status}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! TRL is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    import shutil
    for dir_name in ["./test_ppo_output", "./test_rldk_output"]:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"✅ Cleaned up {dir_name}")

if __name__ == "__main__":
    main()