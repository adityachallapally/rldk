#!/usr/bin/env python3
"""Test TRL with different models and configurations."""

import os
import sys
import time
import torch
import copy
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_multiple_models():
    """Test TRL with different model sizes."""
    print("🤖 Testing TRL with multiple models...")
    
    models_to_test = [
        "gpt2",           # Small model (124M parameters)
        "distilgpt2",     # Even smaller model (82M parameters)
    ]
    
    results = {}
    
    for model_name in models_to_test:
        print(f"\n📦 Testing with {model_name}...")
        
        try:
            from trl import AutoModelForCausalLMWithValueHead
            from transformers import AutoTokenizer
            
            # Download and test model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
            
            # Test text generation
            prompt = "The future of AI is"
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=15,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            param_count = sum(p.numel() for p in model.parameters())
            
            print(f"✅ {model_name} working")
            print(f"   - Parameters: {param_count:,}")
            print(f"   - Generated: '{generated_text}'")
            
            results[model_name] = {
                "success": True,
                "parameters": param_count,
                "generated_text": generated_text
            }
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            results[model_name] = {"success": False, "error": str(e)}
    
    return results

def test_ppo_trainer_with_copy():
    """Test PPOTrainer creation with proper model copying."""
    print("\n🚀 Testing PPOTrainer with proper model setup...")
    
    try:
        from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
        from transformers import AutoTokenizer
        from datasets import Dataset
        
        model_name = "distilgpt2"  # Use smaller model
        print(f"📦 Using {model_name} for PPO test...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create main model
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        
        # Create reference model (copy of main model)
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        
        # Create reward and value models (copies)
        reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        value_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        
        # Create dataset
        dataset = Dataset.from_dict({
            "query": ["Hello", "How are you?", "What is AI?"],
            "response": ["Hi there!", "I'm doing well.", "AI is artificial intelligence."]
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
        
        print(f"✅ PPOTrainer created successfully with {model_name}")
        print(f"   - Model type: {type(trainer.model).__name__}")
        print(f"   - Dataset size: {len(trainer.train_dataset)}")
        print(f"   - Reference model: {type(trainer.ref_model).__name__}")
        
        return trainer
        
    except Exception as e:
        print(f"❌ PPOTrainer creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_rldk_with_different_models():
    """Test RLDK integration with different models."""
    print("\n🔗 Testing RLDK integration with different models...")
    
    try:
        from rldk.integrations.trl import RLDKCallback, PPOMonitor, CheckpointMonitor
        from trl import AutoModelForCausalLMWithValueHead
        from transformers import AutoTokenizer
        
        models_to_test = ["gpt2", "distilgpt2"]
        results = {}
        
        for model_name in models_to_test:
            print(f"📦 Testing RLDK with {model_name}...")
            
            try:
                # Create model and tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokenizer.pad_token = tokenizer.eos_token
                model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
                
                # Create RLDK callbacks
                rldk_callback = RLDKCallback(output_dir=f"./test_rldk_{model_name}")
                ppo_monitor = PPOMonitor(output_dir=f"./test_rldk_{model_name}")
                checkpoint_monitor = CheckpointMonitor(output_dir=f"./test_rldk_{model_name}")
                
                # Test callback initialization
                callbacks = [rldk_callback, ppo_monitor, checkpoint_monitor]
                
                print(f"✅ RLDK callbacks created for {model_name}")
                print(f"   - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
                
                results[model_name] = {
                    "success": True,
                    "callbacks": len(callbacks),
                    "model_params": sum(p.numel() for p in model.parameters())
                }
                
            except Exception as e:
                print(f"❌ RLDK test failed for {model_name}: {e}")
                results[model_name] = {"success": False, "error": str(e)}
        
        return results
        
    except Exception as e:
        print(f"❌ RLDK integration test failed: {e}")
        return None

def test_ppo_training_simulation():
    """Simulate a minimal PPO training step."""
    print("\n🎯 Testing PPO training simulation...")
    
    try:
        from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
        from transformers import AutoTokenizer
        from datasets import Dataset
        
        model_name = "distilgpt2"
        print(f"📦 Simulating PPO training with {model_name}...")
        
        # Setup
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        value_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        
        # Create dataset
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
            output_dir="./test_ppo_simulation",
            bf16=False,
            fp16=False,
            remove_unused_columns=False,
            max_steps=1,  # Just one step for testing
        )
        
        # Create trainer
        trainer = PPOTrainer(
            args=config,
            model=model,
            processing_class=tokenizer,
            ref_model=ref_model,
            reward_model=reward_model,
            value_model=value_model,
            train_dataset=dataset,
        )
        
        print(f"✅ PPOTrainer created for simulation")
        
        # Test that we can access trainer properties
        print(f"   - Model device: {next(trainer.model.parameters()).device}")
        print(f"   - Training dataset size: {len(trainer.train_dataset)}")
        print(f"   - Config learning rate: {trainer.args.learning_rate}")
        
        # Test model forward pass (without actual training)
        sample_input = tokenizer("Test input", return_tensors="pt")
        with torch.no_grad():
            outputs = trainer.model(**sample_input)
        
        print(f"   - Model forward pass successful")
        print(f"   - Output shape: {outputs.logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ PPO training simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_usage():
    """Test memory usage with different models."""
    print("\n💾 Testing memory usage...")
    
    try:
        import psutil
        import torch
        
        def get_memory_usage():
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        
        models_to_test = ["gpt2", "distilgpt2"]
        results = {}
        
        for model_name in models_to_test:
            print(f"📦 Testing memory usage with {model_name}...")
            
            initial_memory = get_memory_usage()
            
            try:
                from trl import AutoModelForCausalLMWithValueHead
                from transformers import AutoTokenizer
                
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
                
                after_load_memory = get_memory_usage()
                memory_increase = after_load_memory - initial_memory
                
                param_count = sum(p.numel() for p in model.parameters())
                
                print(f"✅ {model_name} memory test")
                print(f"   - Parameters: {param_count:,}")
                print(f"   - Memory increase: {memory_increase:.1f} MB")
                
                results[model_name] = {
                    "success": True,
                    "parameters": param_count,
                    "memory_increase_mb": memory_increase
                }
                
                # Clean up
                del model, tokenizer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"❌ Memory test failed for {model_name}: {e}")
                results[model_name] = {"success": False, "error": str(e)}
        
        return results
        
    except Exception as e:
        print(f"❌ Memory usage test failed: {e}")
        return None

def main():
    """Run comprehensive TRL model tests."""
    print("🎯 TRL Multi-Model Testing Suite")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Multiple models
    results["multiple_models"] = test_multiple_models()
    
    # Test 2: PPO trainer with proper setup
    trainer = test_ppo_trainer_with_copy()
    results["ppo_trainer"] = trainer is not None
    
    # Test 3: RLDK with different models
    results["rldk_models"] = test_rldk_with_different_models()
    
    # Test 4: PPO training simulation
    results["ppo_simulation"] = test_ppo_training_simulation()
    
    # Test 5: Memory usage
    results["memory_usage"] = test_memory_usage()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    
    # Count successful tests
    successful_tests = 0
    total_tests = 0
    
    for test_name, result in results.items():
        if isinstance(result, dict) and "success" in str(result):
            # Handle dict results (multiple models, etc.)
            if isinstance(result, dict) and any(isinstance(v, dict) for v in result.values()):
                # Multiple model results
                for model_name, model_result in result.items():
                    if isinstance(model_result, dict) and "success" in model_result:
                        total_tests += 1
                        if model_result["success"]:
                            successful_tests += 1
                        status = "✅ PASSED" if model_result["success"] else "❌ FAILED"
                        print(f"{test_name} ({model_name}): {status}")
            else:
                # Single result
                total_tests += 1
                if result.get("success", False):
                    successful_tests += 1
                status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
                print(f"{test_name.replace('_', ' ').title():<25} {status}")
        else:
            # Boolean results
            total_tests += 1
            if result:
                successful_tests += 1
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name.replace('_', ' ').title():<25} {status}")
    
    print(f"\nResults: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests == total_tests:
        print("🎉 All tests passed! TRL is working excellently with multiple models.")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    import shutil
    for dir_name in ["./test_ppo_output", "./test_ppo_simulation"]:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"✅ Cleaned up {dir_name}")
    
    # Clean up model-specific directories
    for model_name in ["gpt2", "distilgpt2"]:
        dir_name = f"./test_rldk_{model_name}"
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"✅ Cleaned up {dir_name}")

if __name__ == "__main__":
    main()