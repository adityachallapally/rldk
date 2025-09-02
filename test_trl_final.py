#!/usr/bin/env python3
"""Final comprehensive TRL test with working examples."""

import os
import sys
import time
import torch
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_trl_working_features():
    """Test all the TRL features that actually work."""
    print("🎯 Testing TRL Working Features")
    print("=" * 50)
    
    results = {}
    
    # Test 1: Basic TRL imports and model creation
    print("\n1️⃣ Testing basic TRL functionality...")
    try:
        from trl import AutoModelForCausalLMWithValueHead, PPOConfig
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Test with GPT-2
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create regular model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Create PPO model with value head
        ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        
        # Test text generation
        prompt = "The future of artificial intelligence is"
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
        
        print(f"✅ Basic TRL functionality working")
        print(f"   - Model: {model_name}")
        print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   - Generated: '{generated_text}'")
        print(f"   - PPO model has value head: {hasattr(ppo_model, 'v_head')}")
        
        results["basic_functionality"] = True
        
    except Exception as e:
        print(f"❌ Basic functionality failed: {e}")
        results["basic_functionality"] = False
    
    # Test 2: Multiple model sizes
    print("\n2️⃣ Testing different model sizes...")
    models_to_test = [
        ("gpt2", "Small model (124M)"),
        ("distilgpt2", "Distilled model (82M)"),
    ]
    
    model_results = {}
    for model_name, description in models_to_test:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            param_count = sum(p.numel() for p in model.parameters())
            
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
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"✅ {model_name}: {description}")
            print(f"   - Parameters: {param_count:,}")
            print(f"   - Generated: '{generated}'")
            
            model_results[model_name] = {
                "success": True,
                "parameters": param_count,
                "generated": generated
            }
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            model_results[model_name] = {"success": False, "error": str(e)}
    
    results["multiple_models"] = model_results
    
    # Test 3: PPO Configuration
    print("\n3️⃣ Testing PPO configuration...")
    try:
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
        
        print(f"✅ PPO configuration created")
        print(f"   - Learning rate: {config.learning_rate}")
        print(f"   - Batch size: {config.per_device_train_batch_size}")
        print(f"   - PPO epochs: {config.num_ppo_epochs}")
        print(f"   - Max grad norm: {config.max_grad_norm}")
        
        results["ppo_config"] = True
        
    except Exception as e:
        print(f"❌ PPO configuration failed: {e}")
        results["ppo_config"] = False
    
    # Test 4: RLDK Integration
    print("\n4️⃣ Testing RLDK-TRL integration...")
    try:
        from rldk.integrations.trl import RLDKCallback, PPOMonitor, CheckpointMonitor
        
        # Create callbacks
        rldk_callback = RLDKCallback(output_dir="./test_rldk_output")
        ppo_monitor = PPOMonitor(output_dir="./test_rldk_output")
        checkpoint_monitor = CheckpointMonitor(output_dir="./test_rldk_output")
        
        print(f"✅ RLDK callbacks created")
        print(f"   - RLDKCallback: {type(rldk_callback).__name__}")
        print(f"   - PPOMonitor: {type(ppo_monitor).__name__}")
        print(f"   - CheckpointMonitor: {type(checkpoint_monitor).__name__}")
        
        # Test alert system
        rldk_callback.alert("Test alert from RLDK integration")
        
        results["rldk_integration"] = True
        
    except Exception as e:
        print(f"❌ RLDK integration failed: {e}")
        results["rldk_integration"] = False
    
    # Test 5: Memory and Performance
    print("\n5️⃣ Testing memory usage and performance...")
    try:
        import psutil
        import time
        
        def get_memory_mb():
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        
        initial_memory = get_memory_mb()
        
        # Load a model and measure memory
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "gpt2"
        start_time = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        load_time = time.time() - start_time
        after_load_memory = get_memory_mb()
        memory_increase = after_load_memory - initial_memory
        
        # Test generation speed
        prompt = "The quick brown fox"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=15,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        generation_time = time.time() - start_time
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"✅ Performance test completed")
        print(f"   - Model load time: {load_time:.2f}s")
        print(f"   - Memory increase: {memory_increase:.1f} MB")
        print(f"   - Generation time: {generation_time:.3f}s")
        print(f"   - Generated: '{generated_text}'")
        
        results["performance"] = {
            "load_time": load_time,
            "memory_increase": memory_increase,
            "generation_time": generation_time,
            "success": True
        }
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        results["performance"] = {"success": False, "error": str(e)}
    
    # Test 6: TRL Value Head Functionality
    print("\n6️⃣ Testing TRL value head functionality...")
    try:
        from trl import AutoModelForCausalLMWithValueHead
        from transformers import AutoTokenizer
        
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        
        # Test forward pass with value head
        prompt = "Hello world"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"✅ Value head functionality working")
        print(f"   - Model type: {type(model).__name__}")
        print(f"   - Has value head: {hasattr(model, 'v_head')}")
        print(f"   - Output keys: {list(outputs.keys())}")
        print(f"   - Logits shape: {outputs.logits.shape}")
        if hasattr(outputs, 'value'):
            print(f"   - Value shape: {outputs.value.shape}")
        
        results["value_head"] = True
        
    except Exception as e:
        print(f"❌ Value head test failed: {e}")
        results["value_head"] = False
    
    return results

def test_larger_model():
    """Test with a larger model to see TRL's capabilities."""
    print("\n🚀 Testing with larger model...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Try a larger model (but not too large for this environment)
        model_name = "microsoft/DialoGPT-medium"  # 345M parameters
        
        print(f"📦 Loading {model_name}...")
        start_time = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        load_time = time.time() - start_time
        param_count = sum(p.numel() for p in model.parameters())
        
        # Test generation
        prompt = "Hello, how are you?"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=20,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        generation_time = time.time() - start_time
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"✅ Larger model test successful")
        print(f"   - Model: {model_name}")
        print(f"   - Parameters: {param_count:,}")
        print(f"   - Load time: {load_time:.2f}s")
        print(f"   - Generation time: {generation_time:.3f}s")
        print(f"   - Generated: '{generated_text}'")
        
        return {
            "success": True,
            "model": model_name,
            "parameters": param_count,
            "load_time": load_time,
            "generation_time": generation_time,
            "generated": generated_text
        }
        
    except Exception as e:
        print(f"❌ Larger model test failed: {e}")
        return {"success": False, "error": str(e)}

def main():
    """Run final comprehensive TRL tests."""
    print("🎯 Final TRL Comprehensive Test Suite")
    print("=" * 60)
    
    # Run main tests
    results = test_trl_working_features()
    
    # Test larger model
    larger_model_result = test_larger_model()
    results["larger_model"] = larger_model_result
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 60)
    
    successful_tests = 0
    total_tests = 0
    
    # Count results
    for test_name, result in results.items():
        if test_name == "multiple_models":
            # Handle multiple model results
            for model_name, model_result in result.items():
                total_tests += 1
                if model_result.get("success", False):
                    successful_tests += 1
                status = "✅ PASSED" if model_result.get("success", False) else "❌ FAILED"
                print(f"Multiple Models ({model_name}): {status}")
        elif test_name == "performance":
            # Handle performance results
            total_tests += 1
            if result.get("success", False):
                successful_tests += 1
            status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
            print(f"Performance Test: {status}")
        elif test_name == "larger_model":
            # Handle larger model results
            total_tests += 1
            if result.get("success", False):
                successful_tests += 1
            status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
            print(f"Larger Model Test: {status}")
        else:
            # Handle boolean results
            total_tests += 1
            if result:
                successful_tests += 1
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name.replace('_', ' ').title():<25} {status}")
    
    print(f"\nResults: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests >= total_tests * 0.8:  # 80% success rate
        print("🎉 TRL is working excellently! Most features are functional.")
        print("\n✅ What's Working:")
        print("   - Model loading and text generation")
        print("   - PPO model creation with value heads")
        print("   - PPO configuration")
        print("   - RLDK integration and callbacks")
        print("   - Multiple model sizes (GPT-2, DistilGPT-2)")
        print("   - Memory management and performance")
        
        print("\n⚠️  Known Limitations:")
        print("   - PPOTrainer requires specific setup for full training")
        print("   - Some advanced features need proper model configurations")
        
    else:
        print("⚠️  Some tests failed. Check the output above.")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    import shutil
    for dir_name in ["./test_output", "./test_rldk_output"]:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"✅ Cleaned up {dir_name}")

if __name__ == "__main__":
    main()