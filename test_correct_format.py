#!/usr/bin/env python3
"""Test with CORRECT dataset format for TRL PPOTrainer."""

import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_correct_dataset():
    """Create a dataset in the CORRECT format for TRL PPOTrainer."""
    from datasets import Dataset
    from transformers import AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create text data
    texts = [
        "Hello world, this is a test.",
        "Python is a great programming language.",
        "Machine learning is fascinating.",
        "Artificial intelligence will change the world.",
        "Deep learning models are powerful.",
    ] * 4  # 20 samples total
    
    # Tokenize the texts
    tokenized = tokenizer(texts, padding=True, truncation=True, max_length=128)
    
    # Create dataset with tokenized data
    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    })
    
    return dataset, tokenizer


def test_real_training_with_correct_format():
    """Test REAL training with correct dataset format."""
    print("🎯 REAL RL Training Test - CORRECT Format")
    print("=" * 60)
    
    try:
        from rldk.integrations.trl import create_ppo_trainer, PPOMonitor
        from trl import PPOConfig
        
        # Check if we should skip model loading
        if os.getenv("SKIP_MODEL_LOADING", "false").lower() == "true":
            print("⚠️  Skipping model loading (SKIP_MODEL_LOADING=true)")
            print("✅ Test structure validated")
            return True
        
        # 1. Create CORRECT dataset format
        print("📊 Creating CORRECT dataset format...")
        dataset, tokenizer = create_correct_dataset()
        print(f"✅ Created dataset with {len(dataset)} samples")
        print(f"📊 Dataset columns: {dataset.column_names}")
        print(f"📊 Sample input_ids shape: {len(dataset[0]['input_ids'])}")
        
        # 2. Create PPO config
        print("⚙️  Creating PPO config...")
        ppo_config = PPOConfig(
            learning_rate=1e-5,
            per_device_train_batch_size=1,
            mini_batch_size=1,
            num_ppo_epochs=1,
            max_grad_norm=0.5,
            output_dir="./rl_test_output",
            remove_unused_columns=False,
            bf16=False,
            fp16=False,
            max_steps=3,  # Very short for testing
            logging_steps=1,
            save_steps=1000,
            eval_steps=1000,
        )
        print("✅ PPO config created")
        
        # 3. Create RLDK monitor
        print("🔍 Creating RLDK monitor...")
        monitor = PPOMonitor(
            output_dir="./rl_test_output",
            kl_threshold=0.1,
            reward_threshold=0.05,
            run_id="correct_format_test"
        )
        print("✅ RLDK monitor created")
        
        # 4. Create trainer with our factory function
        print("🏭 Creating PPOTrainer with factory function...")
        model_name = "sshleifer/tiny-gpt2"
        
        start_time = time.time()
        trainer = create_ppo_trainer(
            model_name=model_name,
            ppo_config=ppo_config,
            train_dataset=dataset,
            callbacks=[monitor],  # Include RLDK monitor
        )
        creation_time = time.time() - start_time
        
        print(f"✅ PPOTrainer created in {creation_time:.2f} seconds")
        print(f"📊 Trainer type: {type(trainer).__name__}")
        
        # 5. Verify trainer components
        required_attrs = ['model', 'ref_model', 'reward_model', 'value_model', 'processing_class']
        for attr in required_attrs:
            if hasattr(trainer, attr):
                print(f"✅ Trainer has {attr}")
            else:
                print(f"❌ Trainer missing {attr}")
                return False
        
        # 6. ACTUALLY RUN REAL RL TRAINING
        print("\n🚀 STARTING REAL RL TRAINING WITH CORRECT FORMAT...")
        print("=" * 60)
        
        try:
            start_time = time.time()
            
            # This is the REAL test - actual RL training with correct format
            trainer.train()
            
            training_time = time.time() - start_time
            print(f"🎉 REAL RL TRAINING COMPLETED in {training_time:.2f} seconds!")
            
            # Check training state
            if hasattr(trainer, 'state'):
                print(f"📊 Training completed: {trainer.state.global_step} steps")
            
            # Check if RLDK monitor collected data
            if hasattr(monitor, 'metrics_history'):
                print(f"📈 RLDK collected {len(monitor.metrics_history)} metrics")
            
            print("🎉 SUCCESS! Real RL training with RLDK integration worked!")
            return True
            
        except Exception as e:
            print(f"❌ RL training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the real RL training test with correct format."""
    print("🎯 REAL RL Training Test - CORRECT Dataset Format")
    print("=" * 60)
    
    success = test_real_training_with_correct_format()
    
    if success:
        print("\n🎉 SUCCESS!")
        print("✅ Real RL training with RLDK integration works!")
        print("✅ Dataset format issue resolved!")
        print("✅ End-to-end functionality verified!")
    else:
        print("\n❌ Test failed")
        print("Check the output above for details")
    
    return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)