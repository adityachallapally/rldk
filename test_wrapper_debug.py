#!/usr/bin/env python3
"""Debug the wrapper issue."""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_wrapper_debug():
    """Test the wrapper in isolation."""
    print("🔍 Debugging Wrapper Issue")
    print("=" * 50)
    
    try:
        from rldk.integrations.trl import create_ppo_trainer
        from trl import PPOConfig
        from datasets import Dataset
        from transformers import AutoTokenizer
        
        # Check if we should skip model loading
        if os.getenv("SKIP_MODEL_LOADING", "false").lower() == "true":
            print("⚠️  Skipping model loading (SKIP_MODEL_LOADING=true)")
            return True
        
        # Create minimal dataset
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        texts = ["Hello world", "Test text"]
        tokenized = tokenizer(texts, padding=True, truncation=True, max_length=128)
        
        dataset = Dataset.from_dict({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        })
        
        # Create PPO config
        ppo_config = PPOConfig(
            learning_rate=1e-5,
            per_device_train_batch_size=1,
            mini_batch_size=1,
            num_ppo_epochs=1,
            max_grad_norm=0.5,
            output_dir="./debug_output",
            remove_unused_columns=False,
            bf16=False,
            fp16=False,
            max_steps=1,  # Just 1 step for debugging
            logging_steps=1,
            save_steps=1000,
            eval_steps=1000,
        )
        
        # Create trainer
        print("🏭 Creating PPOTrainer...")
        model_name = "sshleifer/tiny-gpt2"
        
        trainer = create_ppo_trainer(
            model_name=model_name,
            ppo_config=ppo_config,
            train_dataset=dataset,
            callbacks=[],  # No callbacks for debugging
        )
        
        print("✅ PPOTrainer created")
        print(f"📊 Trainer type: {type(trainer).__name__}")
        
        # Check the models
        print(f"📊 Model type: {type(trainer.model)}")
        print(f"📊 Ref model type: {type(trainer.ref_model)}")
        print(f"📊 Reward model type: {type(trainer.reward_model)}")
        print(f"📊 Value model type: {type(trainer.value_model)}")
        
        # Test the models directly
        import torch
        
        print("\n🧪 Testing model outputs...")
        
        # Test main model
        test_input = torch.tensor([[50256, 15496, 995, 50256]])  # Example tokens
        test_attention = torch.ones_like(test_input)
        test_position = torch.arange(test_input.size(1)).unsqueeze(0)
        
        with torch.no_grad():
            # Test main model
            output = trainer.model(
                input_ids=test_input,
                attention_mask=test_attention,
                position_ids=test_position,
                return_dict=True,
                output_hidden_states=True,
            )
            print(f"✅ Main model output type: {type(output)}")
            print(f"✅ Has logits: {hasattr(output, 'logits')}")
            
            # Test ref model
            ref_output = trainer.ref_model(
                input_ids=test_input,
                attention_mask=test_attention,
                position_ids=test_position,
                return_dict=True,
                output_hidden_states=True,
            )
            print(f"✅ Ref model output type: {type(ref_output)}")
            print(f"✅ Has logits: {hasattr(ref_output, 'logits')}")
            
            # Test if we can access logits
            if hasattr(ref_output, 'logits'):
                print(f"✅ Ref logits shape: {ref_output.logits.shape}")
                print(f"✅ Can slice logits: {ref_output.logits[:, 1:-1].shape}")
        
        print("\n🚀 Attempting training...")
        try:
            trainer.train()
            print("🎉 Training completed successfully!")
            return True
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_wrapper_debug()
    sys.exit(0 if success else 1)