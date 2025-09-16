#!/usr/bin/env python3
"""Simplified offline PPO script with RLDK monitoring."""

import json
import os
import time
from pathlib import Path

import torch
from datasets import Dataset

# Set offline mode
os.environ["HF_HUB_OFFLINE"] = "1"

# Import TRL components
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

# Import RLDK components
from rldk.integrations.trl import PPOMonitor, RLDKCallback
from rldk.forensics.comprehensive_ppo_forensics import ComprehensivePPOForensics


def create_synthetic_dataset():
    """Create a tiny synthetic dataset of short prompts."""
    prompts = [
        "Write a one word positive review:",
        "Say something nice:",
        "Give a positive comment:",
        "Express something good:",
        "Share something awesome:",
        "Tell me something great:",
        "Write something wonderful:",
        "Say something amazing:",
        "Give a good response:",
        "Express something positive:",
    ] * 7  # 70 total prompts
    
    return Dataset.from_dict({"prompt": prompts})


def compute_reward(text: str) -> float:
    """Simple reward function: +1 if contains positive words, else 0."""
    positive_words = ["good", "great", "awesome", "wonderful", "amazing", "excellent", "fantastic", "perfect"]
    return 1.0 if any(word in text.lower() for word in positive_words) else 0.0


class SimpleRewardModel(torch.nn.Module):
    """Simple reward model that returns constant rewards."""
    
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
    
    def forward(self, **batch):
        """Return constant reward of 0.5."""
        batch_size = 1
        if 'input_ids' in batch:
            batch_size = batch['input_ids'].shape[0]
        return torch.full((batch_size,), 0.5, dtype=torch.float32)


def main():
    """Main PPO training loop."""
    print("🚀 Starting offline PPO training with RLDK monitoring")
    
    # Create output directory
    output_dir = Path("artifacts/phase2_offline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer and model
    print("📥 Loading model and tokenizer...")
    model_path = "assets/tiny_causal"
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model with value head
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
    
    # Add generation config if missing
    from transformers import GenerationConfig
    if not hasattr(model, 'generation_config'):
        model.generation_config = GenerationConfig(
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
    if not hasattr(ref_model, 'generation_config'):
        ref_model.generation_config = GenerationConfig(
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
    
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✅ Tokenizer vocab size: {len(tokenizer)}")
    
    # Create dataset
    dataset = create_synthetic_dataset()
    print(f"✅ Dataset created: {len(dataset)} samples")
    
    # PPO Configuration
    ppo_config = PPOConfig(
        batch_size=2,
        mini_batch_size=1,
        learning_rate=1e-5,
        num_ppo_epochs=2,
        kl_coef=0.1,
        seed=13,
        logging_dir=str(output_dir),
        bf16=False,
        fp16=False,
        use_cpu=True,
    )
    
    # Create PPO trainer with minimal setup
    reward_model = SimpleRewardModel()
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        value_model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
    )
    
    print("✅ PPO Trainer created")
    
    # Initialize RLDK monitoring
    print("🔍 Setting up RLDK monitoring...")
    
    # Create comprehensive PPO forensics
    forensics = ComprehensivePPOForensics(
        kl_target=0.1,
        kl_target_tolerance=0.05,
        enable_kl_schedule_tracking=True,
        enable_gradient_norms_analysis=True,
        enable_advantage_statistics=True,
    )
    
    # Create RLDK monitors
    rldk_callback = RLDKCallback(
        output_dir=str(output_dir),
        log_interval=5,
        run_id="offline_ppo_test"
    )
    
    ppo_monitor = PPOMonitor(
        output_dir=str(output_dir),
        kl_threshold=0.1,
        reward_threshold=0.05,
        run_id="offline_ppo_test"
    )
    
    # Attach monitors to trainer
    ppo_trainer.add_callback(rldk_callback)
    ppo_trainer.add_callback(ppo_monitor)
    
    print("✅ RLDK monitoring attached")
    
    # Training loop
    print("🎯 Starting training loop...")
    
    start_time = time.time()
    step_times = []
    all_metrics = []
    
    for step in range(100):
        step_start = time.time()
        
        try:
            # Sample batch of prompts
            batch_prompts = [dataset[i % len(dataset)]["prompt"] for i in range(ppo_config.batch_size)]
            
            # Tokenize prompts
            query_tensors = [tokenizer.encode(prompt, return_tensors="pt").squeeze() for prompt in batch_prompts]
            
            # Generate responses
            response_tensors = []
            for query_tensor in query_tensors:
                with torch.no_grad():
                    response = model.generate(
                        query_tensor.unsqueeze(0),
                        max_new_tokens=8,
                        do_sample=True,
                        top_k=0,
                        top_p=1.0,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    response_tensors.append(response.squeeze())
            
            # Compute rewards manually
            rewards = []
            for i, response_tensor in enumerate(response_tensors):
                # Decode response
                response_text = tokenizer.decode(response_tensor[len(query_tensors[i]):], skip_special_tokens=True)
                reward = compute_reward(response_text)
                rewards.append(reward)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            
            # PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            # Extract metrics
            kl_mean = stats.get('ppo/policy/kl_mean', 0.0)
            reward_mean = stats.get('ppo/rewards/mean', 0.0)
            advantage_mean = stats.get('ppo/advantages/mean', 0.0)
            grad_norm = stats.get('ppo/policy/grad_norm', 0.0)
            
            # Update comprehensive forensics
            forensics.update(
                step=step,
                kl=kl_mean,
                kl_coef=1.0,
                entropy=stats.get('ppo/policy/entropy', 0.0),
                reward_mean=reward_mean,
                reward_std=stats.get('ppo/rewards/std', 0.0),
                policy_grad_norm=grad_norm,
                value_grad_norm=stats.get('ppo/val/grad_norm', 0.0),
                advantage_mean=advantage_mean,
                advantage_std=stats.get('ppo/advantages/std', 0.0),
                advantage_min=stats.get('ppo/advantages/min', 0.0),
                advantage_max=stats.get('ppo/advantages/max', 0.0),
                advantage_samples=[advantage_mean] * 10,  # Simplified
            )
            
            # Log metrics
            metrics = {
                "step": step,
                "kl_mean": kl_mean,
                "reward_mean": reward_mean,
                "advantage_mean": advantage_mean,
                "grad_norm": grad_norm,
                "timestamp": time.time(),
            }
            all_metrics.append(metrics)
            
            # Log to JSONL
            with open(output_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(metrics) + "\n")
            
            if step % 10 == 0:
                print(f"Step {step}: KL={kl_mean:.4f}, Reward={reward_mean:.4f}, "
                      f"Advantage={advantage_mean:.4f}, GradNorm={grad_norm:.4f}")
        
        except Exception as e:
            print(f"❌ Error at step {step}: {e}")
            continue
        
        step_time = time.time() - step_start
        step_times.append(step_time)
    
    # Final analysis
    total_time = time.time() - start_time
    
    # Get comprehensive analysis
    analysis = forensics.get_comprehensive_analysis()
    health_summary = forensics.get_health_summary()
    
    # Create summary
    summary = {
        "total_steps": 100,
        "total_time_seconds": total_time,
        "steps_per_second": 100 / total_time,
        "average_step_time": sum(step_times) / len(step_times),
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
        "final_kl_mean": all_metrics[-1]["kl_mean"] if all_metrics else 0.0,
        "final_reward_mean": all_metrics[-1]["reward_mean"] if all_metrics else 0.0,
        "final_advantage_mean": all_metrics[-1]["advantage_mean"] if all_metrics else 0.0,
        "comprehensive_analysis": analysis,
        "health_summary": health_summary,
    }
    
    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save comprehensive analysis
    forensics.save_analysis(str(output_dir / "comprehensive_analysis.json"))
    
    print(f"\n🎉 Training completed!")
    print(f"📊 Total time: {total_time:.2f}s")
    print(f"📊 Steps per second: {100/total_time:.2f}")
    print(f"📊 Final KL: {summary['final_kl_mean']:.4f}")
    print(f"📊 Final Reward: {summary['final_reward_mean']:.4f}")
    print(f"📊 Overall Health Score: {analysis['overall_health_score']:.3f}")
    
    # Check for anomalies
    anomalies = forensics.get_anomalies()
    if anomalies:
        print(f"🚨 Detected {len(anomalies)} anomalies:")
        for anomaly in anomalies[-3:]:  # Show last 3
            print(f"   - {anomaly['type']}: {anomaly['message']}")
    else:
        print("✅ No anomalies detected")
    
    print(f"💾 All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()