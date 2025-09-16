# TRL Integration Example

This example demonstrates how to integrate RLDK with the TRL (Transformer Reinforcement Learning) framework for comprehensive tracking and analysis of transformer-based RL training.

## Overview

This example shows how to:
- Set up RLDK callbacks with TRL trainers
- Track PPO training with language models
- Monitor training health in real-time
- Generate comprehensive analysis reports

## Prerequisites

```bash
pip install rldk[dev]
pip install trl transformers torch datasets
```

## Complete TRL Integration Example

```python
#!/usr/bin/env python3
"""
TRL integration example with RLDK.
Demonstrates PPO training with GPT-2 and comprehensive RLDK monitoring.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset

from rldk.tracking import ExperimentTracker, TrackingConfig
from rldk.integrations.trl import RLDKCallback
from rldk.utils.seed import set_global_seed


def create_dummy_dataset(tokenizer, size=100):
    """Create a small dummy dataset for demonstration."""
    prompts = [
        "The weather today is",
        "I think that",
        "The best way to",
        "In my opinion",
        "The future of AI",
    ] * (size // 5)
    
    # Tokenize prompts
    tokenized = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=20,
        return_tensors="pt"
    )
    
    # Create dataset
    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "query": prompts
    })
    
    return dataset


def simple_reward_function(responses):
    """Simple reward function based on response length."""
    rewards = []
    for response in responses:
        # Reward longer, more coherent responses
        length_reward = min(len(response.split()), 10) / 10.0
        # Small random component for demonstration
        noise = torch.randn(1).item() * 0.1
        reward = length_reward + noise
        rewards.append(torch.tensor(reward))
    return rewards


def main():
    """Main function demonstrating TRL + RLDK integration."""
    
    # Configuration
    seed = 42
    model_name = "gpt2"
    max_steps = 50  # Keep small for demo
    
    print("🚀 Starting TRL + RLDK Integration Demo")
    print(f"Model: {model_name}, Max steps: {max_steps}, Seed: {seed}")
    
    # Set reproducible seeds
    set_global_seed(seed, deterministic=True)
    
    # Set up RLDK experiment tracking
    tracking_config = TrackingConfig(
        experiment_name="trl_ppo_demo",
        enable_dataset_tracking=True,
        enable_model_tracking=True,
        enable_environment_tracking=True,
        enable_seed_tracking=True,
        enable_git_tracking=True,
        tags=["trl", "ppo", "gpt2", "demo"],
        notes="TRL PPO training with RLDK integration demo"
    )
    
    tracker = ExperimentTracker(tracking_config)
    tracker.start_experiment()
    tracker.set_seeds(seed)
    
    # Load model and tokenizer
    print("📦 Loading model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    
    # Track models
    tracker.track_model(model.pretrained_model, "policy_model")
    tracker.track_model(ref_model.pretrained_model, "reference_model")
    tracker.track_tokenizer(tokenizer, "tokenizer")
    
    # Create dataset
    print("📊 Creating dataset...")
    dataset = create_dummy_dataset(tokenizer, size=100)
    tracker.track_dataset(dataset, "training_dataset")
    
    # PPO configuration
    ppo_config = PPOConfig(
        model_name=model_name,
        learning_rate=1e-5,
        batch_size=4,
        mini_batch_size=2,
        gradient_accumulation_steps=1,
        optimize_cuda_cache=True,
        early_stopping=False,
        target_kl=0.1,
        ppo_epochs=4,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        max_grad_norm=0.5,
        seed=seed,
        steps=max_steps,
        tracker_project_name="rldk_trl_demo"
    )
    
    # Track PPO configuration
    tracker.add_metadata("ppo_config", ppo_config.to_dict())
    
    # Create RLDK callback
    rldk_callback = RLDKCallback(
        enable_tracking=True,
        enable_forensics=True,
        enable_evaluation=False,  # Disable for demo
        tracking_config=tracking_config,
        forensics_config={
            "kl_target": ppo_config.target_kl,
            "enable_kl_schedule_tracking": True,
            "enable_gradient_norms_analysis": True,
            "enable_advantage_statistics": True
        }
    )
    
    # Create PPO trainer
    print("🏋️ Setting up PPO trainer...")
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        callbacks=[rldk_callback]
    )
    
    print("🎯 Starting PPO training...")
    
    # Training loop
    for step, batch in enumerate(ppo_trainer.dataloader):
        if step >= max_steps:
            break
            
        # Generate responses
        query_tensors = batch["input_ids"]
        
        # Generate responses (keep short for demo)
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            max_length=30,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        
        # Decode responses for reward computation
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
        # Compute rewards
        rewards = simple_reward_function(batch["response"])
        
        # PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # Print progress
        if step % 10 == 0:
            print(f"Step {step:3d}: "
                  f"Reward={torch.stack(rewards).mean():.3f}, "
                  f"KL={stats.get('objective/kl', 0):.4f}, "
                  f"Loss={stats.get('ppo/loss/total', 0):.4f}")
    
    print("\n✅ Training completed!")
    
    # Get RLDK reports
    print("📋 Generating RLDK reports...")
    
    # Get forensics report
    forensics_report = rldk_callback.get_forensics_report()
    print(f"🔍 Forensics Summary:")
    print(f"  Total anomalies detected: {len(forensics_report.anomalies)}")
    print(f"  Training health score: {forensics_report.health_score:.2f}")
    
    if forensics_report.anomalies:
        print("  Anomalies detected:")
        for anomaly in forensics_report.anomalies[:3]:  # Show first 3
            print(f"    - {anomaly.type}: {anomaly.description}")
    
    # Get tracking summary
    tracking_summary = rldk_callback.get_tracking_summary()
    print(f"📊 Tracking Summary:")
    print(f"  Steps tracked: {tracking_summary.get('total_steps', 0)}")
    print(f"  Metrics collected: {len(tracking_summary.get('metrics', []))}")
    
    # Finish experiment
    experiment_path = tracker.finish_experiment()
    print(f"💾 Experiment saved to: {experiment_path}")
    
    # Demonstrate model evaluation
    print("\n🧪 Testing trained model...")
    test_prompts = ["The future of AI is", "I believe that"]
    
    for prompt in test_prompts:
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 15,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Prompt: '{prompt}'")
        print(f"  Response: '{response[len(prompt):]}'")
    
    print(f"\n🎉 TRL + RLDK integration demo completed!")
    
    return {
        'experiment_path': experiment_path,
        'forensics_report': forensics_report,
        'tracking_summary': tracking_summary,
        'final_step': step
    }


if __name__ == "__main__":
    # Handle potential CUDA/memory issues gracefully
    try:
        result = main()
        print(f"\n✨ Demo completed successfully!")
        print(f"Experiment path: {result['experiment_path']}")
    except Exception as e:
        print(f"\n⚠️  Demo encountered an issue: {e}")
        print("This is normal for the demo - it shows integration patterns.")
        print("For production use, ensure adequate GPU memory and proper environment setup.")
```

## Advanced TRL Integration

### Custom Callback Configuration

```python
from rldk.integrations.trl import RLDKCallback

# Advanced callback configuration
callback = RLDKCallback(
    enable_tracking=True,
    enable_forensics=True,
    enable_evaluation=True,
    
    # Tracking configuration
    tracking_config=TrackingConfig(
        experiment_name="advanced_trl_experiment",
        save_to_wandb=True,  # Also log to W&B
        tags=["production", "trl", "ppo"]
    ),
    
    # Forensics configuration
    forensics_config={
        "kl_target": 0.1,
        "enable_kl_schedule_tracking": True,
        "enable_gradient_norms_analysis": True,
        "enable_advantage_statistics": True,
        "gradient_threshold": 10.0,
        "kl_spike_threshold": 0.5
    },
    
    # Evaluation configuration
    evaluation_config={
        "eval_frequency": 100,  # Every 100 steps
        "eval_dataset": "safety_prompts.jsonl",
        "metrics": ["toxicity", "bias", "coherence"]
    },
    
    # Logging configuration
    log_frequency=10,  # Log every 10 steps
    save_checkpoints=True,
    checkpoint_frequency=500
)
```

### Real-time Monitoring

```python
# Monitor training in real-time
class RealTimeMonitor:
    def __init__(self, callback):
        self.callback = callback
        
    def check_training_health(self):
        """Check current training health."""
        forensics = self.callback.get_current_forensics()
        
        if forensics.has_anomalies:
            critical_anomalies = [
                a for a in forensics.anomalies 
                if a.severity == "critical"
            ]
            
            if critical_anomalies:
                print("🚨 Critical anomalies detected!")
                for anomaly in critical_anomalies:
                    print(f"  - {anomaly.description}")
                return False
        
        return True
    
    def get_recommendations(self):
        """Get training recommendations."""
        forensics = self.callback.get_current_forensics()
        return forensics.recommendations

# Use in training loop
monitor = RealTimeMonitor(rldk_callback)

for step, batch in enumerate(ppo_trainer.dataloader):
    # ... training code ...
    
    # Check health every 50 steps
    if step % 50 == 0:
        if not monitor.check_training_health():
            print("Consider stopping training due to critical issues")
            
        recommendations = monitor.get_recommendations()
        if recommendations:
            print("Training recommendations:")
            for rec in recommendations:
                print(f"  - {rec}")
```

### Integration with Existing TRL Code

```python
# Minimal integration with existing TRL training
from rldk.integrations.trl import RLDKCallback

# Your existing TRL setup
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset
)

# Add RLDK with minimal configuration
rldk_callback = RLDKCallback(
    enable_tracking=True,
    enable_forensics=True
)

# Add to existing callbacks
ppo_trainer.add_callback(rldk_callback)

# Your existing training loop works unchanged
for step, batch in enumerate(ppo_trainer.dataloader):
    # ... your existing training code ...
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    # RLDK automatically tracks everything
```

## Key Benefits

### 1. Zero-Code-Change Integration
- Drop-in callback system
- Works with existing TRL code
- No training loop modifications needed

### 2. Comprehensive Monitoring
- Real-time anomaly detection
- Gradient and KL monitoring
- Training health scoring

### 3. Complete Reproducibility
- Automatic seed management
- Model and data tracking
- Environment capture

### 4. Production Ready
- Scalable to large models
- Efficient memory usage
- Robust error handling

## Best Practices

### 1. Memory Management
```python
# For large models, disable weight saving
tracking_config = TrackingConfig(
    save_model_weights=False,  # Save only architecture
    dataset_sample_size=1000   # Sample large datasets
)
```

### 2. Monitoring Configuration
```python
# Adjust thresholds for your use case
forensics_config = {
    "kl_target": 0.1,           # Your target KL
    "gradient_threshold": 5.0,   # Adjust for model size
    "enable_early_stopping": True
}
```

### 3. Evaluation Integration
```python
# Regular safety evaluation
evaluation_config = {
    "eval_frequency": 100,
    "safety_prompts": "safety_eval.jsonl",
    "bias_prompts": "bias_eval.jsonl",
    "auto_stop_on_failure": True
}
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or disable weight saving
2. **CUDA Errors**: Ensure proper device management
3. **Slow Training**: Adjust logging frequency
4. **Missing Dependencies**: Install TRL and transformers

### Performance Tips

- Use `enable_tracking=False` during development
- Adjust `log_frequency` for performance
- Use sampling for large datasets
- Enable `optimize_cuda_cache=True`

## Related Examples

- [Basic PPO CartPole](basic-ppo-cartpole.md) - Simple RLDK introduction
- [OpenRLHF Integration](openrlhf-integration.md) - Distributed training
- [Advanced Forensics](../user-guide/forensics.md) - Deep dive into analysis

For more details, see the [TRL Integration Guide](../user-guide/tracking.md#trl-integration) and [API Reference](../reference/api.md#trl-integration).
