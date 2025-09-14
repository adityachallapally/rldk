# Basic PPO CartPole Example

This example demonstrates a complete RLDK workflow using PPO training on the CartPole environment. It's designed to be CPU-friendly and run quickly for demonstration purposes.

## Overview

This example shows how to:
- Set up experiment tracking with RLDK
- Implement a simple PPO training loop
- Use forensics analysis during training
- Check determinism and reproducibility
- Generate analysis reports

## Prerequisites

```bash
pip install rldk[dev]
pip install gym torch
```

## Complete Example

```python
#!/usr/bin/env python3
"""
Basic PPO CartPole example with RLDK integration.
CPU-friendly demonstration of experiment tracking, forensics, and reproducibility.
"""

import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym

from rldk.tracking import ExperimentTracker, TrackingConfig
from rldk.forensics import ComprehensivePPOForensics
from rldk.determinism import check
from rldk.utils.seed import set_global_seed


class SimplePPOPolicy(nn.Module):
    """Simple policy network for CartPole."""
    
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.policy(state), self.value(state)
    
    def get_action(self, state):
        probs, value = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value


def collect_trajectories(env, policy, num_episodes=5):
    """Collect training trajectories."""
    trajectories = []
    
    for _ in range(num_episodes):
        states, actions, rewards, log_probs, values = [], [], [], [], []
        state = env.reset()
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, value = policy.get_action(state_tensor)
            
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value.item())
            
            state = next_state
        
        trajectories.append({
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'log_probs': log_probs,
            'values': values
        })
    
    return trajectories


def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    """Compute GAE advantages."""
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    return advantages


def ppo_update(policy, optimizer, trajectories, clip_ratio=0.2, epochs=4):
    """Perform PPO update."""
    all_states = []
    all_actions = []
    all_old_log_probs = []
    all_advantages = []
    all_returns = []
    
    # Prepare data
    for traj in trajectories:
        advantages = compute_advantages(traj['rewards'], traj['values'])
        returns = [adv + val for adv, val in zip(advantages, traj['values'])]
        
        all_states.extend(traj['states'])
        all_actions.extend(traj['actions'])
        all_old_log_probs.extend([lp.detach() for lp in traj['log_probs']])
        all_advantages.extend(advantages)
        all_returns.extend(returns)
    
    # Convert to tensors
    states = torch.FloatTensor(all_states)
    actions = torch.LongTensor(all_actions)
    old_log_probs = torch.stack(all_old_log_probs)
    advantages = torch.FloatTensor(all_advantages)
    returns = torch.FloatTensor(all_returns)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    total_policy_loss = 0
    total_value_loss = 0
    total_kl = 0
    
    for _ in range(epochs):
        # Forward pass
        probs, values = policy(states)
        dist = Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        
        # Policy loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values.squeeze(), returns)
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        
        # Track metrics
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        
        # Compute KL divergence
        kl = (old_log_probs - new_log_probs).mean().item()
        total_kl += abs(kl)
    
    return {
        'policy_loss': total_policy_loss / epochs,
        'value_loss': total_value_loss / epochs,
        'kl_divergence': total_kl / epochs,
        'policy_grad_norm': torch.nn.utils.clip_grad_norm_(policy.parameters(), float('inf')).item()
    }


def main():
    """Main training function with RLDK integration."""
    
    # Configuration
    seed = 42
    num_updates = 20
    episodes_per_update = 5
    learning_rate = 3e-4
    
    # Set up RLDK tracking
    tracking_config = TrackingConfig(
        experiment_name="ppo_cartpole_demo",
        enable_dataset_tracking=True,
        enable_model_tracking=True,
        enable_environment_tracking=True,
        enable_seed_tracking=True,
        enable_git_tracking=True,
        tags=["ppo", "cartpole", "demo", "cpu"],
        notes="Basic PPO CartPole example with RLDK integration"
    )
    
    tracker = ExperimentTracker(tracking_config)
    tracker.start_experiment()
    
    # Set reproducible seeds
    set_global_seed(seed, deterministic=True)
    tracker.set_seeds(seed)
    
    # Environment and model setup
    env = gym.make('CartPole-v1')
    policy = SimplePPOPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    
    # Track model and hyperparameters
    tracker.track_model(policy, "ppo_policy")
    tracker.add_metadata("learning_rate", learning_rate)
    tracker.add_metadata("num_updates", num_updates)
    tracker.add_metadata("episodes_per_update", episodes_per_update)
    tracker.add_metadata("environment", "CartPole-v1")
    
    # Set up forensics
    forensics = ComprehensivePPOForensics(
        kl_target=0.01,
        enable_kl_schedule_tracking=True,
        enable_gradient_norms_analysis=True,
        enable_advantage_statistics=True
    )
    
    # Training loop
    metrics_log = []
    
    print("🚀 Starting PPO training with RLDK...")
    print(f"Seed: {seed}, Updates: {num_updates}, Episodes per update: {episodes_per_update}")
    
    for update in range(num_updates):
        # Collect trajectories
        trajectories = collect_trajectories(env, policy, episodes_per_update)
        
        # Compute episode statistics
        episode_rewards = [sum(traj['rewards']) for traj in trajectories]
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        # PPO update
        update_metrics = ppo_update(policy, optimizer, trajectories)
        
        # Combine metrics
        step_metrics = {
            'step': update,
            'seed': seed,
            'reward_mean': mean_reward,
            'reward_std': std_reward,
            'policy_loss': update_metrics['policy_loss'],
            'value_loss': update_metrics['value_loss'],
            'kl_divergence': update_metrics['kl_divergence'],
            'policy_grad_norm': update_metrics['policy_grad_norm'],
            'entropy': 0.5,  # Placeholder for entropy
            'advantage_mean': 0.0,  # Placeholder
            'advantage_std': 1.0,   # Placeholder
        }
        
        # Forensics analysis
        forensics_result = forensics.update(
            step=update,
            kl=step_metrics['kl_divergence'],
            kl_coef=0.1,  # Fixed coefficient for this example
            entropy=step_metrics['entropy'],
            reward_mean=step_metrics['reward_mean'],
            reward_std=step_metrics['reward_std'],
            policy_grad_norm=step_metrics['policy_grad_norm'],
            value_grad_norm=step_metrics['policy_grad_norm'],  # Using same for simplicity
            advantage_mean=step_metrics['advantage_mean'],
            advantage_std=step_metrics['advantage_std']
        )
        
        # Log metrics
        metrics_log.append(step_metrics)
        
        # Print progress
        print(f"Update {update:2d}: Reward={mean_reward:6.2f}±{std_reward:5.2f}, "
              f"KL={step_metrics['kl_divergence']:.4f}, "
              f"Policy Loss={step_metrics['policy_loss']:.4f}")
        
        # Check for anomalies
        if forensics_result.has_anomalies:
            print(f"  ⚠️  {len(forensics_result.anomalies)} anomalies detected:")
            for anomaly in forensics_result.anomalies:
                print(f"    - {anomaly.type}: {anomaly.description}")
    
    # Save metrics
    metrics_file = "ppo_cartpole_metrics.jsonl"
    with open(metrics_file, 'w') as f:
        for metrics in metrics_log:
            f.write(json.dumps(metrics) + '\n')
    
    print(f"\n📊 Metrics saved to {metrics_file}")
    
    # Track final results
    final_reward = metrics_log[-1]['reward_mean']
    tracker.add_metadata("final_reward", final_reward)
    tracker.add_metadata("total_steps", len(metrics_log))
    tracker.add_metadata("metrics_file", metrics_file)
    
    # Get forensics report
    forensics_report = forensics.get_report()
    print(f"\n🔍 Forensics Summary:")
    print(f"  Total anomalies: {len(forensics_report.anomalies)}")
    print(f"  Training health score: {forensics_report.health_score:.2f}")
    
    # Finish experiment tracking
    experiment_path = tracker.finish_experiment()
    print(f"\n✅ Experiment tracking completed: {experiment_path}")
    
    # Demonstrate determinism checking
    print(f"\n🎯 Testing determinism...")
    
    # Create a simple test command
    test_script = "test_determinism.py"
    with open(test_script, 'w') as f:
        f.write(f"""
import random
import numpy as np
import torch
import json

# Set seed from command line
import sys
if len(sys.argv) > 1 and sys.argv[1] == '--seed':
    seed = int(sys.argv[2])
else:
    seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Simple computation
result = {{
    "step": 0,
    "seed": seed,
    "loss": random.random(),
    "reward_mean": np.random.normal(0.5, 0.1)
}}

print(json.dumps(result))
""")
    
    try:
        from rldk.determinism import check
        
        determinism_report = check(
            cmd=f"python {test_script} --seed 42",
            compare=["loss", "reward_mean"],
            replicas=3,
            device="cpu"
        )
        
        if determinism_report.passed:
            print("  ✅ Determinism check passed!")
        else:
            print(f"  ❌ Determinism check failed: {len(determinism_report.mismatches)} mismatches")
            
    except Exception as e:
        print(f"  ⚠️  Determinism check skipped: {e}")
    
    # Clean up
    if os.path.exists(test_script):
        os.remove(test_script)
    
    print(f"\n🎉 PPO CartPole demo completed successfully!")
    print(f"Final reward: {final_reward:.2f}")
    print(f"Experiment data: {experiment_path}")
    
    return {
        'final_reward': final_reward,
        'experiment_path': experiment_path,
        'metrics_file': metrics_file,
        'forensics_report': forensics_report
    }


if __name__ == "__main__":
    main()
```

## Running the Example

```bash
# Run the complete example
python examples/basic_ppo_cartpole.py

# Expected output:
# 🚀 Starting PPO training with RLDK...
# Seed: 42, Updates: 20, Episodes per update: 5
# Update  0: Reward= 23.40±12.34, KL=0.0123, Policy Loss=0.0456
# ...
# ✅ Experiment tracking completed: ./experiments/ppo_cartpole_demo_20240101_120000
# 🎉 PPO CartPole demo completed successfully!
```

## What This Example Demonstrates

### 1. Experiment Tracking
- Complete experiment setup with metadata
- Model and hyperparameter tracking
- Reproducible seed management
- Automatic experiment organization

### 2. Training Integration
- Real-time metrics collection
- Structured logging in JSONL format
- Integration with existing training loops

### 3. Forensics Analysis
- PPO-specific anomaly detection
- Gradient norm monitoring
- KL divergence tracking
- Health score computation

### 4. Reproducibility
- Deterministic seed setting
- Determinism verification
- Consistent environment setup

### 5. CPU-Friendly Design
- Small model and environment
- Fast training (< 1 minute)
- Minimal dependencies
- No GPU requirements

## Key Features Highlighted

- **Zero-overhead integration**: RLDK doesn't interfere with training
- **Comprehensive tracking**: Everything needed for reproducibility
- **Real-time analysis**: Immediate feedback on training health
- **Easy debugging**: Clear anomaly detection and reporting
- **Production ready**: Patterns that scale to larger experiments

## Next Steps

After running this example, try:

1. **Modify hyperparameters** and see how forensics detects changes
2. **Break determinism** by removing seed setting
3. **Add custom metrics** to the tracking system
4. **Scale up** to larger environments and models
5. **Integrate with your existing training code**

## Related Examples

- [TRL Integration](trl-integration.md) - Using RLDK with TRL framework
- [OpenRLHF Integration](openrlhf-integration.md) - Distributed training with RLDK
- [Advanced Forensics](../user-guide/forensics.md) - Deep dive into anomaly detection

For more details, see the [User Guide](../user-guide/tracking.md) and [API Reference](../reference/api.md).
