#!/usr/bin/env python3
"""
RLDK CPU Quickstart - 2 Minute Success Path

This script demonstrates RLDK's core functionality in a CPU-friendly, 
quick demonstration that completes in under 2 minutes.

Features:
- Simple PPO training on CartPole (CPU-only)
- Experiment tracking with RLDK
- Forensics analysis
- JSON and PNG output generation
- W&B disabled by default (respects WANDB_DISABLED)

Usage:
    python examples/quickstart_cpu.py

Output:
    - training_results.json: Training metrics and analysis
    - training_plot.png: Training visualization
    - runs/ directory: Complete experiment tracking data
"""

import json
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# RLDK imports
from rldk.tracking import ExperimentTracker, TrackingConfig
from rldk.forensics import ComprehensivePPOForensics
from rldk.utils.seed import set_global_seed


def simple_ppo_train(env_name="CartPole-v1", episodes=20, max_steps=200):
    """
    Simple PPO implementation for quick demonstration.
    Designed to run fast on CPU.
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Simple linear policy
    policy = np.random.randn(state_dim, action_dim) * 0.1
    value_fn = np.random.randn(state_dim, 1) * 0.1
    
    # Training parameters
    lr = 0.01
    gamma = 0.99
    eps_clip = 0.2
    
    episode_rewards = []
    episode_lengths = []
    training_metrics = []
    
    print(f"🎯 Training PPO on {env_name} for {episodes} episodes")
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        states, actions, rewards, log_probs, values = [], [], [], [], []
        
        for step in range(max_steps):
            # Get action from policy
            logits = state @ policy
            probs = softmax(logits)
            
            # Ensure probabilities are valid
            if np.any(np.isnan(probs)) or np.sum(probs) == 0:
                probs = np.ones(action_dim) / action_dim  # Uniform distribution
            
            action = np.random.choice(action_dim, p=probs)
            log_prob = np.log(probs[action] + 1e-8)
            value = (state @ value_fn).item()
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        # Simple policy update (simplified PPO)
        if len(states) > 1:
            # Compute returns and advantages
            returns = compute_returns(rewards, gamma)
            advantages = [returns[i] - values[i] for i in range(len(rewards))]
            
            # Update policy and value function
            for i, state in enumerate(states):
                action = actions[i]
                advantage = advantages[i]
                
                # Policy update
                logits = state @ policy
                probs = softmax(logits)
                log_prob_new = np.log(probs[action] + 1e-8)
                ratio = np.exp(log_prob_new - log_probs[i])
                
                # Clipped objective
                surr1 = ratio * advantage
                surr2 = np.clip(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
                policy_loss = -np.minimum(surr1, surr2)
                
                # Value update
                value_new = (state @ value_fn).item()
                value_loss = (value_new - returns[i]) ** 2
                
                # Gradient step with clipping
                policy_grad = lr * policy_loss * state.reshape(-1, 1)
                value_grad = lr * value_loss * state.reshape(-1, 1)
                
                # Clip gradients to prevent overflow
                policy_grad = np.clip(policy_grad, -1.0, 1.0)
                value_grad = np.clip(value_grad, -1.0, 1.0)
                
                policy -= policy_grad
                value_fn -= value_grad
            
            # Compute training metrics
            kl_div = compute_kl_divergence(states, policy, log_probs)
            # Compute entropy safely
            entropy_values = []
            for log_prob in log_probs:
                if not np.isnan(log_prob) and log_prob > -np.inf:
                    entropy_values.append(log_prob * np.log(log_prob + 1e-8))
            entropy = -np.mean(entropy_values) if entropy_values else 0.0
            
            training_metrics.append({
                'episode': episode,
                'reward': episode_reward,
                'length': episode_length,
                'kl': kl_div,
                'entropy': entropy,
                'advantage_mean': np.mean(advantages),
                'advantage_std': np.std(advantages)
            })
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Progress update
        if (episode + 1) % 5 == 0:
            avg_reward = np.mean(episode_rewards[-5:])
            print(f"  Episode {episode + 1:2d}: Avg Reward = {avg_reward:6.2f}, Length = {episode_length:3d}")
    
    env.close()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'training_metrics': training_metrics,
        'policy': policy,
        'value_fn': value_fn
    }


def softmax(x):
    """Softmax function with numerical stability."""
    # Add small epsilon to prevent overflow
    x = np.clip(x, -500, 500)
    exp_x = np.exp(x - np.max(x))
    return exp_x / (np.sum(exp_x) + 1e-8)


def compute_returns(rewards, gamma):
    """Compute discounted returns."""
    returns = []
    R = 0
    for reward in reversed(rewards):
        R = reward + gamma * R
        returns.insert(0, R)
    return returns


def compute_kl_divergence(states, policy, old_log_probs):
    """Compute KL divergence between old and new policy."""
    kl_div = 0.0
    for i, state in enumerate(states):
        logits = state @ policy
        probs = softmax(logits)
        action = np.argmax(probs)  # Use most likely action for simplicity
        new_log_prob = np.log(probs[action] + 1e-8)
        kl_div += new_log_prob - old_log_probs[i]
    return kl_div / len(states) if states else 0.0


def main():
    """Main function for CPU quickstart demonstration."""
    
    print("🚀 RLDK CPU Quickstart - 2 Minute Success Path")
    print("=" * 60)
    
    # Set reproducible seed
    seed = set_global_seed(42)
    print(f"🌱 Set global seed to: {seed}")
    
    # Set up experiment tracking with W&B disabled by default
    print("\n📊 Setting up experiment tracking...")
    
    config = TrackingConfig(
        experiment_name="cpu_quickstart_demo",
        enable_dataset_tracking=True,
        enable_model_tracking=True,
        enable_environment_tracking=True,
        enable_seed_tracking=True,
        enable_git_tracking=True,
        save_to_wandb=False,  # Explicitly disable W&B for quickstart
        output_dir=Path("./runs"),
        tags=["quickstart", "cpu", "demo", "2min"],
        notes="RLDK CPU quickstart demonstration - 2 minute success path"
    )
    
    tracker = ExperimentTracker(config)
    tracking_data = tracker.start_experiment()
    
    print(f"✅ Experiment started: {tracking_data['experiment_id']}")
    
    # Run training
    print("\n🎯 Running PPO training...")
    start_time = time.time()
    
    training_results = simple_ppo_train(
        env_name="CartPole-v1",
        episodes=20,  # Reduced for quick demo
        max_steps=200
    )
    
    training_time = time.time() - start_time
    print(f"⏱️  Training completed in {training_time:.2f} seconds")
    
    # Extract results
    episode_rewards = training_results['episode_rewards']
    episode_lengths = training_results['episode_lengths']
    training_metrics = training_results['training_metrics']
    policy = training_results['policy']
    value_fn = training_results['value_fn']
    
    final_avg_reward = np.mean(episode_rewards[-5:])
    print(f"📊 Final average reward: {final_avg_reward:.2f}")
    
    # Track training data
    print("\n📁 Tracking training data...")
    
    metrics_df = pd.DataFrame(training_metrics)
    
    tracker.track_dataset(
        metrics_df,
        "training_metrics",
        {
            "episodes": len(episode_rewards),
            "total_steps": sum(episode_lengths),
            "final_reward": episode_rewards[-1],
            "avg_reward": np.mean(episode_rewards),
            "training_time": training_time
        }
    )
    
    # Track the trained model
    tracker.track_model(
        {"policy": policy, "value_fn": value_fn},
        "ppo_cartpole_model",
        {
            "algorithm": "PPO",
            "environment": "CartPole-v1",
            "state_dim": policy.shape[0],
            "action_dim": policy.shape[1],
            "training_time": training_time
        }
    )
    
    # Add custom metadata
    tracker.add_metadata("training_time_seconds", training_time)
    tracker.add_metadata("final_avg_reward", final_avg_reward)
    tracker.add_metadata("max_reward", max(episode_rewards))
    tracker.add_metadata("min_reward", min(episode_rewards))
    
    # Run forensics analysis
    print("\n🔍 Running forensics analysis...")
    
    forensics = ComprehensivePPOForensics(
        kl_target=0.1,
        enable_kl_schedule_tracking=True,
        enable_gradient_norms_analysis=True,
        enable_advantage_statistics=True
    )
    
    # Update forensics with training data
    for metric in training_metrics:
        forensics.update(
            step=metric['episode'],
            kl=metric['kl'],
            kl_coef=0.2,
            entropy=metric['entropy'],
            reward_mean=metric['reward'],
            reward_std=0.1,  # Simplified
            policy_grad_norm=1.0,  # Simplified
            value_grad_norm=1.0,  # Simplified
            advantage_mean=metric['advantage_mean'],
            advantage_std=metric['advantage_std']
        )
    
    # Get analysis results
    analysis = forensics.get_comprehensive_analysis()
    anomalies = forensics.get_anomalies()
    health_summary = forensics.get_health_summary()
    
    print(f"🔍 Forensics Results:")
    print(f"   Anomalies detected: {len(anomalies)}")
    print(f"   Health score: {health_summary.get('overall_health', 'N/A')}")
    
    if anomalies:
        print("   🚨 Anomalies found:")
        for i, anomaly in enumerate(anomalies[:3]):  # Show first 3
            if isinstance(anomaly, dict):
                rule = anomaly.get('rule', f'Anomaly {i+1}')
                desc = anomaly.get('description', str(anomaly))
                print(f"      - {rule}: {desc}")
            else:
                print(f"      - Anomaly {i+1}: {anomaly}")
    else:
        print("   ✅ No anomalies detected - training looks healthy!")
    
    # Create visualization
    print("\n📊 Creating training visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Episode rewards
    ax1.plot(episode_rewards, alpha=0.7, label='Episode Reward')
    ax1.plot(pd.Series(episode_rewards).rolling(5).mean(), color='red', linewidth=2, label='5-Episode Average')
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Episode lengths
    ax2.plot(episode_lengths, alpha=0.7, label='Episode Length')
    ax2.plot(pd.Series(episode_lengths).rolling(5).mean(), color='red', linewidth=2, label='5-Episode Average')
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("📊 Training plot saved to training_plot.png")
    
    # Create JSON results
    print("\n💾 Creating training results JSON...")
    
    results = {
        "experiment_id": tracking_data['experiment_id'],
        "experiment_name": "cpu_quickstart_demo",
        "training_summary": {
            "episodes": len(episode_rewards),
            "total_steps": sum(episode_lengths),
            "training_time_seconds": training_time,
            "final_avg_reward": final_avg_reward,
            "max_reward": max(episode_rewards),
            "min_reward": min(episode_rewards)
        },
        "forensics_analysis": {
            "anomalies_count": len(anomalies),
            "health_score": health_summary.get('overall_health', 'N/A'),
            "anomalies": anomalies[:5]  # Top 5 anomalies
        },
        "training_metrics": training_metrics,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("💾 Training results saved to training_results.json")
    
    # Finish experiment
    print("\n🏁 Finishing experiment...")
    summary = tracker.finish_experiment()
    
    print("\n🎉 CPU Quickstart completed successfully!")
    print("=" * 60)
    print(f"⏱️  Total time: {time.time() - start_time:.2f} seconds")
    print(f"📊 Final average reward: {final_avg_reward:.2f}")
    print(f"🔍 Anomalies detected: {len(anomalies)}")
    print(f"📁 Experiment ID: {summary['experiment_id']}")
    
    print("\n📁 Generated files:")
    print("   - training_results.json: Complete training results and analysis")
    print("   - training_plot.png: Training visualization")
    print(f"   - {config.output_dir}/{summary['experiment_id']}/: Complete experiment tracking data")
    
    print("\n💡 Next steps:")
    print("   1. Check training_results.json for detailed metrics")
    print("   2. View training_plot.png for visualization")
    print("   3. Explore the runs/ directory for complete tracking data")
    print("   4. Try other RLDK examples for more advanced features")
    print("   5. Integrate RLDK into your own training scripts")
    
    print("\n🚀 RLDK CPU Quickstart - Success! 🎉")


if __name__ == "__main__":
    main()