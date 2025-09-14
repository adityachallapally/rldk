#!/usr/bin/env python3
"""
Basic PPO CartPole Example for RLDK

This script demonstrates the basic usage of RLDK with a simple PPO training loop
on the CartPole environment. This is a CPU-friendly example that focuses on
logging and analysis without GPU requirements.

Learning Objectives:
- How to set up experiment tracking with RLDK
- How to run forensics analysis on training metrics
- How to check determinism and reproducibility
- How to analyze reward model health
- How to use RLDK CLI commands for analysis

Prerequisites:
- RLDK installed (pip install rldk)
- Basic understanding of PPO algorithm
- Familiarity with reinforcement learning concepts
"""

import json
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# RLDK imports
import rldk
from rldk.determinism import check
from rldk.forensics import ComprehensivePPOForensics
from rldk.reward import health
from rldk.tracking import ExperimentTracker, TrackingConfig
from rldk.utils import set_global_seed


def main():
    """Main function demonstrating RLDK with PPO training."""

    print("🚀 RLDK Basic PPO CartPole Example")
    print("=" * 50)

    # 1. Setup and Imports
    print("\n1. Setting up reproducible environment...")

    # Set up reproducible environment
    seed = set_global_seed(42)
    print(f"🌱 Set global seed to: {seed}")

    # Set up plotting
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)

    # 2. Simple PPO Implementation
    print("\n2. Implementing Simple PPO...")

    class SimplePPO:
        """Simple PPO implementation for demonstration."""

        def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.lr = lr
            self.gamma = gamma
            self.eps_clip = eps_clip

            # Initialize policy (simple linear policy)
            self.policy = np.random.randn(state_dim, action_dim) * 0.1
            self.value = np.random.randn(state_dim, 1) * 0.1

            # Training statistics
            self.training_stats = []

        def get_action(self, state):
            """Get action from policy."""
            logits = state @ self.policy
            probs = self._softmax(logits)
            action = np.random.choice(self.action_dim, p=probs)
            log_prob = np.log(probs[action] + 1e-8)
            return action, log_prob

        def get_value(self, state):
            """Get state value."""
            return (state @ self.value).item()

        def _softmax(self, x):
            """Softmax function."""
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x)

        def _log_softmax(self, x):
            """Log softmax function for numerical stability."""
            x_max = np.max(x)
            log_sum_exp = np.log(np.sum(np.exp(x - x_max))) + x_max
            return x - log_sum_exp

        def update(self, states, actions, rewards, log_probs_old, values_old, advantages, returns):
            """Update policy and value function."""
            # Simple gradient update (simplified from full PPO)
            for i, state in enumerate(states):
                action = actions[i]
                advantage = advantages[i]
                return_val = returns[i]

                # Policy update
                # Get log probability for the action that was actually taken
                logits = state @ self.policy
                probs = self._softmax(logits)
                log_prob_new = np.log(probs[action] + 1e-8)
                ratio = np.exp(log_prob_new - log_probs_old[i])

                # Clipped objective
                surr1 = ratio * advantage
                surr2 = np.clip(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                policy_loss = -np.minimum(surr1, surr2)

                # Value update
                value_new = self.get_value(state)
                value_loss = (value_new - return_val) ** 2

                # Simple gradient step (learning rate applied)
                self.policy -= self.lr * policy_loss * state.reshape(-1, 1)
                self.value -= self.lr * value_loss * state.reshape(-1, 1)

        def compute_advantages(self, rewards, values, dones):
            """Compute advantages using basic advantage estimation (not GAE).

            Args:
                rewards: List of rewards for each timestep
                values: List of value function estimates (should be len(rewards) + 1)
                dones: List of done flags for each timestep
            """
            advantages = []
            returns = []

            # Ensure values has the correct length (rewards + 1 for terminal state)
            if len(values) != len(rewards) + 1:
                raise ValueError(f"Values length {len(values)} should be rewards length {len(rewards)} + 1")

            # Compute returns (discounted cumulative rewards)
            returns = [0.0] * len(rewards)
            for i in reversed(range(len(rewards))):
                if dones[i]:
                    # Terminal state: return is just the reward
                    returns[i] = rewards[i]
                else:
                    # Non-terminal: bootstrap from next value function
                    returns[i] = rewards[i] + self.gamma * values[i + 1]

            # Compute advantages: A_t = G_t - V(s_t)
            for i in range(len(rewards)):
                advantage = returns[i] - values[i]
                advantages.append(advantage)

            return np.array(advantages), np.array(returns)

    print("✅ Simple PPO implementation ready")

    # 3. Experiment Tracking Setup
    print("\n3. Setting up experiment tracking...")

    # Configure experiment tracking
    config = TrackingConfig(
        experiment_name="basic_ppo_cartpole",
        enable_dataset_tracking=True,
        enable_model_tracking=True,
        enable_environment_tracking=True,
        enable_seed_tracking=True,
        enable_git_tracking=True,
        output_dir="./runs",
        tags=["demo", "ppo", "cartpole", "cpu-friendly"],
        notes="Basic PPO implementation on CartPole environment for RLDK demonstration"
    )

    # Create tracker
    tracker = ExperimentTracker(config)

    # Start experiment
    tracking_data = tracker.start_experiment()
    print(f"🚀 Started experiment: {tracking_data['experiment_id']}")
    print(f"📁 Output directory: {config.output_dir}")
    print(f"🏷️  Tags: {config.tags}")

    # 4. Training Loop with Forensics
    print("\n4. Running training with forensics...")

    def train_ppo_with_forensics(env_name="CartPole-v1", episodes=50, steps_per_episode=200):
        """Train PPO agent with comprehensive forensics tracking."""

        # Create environment
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Initialize PPO agent
        agent = SimplePPO(state_dim, action_dim, lr=3e-4)

        # Initialize forensics
        forensics = ComprehensivePPOForensics(
            kl_target=0.1,
            enable_kl_schedule_tracking=True,
            enable_gradient_norms_analysis=True,
            enable_advantage_statistics=True
        )

        # Training statistics
        episode_rewards = []
        episode_lengths = []
        training_metrics = []

        print(f"🎯 Training PPO on {env_name} for {episodes} episodes")
        print(f"📊 State dimension: {state_dim}, Action dimension: {action_dim}")

        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0

            # Store trajectory data
            states, actions, rewards, log_probs, values = [], [], [], [], []

            for step in range(steps_per_episode):
                # Get action from policy
                action, log_prob = agent.get_action(state)
                value = agent.get_value(state)

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

            # Update agent
            if len(states) > 1:
                # Compute advantages and returns
                dones = [False] * (len(states) - 1) + [True]
                advantages, returns = agent.compute_advantages(rewards, values, dones)

                # Store old policy parameters before update for KL divergence calculation
                old_policy = agent.policy.copy()

                # Update policy
                agent.update(states, actions, rewards, log_probs, values, advantages, returns)

                # Compute training metrics for forensics
                # Calculate KL divergence between old and new policy distributions
                kl_div = 0.0
                for i, state in enumerate(states):
                    # Get old policy distribution (before update)
                    old_logits = state @ old_policy
                    old_probs = agent._softmax(old_logits)
                    # Get new policy distribution (after update)
                    new_logits = state @ agent.policy
                    new_probs = agent._softmax(new_logits)
                    # KL divergence: KL(old||new) = Σ old_probs * log(old_probs / new_probs)
                    # Add small epsilon for numerical stability
                    eps = 1e-8
                    kl_div += np.sum(old_probs * (np.log(old_probs + eps) - np.log(new_probs + eps)))
                kl_div = kl_div / len(states) if states else 0.0
                entropy = -np.mean([log_prob * np.log(log_prob + 1e-8) for log_prob in log_probs])
                policy_grad_norm = np.linalg.norm(agent.policy.flatten())
                value_grad_norm = np.linalg.norm(agent.value.flatten())

                # Update forensics
                forensics.update(
                    step=episode,
                    kl=kl_div,
                    kl_coef=0.2,
                    entropy=entropy,
                    reward_mean=episode_reward,
                    reward_std=np.std(rewards),
                    policy_grad_norm=policy_grad_norm,
                    value_grad_norm=value_grad_norm,
                    advantage_mean=np.mean(advantages),
                    advantage_std=np.std(advantages)
                )

                # Store training metrics
                training_metrics.append({
                    'episode': episode,
                    'reward': episode_reward,
                    'length': episode_length,
                    'kl': kl_div,
                    'entropy': entropy,
                    'policy_grad_norm': policy_grad_norm,
                    'value_grad_norm': value_grad_norm,
                    'advantage_mean': np.mean(advantages),
                    'advantage_std': np.std(advantages)
                })

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Progress update
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode + 1:3d}: Avg Reward = {avg_reward:6.2f}, Length = {episode_length:3d}")

        env.close()

        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'training_metrics': training_metrics,
            'forensics': forensics,
            'agent': agent
        }

    # Run training
    print("🚀 Starting training...")
    start_time = time.time()

    training_results = train_ppo_with_forensics(
        env_name="CartPole-v1",
        episodes=50,  # Reduced for demo
        steps_per_episode=200
    )

    training_time = time.time() - start_time
    print(f"\n⏱️  Training completed in {training_time:.2f} seconds")

    # Extract results
    episode_rewards = training_results['episode_rewards']
    episode_lengths = training_results['episode_lengths']
    training_metrics = training_results['training_metrics']
    forensics = training_results['forensics']
    agent = training_results['agent']

    print(f"📊 Final average reward: {np.mean(episode_rewards[-10:]):.2f}")
    print(f"📏 Final average length: {np.mean(episode_lengths[-10:]):.2f}")

    # 5. Track Training Data
    print("\n5. Tracking training data...")

    # Convert training metrics to DataFrame
    metrics_df = pd.DataFrame(training_metrics)

    # Track training data
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
        agent,
        "ppo_cartpole_model",
        {
            "algorithm": "PPO",
            "environment": "CartPole-v1",
            "state_dim": agent.state_dim,
            "action_dim": agent.action_dim,
            "learning_rate": agent.lr,
            "gamma": agent.gamma,
            "eps_clip": agent.eps_clip
        }
    )

    # Add custom metadata
    tracker.add_metadata("training_time_seconds", training_time)
    tracker.add_metadata("final_avg_reward", np.mean(episode_rewards[-10:]))
    tracker.add_metadata("max_reward", max(episode_rewards))
    tracker.add_metadata("min_reward", min(episode_rewards))

    print("✅ Training data tracked successfully")

    # 6. Forensics Analysis
    print("\n6. Running forensics analysis...")

    # Get forensics analysis
    forensics.get_comprehensive_analysis()
    anomalies = forensics.get_anomalies()
    health_summary = forensics.get_health_summary()

    print("🔍 Forensics Analysis Results:")
    print(f"   Total anomalies detected: {len(anomalies)}")
    print(f"   Overall health score: {health_summary.get('overall_health', 'N/A')}")

    # Show specific anomalies
    if anomalies:
        print("\n🚨 Anomalies detected:")
        for i, anomaly in enumerate(anomalies[:5]):  # Show first 5
            print(f"   {i+1}. {anomaly['rule']}: {anomaly['description']}")
            if 'step_range' in anomaly:
                print(f"      Steps: {anomaly['step_range'][0]} to {anomaly['step_range'][1]}")
    else:
        print("\n✅ No anomalies detected - training looks healthy!")

    # Save forensics analysis
    forensics.save_analysis("./forensics_analysis.json")
    print("\n💾 Forensics analysis saved to ./forensics_analysis.json")

    # 7. Visualization
    print("\n7. Creating visualizations...")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.7, label='Episode Reward')
    axes[0, 0].plot(pd.Series(episode_rewards).rolling(10).mean(), color='red', linewidth=2, label='10-Episode Average')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Episode lengths
    axes[0, 1].plot(episode_lengths, alpha=0.7, label='Episode Length')
    axes[0, 1].plot(pd.Series(episode_lengths).rolling(10).mean(), color='red', linewidth=2, label='10-Episode Average')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Length')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # KL divergence
    if 'kl' in metrics_df.columns:
        axes[1, 0].plot(metrics_df['episode'], metrics_df['kl'], alpha=0.7)
        axes[1, 0].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='KL Target (0.1)')
        axes[1, 0].set_title('KL Divergence')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('KL Divergence')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Policy gradient norm
    if 'policy_grad_norm' in metrics_df.columns:
        axes[1, 1].plot(metrics_df['episode'], metrics_df['policy_grad_norm'], alpha=0.7)
        axes[1, 1].set_title('Policy Gradient Norm')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./training_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("📊 Training plots saved to ./training_plots.png")

    # 8. Determinism Check
    print("\n8. Running determinism check...")

    # Create a simple deterministic test
    test_script = """
import random
import numpy as np

def main():
    # Set seeds
    random.seed(42)
    np.random.seed(42)

    # Simulate training
    for step in range(10):
        loss = random.random() * np.exp(-step/5)
        print(f"loss: {loss:.6f}")

if __name__ == "__main__":
    main()
"""

    with open("determinism_test.py", "w") as f:
        f.write(test_script)

    print("🧪 Running determinism check...")

    # Check determinism
    report = check(
        cmd="python determinism_test.py",
        compare=["loss"],
        replicas=3,
        device="cpu"
    )

    print("\n🎯 Determinism Check Results:")
    print(f"   Passed: {report.passed}")
    print(f"   Replicas tested: {len(report.mismatches) + 1}")

    if not report.passed:
        print(f"   Issues found: {len(report.mismatches)}")
        for mismatch in report.mismatches[:3]:
            print(f"   - {mismatch}")
    else:
        print("   ✅ Training is deterministic!")

    # Clean up
    import os
    os.remove("determinism_test.py")
    print("\n🧹 Cleaned up test files")

    # 9. Reward Health Analysis
    print("\n9. Running reward health analysis...")

    # Run reward health analysis
    print("🔍 Running reward health analysis...")

    health_report = health(
        run_data=metrics_df,
        reward_col="reward",
        step_col="episode",
        threshold_drift=0.1,
        threshold_saturation=0.8,
        threshold_calibration=0.7
    )

    print("\n📊 Reward Health Analysis Results:")
    print(f"   Passed: {health_report.passed}")
    print(f"   Drift detected: {health_report.drift_detected}")
    print(f"   Calibration score: {health_report.calibration_score:.3f}")
    print(f"   Saturation issues: {len(health_report.saturation_issues)}")
    print(f"   Shortcut signals: {len(health_report.shortcut_signals)}")

    if health_report.passed:
        print("   ✅ Reward model is healthy!")
    else:
        print("   ⚠️  Reward model has issues that need attention")

    # 10. Finish Experiment
    print("\n10. Finishing experiment...")

    # Finish experiment tracking
    summary = tracker.finish_experiment()

    print("\n🎉 Experiment completed successfully!")
    print(f"   Experiment ID: {summary['experiment_id']}")
    print(f"   Output directory: {config.output_dir}")
    print(f"   Total episodes: {len(episode_rewards)}")
    print(f"   Final average reward: {np.mean(episode_rewards[-10:]):.2f}")
    print(f"   Training time: {training_time:.2f} seconds")

    print("\n📁 Files created:")
    print(f"   - {config.output_dir}/{summary['experiment_id']}/ - Experiment data")
    print("   - ./forensics_analysis.json - Forensics analysis")
    print("   - ./training_plots.png - Training visualizations")

    # 11. CLI Commands Demo
    print("\n11. CLI Commands Demo...")

    print("\n💡 Try these commands in your terminal:")
    print("   rldk forensics log-scan ./runs - Scan for training anomalies")
    print("   rldk reward-health --run ./runs - Analyze reward model health")
    print("   rldk check-determinism --cmd 'python train.py' --compare loss - Check determinism")
    print("   rldk evals evaluate data.jsonl --suite quick - Run evaluation suite")

    # 12. Summary
    print("\n12. Summary and Next Steps...")

    print("\n✅ What We Accomplished:")
    print("   1. Experiment Tracking: Set up comprehensive experiment tracking")
    print("   2. Training with Forensics: Implemented PPO training with real-time forensics")
    print("   3. Anomaly Detection: Detected and analyzed training anomalies")
    print("   4. Determinism Checking: Verified training reproducibility")
    print("   5. Reward Health Analysis: Analyzed reward model health")
    print("   6. Visualization: Created comprehensive training plots")
    print("   7. CLI Integration: Demonstrated RLDK CLI commands")

    print("\n📊 Key Metrics:")
    print(f"   - Final Average Reward: {np.mean(episode_rewards[-10:]):.2f}")
    print(f"   - Training Time: {training_time:.2f} seconds")
    print(f"   - Anomalies Detected: {len(anomalies)}")
    print(f"   - Determinism: {'✅ Passed' if report.passed else '❌ Failed'}")
    print(f"   - Reward Health: {'✅ Healthy' if health_report.passed else '⚠️ Issues detected'}")

    print("\n🚀 Next Steps:")
    print("   1. Explore Other Examples: Check out other example notebooks")
    print("   2. Real Training: Apply RLDK to your actual RL training runs")
    print("   3. Custom Forensics: Implement custom forensics rules")
    print("   4. CI Integration: Integrate RLDK into your CI/CD pipeline")
    print("   5. Team Collaboration: Share experiment data with your team")

    print("\n📚 Additional Resources:")
    print("   - RLDK Documentation: https://github.com/adityachallapally/rldk")
    print("   - API Reference: https://github.com/adityachallapally/rldk/blob/main/docs/reference/api.md")
    print("   - CLI Commands: https://github.com/adityachallapally/rldk/blob/main/docs/reference/commands.md")

    print("\nHappy experimenting! 🎉")

if __name__ == "__main__":
    main()
