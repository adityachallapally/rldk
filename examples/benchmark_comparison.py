#!/usr/bin/env python3
"""
Benchmark Comparison Example for RLDK

This example demonstrates how to use RLDK for systematic benchmarking of
different RL algorithms, environments, and hyperparameters. We'll implement
a comprehensive benchmarking framework that compares multiple algorithms
across different environments with statistical rigor.

Learning Objectives:
- How to structure benchmarking experiments with RLDK
- How to compare multiple RL algorithms systematically
- How to handle statistical significance testing
- How to create reproducible benchmark suites
- How to analyze algorithm performance across environments
- How to detect algorithm-specific failure modes

Prerequisites:
- RLDK installed (pip install rldk)
- Basic understanding of RL algorithms
- Familiarity with statistical testing
- Understanding of benchmarking best practices
"""

import json
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

import gymnasium as gym

# RLDK imports
import rldk
from rldk.diff import first_divergence
from rldk.forensics import ComprehensivePPOForensics
from rldk.tracking import ExperimentTracker, TrackingConfig
from rldk.utils import set_global_seed, validate_numeric_range


class SimplePPO:
    """Simple PPO implementation for benchmarking."""

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2,
                 entropy_coef=0.01, value_coef=0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        # Initialize policy and value networks
        self.policy = np.random.randn(state_dim, action_dim) * 0.1
        self.value = np.random.randn(state_dim, 1) * 0.1

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

    def update(self, states, actions, rewards, log_probs_old, values_old, advantages, returns):
        """Update policy and value function."""
        for i, state in enumerate(states):
            action = actions[i]
            advantage = advantages[i]
            return_val = returns[i]

            # Policy update - get log probability for the action that was actually taken
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
            value_loss = self.value_coef * (value_new - return_val) ** 2

            # Entropy bonus - use current policy distribution
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            entropy_loss = -self.entropy_coef * entropy

            # Total loss
            policy_loss + value_loss + entropy_loss

            # Simple gradient step
            self.policy -= self.lr * policy_loss * state.reshape(-1, 1)
            self.value -= self.lr * value_loss * state.reshape(-1, 1)

    def compute_advantages(self, rewards, values, dones):
        """Compute advantages using GAE."""
        advantages = []
        returns = []

        for i in range(len(rewards)):
            if i == len(rewards) - 1 or dones[i]:
                advantage = rewards[i] - values[i]
                return_val = rewards[i]
            else:
                advantage = rewards[i] + self.gamma * values[i+1] - values[i]
                return_val = rewards[i] + self.gamma * values[i+1]

            advantages.append(advantage)
            returns.append(return_val)

        return np.array(advantages), np.array(returns)

class SimpleDQN:
    """Simple DQN implementation for benchmarking."""

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, memory_size=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory_size = memory_size

        # Initialize Q-network
        self.q_network = np.random.randn(state_dim, action_dim) * 0.1
        self.target_network = self.q_network.copy()

        # Experience replay buffer
        self.memory = []

    def get_action(self, state):
        """Get action using epsilon-greedy policy."""
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            q_values = state @ self.q_network
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def replay(self, batch_size=32):
        """Train on a batch of experiences."""
        if len(self.memory) < batch_size:
            return

        batch = np.random.choice(len(self.memory), batch_size, replace=False)

        for i in batch:
            state, action, reward, next_state, done = self.memory[i]

            target = reward
            if not done:
                target = reward + self.gamma * np.max(next_state @ self.target_network)

            current_q = state @ self.q_network
            current_q[action] = target

            # Simple gradient step
            self.q_network -= self.lr * (current_q - state @ self.q_network) * state.reshape(-1, 1)

        # Update target network
        if np.random.random() < 0.1:  # Update target network occasionally
            self.target_network = self.q_network.copy()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class SimpleA2C:
    """Simple A2C implementation for benchmarking."""

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, entropy_coef=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        # Initialize policy and value networks
        self.policy = np.random.randn(state_dim, action_dim) * 0.1
        self.value = np.random.randn(state_dim, 1) * 0.1

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

    def update(self, states, actions, rewards, log_probs, values, dones):
        """Update policy and value function."""
        for i, state in enumerate(states):
            reward = rewards[i]
            log_prob = log_probs[i]
            value = values[i]
            done = dones[i]

            # Compute target value
            if done:
                target_value = reward
            else:
                target_value = reward + self.gamma * self.get_value(state)

            # Value loss
            value_loss = (value - target_value) ** 2

            # Policy loss
            advantage = target_value - value
            policy_loss = -log_prob * advantage

            # Entropy loss
            probs = self._softmax(state @ self.policy)
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            entropy_loss = -self.entropy_coef * entropy

            # Total loss
            policy_loss + value_loss + entropy_loss

            # Simple gradient step
            self.policy -= self.lr * policy_loss * state.reshape(-1, 1)
            self.value -= self.lr * value_loss * state.reshape(-1, 1)

class BenchmarkSuite:
    """Comprehensive benchmarking suite for RL algorithms."""

    def __init__(self, algorithms: List[Any], environments: List[str],
                 seeds: List[int], episodes: int = 100):
        self.algorithms = algorithms
        self.environments = environments
        self.seeds = seeds
        self.episodes = episodes

        # Results storage
        self.results = []
        self.algorithm_names = [alg.__class__.__name__ for alg in algorithms]

        # Initialize forensics for each algorithm
        self.forensics = {
            name: ComprehensivePPOForensics(kl_target=0.1)
            for name in self.algorithm_names
        }

    def run_single_experiment(self, algorithm, env_name, seed, episodes=None):
        """Run a single experiment with given algorithm, environment, and seed."""

        if episodes is None:
            episodes = self.episodes

        # Set seed
        set_global_seed(seed)

        # Create environment
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Initialize algorithm
        if algorithm.__class__.__name__ == 'SimplePPO':
            agent = SimplePPO(state_dim, action_dim)
        elif algorithm.__class__.__name__ == 'SimpleDQN':
            agent = SimpleDQN(state_dim, action_dim)
        elif algorithm.__class__.__name__ == 'SimpleA2C':
            agent = SimpleA2C(state_dim, action_dim)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm.__class__.__name__}")

        # Training statistics
        episode_rewards = []
        episode_lengths = []
        training_metrics = []

        # Training loop
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0

            # Store trajectory data
            states, actions, rewards, log_probs, values = [], [], [], [], []

            for step in range(200):  # Max steps per episode
                # Get action from policy
                if algorithm.__class__.__name__ == 'SimpleDQN':
                    action = agent.get_action(state)
                    log_prob = 0.0  # DQN doesn't have explicit log probabilities
                    value = 0.0  # DQN doesn't have explicit values
                else:
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
                if algorithm.__class__.__name__ == 'SimpleDQN':
                    # DQN experience replay
                    for i in range(len(states) - 1):
                        agent.remember(states[i], actions[i], rewards[i], states[i+1], done)
                    agent.replay()
                else:
                    # Store old policy parameters before update for KL divergence calculation
                    old_policy = agent.policy.copy()

                    # PPO/A2C update
                    if algorithm.__class__.__name__ == 'SimplePPO':
                        # Compute advantages and returns
                        dones = [False] * (len(states) - 1) + [True]
                        advantages, returns = agent.compute_advantages(rewards, values, dones)

                        # Update policy
                        agent.update(states, actions, rewards, log_probs, values, advantages, returns)
                    else:  # A2C
                        # Simple A2C update
                        dones = [False] * (len(states) - 1) + [True]
                        agent.update(states, actions, rewards, log_probs, values, dones)

                # Compute training metrics for forensics
                if algorithm.__class__.__name__ != 'SimpleDQN':
                    # Calculate KL divergence between old and new policy distributions
                    kl_div = 0.0
                    for i, state in enumerate(states):
                        # Get old policy distribution (before update)
                        old_logits = state @ old_policy
                        old_probs = agent._softmax(old_logits)
                        # Get new policy distribution (after update)
                        new_logits = state @ agent.policy
                        new_probs = agent._softmax(new_logits)
                        kl_div += np.sum(old_probs * np.log((old_probs + 1e-8) / (new_probs + 1e-8)))
                    kl_div = kl_div / len(states) if states else 0.0
                    entropy = -np.mean([log_prob * np.log(log_prob + 1e-8) for log_prob in log_probs])
                    policy_grad_norm = np.linalg.norm(agent.policy.flatten())
                    value_grad_norm = np.linalg.norm(agent.value.flatten())

                    # Update forensics
                    self.forensics[algorithm.__class__.__name__].update(
                        step=episode,
                        kl=kl_div,
                        kl_coef=0.2,
                        entropy=entropy,
                        reward_mean=episode_reward,
                        reward_std=np.std(rewards),
                        policy_grad_norm=policy_grad_norm,
                        value_grad_norm=value_grad_norm,
                        advantage_mean=0.0,  # Placeholder
                        advantage_std=0.1
                    )

                    # Store training metrics
                    training_metrics.append({
                        'episode': episode,
                        'reward': episode_reward,
                        'length': episode_length,
                        'kl': kl_div,
                        'entropy': entropy,
                        'policy_grad_norm': policy_grad_norm,
                        'value_grad_norm': value_grad_norm
                    })
                else:
                    # DQN metrics
                    training_metrics.append({
                        'episode': episode,
                        'reward': episode_reward,
                        'length': episode_length,
                        'epsilon': agent.epsilon,
                        'q_value_mean': np.mean(state @ agent.q_network)
                    })

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        env.close()

        # Get forensics analysis
        analysis = self.forensics[algorithm.__class__.__name__].get_comprehensive_analysis()
        anomalies = self.forensics[algorithm.__class__.__name__].get_anomalies()
        health_summary = self.forensics[algorithm.__class__.__name__].get_health_summary()

        return {
            'algorithm': algorithm.__class__.__name__,
            'environment': env_name,
            'seed': seed,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'training_metrics': training_metrics,
            'analysis': analysis,
            'anomalies': anomalies,
            'health_summary': health_summary
        }

    def run_benchmark(self, episodes=None):
        """Run complete benchmark suite."""

        if episodes is None:
            episodes = self.episodes

        print("🚀 Starting benchmark suite...")
        print(f"   Algorithms: {self.algorithm_names}")
        print(f"   Environments: {self.environments}")
        print(f"   Seeds: {self.seeds}")
        print(f"   Episodes: {episodes}")
        print(f"   Total experiments: {len(self.algorithms) * len(self.environments) * len(self.seeds)}")

        total_experiments = len(self.algorithms) * len(self.environments) * len(self.seeds)
        experiment_count = 0

        for algorithm in self.algorithms:
            for env_name in self.environments:
                for seed in self.seeds:
                    experiment_count += 1

                    print(f"\n📊 Experiment {experiment_count}/{total_experiments}")
                    print(f"   Algorithm: {algorithm.__class__.__name__}")
                    print(f"   Environment: {env_name}")
                    print(f"   Seed: {seed}")

                    try:
                        result = self.run_single_experiment(algorithm, env_name, seed, episodes)
                        self.results.append(result)

                        # Calculate summary metrics
                        final_avg_reward = np.mean(result['episode_rewards'][-10:])
                        total_anomalies = len(result['anomalies'])
                        health_score = result['health_summary'].get('overall_health', 0)

                        print(f"   Final avg reward: {final_avg_reward:.2f}")
                        print(f"   Anomalies: {total_anomalies}")
                        print(f"   Health score: {health_score:.3f}")

                    except Exception as e:
                        print(f"   ❌ Error: {e}")
                        self.results.append({
                            'algorithm': algorithm.__class__.__name__,
                            'environment': env_name,
                            'seed': seed,
                            'episode_rewards': [],
                            'episode_lengths': [],
                            'training_metrics': [],
                            'analysis': {},
                            'anomalies': [],
                            'health_summary': {},
                            'error': str(e)
                        })

        print("\n✅ Benchmark suite completed!")
        print(f"   Total experiments: {len(self.results)}")
        print(f"   Successful experiments: {len([r for r in self.results if 'error' not in r])}")

        return self.results

    def analyze_results(self):
        """Analyze benchmark results with statistical tests."""

        print("\n📊 Analyzing benchmark results...")

        # Convert results to DataFrame
        df_data = []
        for result in self.results:
            if 'error' not in result:
                # Calculate summary metrics
                final_avg_reward = np.mean(result['episode_rewards'][-10:])
                total_anomalies = len(result['anomalies'])
                health_score = result['health_summary'].get('overall_health', 0)

                df_data.append({
                    'algorithm': result['algorithm'],
                    'environment': result['environment'],
                    'seed': result['seed'],
                    'final_avg_reward': final_avg_reward,
                    'total_anomalies': total_anomalies,
                    'health_score': health_score,
                    'episode_rewards': result['episode_rewards'],
                    'training_metrics': result['training_metrics']
                })

        df = pd.DataFrame(df_data)

        if df.empty:
            print("❌ No valid results to analyze!")
            return None, None

        # Statistical analysis
        print("\n📈 Statistical Analysis:")

        # 1. Algorithm comparison
        algorithm_stats = df.groupby('algorithm')['final_avg_reward'].agg(['mean', 'std', 'count'])
        print("\nAlgorithm Performance:")
        for alg, stats in algorithm_stats.iterrows():
            print(f"   {alg}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")

        # 2. Environment comparison
        env_stats = df.groupby('environment')['final_avg_reward'].agg(['mean', 'std', 'count'])
        print("\nEnvironment Difficulty:")
        for env, stats in env_stats.iterrows():
            print(f"   {env}: {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['count']})")

        # 3. Algorithm-Environment interaction
        interaction_stats = df.groupby(['algorithm', 'environment'])['final_avg_reward'].agg(['mean', 'std'])
        print("\nAlgorithm-Environment Interaction:")
        for (alg, env), stats in interaction_stats.iterrows():
            print(f"   {alg} on {env}: {stats['mean']:.3f} ± {stats['std']:.3f}")

        # 4. Statistical significance tests
        print("\nStatistical Significance Tests:")

        # Pairwise t-tests between algorithms
        algorithms = df['algorithm'].unique()
        for i, alg1 in enumerate(algorithms):
            for alg2 in algorithms[i+1:]:
                alg1_rewards = df[df['algorithm'] == alg1]['final_avg_reward']
                alg2_rewards = df[df['algorithm'] == alg2]['final_avg_reward']

                if len(alg1_rewards) > 1 and len(alg2_rewards) > 1:
                    t_stat, p_value = stats.ttest_ind(alg1_rewards, alg2_rewards)
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    print(f"   {alg1} vs {alg2}: t={t_stat:.3f}, p={p_value:.3f} {significance}")

        # 5. Anomaly analysis
        print("\nAnomaly Analysis:")
        anomaly_stats = df.groupby('algorithm')['total_anomalies'].agg(['mean', 'std'])
        for alg, stats in anomaly_stats.iterrows():
            print(f"   {alg}: {stats['mean']:.1f} ± {stats['std']:.1f} anomalies")

        # 6. Health score analysis
        print("\nHealth Score Analysis:")
        health_stats = df.groupby('algorithm')['health_score'].agg(['mean', 'std'])
        for alg, stats in health_stats.iterrows():
            print(f"   {alg}: {stats['mean']:.3f} ± {stats['std']:.3f} health score")

        return df, interaction_stats

    def create_visualizations(self, df, interaction_stats):
        """Create comprehensive visualizations for benchmark results."""

        if df is None or df.empty:
            print("❌ No data to visualize!")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Algorithm performance comparison
        sns.boxplot(data=df, x='algorithm', y='final_avg_reward', ax=axes[0, 0])
        axes[0, 0].set_title('Algorithm Performance Comparison')
        axes[0, 0].set_xlabel('Algorithm')
        axes[0, 0].set_ylabel('Final Average Reward')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Environment difficulty comparison
        sns.boxplot(data=df, x='environment', y='final_avg_reward', ax=axes[0, 1])
        axes[0, 1].set_title('Environment Difficulty Comparison')
        axes[0, 1].set_xlabel('Environment')
        axes[0, 1].set_ylabel('Final Average Reward')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Algorithm-Environment heatmap
        if interaction_stats is not None:
            pivot_table = interaction_stats['mean'].unstack()
            sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[0, 2])
            axes[0, 2].set_title('Algorithm-Environment Performance Heatmap')
            axes[0, 2].set_xlabel('Environment')
            axes[0, 2].set_ylabel('Algorithm')

        # 4. Anomaly distribution
        sns.boxplot(data=df, x='algorithm', y='total_anomalies', ax=axes[1, 0])
        axes[1, 0].set_title('Anomaly Distribution by Algorithm')
        axes[1, 0].set_xlabel('Algorithm')
        axes[1, 0].set_ylabel('Total Anomalies')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Health score distribution
        sns.boxplot(data=df, x='algorithm', y='health_score', ax=axes[1, 1])
        axes[1, 1].set_title('Health Score Distribution by Algorithm')
        axes[1, 1].set_xlabel('Algorithm')
        axes[1, 1].set_ylabel('Health Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Learning curves (average across seeds)
        for algorithm in df['algorithm'].unique():
            alg_data = df[df['algorithm'] == algorithm]

            # Average episode rewards across seeds
            all_rewards = []
            for _, row in alg_data.iterrows():
                if len(row['episode_rewards']) > 0:
                    all_rewards.append(row['episode_rewards'])

            if all_rewards:
                # Pad to same length
                max_length = max(len(rewards) for rewards in all_rewards)
                padded_rewards = []
                for rewards in all_rewards:
                    padded = rewards + [rewards[-1]] * (max_length - len(rewards))
                    padded_rewards.append(padded)

                # Average across seeds
                avg_rewards = np.mean(padded_rewards, axis=0)
                std_rewards = np.std(padded_rewards, axis=0)

                axes[1, 2].plot(avg_rewards, label=algorithm, alpha=0.8)
                axes[1, 2].fill_between(range(len(avg_rewards)),
                                      avg_rewards - std_rewards,
                                      avg_rewards + std_rewards,
                                      alpha=0.2)

        axes[1, 2].set_title('Learning Curves (Average ± Std)')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Episode Reward')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('./benchmark_comparison_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("📊 Benchmark comparison plots saved to ./benchmark_comparison_plots.png")

def main():
    """Main function demonstrating benchmarking with RLDK."""

    print("🚀 RLDK Benchmark Comparison Example")
    print("=" * 60)

    # 1. Setup
    print("\n1. Setting up reproducible environment...")
    seed = set_global_seed(42)
    print(f"🌱 Set global seed to: {seed}")

    # 2. Define algorithms and environments
    print("\n2. Defining algorithms and environments...")

    # Algorithms
    algorithms = [
        SimplePPO(),
        SimpleDQN(),
        SimpleA2C()
    ]

    # Environments
    environments = [
        "CartPole-v1",
        "Acrobot-v1",
        "MountainCar-v0"
    ]

    # Seeds for statistical robustness
    seeds = [42, 123, 456, 789, 999]

    print(f"   Algorithms: {[alg.__class__.__name__ for alg in algorithms]}")
    print(f"   Environments: {environments}")
    print(f"   Seeds: {seeds}")

    # 3. Initialize Benchmark Suite
    print("\n3. Initializing benchmark suite...")

    benchmark = BenchmarkSuite(
        algorithms=algorithms,
        environments=environments,
        seeds=seeds,
        episodes=50  # Reduced for demo
    )

    # 4. Experiment Tracking Setup
    print("\n4. Setting up experiment tracking...")

    config = TrackingConfig(
        experiment_name="benchmark_comparison",
        enable_dataset_tracking=True,
        enable_model_tracking=True,
        enable_environment_tracking=True,
        enable_seed_tracking=True,
        enable_git_tracking=True,
        output_dir="./runs",
        tags=["demo", "benchmark", "comparison", "statistical-analysis"],
        notes="Comprehensive benchmark comparison of RL algorithms"
    )

    tracker = ExperimentTracker(config)
    tracking_data = tracker.start_experiment()
    print(f"🚀 Started experiment: {tracking_data['experiment_id']}")

    # 5. Run Benchmark
    print("\n5. Running benchmark suite...")

    start_time = time.time()

    results = benchmark.run_benchmark(episodes=50)

    benchmark_time = time.time() - start_time
    print(f"\n⏱️  Benchmark completed in {benchmark_time:.2f} seconds")

    # 6. Analyze Results
    print("\n6. Analyzing benchmark results...")

    df, interaction_stats = benchmark.analyze_results()

    # 7. Create Visualizations
    print("\n7. Creating benchmark visualizations...")

    benchmark.create_visualizations(df, interaction_stats)

    # 8. Track Results
    print("\n8. Tracking benchmark results...")

    # Create summary DataFrame
    if df is not None:
        # Track results
        tracker.track_dataset(
            df,
            "benchmark_results",
            {
                "total_experiments": len(results),
                "successful_experiments": len([r for r in results if 'error' not in r]),
                "benchmark_time": benchmark_time,
                "algorithms": [alg.__class__.__name__ for alg in algorithms],
                "environments": environments,
                "seeds": seeds
            }
        )

        # Track best performing algorithm
        best_algorithm = df.groupby('algorithm')['final_avg_reward'].mean().idxmax()
        best_reward = df.groupby('algorithm')['final_avg_reward'].mean().max()

        tracker.track_model(
            algorithms[0],  # Placeholder model
            "best_algorithm",
            {
                "algorithm": best_algorithm,
                "average_reward": best_reward,
                "total_experiments": len(results)
            }
        )

    # Add metadata
    tracker.add_metadata("benchmark_time", benchmark_time)
    tracker.add_metadata("total_experiments", len(results))
    tracker.add_metadata("successful_experiments", len([r for r in results if 'error' not in r]))
    tracker.add_metadata("algorithms", str([alg.__class__.__name__ for alg in algorithms]))
    tracker.add_metadata("environments", str(environments))
    tracker.add_metadata("seeds", str(seeds))

    print("✅ Benchmark results tracked successfully")

    # 9. Save Detailed Results
    print("\n9. Saving detailed results...")

    detailed_results = {
        "benchmark_config": {
            "algorithms": [alg.__class__.__name__ for alg in algorithms],
            "environments": environments,
            "seeds": seeds,
            "episodes": 50
        },
        "results": results,
        "summary_statistics": {
            "total_experiments": len(results),
            "successful_experiments": len([r for r in results if 'error' not in r]),
            "benchmark_time": benchmark_time
        }
    }

    with open("./benchmark_comparison_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2, default=str)

    print("💾 Detailed results saved to ./benchmark_comparison_results.json")

    # 10. Finish Experiment
    print("\n10. Finishing experiment...")

    summary = tracker.finish_experiment()

    print("\n🎉 Benchmark Comparison Example completed successfully!")
    print(f"   Experiment ID: {summary['experiment_id']}")
    print(f"   Output directory: {config.output_dir}")
    print(f"   Total experiments: {len(results)}")
    print(f"   Benchmark time: {benchmark_time:.2f} seconds")

    print("\n📁 Files created:")
    print(f"   - {config.output_dir}/{summary['experiment_id']}/ - Experiment data")
    print("   - ./benchmark_comparison_results.json - Detailed results")
    print("   - ./benchmark_comparison_plots.png - Analysis plots")

    # 11. Summary
    print("\n11. Summary and Next Steps...")

    print("\n✅ What We Accomplished:")
    print("   1. Algorithm Comparison: Compared PPO, DQN, and A2C")
    print("   2. Environment Testing: Tested across multiple environments")
    print("   3. Statistical Analysis: Performed significance tests")
    print("   4. Forensics Integration: Used RLDK to identify issues")
    print("   5. Reproducible Results: Ensured all experiments are reproducible")
    print("   6. Comprehensive Visualization: Created detailed analysis plots")

    if df is not None:
        print("\n📊 Key Findings:")
        best_algorithm = df.groupby('algorithm')['final_avg_reward'].mean().idxmax()
        best_reward = df.groupby('algorithm')['final_avg_reward'].mean().max()
        print(f"   - Best algorithm: {best_algorithm}")
        print(f"   - Best average reward: {best_reward:.3f}")
        print(f"   - Total experiments: {len(results)}")
        print(f"   - Benchmark time: {benchmark_time:.2f} seconds")

    print("\n🚀 Next Steps:")
    print("   1. More Algorithms: Add SAC, TD3, and other algorithms")
    print("   2. More Environments: Test on Atari and MuJoCo environments")
    print("   3. Hyperparameter Tuning: Optimize hyperparameters for each algorithm")
    print("   4. Statistical Power: Increase sample size for better statistics")
    print("   5. Computational Cost: Analyze training time and memory usage")
    print("   6. Failure Mode Analysis: Deep dive into algorithm-specific issues")

    print("\n📚 Key Takeaways:")
    print("   - RLDK makes benchmarking systematic and reproducible")
    print("   - Statistical significance testing is crucial")
    print("   - Different algorithms excel in different environments")
    print("   - Forensics help identify algorithm-specific issues")
    print("   - Multiple seeds provide robust performance estimates")
    print("   - Visualization is key for understanding results")

    print("\nHappy benchmarking! 🎉")

if __name__ == "__main__":
    main()
