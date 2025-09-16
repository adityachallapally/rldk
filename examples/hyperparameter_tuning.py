#!/usr/bin/env python3
"""
Hyperparameter Tuning Example for RLDK

This example demonstrates how to use RLDK for systematic hyperparameter tuning
in reinforcement learning experiments. We'll implement grid search and random search
on a toy PPO training problem, showing how to track and compare multiple runs.

Learning Objectives:
- How to structure hyperparameter tuning experiments with RLDK
- How to track multiple runs and compare their performance
- How to use RLDK's forensics to identify good vs bad hyperparameters
- How to analyze hyperparameter sensitivity and interactions
- How to create reproducible hyperparameter search workflows

Prerequisites:
- RLDK installed (pip install rldk)
- Basic understanding of hyperparameter tuning
- Familiarity with grid search and random search
"""

import json
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# RLDK imports
import rldk
from rldk.diff import first_divergence
from rldk.forensics import ComprehensivePPOForensics
from rldk.tracking import ExperimentTracker, TrackingConfig
from rldk.utils import set_global_seed, validate_numeric_range


class SimplePPO:
    """Simple PPO implementation for hyperparameter tuning demo."""

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

    def _log_softmax(self, x):
        """Log softmax function for numerical stability."""
        x_max = np.max(x)
        log_sum_exp = np.log(np.sum(np.exp(x - x_max))) + x_max
        return x - log_sum_exp

    def update(self, states, actions, rewards, log_probs_old, values_old, advantages, returns):
        """Update policy and value function."""
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
            value_loss = self.value_coef * (value_new - return_val) ** 2

            # Entropy bonus
            entropy = -np.sum(log_prob_new * np.log(log_prob_new + 1e-8))
            entropy_loss = -self.entropy_coef * entropy

            # Total loss
            policy_loss + value_loss + entropy_loss

            # Simple gradient step
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

def train_single_run(env_name, hyperparams, episodes=50, seed=42):
    """Train a single run with given hyperparameters."""

    # Set seed for reproducibility
    set_global_seed(seed)

    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize agent with hyperparameters
    agent = SimplePPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=hyperparams['lr'],
        gamma=hyperparams['gamma'],
        eps_clip=hyperparams['eps_clip'],
        entropy_coef=hyperparams['entropy_coef'],
        value_coef=hyperparams['value_coef']
    )

    # Initialize forensics
    forensics = ComprehensivePPOForensics(kl_target=0.1)

    # Training statistics
    episode_rewards = []
    episode_lengths = []
    training_metrics = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        # Store trajectory data
        states, actions, rewards, log_probs, values = [], [], [], [], []

        for step in range(200):  # Max steps per episode
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

    env.close()

    # Get forensics analysis
    analysis = forensics.get_comprehensive_analysis()
    anomalies = forensics.get_anomalies()
    health_summary = forensics.get_health_summary()

    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'training_metrics': training_metrics,
        'forensics': forensics,
        'analysis': analysis,
        'anomalies': anomalies,
        'health_summary': health_summary,
        'agent': agent
    }

def grid_search_hyperparameters(env_name, param_grid, episodes=50, seeds=None):
    """Perform grid search over hyperparameters."""

    if seeds is None:
        seeds = [42, 123, 456]  # Multiple seeds for robustness

    print(f"🔍 Starting grid search over {len(list(product(*param_grid.values())))} combinations")
    print(f"   Parameters: {list(param_grid.keys())}")
    print(f"   Seeds per combination: {len(seeds)}")

    results = []

    for i, hyperparams in enumerate(product(*param_grid.values())):
        hyperparams_dict = dict(zip(param_grid.keys(), hyperparams))

        print(f"\n📊 Combination {i+1}/{len(list(product(*param_grid.values())))}: {hyperparams_dict}")

        combination_results = []

        for seed in seeds:
            print(f"   🌱 Seed {seed}...")

            try:
                result = train_single_run(env_name, hyperparams_dict, episodes, seed)

                # Calculate summary metrics
                final_avg_reward = np.mean(result['episode_rewards'][-10:])
                total_anomalies = len(result['anomalies'])
                health_score = result['health_summary'].get('overall_health', 0)

                combination_results.append({
                    'seed': seed,
                    'final_avg_reward': final_avg_reward,
                    'total_anomalies': total_anomalies,
                    'health_score': health_score,
                    'episode_rewards': result['episode_rewards'],
                    'training_metrics': result['training_metrics'],
                    'anomalies': result['anomalies'],
                    'hyperparams': hyperparams_dict
                })

                print(f"      Final avg reward: {final_avg_reward:.2f}")
                print(f"      Anomalies: {total_anomalies}")

            except Exception as e:
                print(f"      ❌ Error: {e}")
                combination_results.append({
                    'seed': seed,
                    'final_avg_reward': 0,
                    'total_anomalies': 999,
                    'health_score': 0,
                    'episode_rewards': [],
                    'training_metrics': [],
                    'anomalies': [],
                    'hyperparams': hyperparams_dict,
                    'error': str(e)
                })

        # Aggregate results for this combination
        valid_results = [r for r in combination_results if 'error' not in r]

        if valid_results:
            avg_reward = np.mean([r['final_avg_reward'] for r in valid_results])
            avg_anomalies = np.mean([r['total_anomalies'] for r in valid_results])
            avg_health = np.mean([r['health_score'] for r in valid_results])
            std_reward = np.std([r['final_avg_reward'] for r in valid_results])
        else:
            avg_reward = 0
            avg_anomalies = 999
            avg_health = 0
            std_reward = 0

        results.append({
            'combination_id': i,
            'hyperparams': hyperparams_dict,
            'seeds': seeds,
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'avg_anomalies': avg_anomalies,
            'avg_health': avg_health,
            'individual_results': combination_results
        })

        print(f"   📈 Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"   🚨 Average anomalies: {avg_anomalies:.1f}")

    return results

def random_search_hyperparameters(env_name, param_ranges, n_trials=20, episodes=50, seeds=None):
    """Perform random search over hyperparameters."""

    if seeds is None:
        seeds = [42]  # Single seed for random search (faster)

    print(f"🎲 Starting random search with {n_trials} trials")
    print(f"   Parameter ranges: {param_ranges}")
    print(f"   Seeds per trial: {len(seeds)}")

    results = []

    for i in range(n_trials):
        # Sample random hyperparameters
        hyperparams = {}
        for param_name, param_range in param_ranges.items():
            if isinstance(param_range, list):
                hyperparams[param_name] = np.random.choice(param_range)
            elif isinstance(param_range, tuple) and len(param_range) == 2:
                # Continuous range
                if param_name in ['lr']:
                    # Log scale for learning rate
                    hyperparams[param_name] = np.exp(np.random.uniform(
                        np.log(param_range[0]), np.log(param_range[1])
                    ))
                else:
                    hyperparams[param_name] = np.random.uniform(param_range[0], param_range[1])
            else:
                hyperparams[param_name] = param_range

        print(f"\n🎯 Trial {i+1}/{n_trials}: {hyperparams}")

        trial_results = []

        for seed in seeds:
            print(f"   🌱 Seed {seed}...")

            try:
                result = train_single_run(env_name, hyperparams, episodes, seed)

                # Calculate summary metrics
                final_avg_reward = np.mean(result['episode_rewards'][-10:])
                total_anomalies = len(result['anomalies'])
                health_score = result['health_summary'].get('overall_health', 0)

                trial_results.append({
                    'seed': seed,
                    'final_avg_reward': final_avg_reward,
                    'total_anomalies': total_anomalies,
                    'health_score': health_score,
                    'episode_rewards': result['episode_rewards'],
                    'training_metrics': result['training_metrics'],
                    'anomalies': result['anomalies'],
                    'hyperparams': hyperparams
                })

                print(f"      Final avg reward: {final_avg_reward:.2f}")
                print(f"      Anomalies: {total_anomalies}")

            except Exception as e:
                print(f"      ❌ Error: {e}")
                trial_results.append({
                    'seed': seed,
                    'final_avg_reward': 0,
                    'total_anomalies': 999,
                    'health_score': 0,
                    'episode_rewards': [],
                    'training_metrics': [],
                    'anomalies': [],
                    'hyperparams': hyperparams,
                    'error': str(e)
                })

        # Aggregate results for this trial
        valid_results = [r for r in trial_results if 'error' not in r]

        if valid_results:
            avg_reward = np.mean([r['final_avg_reward'] for r in valid_results])
            avg_anomalies = np.mean([r['total_anomalies'] for r in valid_results])
            avg_health = np.mean([r['health_score'] for r in valid_results])
            std_reward = np.std([r['final_avg_reward'] for r in valid_results])
        else:
            avg_reward = 0
            avg_anomalies = 999
            avg_health = 0
            std_reward = 0

        results.append({
            'trial_id': i,
            'hyperparams': hyperparams,
            'seeds': seeds,
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'avg_anomalies': avg_anomalies,
            'avg_health': avg_health,
            'individual_results': trial_results
        })

        print(f"   📈 Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"   🚨 Average anomalies: {avg_anomalies:.1f}")

    return results

def analyze_hyperparameter_results(results, search_type="grid"):
    """Analyze hyperparameter search results."""

    print(f"\n📊 Analyzing {search_type} search results...")

    # Convert to DataFrame for easier analysis
    df_data = []
    for result in results:
        row = result['hyperparams'].copy()
        row.update({
            'avg_reward': result['avg_reward'],
            'std_reward': result['std_reward'],
            'avg_anomalies': result['avg_anomalies'],
            'avg_health': result['avg_health']
        })
        df_data.append(row)

    df = pd.DataFrame(df_data)

    # Find best hyperparameters
    best_idx = df['avg_reward'].idxmax()
    best_result = results[best_idx]

    print(f"\n🏆 Best {search_type} search result:")
    print(f"   Hyperparameters: {best_result['hyperparams']}")
    print(f"   Average reward: {best_result['avg_reward']:.3f} ± {best_result['std_reward']:.3f}")
    print(f"   Average anomalies: {best_result['avg_anomalies']:.1f}")
    print(f"   Average health: {best_result['avg_health']:.3f}")

    # Parameter sensitivity analysis
    print("\n📈 Parameter sensitivity analysis:")

    for param in df.columns:
        if param not in ['avg_reward', 'std_reward', 'avg_anomalies', 'avg_health']:
            if df[param].dtype in ['float64', 'int64']:
                # Continuous parameter
                correlation = df[param].corr(df['avg_reward'])
                print(f"   {param}: correlation with reward = {correlation:.3f}")
            else:
                # Categorical parameter
                param_means = df.groupby(param)['avg_reward'].mean()
                print(f"   {param}: {dict(param_means)}")

    return df, best_result

def create_hyperparameter_visualizations(grid_results, random_results):
    """Create visualizations for hyperparameter tuning results."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Reward distribution comparison
    grid_rewards = [r['avg_reward'] for r in grid_results]
    random_rewards = [r['avg_reward'] for r in random_results]

    axes[0, 0].hist(grid_rewards, alpha=0.7, label='Grid Search', bins=10)
    axes[0, 0].hist(random_rewards, alpha=0.7, label='Random Search', bins=10)
    axes[0, 0].set_title('Reward Distribution Comparison')
    axes[0, 0].set_xlabel('Average Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Best reward over trials (for random search)
    random_rewards_sorted = sorted(random_rewards, reverse=True)
    axes[0, 1].plot(random_rewards_sorted, 'o-', label='Random Search')
    axes[0, 1].axhline(y=max(grid_rewards), color='red', linestyle='--', label='Best Grid Search')
    axes[0, 1].set_title('Best Reward vs Trials (Random Search)')
    axes[0, 1].set_xlabel('Trial Rank')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Learning rate vs reward (if available)
    if 'lr' in grid_results[0]['hyperparams']:
        lr_values = [r['hyperparams']['lr'] for r in grid_results]
        rewards = [r['avg_reward'] for r in grid_results]

        axes[0, 2].scatter(lr_values, rewards, alpha=0.7)
        axes[0, 2].set_xlabel('Learning Rate')
        axes[0, 2].set_ylabel('Average Reward')
        axes[0, 2].set_title('Learning Rate vs Reward')
        axes[0, 2].set_xscale('log')
        axes[0, 2].grid(True, alpha=0.3)

    # 4. Gamma vs reward (if available)
    if 'gamma' in grid_results[0]['hyperparams']:
        gamma_values = [r['hyperparams']['gamma'] for r in grid_results]
        rewards = [r['avg_reward'] for r in grid_results]

        axes[1, 0].scatter(gamma_values, rewards, alpha=0.7)
        axes[1, 0].set_xlabel('Gamma')
        axes[1, 0].set_ylabel('Average Reward')
        axes[1, 0].set_title('Gamma vs Reward')
        axes[1, 0].grid(True, alpha=0.3)

    # 5. Epsilon clip vs reward (if available)
    if 'eps_clip' in grid_results[0]['hyperparams']:
        eps_values = [r['hyperparams']['eps_clip'] for r in grid_results]
        rewards = [r['avg_reward'] for r in grid_results]

        axes[1, 1].scatter(eps_values, rewards, alpha=0.7)
        axes[1, 1].set_xlabel('Epsilon Clip')
        axes[1, 1].set_ylabel('Average Reward')
        axes[1, 1].set_title('Epsilon Clip vs Reward')
        axes[1, 1].grid(True, alpha=0.3)

    # 6. Anomalies vs reward
    grid_anomalies = [r['avg_anomalies'] for r in grid_results]
    grid_rewards = [r['avg_reward'] for r in grid_results]

    axes[1, 2].scatter(grid_anomalies, grid_rewards, alpha=0.7, label='Grid Search')

    random_anomalies = [r['avg_anomalies'] for r in random_results]
    random_rewards = [r['avg_reward'] for r in random_results]

    axes[1, 2].scatter(random_anomalies, random_rewards, alpha=0.7, label='Random Search')
    axes[1, 2].set_xlabel('Average Anomalies')
    axes[1, 2].set_ylabel('Average Reward')
    axes[1, 2].set_title('Anomalies vs Reward')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./hyperparameter_tuning_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("📊 Hyperparameter tuning plots saved to ./hyperparameter_tuning_plots.png")

def main():
    """Main function demonstrating hyperparameter tuning with RLDK."""

    print("🚀 RLDK Hyperparameter Tuning Example")
    print("=" * 50)

    # 1. Setup
    print("\n1. Setting up reproducible environment...")
    seed = set_global_seed(42)
    print(f"🌱 Set global seed to: {seed}")

    # 2. Define hyperparameter spaces
    print("\n2. Defining hyperparameter search spaces...")

    # Grid search parameters
    grid_param_grid = {
        'lr': [1e-4, 3e-4, 1e-3],
        'gamma': [0.9, 0.95, 0.99],
        'eps_clip': [0.1, 0.2, 0.3],
        'entropy_coef': [0.01, 0.05, 0.1]
    }

    # Random search parameters
    random_param_ranges = {
        'lr': (1e-4, 1e-2),  # Log scale
        'gamma': (0.9, 0.99),
        'eps_clip': (0.1, 0.3),
        'entropy_coef': (0.01, 0.1),
        'value_coef': (0.1, 1.0)
    }

    print(f"   Grid search: {len(list(product(*grid_param_grid.values())))} combinations")
    print("   Random search: 20 trials")

    # 3. Experiment Tracking Setup
    print("\n3. Setting up experiment tracking...")

    config = TrackingConfig(
        experiment_name="hyperparameter_tuning",
        enable_dataset_tracking=True,
        enable_model_tracking=True,
        enable_environment_tracking=True,
        enable_seed_tracking=True,
        enable_git_tracking=True,
        output_dir="./runs",
        tags=["demo", "hyperparameter-tuning", "grid-search", "random-search"],
        notes="Hyperparameter tuning example comparing grid search and random search"
    )

    tracker = ExperimentTracker(config)
    tracking_data = tracker.start_experiment()
    print(f"🚀 Started experiment: {tracking_data['experiment_id']}")

    # 4. Grid Search
    print("\n4. Running grid search...")

    start_time = time.time()

    grid_results = grid_search_hyperparameters(
        env_name="CartPole-v1",
        param_grid=grid_param_grid,
        episodes=30,  # Reduced for demo
        seeds=[42, 123]  # Reduced for demo
    )

    grid_time = time.time() - start_time
    print(f"\n⏱️  Grid search completed in {grid_time:.2f} seconds")

    # 5. Random Search
    print("\n5. Running random search...")

    start_time = time.time()

    random_results = random_search_hyperparameters(
        env_name="CartPole-v1",
        param_ranges=random_param_ranges,
        n_trials=15,  # Reduced for demo
        episodes=30,  # Reduced for demo
        seeds=[42]  # Single seed for speed
    )

    random_time = time.time() - start_time
    print(f"\n⏱️  Random search completed in {random_time:.2f} seconds")

    # 6. Analyze Results
    print("\n6. Analyzing hyperparameter tuning results...")

    # Analyze grid search
    grid_df, best_grid_result = analyze_hyperparameter_results(grid_results, "grid")

    # Analyze random search
    random_df, best_random_result = analyze_hyperparameter_results(random_results, "random")

    # Compare best results
    print("\n🏆 Best Results Comparison:")
    print(f"   Grid Search Best Reward: {best_grid_result['avg_reward']:.3f}")
    print(f"   Random Search Best Reward: {best_random_result['avg_reward']:.3f}")

    if best_grid_result['avg_reward'] > best_random_result['avg_reward']:
        print("   🥇 Grid search found better hyperparameters!")
        best_overall = best_grid_result
    else:
        print("   🥇 Random search found better hyperparameters!")
        best_overall = best_random_result

    # 7. Create Visualizations
    print("\n7. Creating hyperparameter tuning visualizations...")

    create_hyperparameter_visualizations(grid_results, random_results)

    # 8. Track Results
    print("\n8. Tracking hyperparameter tuning results...")

    # Create summary DataFrame
    all_results = []

    for result in grid_results:
        row = result['hyperparams'].copy()
        row.update({
            'search_type': 'grid',
            'avg_reward': result['avg_reward'],
            'std_reward': result['std_reward'],
            'avg_anomalies': result['avg_anomalies'],
            'avg_health': result['avg_health']
        })
        all_results.append(row)

    for result in random_results:
        row = result['hyperparams'].copy()
        row.update({
            'search_type': 'random',
            'avg_reward': result['avg_reward'],
            'std_reward': result['std_reward'],
            'avg_anomalies': result['avg_anomalies'],
            'avg_health': result['avg_health']
        })
        all_results.append(row)

    results_df = pd.DataFrame(all_results)

    # Track results
    tracker.track_dataset(
        results_df,
        "hyperparameter_tuning_results",
        {
            "grid_search_combinations": len(grid_results),
            "random_search_trials": len(random_results),
            "grid_search_time": grid_time,
            "random_search_time": random_time,
            "best_overall_reward": best_overall['avg_reward'],
            "best_hyperparams": best_overall['hyperparams']
        }
    )

    # Track best model (simulate training with best hyperparams)
    print("🎯 Training final model with best hyperparameters...")

    final_result = train_single_run(
        env_name="CartPole-v1",
        hyperparams=best_overall['hyperparams'],
        episodes=50,
        seed=42
    )

    tracker.track_model(
        final_result['agent'],
        "best_hyperparameter_model",
        {
            "algorithm": "SimplePPO",
            "environment": "CartPole-v1",
            "hyperparameters": best_overall['hyperparams'],
            "final_reward": np.mean(final_result['episode_rewards'][-10:]),
            "total_anomalies": len(final_result['anomalies']),
            "search_type": "grid" if best_overall == best_grid_result else "random"
        }
    )

    # Add metadata
    tracker.add_metadata("grid_search_time", grid_time)
    tracker.add_metadata("random_search_time", random_time)
    tracker.add_metadata("best_reward", best_overall['avg_reward'])
    tracker.add_metadata("best_hyperparams", str(best_overall['hyperparams']))
    tracker.add_metadata("total_experiments", len(grid_results) + len(random_results))

    print("✅ Hyperparameter tuning results tracked successfully")

    # 9. Save Detailed Results
    print("\n9. Saving detailed results...")

    detailed_results = {
        "grid_search_results": grid_results,
        "random_search_results": random_results,
        "best_grid_result": best_grid_result,
        "best_random_result": best_random_result,
        "best_overall_result": best_overall,
        "summary_statistics": {
            "grid_search_time": grid_time,
            "random_search_time": random_time,
            "total_experiments": len(grid_results) + len(random_results),
            "best_overall_reward": best_overall['avg_reward']
        }
    }

    with open("./hyperparameter_tuning_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2, default=str)

    print("💾 Detailed results saved to ./hyperparameter_tuning_results.json")

    # 10. Finish Experiment
    print("\n10. Finishing experiment...")

    summary = tracker.finish_experiment()

    print("\n🎉 Hyperparameter Tuning Example completed successfully!")
    print(f"   Experiment ID: {summary['experiment_id']}")
    print(f"   Output directory: {config.output_dir}")
    print(f"   Total experiments: {len(grid_results) + len(random_results)}")
    print(f"   Best reward: {best_overall['avg_reward']:.3f}")
    print(f"   Total time: {grid_time + random_time:.2f} seconds")

    print("\n📁 Files created:")
    print(f"   - {config.output_dir}/{summary['experiment_id']}/ - Experiment data")
    print("   - ./hyperparameter_tuning_results.json - Detailed results")
    print("   - ./hyperparameter_tuning_plots.png - Analysis plots")

    # 11. Summary
    print("\n11. Summary and Next Steps...")

    print("\n✅ What We Accomplished:")
    print("   1. Grid Search: Systematically explored hyperparameter space")
    print("   2. Random Search: Efficiently sampled hyperparameter space")
    print("   3. Performance Comparison: Compared both search strategies")
    print("   4. Parameter Sensitivity: Analyzed parameter-reward relationships")
    print("   5. Forensics Integration: Used RLDK to identify good vs bad hyperparams")
    print("   6. Reproducible Results: Ensured all experiments are reproducible")

    print("\n📊 Key Findings:")
    print(f"   - Best hyperparameters: {best_overall['hyperparams']}")
    print(f"   - Best reward: {best_overall['avg_reward']:.3f}")
    print(f"   - Grid search time: {grid_time:.2f} seconds")
    print(f"   - Random search time: {random_time:.2f} seconds")
    print(f"   - Total experiments: {len(grid_results) + len(random_results)}")

    print("\n🚀 Next Steps:")
    print("   1. Bayesian Optimization: Try more sophisticated search methods")
    print("   2. Multi-Objective: Optimize for both reward and stability")
    print("   3. Early Stopping: Implement early stopping for efficiency")
    print("   4. Cross-Validation: Use multiple environments for robustness")
    print("   5. Hyperparameter Importance: Analyze which parameters matter most")

    print("\n📚 Key Takeaways:")
    print("   - RLDK makes hyperparameter tuning systematic and reproducible")
    print("   - Forensics help identify why certain hyperparams work better")
    print("   - Grid search is thorough but expensive")
    print("   - Random search is often more efficient")
    print("   - Multiple seeds provide robustness estimates")

    print("\nHappy hyperparameter tuning! 🎉")

if __name__ == "__main__":
    main()
