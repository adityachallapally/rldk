#!/usr/bin/env python3
"""
Multi-Run Analysis Example for RLDK

This example demonstrates how to use RLDK for analyzing multiple training runs,
comparing their performance, and identifying patterns across runs.

Learning Objectives:
- How to analyze multiple training runs with RLDK
- How to compare run performance and identify patterns
- How to detect run-specific issues and anomalies
- How to create comprehensive run comparison reports
- How to implement run clustering and categorization

Prerequisites:
- RLDK installed (pip install rldk)
- Basic understanding of statistical analysis
- Familiarity with run comparison techniques
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# RLDK imports
import rldk
from rldk.diff import first_divergence
from rldk.forensics import ComprehensivePPOForensics
from rldk.tracking import ExperimentTracker, TrackingConfig
from rldk.utils import set_global_seed, validate_numeric_range


class SimplePPO:
    """Simple PPO implementation for multi-run analysis."""

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip

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
            value_loss = 0.5 * (value_new - return_val) ** 2

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

def run_single_experiment(env_name, hyperparams, episodes=100, seed=42):
    """Run a single experiment with given hyperparameters."""

    # Set seed for reproducibility
    set_global_seed(seed)

    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize agent
    agent = SimplePPO(state_dim, action_dim, **hyperparams)

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
                kl_div += np.sum(old_probs * np.log((old_probs + 1e-8) / (new_probs + 1e-8)))
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
                'value_grad_norm': value_grad_norm
            })

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    env.close()

    # Get forensics analysis
    analysis = forensics.get_comprehensive_analysis()
    anomalies = forensics.get_anomalies()
    health_summary = forensics.get_health_summary()

    return {
        'hyperparams': hyperparams,
        'seed': seed,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'training_metrics': training_metrics,
        'forensics': forensics,
        'analysis': analysis,
        'anomalies': anomalies,
        'health_summary': health_summary
    }

def run_multiple_experiments(env_name, hyperparams_list, episodes=100, seeds=None):
    """Run multiple experiments with different hyperparameters."""

    if seeds is None:
        seeds = [42, 123, 456, 789, 999]

    print(f"🚀 Running {len(hyperparams_list)} experiments with {len(seeds)} seeds each")
    print(f"   Total runs: {len(hyperparams_list) * len(seeds)}")

    results = []

    for i, hyperparams in enumerate(hyperparams_list):
        print(f"\n📊 Experiment {i+1}/{len(hyperparams_list)}: {hyperparams}")

        for j, seed in enumerate(seeds):
            print(f"   🌱 Seed {seed} ({j+1}/{len(seeds)})...")

            try:
                result = run_single_experiment(env_name, hyperparams, episodes, seed)
                results.append(result)

                # Calculate summary metrics
                final_avg_reward = np.mean(result['episode_rewards'][-min(10, len(result['episode_rewards'])):]) if result['episode_rewards'] else 0
                total_anomalies = len(result['anomalies'])
                health_score = result['health_summary'].get('overall_health', 0)

                print(f"      Final avg reward: {final_avg_reward:.2f}")
                print(f"      Anomalies: {total_anomalies}")
                print(f"      Health score: {health_score:.2f}")

            except Exception as e:
                print(f"      ❌ Error: {e}")
                results.append({
                    'hyperparams': hyperparams,
                    'seed': seed,
                    'episode_rewards': [],
                    'episode_lengths': [],
                    'training_metrics': [],
                    'forensics': None,
                    'analysis': {},
                    'anomalies': [],
                    'health_summary': {},
                    'error': str(e)
                })

    print(f"\n✅ Completed {len(results)} runs")
    return results

def analyze_multiple_runs(results):
    """Analyze multiple training runs."""

    print(f"\n📊 Analyzing {len(results)} training runs...")

    # Convert results to DataFrame
    df_data = []
    for i, result in enumerate(results):
        if 'error' not in result:
            # Calculate summary metrics
            final_avg_reward = np.mean(result['episode_rewards'][-min(10, len(result['episode_rewards'])):]) if result['episode_rewards'] else 0
            total_anomalies = len(result['anomalies'])
            health_score = result['health_summary'].get('overall_health', 0)

            # Extract hyperparameters
            hyperparams = result['hyperparams']

            df_data.append({
                'run_id': i,
                'hyperparams': hyperparams,
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
        return None

    # Statistical analysis
    print("\n📈 Statistical Analysis:")

    # 1. Overall performance statistics
    performance_stats = df['final_avg_reward'].agg(['mean', 'std', 'min', 'max', 'count'])
    print("\nOverall Performance Statistics:")
    print(f"   Mean: {performance_stats['mean']:.3f}")
    print(f"   Std: {performance_stats['std']:.3f}")
    print(f"   Min: {performance_stats['min']:.3f}")
    print(f"   Max: {performance_stats['max']:.3f}")
    print(f"   Count: {performance_stats['count']}")

    # 2. Hyperparameter analysis
    print("\nHyperparameter Analysis:")

    # Extract hyperparameters for analysis
    hyperparam_cols = []
    for col in ['lr', 'gamma', 'eps_clip']:
        if col in df['hyperparams'].iloc[0]:
            df[col] = df['hyperparams'].apply(lambda x: x.get(col, 0))
            hyperparam_cols.append(col)

    # Correlation analysis
    for col in hyperparam_cols:
        correlation = df[col].corr(df['final_avg_reward'])
        print(f"   {col} correlation with reward: {correlation:.3f}")

    # 3. Run clustering
    print("\nRun Clustering:")

    # Prepare features for clustering
    features = ['final_avg_reward', 'total_anomalies', 'health_score']
    if hyperparam_cols:
        features.extend(hyperparam_cols)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    # Perform clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

    # Analyze clusters
    cluster_stats = df.groupby('cluster')[features].agg(['mean', 'std'])
    print(f"   Cluster 0: {len(df[df['cluster'] == 0])} runs")
    print(f"   Cluster 1: {len(df[df['cluster'] == 1])} runs")
    print(f"   Cluster 2: {len(df[df['cluster'] == 2])} runs")

    # 4. Best and worst runs
    print("\nBest and Worst Runs:")

    best_run = df.loc[df['final_avg_reward'].idxmax()]
    worst_run = df.loc[df['final_avg_reward'].idxmin()]

    print(f"   Best run: {best_run['run_id']} (reward: {best_run['final_avg_reward']:.3f})")
    print(f"   Worst run: {worst_run['run_id']} (reward: {worst_run['final_avg_reward']:.3f})")

    return df, cluster_stats

def create_multi_run_visualizations(df, cluster_stats):
    """Create visualizations for multi-run analysis."""

    if df is None or df.empty:
        print("❌ No data to visualize!")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Performance distribution
    axes[0, 0].hist(df['final_avg_reward'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(df['final_avg_reward'].mean(), color='red', linestyle='--',
                      linewidth=2, label=f'Mean: {df["final_avg_reward"].mean():.2f}')
    axes[0, 0].set_title('Performance Distribution Across Runs')
    axes[0, 0].set_xlabel('Final Average Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Performance vs anomalies
    scatter = axes[0, 1].scatter(df['total_anomalies'], df['final_avg_reward'],
                                c=df['cluster'], cmap='viridis', alpha=0.7, s=60)
    axes[0, 1].set_title('Performance vs Anomalies (Colored by Cluster)')
    axes[0, 1].set_xlabel('Total Anomalies')
    axes[0, 1].set_ylabel('Final Average Reward')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 1], label='Cluster')

    # 3. Learning curves (average across runs)
    all_rewards = []
    for _, row in df.iterrows():
        if len(row['episode_rewards']) > 0:
            all_rewards.append(row['episode_rewards'])

    if all_rewards:
        # Pad to same length
        max_length = max(len(rewards) for rewards in all_rewards)
        padded_rewards = []
        for rewards in all_rewards:
            padded = rewards + [rewards[-1]] * (max_length - len(rewards))
            padded_rewards.append(padded)

        # Average across runs
        avg_rewards = np.mean(padded_rewards, axis=0)
        std_rewards = np.std(padded_rewards, axis=0)

        axes[0, 2].plot(avg_rewards, 'b-', linewidth=2, label='Mean')
        axes[0, 2].fill_between(range(len(avg_rewards)),
                              avg_rewards - std_rewards,
                              avg_rewards + std_rewards,
                              alpha=0.3, color='blue', label='±1 Std')
        axes[0, 2].set_title('Average Learning Curve Across Runs')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Episode Reward')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

    # 4. Cluster analysis
    cluster_means = df.groupby('cluster')['final_avg_reward'].mean()
    cluster_stds = df.groupby('cluster')['final_avg_reward'].std()

    axes[1, 0].bar(cluster_means.index, cluster_means.values,
                   yerr=cluster_stds.values, capsize=5, alpha=0.7, color='lightcoral')
    axes[1, 0].set_title('Performance by Cluster')
    axes[1, 0].set_xlabel('Cluster')
    axes[1, 0].set_ylabel('Final Average Reward')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Hyperparameter analysis (if available)
    hyperparam_cols = [col for col in ['lr', 'gamma', 'eps_clip'] if col in df.columns]
    if hyperparam_cols:
        for col in hyperparam_cols:
            correlation = df[col].corr(df['final_avg_reward'])
            axes[1, 1].scatter(df[col], df['final_avg_reward'], alpha=0.7, label=f'{col} (r={correlation:.3f})')

        axes[1, 1].set_title('Hyperparameter vs Performance')
        axes[1, 1].set_xlabel('Hyperparameter Value')
        axes[1, 1].set_ylabel('Final Average Reward')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No hyperparameter data available',
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Hyperparameter Analysis')

    # 6. Run ranking
    df_sorted = df.sort_values('final_avg_reward', ascending=False)
    top_runs = df_sorted.head(10)

    axes[1, 2].barh(range(len(top_runs)), top_runs['final_avg_reward'],
                    color='lightgreen', alpha=0.7)
    axes[1, 2].set_title('Top 10 Runs by Performance')
    axes[1, 2].set_xlabel('Final Average Reward')
    axes[1, 2].set_ylabel('Run Rank')
    axes[1, 2].set_yticks(range(len(top_runs)))
    axes[1, 2].set_yticklabels([f'Run {i}' for i in top_runs['run_id']])
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./multi_run_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("📊 Multi-run analysis plots saved to ./multi_run_analysis_plots.png")

def main():
    """Main function demonstrating multi-run analysis with RLDK."""

    print("🚀 RLDK Multi-Run Analysis Example")
    print("=" * 60)

    # 1. Setup
    print("\n1. Setting up reproducible environment...")
    seed = set_global_seed(42)
    print(f"🌱 Set global seed to: {seed}")

    # 2. Define hyperparameter combinations
    print("\n2. Defining hyperparameter combinations...")

    hyperparams_list = [
        {'lr': 1e-4, 'gamma': 0.9, 'eps_clip': 0.1},
        {'lr': 3e-4, 'gamma': 0.95, 'eps_clip': 0.2},
        {'lr': 1e-3, 'gamma': 0.99, 'eps_clip': 0.3},
        {'lr': 3e-4, 'gamma': 0.9, 'eps_clip': 0.2},
        {'lr': 1e-4, 'gamma': 0.99, 'eps_clip': 0.1},
        {'lr': 1e-3, 'gamma': 0.95, 'eps_clip': 0.3}
    ]

    seeds = [42, 123, 456, 789, 999]

    print(f"   Hyperparameter combinations: {len(hyperparams_list)}")
    print(f"   Seeds per combination: {len(seeds)}")
    print(f"   Total runs: {len(hyperparams_list) * len(seeds)}")

    # 3. Experiment Tracking Setup
    print("\n3. Setting up experiment tracking...")

    config = TrackingConfig(
        experiment_name="multi_run_analysis",
        enable_dataset_tracking=True,
        enable_model_tracking=True,
        enable_environment_tracking=True,
        enable_seed_tracking=True,
        enable_git_tracking=True,
        output_dir="./runs",
        tags=["demo", "multi-run", "analysis", "comparison"],
        notes="Multi-run analysis example with different hyperparameters"
    )

    tracker = ExperimentTracker(config)
    tracking_data = tracker.start_experiment()
    print(f"🚀 Started experiment: {tracking_data['experiment_id']}")

    # 4. Run Multiple Experiments
    print("\n4. Running multiple experiments...")

    start_time = time.time()

    results = run_multiple_experiments(
        env_name="CartPole-v1",
        hyperparams_list=hyperparams_list,
        episodes=50,  # Reduced for demo
        seeds=seeds
    )

    experiment_time = time.time() - start_time
    print(f"\n⏱️  Experiments completed in {experiment_time:.2f} seconds")

    # 5. Analyze Results
    print("\n5. Analyzing multiple runs...")

    df, cluster_stats = analyze_multiple_runs(results)

    # 6. Create Visualizations
    print("\n6. Creating multi-run visualizations...")

    create_multi_run_visualizations(df, cluster_stats)

    # 7. Track Results
    print("\n7. Tracking multi-run results...")

    if df is not None:
        # Track results
        tracker.track_dataset(
            df,
            "multi_run_results",
            {
                "total_runs": len(results),
                "successful_runs": len([r for r in results if 'error' not in r]),
                "experiment_time": experiment_time,
                "hyperparameter_combinations": len(hyperparams_list),
                "seeds_per_combination": len(seeds)
            }
        )

        # Track best run
        best_run = df.loc[df['final_avg_reward'].idxmax()]
        tracker.track_model(
            None,  # Placeholder model
            "best_run",
            {
                "run_id": int(best_run['run_id']),
                "final_avg_reward": float(best_run['final_avg_reward']),
                "hyperparams": best_run['hyperparams'],
                "seed": int(best_run['seed'])
            }
        )

    # Add metadata
    tracker.add_metadata("experiment_time", experiment_time)
    tracker.add_metadata("total_runs", len(results))
    tracker.add_metadata("successful_runs", len([r for r in results if 'error' not in r]))
    tracker.add_metadata("hyperparameter_combinations", len(hyperparams_list))
    tracker.add_metadata("seeds_per_combination", len(seeds))

    print("✅ Multi-run results tracked successfully")

    # 8. Save Detailed Results
    print("\n8. Saving detailed results...")

    detailed_results = {
        "experiment_config": {
            "hyperparameter_combinations": hyperparams_list,
            "seeds": seeds,
            "episodes": 50,
            "environment": "CartPole-v1"
        },
        "results": results,
        "summary_statistics": {
            "total_runs": len(results),
            "successful_runs": len([r for r in results if 'error' not in r]),
            "experiment_time": experiment_time
        }
    }

    with open("./multi_run_analysis_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2, default=str)

    print("💾 Detailed results saved to ./multi_run_analysis_results.json")

    # 9. Finish Experiment
    print("\n9. Finishing experiment...")

    summary = tracker.finish_experiment()

    print("\n🎉 Multi-Run Analysis Example completed successfully!")
    print(f"   Experiment ID: {summary['experiment_id']}")
    print(f"   Output directory: {config.output_dir}")
    print(f"   Total runs: {len(results)}")
    print(f"   Experiment time: {experiment_time:.2f} seconds")

    print("\n📁 Files created:")
    print(f"   - {config.output_dir}/{summary['experiment_id']}/ - Experiment data")
    print("   - ./multi_run_analysis_results.json - Detailed results")
    print("   - ./multi_run_analysis_plots.png - Analysis plots")

    # 10. Summary
    print("\n10. Summary and Next Steps...")

    print("\n✅ What We Accomplished:")
    print("   1. Multiple Runs: Executed runs with different hyperparameters")
    print("   2. Statistical Analysis: Analyzed performance across runs")
    print("   3. Run Clustering: Identified run patterns and clusters")
    print("   4. Performance Comparison: Compared run performance")
    print("   5. Visualization: Created comprehensive analysis plots")
    print("   6. RLDK Integration: Tracked all runs and results")

    if df is not None:
        print("\n📊 Key Findings:")
        print(f"   - Total runs: {len(results)}")
        print(f"   - Successful runs: {len([r for r in results if 'error' not in r])}")
        print(f"   - Best performance: {df['final_avg_reward'].max():.3f}")
        print(f"   - Average performance: {df['final_avg_reward'].mean():.3f}")
        print(f"   - Performance std: {df['final_avg_reward'].std():.3f}")

    print("\n🚀 Next Steps:")
    print("   1. Hyperparameter Optimization: Use results to guide hyperparameter tuning")
    print("   2. Run Selection: Select best runs for further analysis")
    print("   3. Failure Analysis: Analyze failed runs to identify issues")
    print("   4. Statistical Testing: Perform significance tests between run groups")
    print("   5. Ensemble Methods: Combine best runs for improved performance")
    print("   6. Automated Analysis: Implement automated run analysis pipelines")

    print("\n📚 Key Takeaways:")
    print("   - RLDK makes multi-run analysis systematic and reproducible")
    print("   - Statistical analysis reveals performance patterns")
    print("   - Run clustering identifies different training behaviors")
    print("   - Visualization helps understand run relationships")
    print("   - Multiple seeds provide robustness estimates")
    print("   - Hyperparameter analysis guides optimization")

    print("\nHappy multi-run analysis! 🎉")

if __name__ == "__main__":
    main()
