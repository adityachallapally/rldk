#!/usr/bin/env python3
"""
Research Reproducibility Workflow Example for RLDK

This example demonstrates a complete research workflow using RLDK for
reproducible reinforcement learning experiments. We'll implement a
systematic approach to experiment design, execution, and analysis that
ensures reproducibility and enables easy sharing of research results.

Learning Objectives:
- How to structure reproducible research workflows with RLDK
- How to implement systematic experiment design and execution
- How to handle experiment versioning and branching
- How to create comprehensive experiment reports
- How to implement cross-validation and statistical testing
- How to generate publication-ready figures and tables
- How to create experiment snapshots for sharing

Prerequisites:
- RLDK installed (pip install rldk)
- Basic understanding of research methodology
- Familiarity with statistical testing
- Understanding of reproducibility best practices
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
import yaml
from scipy import stats

warnings.filterwarnings('ignore')

import gymnasium as gym

# RLDK imports
import rldk
from rldk.diff import first_divergence
from rldk.forensics import ComprehensivePPOForensics
from rldk.tracking import ExperimentTracker, TrackingConfig
from rldk.utils import set_global_seed, validate_numeric_range


class ReproduciblePPO:
    """Reproducible PPO implementation for research workflows."""

    def __init__(self, state_dim, action_dim, config: Dict[str, Any]):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Extract hyperparameters from config
        self.lr = config.get('lr', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.eps_clip = config.get('eps_clip', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.clip_value_loss = config.get('clip_value_loss', True)
        self.normalize_advantages = config.get('normalize_advantages', True)

        # Initialize policy and value networks
        self.policy = np.random.randn(state_dim, action_dim) * 0.1
        self.value = np.random.randn(state_dim, 1) * 0.1
        self.training_metrics = []

        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'kl_divergences': [],
            'gradient_norms': [],
            'advantage_means': [],
            'advantage_stds': []
        }

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

    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        returns = []

        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]

            if dones[i]:
                delta = rewards[i] - values[i]
                gae = delta
            else:
                delta = rewards[i] + self.gamma * next_value - values[i]
                gae = delta + self.gamma * self.gae_lambda * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        advantages = np.array(advantages)
        returns = np.array(returns)

        # Normalize advantages if specified
        if self.normalize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages, returns

    def update(self, states, actions, rewards, log_probs_old, values_old, dones):
        """Update policy and value function."""

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values_old, dones)

        # Store training statistics
        self.training_stats['advantage_means'].append(np.mean(advantages))
        self.training_stats['advantage_stds'].append(np.std(advantages))

        # Policy and value updates
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kl_divergences = []
        gradient_norms = []

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
            if self.clip_value_loss:
                value_loss = self.value_coef * np.maximum(
                    (value_new - return_val) ** 2,
                    np.clip(value_new, value_new - self.eps_clip, value_new + self.eps_clip) - return_val
                ) ** 2
            else:
                value_loss = self.value_coef * (value_new - return_val) ** 2

            # Entropy bonus
            probs = self._softmax(state @ self.policy)
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            entropy_loss = -self.entropy_coef * entropy

            # KL divergence
            kl_div = log_probs_old[i] - log_prob_new

            # Total loss
            total_loss = policy_loss + value_loss + entropy_loss

            self.training_metrics.append({
                'step': i,
                'total_loss': total_loss,
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'entropy_loss': entropy_loss
            })

            # Gradient step
            self.policy -= self.lr * policy_loss * state.reshape(-1, 1)
            self.value -= self.lr * value_loss * state.reshape(-1, 1)

            # Store losses
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            entropy_losses.append(entropy_loss)
            kl_divergences.append(kl_div)

            # Compute gradient norm
            grad_norm = np.linalg.norm(self.policy.flatten()) + np.linalg.norm(self.value.flatten())
            gradient_norms.append(grad_norm)

        # Store training statistics
        self.training_stats['policy_losses'].append(np.mean(policy_losses))
        self.training_stats['value_losses'].append(np.mean(value_losses))
        self.training_stats['entropy_losses'].append(np.mean(entropy_losses))
        self.training_stats['kl_divergences'].append(np.mean(kl_divergences))
        self.training_stats['gradient_norms'].append(np.mean(gradient_norms))

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'kl_divergence': np.mean(kl_divergences),
            'gradient_norm': np.mean(gradient_norms),
            'advantage_mean': np.mean(advantages),
            'advantage_std': np.std(advantages)
        }

class ResearchWorkflow:
    """Complete research workflow for reproducible RL experiments."""

    def __init__(self, config_path: str):
        """Initialize research workflow from configuration file."""

        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Extract configuration sections
        self.experiment_config = self.config['experiment']
        self.algorithm_config = self.config['algorithm']
        self.environment_config = self.config['environment']
        self.training_config = self.config['training']
        self.analysis_config = self.config['analysis']

        # Initialize components
        self.tracker = None
        self.forensics = None
        self.results = []

        # Create output directory
        self.output_dir = Path(self.experiment_config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup_experiment_tracking(self):
        """Set up experiment tracking with RLDK."""

        tracking_config = TrackingConfig(
            experiment_name=self.experiment_config['name'],
            enable_dataset_tracking=True,
            enable_model_tracking=True,
            enable_environment_tracking=True,
            enable_seed_tracking=True,
            enable_git_tracking=True,
            output_dir=str(self.output_dir),
            tags=self.experiment_config.get('tags', []),
            notes=self.experiment_config.get('notes', '')
        )

        self.tracker = ExperimentTracker(tracking_config)
        tracking_data = self.tracker.start_experiment()

        # Add experiment metadata
        self.tracker.add_metadata("config_file", self.config)
        self.tracker.add_metadata("algorithm_config", self.algorithm_config)
        self.tracker.add_metadata("environment_config", self.environment_config)
        self.tracker.add_metadata("training_config", self.training_config)
        self.tracker.add_metadata("analysis_config", self.analysis_config)

        return tracking_data

    def run_single_experiment(self, seed: int, trial_id: int = 0):
        """Run a single experiment with given seed."""

        # Set seed for reproducibility
        set_global_seed(seed)

        # Create environment
        env = gym.make(self.environment_config['name'])
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Initialize algorithm
        agent = ReproduciblePPO(state_dim, action_dim, self.algorithm_config)

        # Initialize forensics
        self.forensics = ComprehensivePPOForensics(kl_target=0.1)

        # Training statistics
        episode_rewards = []
        episode_lengths = []
        training_metrics = []

        # Training loop
        for episode in range(self.training_config['episodes']):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0

            # Store trajectory data
            states, actions, rewards, log_probs, values = [], [], [], [], []

            for step in range(self.training_config['max_steps_per_episode']):
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
                dones = [False] * (len(states) - 1) + [True]
                update_metrics = agent.update(states, actions, rewards, log_probs, values, dones)

                # Update forensics
                self.forensics.update(
                    step=episode,
                    kl=update_metrics['kl_divergence'],
                    kl_coef=0.2,
                    entropy=update_metrics['entropy_loss'],
                    reward_mean=episode_reward,
                    reward_std=np.std(rewards),
                    policy_grad_norm=update_metrics['gradient_norm'],
                    value_grad_norm=update_metrics['gradient_norm'],
                    advantage_mean=update_metrics['advantage_mean'],
                    advantage_std=update_metrics['advantage_std']
                )

                # Store training metrics
                training_metrics.append({
                    'episode': episode,
                    'reward': episode_reward,
                    'length': episode_length,
                    'policy_loss': update_metrics['policy_loss'],
                    'value_loss': update_metrics['value_loss'],
                    'entropy_loss': update_metrics['entropy_loss'],
                    'kl_divergence': update_metrics['kl_divergence'],
                    'gradient_norm': update_metrics['gradient_norm'],
                    'advantage_mean': update_metrics['advantage_mean'],
                    'advantage_std': update_metrics['advantage_std']
                })

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        env.close()

        # Get forensics analysis
        analysis = self.forensics.get_comprehensive_analysis()
        anomalies = self.forensics.get_anomalies()
        health_summary = self.forensics.get_health_summary()

        return {
            'seed': seed,
            'trial_id': trial_id,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'training_metrics': training_metrics,
            'agent_training_stats': agent.training_stats,
            'forensics_analysis': analysis,
            'anomalies': anomalies,
            'health_summary': health_summary,
            'agent': agent
        }

    def run_cross_validation(self):
        """Run cross-validation experiments."""

        print("🚀 Starting cross-validation experiments...")
        print(f"   Seeds: {self.training_config['seeds']}")
        print(f"   Episodes: {self.training_config['episodes']}")
        print(f"   Total experiments: {len(self.training_config['seeds'])}")

        results = []

        for i, seed in enumerate(self.training_config['seeds']):
            print(f"\n📊 Experiment {i+1}/{len(self.training_config['seeds'])} (Seed: {seed})")

            try:
                result = self.run_single_experiment(seed, trial_id=i)
                results.append(result)

                # Calculate summary metrics
                final_avg_reward = np.mean(result['episode_rewards'][-10:])
                total_anomalies = len(result['anomalies'])
                health_score = result['health_summary'].get('overall_health', 0)

                print(f"   Final avg reward: {final_avg_reward:.2f}")
                print(f"   Anomalies: {total_anomalies}")
                print(f"   Health score: {health_score:.3f}")

            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append({
                    'seed': seed,
                    'trial_id': i,
                    'episode_rewards': [],
                    'episode_lengths': [],
                    'training_metrics': [],
                    'agent_training_stats': {},
                    'forensics_analysis': {},
                    'anomalies': [],
                    'health_summary': {},
                    'agent': None,
                    'error': str(e)
                })

        self.results = results
        print("\n✅ Cross-validation completed!")
        print(f"   Total experiments: {len(results)}")
        print(f"   Successful experiments: {len([r for r in results if 'error' not in r])}")

        return results

    def analyze_results(self):
        """Analyze cross-validation results with statistical tests."""

        print("\n📊 Analyzing cross-validation results...")

        # Convert results to DataFrame
        df_data = []
        for result in self.results:
            if 'error' not in result:
                # Calculate summary metrics
                final_avg_reward = np.mean(result['episode_rewards'][-10:])
                total_anomalies = len(result['anomalies'])
                health_score = result['health_summary'].get('overall_health', 0)

                df_data.append({
                    'seed': result['seed'],
                    'trial_id': result['trial_id'],
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

        # 1. Performance statistics
        performance_stats = df['final_avg_reward'].agg(['mean', 'std', 'min', 'max', 'count'])
        print("\nPerformance Statistics:")
        print(f"   Mean: {performance_stats['mean']:.3f}")
        print(f"   Std: {performance_stats['std']:.3f}")
        print(f"   Min: {performance_stats['min']:.3f}")
        print(f"   Max: {performance_stats['max']:.3f}")
        print(f"   Count: {performance_stats['count']}")

        # 2. Confidence intervals
        confidence_level = 0.95
        n = len(df)
        mean = df['final_avg_reward'].mean()
        std = df['final_avg_reward'].std()
        se = std / np.sqrt(n)

        # t-distribution confidence interval
        t_critical = stats.t.ppf((1 + confidence_level) / 2, n - 1)
        margin_error = t_critical * se
        ci_lower = mean - margin_error
        ci_upper = mean + margin_error

        print(f"\nConfidence Intervals ({confidence_level*100}%):")
        print(f"   Mean: {mean:.3f} ± {margin_error:.3f}")
        print(f"   CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

        # 3. Normality test
        shapiro_stat, shapiro_p = stats.shapiro(df['final_avg_reward'])
        print("\nNormality Test (Shapiro-Wilk):")
        print(f"   Statistic: {shapiro_stat:.3f}")
        print(f"   p-value: {shapiro_p:.3f}")
        print(f"   Normal: {'Yes' if shapiro_p > 0.05 else 'No'}")

        # 4. Anomaly analysis
        anomaly_stats = df['total_anomalies'].agg(['mean', 'std', 'min', 'max'])
        print("\nAnomaly Statistics:")
        print(f"   Mean: {anomaly_stats['mean']:.1f}")
        print(f"   Std: {anomaly_stats['std']:.1f}")
        print(f"   Min: {anomaly_stats['min']:.1f}")
        print(f"   Max: {anomaly_stats['max']:.1f}")

        # 5. Health score analysis
        health_stats = df['health_score'].agg(['mean', 'std', 'min', 'max'])
        print("\nHealth Score Statistics:")
        print(f"   Mean: {health_stats['mean']:.3f}")
        print(f"   Std: {health_stats['std']:.3f}")
        print(f"   Min: {health_stats['min']:.3f}")
        print(f"   Max: {health_stats['max']:.3f}")

        return df

    def create_publication_figures(self, df):
        """Create publication-ready figures."""

        if df is None or df.empty:
            print("❌ No data to visualize!")
            return

        # Set publication style
        plt.style.use('seaborn-v0_8-whitegrid')

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Learning curves (average across seeds)
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

            # Average across seeds
            avg_rewards = np.mean(padded_rewards, axis=0)
            std_rewards = np.std(padded_rewards, axis=0)

            axes[0, 0].plot(avg_rewards, 'b-', linewidth=2, label='Mean')
            axes[0, 0].fill_between(range(len(avg_rewards)),
                                  avg_rewards - std_rewards,
                                  avg_rewards + std_rewards,
                                  alpha=0.3, color='blue', label='±1 Std')
            axes[0, 0].set_title('Learning Curve', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Episode', fontsize=12)
            axes[0, 0].set_ylabel('Episode Reward', fontsize=12)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Performance distribution
        axes[0, 1].hist(df['final_avg_reward'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(df['final_avg_reward'].mean(), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {df["final_avg_reward"].mean():.2f}')
        axes[0, 1].set_title('Performance Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Final Average Reward', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Confidence interval plot
        mean = df['final_avg_reward'].mean()
        std = df['final_avg_reward'].std()
        n = len(df)
        se = std / np.sqrt(n)
        t_critical = stats.t.ppf(0.975, n - 1)
        margin_error = t_critical * se

        axes[0, 2].errorbar(0, mean, yerr=margin_error, fmt='o', capsize=10, capthick=2,
                           markersize=8, color='red', label='95% CI')
        axes[0, 2].set_xlim(-0.5, 0.5)
        axes[0, 2].set_title('Performance with Confidence Interval', fontsize=14, fontweight='bold')
        axes[0, 2].set_ylabel('Final Average Reward', fontsize=12)
        axes[0, 2].set_xticks([])
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Anomaly distribution
        axes[1, 0].hist(df['total_anomalies'], bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 0].axvline(df['total_anomalies'].mean(), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {df["total_anomalies"].mean():.1f}')
        axes[1, 0].set_title('Anomaly Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Total Anomalies', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Health score distribution
        axes[1, 1].hist(df['health_score'], bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 1].axvline(df['health_score'].mean(), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {df["health_score"].mean():.3f}')
        axes[1, 1].set_title('Health Score Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Health Score', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Performance vs anomalies scatter
        axes[1, 2].scatter(df['total_anomalies'], df['final_avg_reward'],
                          alpha=0.7, s=60, color='purple', edgecolor='black')
        axes[1, 2].set_title('Performance vs Anomalies', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Total Anomalies', fontsize=12)
        axes[1, 2].set_ylabel('Final Average Reward', fontsize=12)
        axes[1, 2].grid(True, alpha=0.3)

        # Add correlation coefficient
        correlation = df['total_anomalies'].corr(df['final_avg_reward'])
        axes[1, 2].text(0.05, 0.95, f'r = {correlation:.3f}',
                       transform=axes[1, 2].transAxes, fontsize=12,
                       bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})

        plt.tight_layout()

        # Save figure
        figure_path = self.output_dir / 'publication_figures.png'
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 Publication figures saved to {figure_path}")

    def generate_experiment_report(self, df):
        """Generate comprehensive experiment report."""

        if df is None or df.empty:
            print("❌ No data to generate report!")
            return

        # Create report
        report = {
            'experiment_info': {
                'name': self.experiment_config['name'],
                'description': self.experiment_config.get('description', ''),
                'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'algorithm': 'ReproduciblePPO',
                'environment': self.environment_config['name'],
                'total_experiments': len(self.results),
                'successful_experiments': len([r for r in self.results if 'error' not in r])
            },
            'configuration': {
                'algorithm_config': self.algorithm_config,
                'environment_config': self.environment_config,
                'training_config': self.training_config,
                'analysis_config': self.analysis_config
            },
            'results': {
                'performance_statistics': {
                    'mean': float(df['final_avg_reward'].mean()),
                    'std': float(df['final_avg_reward'].std()),
                    'min': float(df['final_avg_reward'].min()),
                    'max': float(df['final_avg_reward'].max()),
                    'count': int(len(df))
                },
                'confidence_intervals': {
                    'level': 0.95,
                    'mean': float(df['final_avg_reward'].mean()),
                    'margin_error': float(stats.t.ppf(0.975, len(df) - 1) * df['final_avg_reward'].std() / np.sqrt(len(df))),
                    'lower': float(df['final_avg_reward'].mean() - stats.t.ppf(0.975, len(df) - 1) * df['final_avg_reward'].std() / np.sqrt(len(df))),
                    'upper': float(df['final_avg_reward'].mean() + stats.t.ppf(0.975, len(df) - 1) * df['final_avg_reward'].std() / np.sqrt(len(df)))
                },
                'normality_test': {
                    'statistic': float(stats.shapiro(df['final_avg_reward'])[0]),
                    'p_value': float(stats.shapiro(df['final_avg_reward'])[1]),
                    'normal': bool(stats.shapiro(df['final_avg_reward'])[1] > 0.05)
                },
                'anomaly_statistics': {
                    'mean': float(df['total_anomalies'].mean()),
                    'std': float(df['total_anomalies'].std()),
                    'min': float(df['total_anomalies'].min()),
                    'max': float(df['total_anomalies'].max())
                },
                'health_statistics': {
                    'mean': float(df['health_score'].mean()),
                    'std': float(df['health_score'].std()),
                    'min': float(df['health_score'].min()),
                    'max': float(df['health_score'].max())
                }
            },
            'recommendations': {
                'performance': 'Good' if df['final_avg_reward'].mean() > 100 else 'Needs improvement',
                'stability': 'Stable' if df['final_avg_reward'].std() < 50 else 'Unstable',
                'anomalies': 'Low' if df['total_anomalies'].mean() < 5 else 'High',
                'health': 'Healthy' if df['health_score'].mean() > 0.7 else 'Unhealthy'
            }
        }

        # Save report
        report_path = self.output_dir / 'experiment_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📄 Experiment report saved to {report_path}")

        return report

    def create_experiment_snapshot(self):
        """Create experiment snapshot for sharing."""

        snapshot_dir = self.output_dir / 'snapshot'
        snapshot_dir.mkdir(exist_ok=True)

        # Copy configuration
        config_path = snapshot_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        # Copy results
        results_path = snapshot_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        # Copy figures
        figure_path = self.output_dir / 'publication_figures.png'
        if figure_path.exists():
            snapshot_figure_path = snapshot_dir / 'publication_figures.png'
            snapshot_figure_path.write_bytes(figure_path.read_bytes())

        # Create README
        readme_path = snapshot_dir / 'README.md'
        readme_content = f"""# Experiment Snapshot: {self.experiment_config['name']}

## Description
{self.experiment_config.get('description', 'No description provided')}

## Configuration
- Algorithm: ReproduciblePPO
- Environment: {self.environment_config['name']}
- Episodes: {self.training_config['episodes']}
- Seeds: {len(self.training_config['seeds'])}

## Results
- Total experiments: {len(self.results)}
- Successful experiments: {len([r for r in self.results if 'error' not in r])}

## Files
- `config.yaml`: Experiment configuration
- `results.json`: Raw experiment results
- `publication_figures.png`: Publication-ready figures

## Reproducibility
This experiment was run with RLDK for full reproducibility. To reproduce:
1. Install RLDK: `pip install rldk`
2. Run: `python research_reproducibility_workflow.py`
3. Use the same configuration file: `config.yaml`

## Contact
For questions about this experiment, please contact the experimenter.
"""

        with open(readme_path, 'w') as f:
            f.write(readme_content)

        print(f"📦 Experiment snapshot created at {snapshot_dir}")

    def finish_experiment(self):
        """Finish experiment and save all results."""

        # Track results
        if self.results:
            # Create summary DataFrame
            df_data = []
            for result in self.results:
                if 'error' not in result:
                    final_avg_reward = np.mean(result['episode_rewards'][-10:])
                    total_anomalies = len(result['anomalies'])
                    health_score = result['health_summary'].get('overall_health', 0)

                    df_data.append({
                        'seed': result['seed'],
                        'trial_id': result['trial_id'],
                        'final_avg_reward': final_avg_reward,
                        'total_anomalies': total_anomalies,
                        'health_score': health_score
                    })

            if df_data:
                df = pd.DataFrame(df_data)
                self.tracker.track_dataset(
                    df,
                    "cross_validation_results",
                    {
                        "total_experiments": len(self.results),
                        "successful_experiments": len([r for r in self.results if 'error' not in r]),
                        "mean_performance": float(df['final_avg_reward'].mean()),
                        "std_performance": float(df['final_avg_reward'].std())
                    }
                )

        # Finish tracking
        summary = self.tracker.finish_experiment()

        return summary

def main():
    """Main function demonstrating research reproducibility workflow."""

    print("🚀 RLDK Research Reproducibility Workflow Example")
    print("=" * 60)

    # 1. Create configuration file
    print("\n1. Creating experiment configuration...")

    config = {
        'experiment': {
            'name': 'reproducible_ppo_research',
            'description': 'Reproducible PPO research workflow example',
            'output_dir': './research_runs',
            'tags': ['research', 'reproducible', 'ppo', 'cross-validation'],
            'notes': 'Demonstrating research reproducibility workflow with RLDK'
        },
        'algorithm': {
            'lr': 3e-4,
            'gamma': 0.99,
            'eps_clip': 0.2,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'gae_lambda': 0.95,
            'max_grad_norm': 0.5,
            'clip_value_loss': True,
            'normalize_advantages': True
        },
        'environment': {
            'name': 'CartPole-v1'
        },
        'training': {
            'episodes': 100,
            'max_steps_per_episode': 200,
            'seeds': [42, 123, 456, 789, 999]
        },
        'analysis': {
            'confidence_level': 0.95,
            'statistical_tests': ['shapiro', 'ttest'],
            'visualizations': ['learning_curves', 'performance_distribution', 'confidence_intervals']
        }
    }

    # Save configuration
    config_path = './research_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"💾 Configuration saved to {config_path}")

    # 2. Initialize research workflow
    print("\n2. Initializing research workflow...")

    workflow = ResearchWorkflow(config_path)

    # 3. Setup experiment tracking
    print("\n3. Setting up experiment tracking...")

    tracking_data = workflow.setup_experiment_tracking()
    print(f"🚀 Started experiment: {tracking_data['experiment_id']}")

    # 4. Run cross-validation
    print("\n4. Running cross-validation experiments...")

    results = workflow.run_cross_validation()

    # 5. Analyze results
    print("\n5. Analyzing results...")

    df = workflow.analyze_results()

    # 6. Create publication figures
    print("\n6. Creating publication figures...")

    workflow.create_publication_figures(df)

    # 7. Generate experiment report
    print("\n7. Generating experiment report...")

    workflow.generate_experiment_report(df)

    # 8. Create experiment snapshot
    print("\n8. Creating experiment snapshot...")

    workflow.create_experiment_snapshot()

    # 9. Finish experiment
    print("\n9. Finishing experiment...")

    summary = workflow.finish_experiment()

    print("\n🎉 Research Reproducibility Workflow completed successfully!")
    print(f"   Experiment ID: {summary['experiment_id']}")
    print(f"   Output directory: {workflow.output_dir}")
    print(f"   Total experiments: {len(results)}")

    print("\n📁 Files created:")
    print(f"   - {workflow.output_dir}/{summary['experiment_id']}/ - Experiment data")
    print(f"   - {workflow.output_dir}/publication_figures.png - Publication figures")
    print(f"   - {workflow.output_dir}/experiment_report.json - Experiment report")
    print(f"   - {workflow.output_dir}/snapshot/ - Experiment snapshot")

    # 10. Summary
    print("\n10. Summary and Next Steps...")

    print("\n✅ What We Accomplished:")
    print("   1. Reproducible Configuration: YAML-based experiment configuration")
    print("   2. Cross-Validation: Multiple seeds for statistical robustness")
    print("   3. Statistical Analysis: Confidence intervals and significance tests")
    print("   4. Publication Figures: Professional-quality visualizations")
    print("   5. Experiment Report: Comprehensive results summary")
    print("   6. Experiment Snapshot: Shareable experiment package")
    print("   7. RLDK Integration: Full tracking and forensics")

    if df is not None:
        print("\n📊 Key Findings:")
        print(f"   - Mean performance: {df['final_avg_reward'].mean():.3f}")
        print(f"   - Performance std: {df['final_avg_reward'].std():.3f}")
        print(f"   - Total experiments: {len(results)}")
        print(f"   - Successful experiments: {len([r for r in results if 'error' not in r])}")

    print("\n🚀 Next Steps:")
    print("   1. Hyperparameter Optimization: Systematic hyperparameter tuning")
    print("   2. Ablation Studies: Component-wise analysis")
    print("   3. Comparison Studies: Compare with other algorithms")
    print("   4. Sensitivity Analysis: Analyze parameter sensitivity")
    print("   5. Publication: Prepare for academic publication")
    print("   6. Open Science: Share experiment snapshots")

    print("\n📚 Key Takeaways:")
    print("   - RLDK enables fully reproducible research workflows")
    print("   - YAML configuration ensures experiment reproducibility")
    print("   - Cross-validation provides statistical robustness")
    print("   - Publication figures are automatically generated")
    print("   - Experiment snapshots enable easy sharing")
    print("   - Statistical analysis ensures scientific rigor")

    print("\nHappy research! 🎉")

if __name__ == "__main__":
    main()
