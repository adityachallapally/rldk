#!/usr/bin/env python3
"""
Custom Environment Tutorial for RLDK

This tutorial demonstrates how to integrate RLDK with custom reinforcement learning
environments. We'll create a simple custom environment and show how to track
experiments, run forensics analysis, and ensure reproducibility.

Learning Objectives:
- How to create custom RL environments compatible with RLDK
- How to track custom environment state and dynamics
- How to implement custom forensics rules for domain-specific issues
- How to ensure reproducibility with custom environments
- How to analyze training dynamics in custom environments

Prerequisites:
- RLDK installed (pip install rldk)
- Basic understanding of RL environments
- Familiarity with gymnasium/gym environments
"""

import json
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces

# RLDK imports
import rldk
from rldk.forensics import ComprehensivePPOForensics
from rldk.tracking import ExperimentTracker, TrackingConfig
from rldk.utils import set_global_seed, validate_numeric_range


class CustomGridWorld(gym.Env):
    """
    Custom Grid World environment for RLDK demonstration.

    This is a simple grid world where an agent must navigate to a goal
    while avoiding obstacles. The environment includes:
    - Stochastic dynamics (actions sometimes fail)
    - Rewards that change over time (non-stationary)
    - Custom observation space with additional context
    - Configurable difficulty levels
    """

    def __init__(self, size=8, difficulty="easy", stochasticity=0.1):
        super().__init__()

        self.size = size
        self.difficulty = difficulty
        self.stochasticity = stochasticity

        # Action space: 4 directions
        self.action_space = spaces.Discrete(4)

        # Observation space: position + goal direction + time + difficulty
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6,), dtype=np.float32
        )

        # Environment state
        self.agent_pos = None
        self.goal_pos = None
        self.obstacles = None
        self.time_step = 0
        self.max_steps = size * 4

        # Action mappings
        self.action_map = {
            0: np.array([-1, 0]),  # Up
            1: np.array([1, 0]),   # Down
            2: np.array([0, -1]),  # Left
            3: np.array([0, 1])    # Right
        }

        # Setup environment based on difficulty
        self._setup_environment()

    def _setup_environment(self):
        """Setup the environment based on difficulty level."""
        if self.difficulty == "easy":
            # Few obstacles, goal close to start
            self.obstacle_count = 2
            self.goal_distance = 3
        elif self.difficulty == "medium":
            # More obstacles, goal further away
            self.obstacle_count = 5
            self.goal_distance = 5
        elif self.difficulty == "hard":
            # Many obstacles, goal far away
            self.obstacle_count = 8
            self.goal_distance = 7
        else:
            raise ValueError(f"Unknown difficulty: {self.difficulty}")

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset state
        self.time_step = 0

        # Place agent at random position
        self.agent_pos = np.array([0, 0])

        # Place goal
        self.goal_pos = self._generate_goal_position()

        # Generate obstacles
        self.obstacles = self._generate_obstacles()

        return self._get_observation(), {}

    def _generate_goal_position(self):
        """Generate goal position based on difficulty."""
        # Ensure goal is reachable and not at start
        attempts = 0
        while attempts < 100:
            if self.difficulty == "easy":
                goal = np.random.randint(1, self.goal_distance + 1, 2)
            elif self.difficulty == "medium":
                goal = np.random.randint(2, self.goal_distance + 1, 2)
            else:  # hard
                goal = np.random.randint(3, self.goal_distance + 1, 2)

            # Ensure goal is within bounds
            if np.all(goal < self.size):
                return goal
            attempts += 1

        # Fallback
        return np.array([self.size - 1, self.size - 1])

    def _generate_obstacles(self):
        """Generate obstacles based on difficulty."""
        obstacles = []
        attempts = 0

        while len(obstacles) < self.obstacle_count and attempts < 1000:
            obstacle = np.random.randint(0, self.size, 2)

            # Don't place obstacles at start or goal
            if (not np.array_equal(obstacle, self.agent_pos) and
                not np.array_equal(obstacle, self.goal_pos) and
                not any(np.array_equal(obstacle, obs) for obs in obstacles)):
                obstacles.append(obstacle)

            attempts += 1

        return obstacles

    def _get_observation(self):
        """Get current observation."""
        # Position (normalized)
        pos_norm = self.agent_pos / (self.size - 1)

        # Goal direction (normalized)
        goal_direction = (self.goal_pos - self.agent_pos) / (self.size - 1)

        # Time step (normalized)
        time_norm = self.time_step / self.max_steps

        # Difficulty encoding
        difficulty_encoding = {
            "easy": [1, 0, 0],
            "medium": [0, 1, 0],
            "hard": [0, 0, 1]
        }[self.difficulty]

        observation = np.concatenate([
            pos_norm,
            goal_direction,
            [time_norm],
            difficulty_encoding
        ]).astype(np.float32)

        return observation

    def step(self, action):
        """Take a step in the environment."""
        self.time_step += 1

        # Get intended movement
        movement = self.action_map[action]

        # Apply stochasticity (action sometimes fails)
        if np.random.random() < self.stochasticity:
            # Random action instead
            movement = self.action_map[np.random.randint(4)]

        # Calculate new position
        new_pos = self.agent_pos + movement

        # Check bounds
        new_pos = np.clip(new_pos, 0, self.size - 1)

        # Check obstacles
        if any(np.array_equal(new_pos, obstacle) for obstacle in self.obstacles):
            # Hit obstacle, stay in place
            new_pos = self.agent_pos.copy()
            reward = -0.1
        else:
            # Valid move
            self.agent_pos = new_pos

            # Calculate reward
            reward = self._calculate_reward()

        # Check termination
        terminated = np.array_equal(self.agent_pos, self.goal_pos)
        truncated = self.time_step >= self.max_steps

        # Check if agent hit an obstacle
        hit_obstacle = False
        for obstacle in self.obstacles:
            if np.array_equal(self.agent_pos, obstacle):
                hit_obstacle = True
                break

        # Additional info
        info = {
            "hit_obstacle": hit_obstacle,
            "time_step": self.time_step,
            "distance_to_goal": np.linalg.norm(self.goal_pos - self.agent_pos)
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _calculate_reward(self):
        """Calculate reward for current state."""
        # Distance-based reward
        distance = np.linalg.norm(self.goal_pos - self.agent_pos)
        max_distance = np.sqrt(2) * (self.size - 1)
        distance_reward = 1.0 - (distance / max_distance)

        # Time penalty
        time_penalty = -0.01

        # Goal bonus
        goal_bonus = 10.0 if np.array_equal(self.agent_pos, self.goal_pos) else 0.0

        # Non-stationary reward (changes over time)
        time_factor = 1.0 + 0.1 * np.sin(self.time_step / 10.0)

        total_reward = (distance_reward + time_penalty + goal_bonus) * time_factor

        return total_reward

    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            grid = np.zeros((self.size, self.size))

            # Mark obstacles
            for obstacle in self.obstacles:
                grid[obstacle[0], obstacle[1]] = -1

            # Mark agent
            grid[self.agent_pos[0], self.agent_pos[1]] = 1

            # Mark goal
            grid[self.goal_pos[0], self.goal_pos[1]] = 2

            print("\\nGrid World:")
            print("A = Agent, G = Goal, X = Obstacle, . = Empty")
            for i in range(self.size):
                row = ""
                for j in range(self.size):
                    if grid[i, j] == 1:
                        row += "A "
                    elif grid[i, j] == 2:
                        row += "G "
                    elif grid[i, j] == -1:
                        row += "X "
                    else:
                        row += ". "
                print(row)
            print(f"Time: {self.time_step}/{self.max_steps}")
            print()

class CustomPPO:
    """Custom PPO implementation for the grid world environment."""

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip

        # Initialize policy and value networks (simple linear)
        self.policy = np.random.randn(state_dim, action_dim) * 0.1
        self.value = np.random.randn(state_dim, 1) * 0.1

        # Custom tracking for grid world specific metrics
        self.grid_metrics = []

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

    def update(self, states, actions, rewards, log_probs_old, values_old, advantages, returns, info_list):
        """Update policy and value function with custom metrics."""
        # Standard PPO update
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

            # Simple gradient step
            self.policy -= self.lr * policy_loss * state.reshape(-1, 1)
            self.value -= self.lr * value_loss * state.reshape(-1, 1)

        # Track custom metrics
        if info_list:
            avg_distance = np.mean([info.get('distance_to_goal', 0) for info in info_list])
            obstacle_hits = sum([info.get('hit_obstacle', False) for info in info_list])

            self.grid_metrics.append({
                'avg_distance_to_goal': avg_distance,
                'obstacle_hits': obstacle_hits,
                'exploration_efficiency': 1.0 / (avg_distance + 1e-8)
            })

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

class CustomForensics(ComprehensivePPOForensics):
    """Custom forensics for grid world environment."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid_specific_metrics = []

    def update(self, step, kl, kl_coef, entropy, reward_mean, reward_std,
               policy_grad_norm, value_grad_norm, advantage_mean, advantage_std,
               distance_to_goal=None, obstacle_hits=None, exploration_efficiency=None):
        """Update forensics with custom grid world metrics."""

        # Standard PPO forensics
        super().update(
            step=step, kl=kl, kl_coef=kl_coef, entropy=entropy,
            reward_mean=reward_mean, reward_std=reward_std,
            policy_grad_norm=policy_grad_norm, value_grad_norm=value_grad_norm,
            advantage_mean=advantage_mean, advantage_std=advantage_std
        )

        # Custom grid world metrics
        if distance_to_goal is not None:
            self.grid_specific_metrics.append({
                'step': step,
                'distance_to_goal': distance_to_goal,
                'obstacle_hits': obstacle_hits or 0,
                'exploration_efficiency': exploration_efficiency or 0
            })

    def get_grid_analysis(self):
        """Get grid world specific analysis."""
        if not self.grid_specific_metrics:
            return {"error": "No grid metrics available"}

        df = pd.DataFrame(self.grid_specific_metrics)

        analysis = {
            "avg_distance_to_goal": df['distance_to_goal'].mean(),
            "min_distance_to_goal": df['distance_to_goal'].min(),
            "total_obstacle_hits": df['obstacle_hits'].sum(),
            "avg_exploration_efficiency": df['exploration_efficiency'].mean(),
            "distance_trend": "improving" if df['distance_to_goal'].iloc[-10:].mean() < df['distance_to_goal'].iloc[:10].mean() else "not improving"
        }

        return analysis

def main():
    """Main function demonstrating RLDK with custom environment."""

    print("🚀 RLDK Custom Environment Tutorial")
    print("=" * 50)

    # 1. Setup
    print("\n1. Setting up reproducible environment...")
    seed = set_global_seed(42)
    print(f"🌱 Set global seed to: {seed}")

    # 2. Create Custom Environment
    print("\n2. Creating custom Grid World environment...")

    # Test different difficulties
    difficulties = ["easy", "medium", "hard"]

    for difficulty in difficulties:
        print(f"\n🎯 Testing {difficulty} difficulty...")

        env = CustomGridWorld(size=8, difficulty=difficulty, stochasticity=0.1)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        print(f"   State dimension: {state_dim}")
        print(f"   Action dimension: {action_dim}")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")

        # Test environment
        obs, info = env.reset()
        print(f"   Initial observation: {obs}")
        print(f"   Initial info: {info}")

        # Take a few random actions
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"   Step {i+1}: action={action}, reward={reward:.3f}, done={terminated or truncated}")
            if terminated or truncated:
                break

        env.close()

    # 3. Experiment Tracking Setup
    print("\n3. Setting up experiment tracking...")

    config = TrackingConfig(
        experiment_name="custom_grid_world",
        enable_dataset_tracking=True,
        enable_model_tracking=True,
        enable_environment_tracking=True,
        enable_seed_tracking=True,
        enable_git_tracking=True,
        output_dir="./runs",
        tags=["demo", "custom-env", "grid-world", "tutorial"],
        notes="Custom Grid World environment tutorial for RLDK demonstration"
    )

    tracker = ExperimentTracker(config)
    tracking_data = tracker.start_experiment()
    print(f"🚀 Started experiment: {tracking_data['experiment_id']}")

    # 4. Training with Custom Environment
    print("\n4. Training with custom environment...")

    def train_custom_environment(difficulty="medium", episodes=100):
        """Train on custom environment with comprehensive tracking."""

        # Create environment
        env = CustomGridWorld(size=8, difficulty=difficulty, stochasticity=0.1)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Initialize agent and forensics
        agent = CustomPPO(state_dim, action_dim, lr=3e-4)
        forensics = CustomForensics(kl_target=0.1)

        # Training statistics
        episode_rewards = []
        episode_lengths = []
        training_metrics = []

        print(f"🎯 Training on {difficulty} Grid World for {episodes} episodes")

        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0

            # Store trajectory data
            states, actions, rewards, log_probs, values, info_list = [], [], [], [], [], []

            for step in range(env.max_steps):
                # Get action from policy
                action, log_prob = agent.get_action(state)
                value = agent.get_value(state)

                # Take action
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Store experience
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                info_list.append(info)

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

                # Update policy with custom metrics
                agent.update(states, actions, rewards, log_probs, values, advantages, returns, info_list)

                # Compute training metrics
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

                # Custom metrics
                avg_distance = np.mean([info.get('distance_to_goal', 0) for info in info_list])
                obstacle_hits = sum([info.get('hit_obstacle', False) for info in info_list])
                exploration_efficiency = 1.0 / (avg_distance + 1e-8)

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
                    advantage_std=np.std(advantages),
                    distance_to_goal=avg_distance,
                    obstacle_hits=obstacle_hits,
                    exploration_efficiency=exploration_efficiency
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
                    'advantage_std': np.std(advantages),
                    'avg_distance_to_goal': avg_distance,
                    'obstacle_hits': obstacle_hits,
                    'exploration_efficiency': exploration_efficiency
                })

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Progress update
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(episode_rewards[-20:])
                avg_length = np.mean(episode_lengths[-20:])
                print(f"Episode {episode + 1:3d}: Avg Reward = {avg_reward:6.2f}, Avg Length = {avg_length:3.1f}")

        env.close()

        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'training_metrics': training_metrics,
            'forensics': forensics,
            'agent': agent,
            'grid_metrics': agent.grid_metrics
        }

    # Run training
    print("🚀 Starting training...")
    start_time = time.time()

    training_results = train_custom_environment(difficulty="medium", episodes=100)

    training_time = time.time() - start_time
    print(f"\n⏱️  Training completed in {training_time:.2f} seconds")

    # Extract results
    episode_rewards = training_results['episode_rewards']
    episode_lengths = training_results['episode_lengths']
    training_metrics = training_results['training_metrics']
    forensics = training_results['forensics']
    agent = training_results['agent']
    grid_metrics = training_results['grid_metrics']

    print(f"📊 Final average reward: {np.mean(episode_rewards[-10:]):.2f}")
    print(f"📏 Final average length: {np.mean(episode_lengths[-10:]):.2f}")

    # 5. Track Custom Data
    print("\n5. Tracking custom environment data...")

    # Convert training metrics to DataFrame
    metrics_df = pd.DataFrame(training_metrics)

    # Track training data with custom metrics
    tracker.track_dataset(
        metrics_df,
        "custom_grid_world_metrics",
        {
            "episodes": len(episode_rewards),
            "total_steps": sum(episode_lengths),
            "final_reward": episode_rewards[-1],
            "avg_reward": np.mean(episode_rewards),
            "training_time": training_time,
            "environment_type": "CustomGridWorld",
            "difficulty": "medium",
            "grid_size": 8,
            "stochasticity": 0.1
        }
    )

    # Track the trained model
    tracker.track_model(
        agent,
        "custom_grid_world_model",
        {
            "algorithm": "CustomPPO",
            "environment": "CustomGridWorld",
            "state_dim": agent.state_dim,
            "action_dim": agent.action_dim,
            "learning_rate": agent.lr,
            "gamma": agent.gamma,
            "eps_clip": agent.eps_clip,
            "grid_size": 8,
            "difficulty": "medium"
        }
    )

    # Add custom metadata
    tracker.add_metadata("environment_type", "CustomGridWorld")
    tracker.add_metadata("grid_size", 8)
    tracker.add_metadata("difficulty", "medium")
    tracker.add_metadata("stochasticity", 0.1)
    tracker.add_metadata("final_avg_reward", np.mean(episode_rewards[-10:]))
    tracker.add_metadata("total_obstacle_hits", sum([m['obstacle_hits'] for m in grid_metrics]))

    print("✅ Custom environment data tracked successfully")

    # 6. Custom Forensics Analysis
    print("\n6. Running custom forensics analysis...")

    # Standard forensics
    analysis = forensics.get_comprehensive_analysis()
    anomalies = forensics.get_anomalies()
    health_summary = forensics.get_health_summary()

    print("🔍 Standard Forensics Analysis:")
    print(f"   Total anomalies detected: {len(anomalies)}")
    print(f"   Overall health score: {health_summary.get('overall_health', 'N/A')}")

    # Custom grid world analysis
    grid_analysis = forensics.get_grid_analysis()

    print("\n🎯 Custom Grid World Analysis:")
    print(f"   Average distance to goal: {grid_analysis['avg_distance_to_goal']:.3f}")
    print(f"   Minimum distance to goal: {grid_analysis['min_distance_to_goal']:.3f}")
    print(f"   Total obstacle hits: {grid_analysis['total_obstacle_hits']}")
    print(f"   Average exploration efficiency: {grid_analysis['avg_exploration_efficiency']:.3f}")
    print(f"   Distance trend: {grid_analysis['distance_trend']}")

    # Save custom analysis
    custom_analysis = {
        "standard_forensics": analysis,
        "grid_world_analysis": grid_analysis,
        "anomalies": anomalies,
        "health_summary": health_summary
    }

    with open("./custom_forensics_analysis.json", "w") as f:
        json.dump(custom_analysis, f, indent=2, default=str)

    print("\n💾 Custom forensics analysis saved to ./custom_forensics_analysis.json")

    # 7. Custom Visualizations
    print("\n7. Creating custom visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.7, label='Episode Reward')
    axes[0, 0].plot(pd.Series(episode_rewards).rolling(20).mean(), color='red', linewidth=2, label='20-Episode Average')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Episode lengths
    axes[0, 1].plot(episode_lengths, alpha=0.7, label='Episode Length')
    axes[0, 1].plot(pd.Series(episode_lengths).rolling(20).mean(), color='red', linewidth=2, label='20-Episode Average')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Length')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Distance to goal
    if 'avg_distance_to_goal' in metrics_df.columns:
        axes[0, 2].plot(metrics_df['episode'], metrics_df['avg_distance_to_goal'], alpha=0.7)
        axes[0, 2].set_title('Average Distance to Goal')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Distance')
        axes[0, 2].grid(True, alpha=0.3)

    # Obstacle hits
    if 'obstacle_hits' in metrics_df.columns:
        axes[1, 0].plot(metrics_df['episode'], metrics_df['obstacle_hits'], alpha=0.7)
        axes[1, 0].set_title('Obstacle Hits per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Obstacle Hits')
        axes[1, 0].grid(True, alpha=0.3)

    # Exploration efficiency
    if 'exploration_efficiency' in metrics_df.columns:
        axes[1, 1].plot(metrics_df['episode'], metrics_df['exploration_efficiency'], alpha=0.7)
        axes[1, 1].set_title('Exploration Efficiency')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Efficiency')
        axes[1, 1].grid(True, alpha=0.3)

    # KL divergence
    if 'kl' in metrics_df.columns:
        axes[1, 2].plot(metrics_df['episode'], metrics_df['kl'], alpha=0.7)
        axes[1, 2].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='KL Target (0.1)')
        axes[1, 2].set_title('KL Divergence')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('KL Divergence')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./custom_environment_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("📊 Custom environment plots saved to ./custom_environment_plots.png")

    # 8. Environment Validation
    print("\n8. Validating custom environment...")

    # Test environment consistency
    print("🧪 Testing environment consistency...")

    env = CustomGridWorld(size=8, difficulty="easy", stochasticity=0.1)

    # Test reset consistency
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)

    if np.allclose(obs1, obs2):
        print("   ✅ Reset consistency: PASSED")
    else:
        print("   ❌ Reset consistency: FAILED")

    # Test action space
    for action in range(env.action_space.n):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Action {action}: reward={reward:.3f}, done={terminated or truncated}")
        if terminated or truncated:
            break

    env.close()

    # 9. Finish Experiment
    print("\n9. Finishing experiment...")

    summary = tracker.finish_experiment()

    print("\n🎉 Custom Environment Tutorial completed successfully!")
    print(f"   Experiment ID: {summary['experiment_id']}")
    print(f"   Output directory: {config.output_dir}")
    print(f"   Total episodes: {len(episode_rewards)}")
    print(f"   Final average reward: {np.mean(episode_rewards[-10:]):.2f}")
    print(f"   Training time: {training_time:.2f} seconds")

    print("\n📁 Files created:")
    print(f"   - {config.output_dir}/{summary['experiment_id']}/ - Experiment data")
    print("   - ./custom_forensics_analysis.json - Custom forensics analysis")
    print("   - ./custom_environment_plots.png - Custom environment visualizations")

    # 10. Summary
    print("\n10. Summary and Next Steps...")

    print("\n✅ What We Accomplished:")
    print("   1. Custom Environment: Created a Grid World environment with custom features")
    print("   2. Custom Agent: Implemented a PPO agent with grid-specific metrics")
    print("   3. Custom Forensics: Extended forensics for domain-specific analysis")
    print("   4. Experiment Tracking: Tracked custom environment and agent data")
    print("   5. Custom Visualizations: Created environment-specific plots")
    print("   6. Environment Validation: Tested environment consistency and correctness")

    print("\n📊 Key Metrics:")
    print(f"   - Final Average Reward: {np.mean(episode_rewards[-10:]):.2f}")
    print(f"   - Training Time: {training_time:.2f} seconds")
    print(f"   - Total Obstacle Hits: {sum([m['obstacle_hits'] for m in grid_metrics])}")
    print(f"   - Exploration Efficiency: {grid_analysis['avg_exploration_efficiency']:.3f}")
    print(f"   - Distance Trend: {grid_analysis['distance_trend']}")

    print("\n🚀 Next Steps:")
    print("   1. Experiment with different difficulties and grid sizes")
    print("   2. Add more complex custom metrics and forensics rules")
    print("   3. Implement custom reward functions and dynamics")
    print("   4. Create multi-agent environments")
    print("   5. Integrate with your own RL algorithms")

    print("\n📚 Key Takeaways:")
    print("   - RLDK works seamlessly with custom environments")
    print("   - You can extend forensics for domain-specific analysis")
    print("   - Custom metrics provide deeper insights into training")
    print("   - Environment validation ensures reproducibility")
    print("   - Experiment tracking captures all custom data")

    print("\nHappy experimenting with custom environments! 🎉")

if __name__ == "__main__":
    main()
