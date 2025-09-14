#!/usr/bin/env python3
"""
Real RL Performance Test for RLDK Tracking System

This script tests the performance improvements with actual RL models,
environments, and training loops to ensure the tracking system works
efficiently in real-world RL scenarios.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import json
from pathlib import Path
import tempfile
import shutil
import asyncio
from typing import Dict, Any, List, Tuple
import psutil
import gc

# Try to import gym and stable-baselines3, but handle gracefully if not available
try:
    import gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    print("Warning: gym not available, will use mock environment")

try:
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not available, will use custom implementations")

# Import RLDK components
from src.rldk.config.settings import RLDKSettings
from src.rldk.tracking.config import TrackingConfig
from src.rldk.tracking.tracker import ExperimentTracker
from src.rldk.tracking.dataset_tracker import DatasetTracker
from src.rldk.tracking.model_tracker import ModelTracker


class MockRLEnvironment:
    """Mock RL environment for testing when gym is not available."""
    
    def __init__(self, state_dim=4, action_dim=2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32) if GYM_AVAILABLE else None
        self.action_space = gym.spaces.Discrete(action_dim) if GYM_AVAILABLE else None
        
    def reset(self):
        return np.random.randn(self.state_dim).astype(np.float32)
    
    def step(self, action):
        state = np.random.randn(self.state_dim).astype(np.float32)
        reward = np.random.randn()
        done = np.random.random() < 0.1
        info = {}
        return state, reward, done, info


class SimplePPOAgent:
    """Simple PPO implementation for testing."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=lr)
        
    def get_action(self, state):
        with torch.no_grad():
            logits = self.policy_net(torch.FloatTensor(state))
            action_probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(action_probs, 1).item()
            return action, action_probs[action].item()
    
    def get_value(self, state):
        with torch.no_grad():
            return self.value_net(torch.FloatTensor(state)).item()
    
    def update(self, states, actions, rewards, values, old_probs):
        # Simple policy gradient update
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        values = torch.FloatTensor(values)
        old_probs = torch.FloatTensor(old_probs)
        
        # Compute new action probabilities
        logits = self.policy_net(states)
        new_probs = torch.softmax(logits, dim=-1)
        new_action_probs = new_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute advantages
        advantages = rewards - values
        
        # Policy loss
        ratio = new_action_probs / (old_probs + 1e-8)
        policy_loss = -torch.mean(ratio * advantages)
        
        # Value loss
        new_values = self.value_net(states).squeeze()
        value_loss = torch.mean((new_values - rewards) ** 2)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()


class SimpleDQNAgent:
    """Simple DQN implementation for testing."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Q-network
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            q_values = self.q_net(torch.FloatTensor(state))
            return q_values.argmax().item()
    
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.q_net(next_states).max(1)[0]
            target_q_values = rewards + 0.99 * next_q_values * (~dones)
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()


class RLPerformanceTester:
    """Test RLDK tracking performance with real RL models and training."""
    
    def __init__(self):
        self.results = {}
        self.temp_dir = Path(tempfile.mkdtemp())
        self.env = None
        self.setup_environment()
        
    def setup_environment(self):
        """Set up RL environment."""
        if GYM_AVAILABLE:
            try:
                self.env = gym.make('CartPole-v1')
                print("Using CartPole-v1 environment")
            except:
                self.env = MockRLEnvironment()
                print("Using mock environment (gym failed)")
        else:
            self.env = MockRLEnvironment()
            print("Using mock environment (gym not available)")
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def measure_memory_usage(self):
        """Measure current memory usage."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def test_settings_performance_with_rl(self):
        """Test settings performance in RL context."""
        print("Testing settings performance with RL context...")
        
        start_time = time.time()
        settings = RLDKSettings()
        
        # Configure for RL workloads
        settings.tracking_timeout = 60.0  # Longer timeout for RL
        settings.dataset_sample_size = 5000  # Larger sample for RL data
        settings.model_fingerprint_limit = 1000000000  # 1B parameters for large RL models
        settings.enable_async_init = True
        settings.cache_environment = True
        settings.cache_git_info = True
        
        init_time = time.time() - start_time
        
        # Test performance config
        perf_config = settings.get_performance_config()
        
        return {
            "init_time": init_time,
            "performance_config": perf_config,
            "memory_usage_mb": self.measure_memory_usage()
        }
    
    def test_dataset_tracking_with_rl_data(self):
        """Test dataset tracking with RL training data."""
        print("Testing dataset tracking with RL training data...")
        
        tracker = DatasetTracker()
        results = {}
        
        # Generate RL training data of various sizes
        data_sizes = [
            (1000, "Small episode batch"),
            (10000, "Medium episode batch"), 
            (100000, "Large episode batch"),
            (1000000, "Very large episode batch")
        ]
        
        for size, name in data_sizes:
            # Generate realistic RL data (states, actions, rewards)
            states = np.random.randn(size, 4).astype(np.float32)
            actions = np.random.randint(0, 2, size)
            rewards = np.random.randn(size)
            dones = np.random.random(size) < 0.1
            
            # Create dataset as structured data
            rl_data = {
                'states': states,
                'actions': actions, 
                'rewards': rewards,
                'dones': dones
            }
            
            start_time = time.time()
            tracking_info = tracker.track_dataset(rl_data, f"rl_data_{size}")
            elapsed = time.time() - start_time
            
            results[name] = {
                "time": elapsed,
                "checksum": tracking_info.get("checksum", "unknown"),
                "size_bytes": tracking_info.get("size_bytes", 0)
            }
            
            print(f"  {name}: {elapsed:.3f}s, checksum: {tracking_info.get('checksum', 'unknown')[:16]}...")
        
        return results
    
    def test_model_tracking_with_rl_models(self):
        """Test model tracking with RL models of various sizes."""
        print("Testing model tracking with RL models...")
        
        tracker = ModelTracker()
        results = {}
        
        # Test different RL model architectures
        model_configs = [
            (4, 2, 32, "Small PPO (4->32->2)"),
            (4, 2, 64, "Medium PPO (4->64->2)"),
            (4, 2, 128, "Large PPO (4->128->2)"),
            (8, 4, 256, "Very Large PPO (8->256->4)"),
            (16, 8, 512, "Huge PPO (16->512->8)")
        ]
        
        for state_dim, action_dim, hidden_dim, name in model_configs:
            # Create RL model
            model = SimplePPOAgent(state_dim, action_dim, hidden_dim)
            
            start_time = time.time()
            tracking_info = tracker.track_model(model, f"rl_model_{name}")
            elapsed = time.time() - start_time
            
            results[name] = {
                "time": elapsed,
                "architecture_checksum": tracking_info.get("architecture_checksum", "unknown"),
                "num_parameters": tracking_info.get("num_parameters", 0),
                "weights_size_bytes": tracking_info.get("weights_size_bytes", 0)
            }
            
            print(f"  {name}: {elapsed:.3f}s, params: {tracking_info.get('num_parameters', 0):,}")
        
        return results
    
    def test_training_loop_with_tracking(self, agent_type="PPO", episodes=100):
        """Test RL training loop with tracking enabled."""
        print(f"Testing {agent_type} training loop with tracking...")
        
        # Create tracking config
        config = TrackingConfig(
            experiment_name=f"rl_training_{agent_type.lower()}",
            output_dir=self.temp_dir / "rl_experiments"
        )
        
        # Create tracker
        tracker = ExperimentTracker(config)
        
        # Initialize tracker
        start_time = time.time()
        if config.enable_async_init:
            try:
                asyncio.run(tracker.initialize_async())
            except Exception as e:
                print(f"Async init failed, falling back to sync: {e}")
                tracker.initialize_sync()
        else:
            tracker.initialize_sync()
        init_time = time.time() - start_time
        
        # Start experiment
        start_time = time.time()
        tracking_data = tracker.start_experiment()
        start_time_total = time.time() - start_time
        
        # Create RL agent
        if agent_type == "PPO":
            agent = SimplePPOAgent(4, 2, 64)
        elif agent_type == "DQN":
            agent = SimpleDQNAgent(4, 2, 64)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Track the model
        start_time = time.time()
        model_info = tracker.track_model(agent, f"{agent_type.lower()}_model")
        model_tracking_time = time.time() - start_time
        
        # Training loop
        training_data = []
        episode_rewards = []
        
        start_time = time.time()
        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_data = []
            
            for step in range(1000):  # Max steps per episode
                action, prob = agent.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                episode_data.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done
                })
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            training_data.extend(episode_data)
            
            # Update agent every 10 episodes
            if episode % 10 == 0 and len(training_data) > 0:
                batch_size = min(1000, len(training_data))
                batch = training_data[-batch_size:]
                
                states = [d['state'] for d in batch]
                actions = [d['action'] for d in batch]
                rewards = [d['reward'] for d in batch]
                
                if agent_type == "PPO":
                    values = [agent.get_value(s) for s in states]
                    old_probs = [0.5] * len(actions)  # Simplified
                    loss = agent.update(states, actions, rewards, values, old_probs)
                elif agent_type == "DQN":
                    next_states = [d['next_state'] for d in batch]
                    dones = [d['done'] for d in batch]
                    loss = agent.update(states, actions, rewards, next_states, dones)
                
                # Track training data periodically
                if episode % 50 == 0:
                    start_time = time.time()
                    tracker.track_dataset(training_data[-1000:], f"training_data_ep_{episode}")
                    data_tracking_time = time.time() - start_time
                    
                    print(f"  Episode {episode}: reward={episode_reward:.2f}, loss={loss:.4f}, data_tracking={data_tracking_time:.3f}s")
        
        training_time = time.time() - start_time
        
        # Track final model
        start_time = time.time()
        final_model_info = tracker.track_model(agent, f"{agent_type.lower()}_final_model")
        final_model_tracking_time = time.time() - start_time
        
        # Track final training data
        start_time = time.time()
        tracker.track_dataset(training_data, f"{agent_type.lower()}_final_training_data")
        final_data_tracking_time = time.time() - start_time
        
        # Finish experiment
        tracker.finish_experiment()
        
        return {
            "init_time": init_time,
            "start_time": start_time_total,
            "model_tracking_time": model_tracking_time,
            "training_time": training_time,
            "final_model_tracking_time": final_model_tracking_time,
            "final_data_tracking_time": final_data_tracking_time,
            "episodes": episodes,
            "avg_reward": np.mean(episode_rewards[-10:]) if episode_rewards else 0,
            "total_data_points": len(training_data),
            "memory_usage_mb": self.measure_memory_usage()
        }
    
    def test_large_scale_rl_training(self):
        """Test with large-scale RL training scenario."""
        print("Testing large-scale RL training...")
        
        # Create large environment
        large_env = MockRLEnvironment(state_dim=16, action_dim=8)
        
        # Create large model
        large_agent = SimplePPOAgent(16, 8, 512)  # Large model
        
        # Create tracking config optimized for large scale
        config = TrackingConfig(
            experiment_name="large_scale_rl",
            output_dir=self.temp_dir / "large_scale_experiments"
        )
        
        # Configure for large scale
        config.dataset_sample_size = 10000
        config.model_fingerprint_limit = 1000000000  # 1B parameters
        
        tracker = ExperimentTracker(config)
        
        # Initialize
        start_time = time.time()
        try:
            asyncio.run(tracker.initialize_async())
        except RuntimeError:
            # Already in event loop, use create_task
            loop = asyncio.get_event_loop()
            loop.run_until_complete(tracker.initialize_async())
        init_time = time.time() - start_time
        
        # Start experiment
        start_time = time.time()
        tracking_data = tracker.start_experiment()
        start_time_total = time.time() - start_time
        
        # Track large model
        start_time = time.time()
        model_info = tracker.track_model(large_agent, "large_ppo_model")
        model_tracking_time = time.time() - start_time
        
        # Generate large dataset
        print("  Generating large RL dataset...")
        large_dataset = []
        for i in range(10000):  # 10K samples
            state = large_env.reset()
            action = np.random.randint(0, 8)
            next_state, reward, done, _ = large_env.step(action)
            large_dataset.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })
        
        # Track large dataset
        start_time = time.time()
        dataset_info = tracker.track_dataset(large_dataset, "large_rl_dataset")
        dataset_tracking_time = time.time() - start_time
        
        # Finish experiment
        tracker.finish_experiment()
        
        return {
            "init_time": init_time,
            "start_time": start_time_total,
            "model_tracking_time": model_tracking_time,
            "dataset_tracking_time": dataset_tracking_time,
            "dataset_size": len(large_dataset),
            "model_parameters": model_info.get("num_parameters", 0),
            "memory_usage_mb": self.measure_memory_usage()
        }
    
    def test_timeout_handling_with_rl(self):
        """Test timeout handling with RL workloads."""
        print("Testing timeout handling with RL workloads...")
        
        # Create very large model that might timeout
        huge_agent = SimplePPOAgent(64, 16, 2048)  # Very large model
        
        tracker = ModelTracker()
        
        # Test with timeout
        start_time = time.time()
        try:
            model_info = tracker.track_model(huge_agent, "huge_rl_model")
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
        
        tracking_time = time.time() - start_time
        
        return {
            "success": success,
            "tracking_time": tracking_time,
            "error": error,
            "model_parameters": sum(p.numel() for p in huge_agent.policy_net.parameters()) if hasattr(huge_agent, 'policy_net') else 0
        }
    
    def run_all_tests(self):
        """Run all RL performance tests."""
        print("Running RLDK RL Performance Tests")
        print("=" * 50)
        
        all_results = {}
        
        try:
            # Test settings with RL context
            all_results["settings_rl"] = self.test_settings_performance_with_rl()
            print()
            
            # Test dataset tracking with RL data
            all_results["dataset_rl"] = self.test_dataset_tracking_with_rl_data()
            print()
            
            # Test model tracking with RL models
            all_results["model_rl"] = self.test_model_tracking_with_rl_models()
            print()
            
            # Test PPO training loop
            all_results["ppo_training"] = self.test_training_loop_with_tracking("PPO", episodes=50)
            print()
            
            # Test DQN training loop
            all_results["dqn_training"] = self.test_training_loop_with_tracking("DQN", episodes=50)
            print()
            
            # Test large scale training
            all_results["large_scale"] = self.test_large_scale_rl_training()
            print()
            
            # Test timeout handling
            all_results["timeout_handling"] = self.test_timeout_handling_with_rl()
            print()
            
        except Exception as e:
            print(f"Error during testing: {e}")
            all_results["error"] = str(e)
        
        finally:
            self.cleanup()
        
        return all_results
    
    def print_results(self, results):
        """Print RL performance test results."""
        print("\n" + "=" * 50)
        print("RL PERFORMANCE TEST RESULTS")
        print("=" * 50)
        
        # Settings performance
        if "settings_rl" in results:
            settings = results["settings_rl"]
            print(f"\nSettings Performance:")
            print(f"  Initialization time: {settings['init_time']:.3f}s")
            print(f"  Memory usage: {settings['memory_usage_mb']:.1f} MB")
        
        # Dataset tracking performance
        if "dataset_rl" in results:
            dataset = results["dataset_rl"]
            print(f"\nDataset Tracking Performance:")
            for name, data in dataset.items():
                print(f"  {name}: {data['time']:.3f}s, size: {data['size_bytes']:,} bytes")
        
        # Model tracking performance
        if "model_rl" in results:
            model = results["model_rl"]
            print(f"\nModel Tracking Performance:")
            for name, data in model.items():
                print(f"  {name}: {data['time']:.3f}s, params: {data['num_parameters']:,}")
        
        # Training loop performance
        if "ppo_training" in results:
            ppo = results["ppo_training"]
            print(f"\nPPO Training Performance:")
            print(f"  Initialization: {ppo['init_time']:.3f}s")
            print(f"  Start experiment: {ppo['start_time']:.3f}s")
            print(f"  Model tracking: {ppo['model_tracking_time']:.3f}s")
            print(f"  Training time: {ppo['training_time']:.3f}s")
            print(f"  Final model tracking: {ppo['final_model_tracking_time']:.3f}s")
            print(f"  Final data tracking: {ppo['final_data_tracking_time']:.3f}s")
            print(f"  Episodes: {ppo['episodes']}")
            print(f"  Average reward: {ppo['avg_reward']:.2f}")
            print(f"  Total data points: {ppo['total_data_points']:,}")
            print(f"  Memory usage: {ppo['memory_usage_mb']:.1f} MB")
        
        if "dqn_training" in results:
            dqn = results["dqn_training"]
            print(f"\nDQN Training Performance:")
            print(f"  Initialization: {dqn['init_time']:.3f}s")
            print(f"  Start experiment: {dqn['start_time']:.3f}s")
            print(f"  Model tracking: {dqn['model_tracking_time']:.3f}s")
            print(f"  Training time: {dqn['training_time']:.3f}s")
            print(f"  Episodes: {dqn['episodes']}")
            print(f"  Average reward: {dqn['avg_reward']:.2f}")
            print(f"  Total data points: {dqn['total_data_points']:,}")
            print(f"  Memory usage: {dqn['memory_usage_mb']:.1f} MB")
        
        # Large scale performance
        if "large_scale" in results:
            large = results["large_scale"]
            print(f"\nLarge Scale Performance:")
            print(f"  Initialization: {large['init_time']:.3f}s")
            print(f"  Start experiment: {large['start_time']:.3f}s")
            print(f"  Model tracking: {large['model_tracking_time']:.3f}s")
            print(f"  Dataset tracking: {large['dataset_tracking_time']:.3f}s")
            print(f"  Dataset size: {large['dataset_size']:,}")
            print(f"  Model parameters: {large['model_parameters']:,}")
            print(f"  Memory usage: {large['memory_usage_mb']:.1f} MB")
        
        # Timeout handling
        if "timeout_handling" in results:
            timeout = results["timeout_handling"]
            print(f"\nTimeout Handling:")
            print(f"  Success: {timeout['success']}")
            print(f"  Tracking time: {timeout['tracking_time']:.3f}s")
            if timeout['error']:
                print(f"  Error: {timeout['error']}")
        
        print("\n" + "=" * 50)
        print("RL Performance tests completed!")
        print("=" * 50)


def main():
    """Main function to run RL performance tests."""
    tester = RLPerformanceTester()
    
    try:
        results = tester.run_all_tests()
        tester.print_results(results)
        
        # Save results to file
        with open("rl_tracking_performance_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to rl_tracking_performance_results.json")
        
    except Exception as e:
        print(f"RL Performance testing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())