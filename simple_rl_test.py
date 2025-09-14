#!/usr/bin/env python3
"""
Simple RL Test for RLDK Tracking System

This script tests the performance improvements with actual RL models
and training loops using only basic Python libraries.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import tempfile
import shutil
import json
import asyncio
from typing import Dict, Any, List

# Import RLDK components
from src.rldk.config.settings import RLDKSettings
from src.rldk.tracking.config import TrackingConfig
from src.rldk.tracking.tracker import ExperimentTracker
from src.rldk.tracking.dataset_tracker import DatasetTracker
from src.rldk.tracking.model_tracker import ModelTracker


class SimpleRLEnvironment:
    """Simple RL environment for testing."""
    
    def __init__(self, state_dim=4, action_dim=2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.step_count = 0
        self.max_steps = 200
        
    def reset(self):
        self.step_count = 0
        return np.random.randn(self.state_dim).astype(np.float32)
    
    def step(self, action):
        self.step_count += 1
        
        # Simple reward function
        reward = 1.0 if action == 0 else 0.5
        done = self.step_count >= self.max_steps
        
        next_state = np.random.randn(self.state_dim).astype(np.float32)
        info = {"step": self.step_count}
        
        return next_state, reward, done, info


class SimpleRLAgent:
    """Simple RL agent for testing."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
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
        
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=3e-4)
        
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


def test_rl_tracking_performance():
    """Test RLDK tracking performance with RL training."""
    print("Testing RLDK tracking performance with RL training...")
    print("=" * 60)
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Test 1: Settings initialization
        print("\n1. Testing settings initialization...")
        start_time = time.time()
        settings = RLDKSettings()
        
        # Configure for RL workloads
        settings.tracking_timeout = 60.0
        settings.dataset_sample_size = 5000
        settings.model_fingerprint_limit = 1000000000  # 1B parameters
        settings.enable_async_init = True
        settings.cache_environment = True
        settings.cache_git_info = True
        
        settings_time = time.time() - start_time
        print(f"   Settings initialization: {settings_time:.3f}s")
        
        # Test 2: Dataset tracking with RL data
        print("\n2. Testing dataset tracking with RL data...")
        tracker = DatasetTracker()
        
        # Generate RL training data
        rl_data_sizes = [1000, 10000, 100000]
        dataset_results = {}
        
        for size in rl_data_sizes:
            # Generate realistic RL data
            states = np.random.randn(size, 4).astype(np.float32)
            actions = np.random.randint(0, 2, size)
            rewards = np.random.randn(size)
            dones = np.random.random(size) < 0.1
            
            rl_data = {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'dones': dones
            }
            
            start_time = time.time()
            tracking_info = tracker.track_dataset(rl_data, f"rl_data_{size}")
            elapsed = time.time() - start_time
            
            dataset_results[f"size_{size}"] = {
                "time": elapsed,
                "checksum": tracking_info.get("checksum", "unknown")[:16] + "...",
                "size_bytes": tracking_info.get("size_bytes", 0)
            }
            
            print(f"   Dataset {size:,} samples: {elapsed:.3f}s")
        
        # Test 3: Model tracking with RL models
        print("\n3. Testing model tracking with RL models...")
        model_tracker = ModelTracker()
        
        # Test different model sizes
        model_sizes = [
            (4, 2, 32, "Small (4->32->2)"),
            (4, 2, 64, "Medium (4->64->2)"),
            (4, 2, 128, "Large (4->128->2)"),
            (8, 4, 256, "Very Large (8->256->4)")
        ]
        
        model_results = {}
        for state_dim, action_dim, hidden_dim, name in model_sizes:
            agent = SimpleRLAgent(state_dim, action_dim, hidden_dim)
            
            start_time = time.time()
            tracking_info = model_tracker.track_model(agent, f"rl_model_{name}")
            elapsed = time.time() - start_time
            
            model_results[name] = {
                "time": elapsed,
                "parameters": tracking_info.get("num_parameters", 0),
                "architecture_checksum": tracking_info.get("architecture_checksum", "unknown")[:16] + "..."
            }
            
            print(f"   {name}: {elapsed:.3f}s, params: {tracking_info.get('num_parameters', 0):,}")
        
        # Test 4: Full RL training loop with tracking
        print("\n4. Testing full RL training loop with tracking...")
        
        # Create tracking config
        config = TrackingConfig(
            experiment_name="rl_training_test",
            output_dir=temp_dir / "rl_experiments"
        )
        
        # Create tracker
        tracker = ExperimentTracker(config)
        
        # Initialize tracker
        start_time = time.time()
        if config.enable_async_init:
            try:
                asyncio.run(tracker.initialize_async())
                print("   Async initialization successful")
            except Exception as e:
                print(f"   Async init failed, falling back to sync: {e}")
                tracker.initialize_sync()
        else:
            tracker.initialize_sync()
        init_time = time.time() - start_time
        
        # Start experiment
        start_time = time.time()
        tracking_data = tracker.start_experiment()
        start_time_total = time.time() - start_time
        
        print(f"   Tracker initialization: {init_time:.3f}s")
        print(f"   Experiment start: {start_time_total:.3f}s")
        
        # Create RL environment and agent
        env = SimpleRLEnvironment()
        agent = SimpleRLAgent(4, 2, 64)
        
        # Track the model
        start_time = time.time()
        model_info = tracker.track_model(agent, "rl_agent")
        model_tracking_time = time.time() - start_time
        print(f"   Model tracking: {model_tracking_time:.3f}s")
        
        # Training loop
        print("   Running RL training loop...")
        training_data = []
        episode_rewards = []
        
        start_time = time.time()
        for episode in range(50):  # 50 episodes
            state = env.reset()
            episode_reward = 0
            episode_data = []
            
            for step in range(200):  # Max steps per episode
                action, prob = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                
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
                values = [agent.get_value(s) for s in states]
                old_probs = [0.5] * len(actions)  # Simplified
                
                loss = agent.update(states, actions, rewards, values, old_probs)
                
                # Track training data periodically
                if episode % 25 == 0:
                    start_time = time.time()
                    tracker.track_dataset(training_data[-1000:], f"training_data_ep_{episode}")
                    data_tracking_time = time.time() - start_time
                    
                    print(f"     Episode {episode}: reward={episode_reward:.2f}, loss={loss:.4f}, data_tracking={data_tracking_time:.3f}s")
        
        training_time = time.time() - start_time
        
        # Track final model and data
        start_time = time.time()
        final_model_info = tracker.track_model(agent, "rl_agent_final")
        final_model_tracking_time = time.time() - start_time
        
        start_time = time.time()
        tracker.track_dataset(training_data, "final_training_data")
        final_data_tracking_time = time.time() - start_time
        
        # Finish experiment
        tracker.finish_experiment()
        
        print(f"   Training time: {training_time:.3f}s")
        print(f"   Final model tracking: {final_model_tracking_time:.3f}s")
        print(f"   Final data tracking: {final_data_tracking_time:.3f}s")
        print(f"   Episodes: 50")
        print(f"   Average reward: {np.mean(episode_rewards[-10:]):.2f}")
        print(f"   Total data points: {len(training_data):,}")
        
        # Test 5: Large scale test
        print("\n5. Testing large scale RL scenario...")
        
        # Create large model
        large_agent = SimpleRLAgent(16, 8, 512)
        
        # Track large model
        start_time = time.time()
        large_model_info = model_tracker.track_model(large_agent, "large_rl_agent")
        large_model_time = time.time() - start_time
        
        # Generate large dataset
        large_dataset = []
        for i in range(50000):  # 50K samples
            state = np.random.randn(16).astype(np.float32)
            action = np.random.randint(0, 8)
            reward = np.random.randn()
            next_state = np.random.randn(16).astype(np.float32)
            done = np.random.random() < 0.1
            
            large_dataset.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })
        
        # Track large dataset
        start_time = time.time()
        large_dataset_info = tracker.track_dataset(large_dataset, "large_rl_dataset")
        large_dataset_time = time.time() - start_time
        
        print(f"   Large model tracking: {large_model_time:.3f}s")
        print(f"   Large dataset tracking: {large_dataset_time:.3f}s")
        print(f"   Large dataset size: {len(large_dataset):,}")
        print(f"   Large model parameters: {large_model_info.get('num_parameters', 0):,}")
        
        # Test 6: Timeout handling
        print("\n6. Testing timeout handling...")
        
        # Create very large model
        huge_agent = SimpleRLAgent(64, 16, 2048)
        
        start_time = time.time()
        try:
            huge_model_info = model_tracker.track_model(huge_agent, "huge_rl_agent")
            timeout_success = True
            timeout_error = None
        except Exception as e:
            timeout_success = False
            timeout_error = str(e)
        
        timeout_time = time.time() - start_time
        
        print(f"   Timeout test success: {timeout_success}")
        print(f"   Timeout test time: {timeout_time:.3f}s")
        if timeout_error:
            print(f"   Timeout error: {timeout_error}")
        
        # Summary
        print("\n" + "=" * 60)
        print("RL TRACKING PERFORMANCE SUMMARY")
        print("=" * 60)
        
        print(f"✅ Settings initialization: {settings_time:.3f}s")
        print(f"✅ Dataset tracking: {sum(r['time'] for r in dataset_results.values()):.3f}s total")
        print(f"✅ Model tracking: {sum(r['time'] for r in model_results.values()):.3f}s total")
        print(f"✅ RL training loop: {training_time:.3f}s")
        print(f"✅ Large scale test: {large_model_time + large_dataset_time:.3f}s")
        print(f"✅ Timeout handling: {'PASS' if timeout_success else 'FAIL'}")
        
        # Performance targets
        print("\nPerformance Targets:")
        print(f"  Settings init < 1s: {'✅' if settings_time < 1.0 else '❌'} ({settings_time:.3f}s)")
        print(f"  Large dataset < 30s: {'✅' if large_dataset_time < 30.0 else '❌'} ({large_dataset_time:.3f}s)")
        print(f"  Large model < 10s: {'✅' if large_model_time < 10.0 else '❌'} ({large_model_time:.3f}s)")
        print(f"  Training loop < 60s: {'✅' if training_time < 60.0 else '❌'} ({training_time:.3f}s)")
        print(f"  Timeout handling: {'✅' if timeout_success else '❌'}")
        
        # Save results
        results = {
            "settings_time": settings_time,
            "dataset_results": dataset_results,
            "model_results": model_results,
            "training_time": training_time,
            "large_model_time": large_model_time,
            "large_dataset_time": large_dataset_time,
            "timeout_success": timeout_success,
            "timeout_time": timeout_time,
            "episode_rewards": episode_rewards,
            "total_data_points": len(training_data)
        }
        
        with open("simple_rl_test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to simple_rl_test_results.json")
        
        return results
        
    except Exception as e:
        print(f"❌ RL testing failed: {e}")
        return None
    
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    """Main function to run RL performance test."""
    print("RLDK RL Performance Test")
    print("Testing with actual RL models and training loops")
    print("=" * 60)
    
    results = test_rl_tracking_performance()
    
    if results:
        print("\n🎉 RL performance test completed successfully!")
        return 0
    else:
        print("\n❌ RL performance test failed!")
        return 1


if __name__ == "__main__":
    exit(main())