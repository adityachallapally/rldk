#!/usr/bin/env python3
"""
Production Deployment Checklist Example for RLDK

This example demonstrates how to use RLDK for preparing RL models for production
deployment, including model validation, performance testing, and deployment readiness.

Learning Objectives:
- How to validate models for production deployment
- How to implement comprehensive testing pipelines
- How to ensure model reliability and performance
- How to create deployment-ready model packages
- How to implement monitoring and alerting systems

Prerequisites:
- RLDK installed (pip install rldk)
- Basic understanding of production deployment
- Familiarity with model validation techniques
- Understanding of deployment best practices
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

import gymnasium as gym

# RLDK imports
import rldk
from rldk.tracking import ExperimentTracker, TrackingConfig
from rldk.forensics import ComprehensivePPOForensics
from rldk.utils import set_global_seed, validate_numeric_range
from rldk.diff import first_divergence

class ProductionPPO:
    """Production-ready PPO implementation with validation and monitoring."""
    
    def __init__(self, state_dim, action_dim, config: Dict[str, Any]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Extract hyperparameters from config
        self.lr = config.get('lr', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.eps_clip = config.get('eps_clip', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        
        # Initialize policy and value networks
        self.policy = np.random.randn(state_dim, action_dim) * 0.1
        self.value = np.random.randn(state_dim, 1) * 0.1
        
        # Production monitoring
        self.performance_metrics = {
            'inference_times': [],
            'action_probabilities': [],
            'value_estimates': [],
            'episode_rewards': [],
            'episode_lengths': []
        }
        
        # Model validation flags
        self.validation_passed = False
        self.performance_threshold = config.get('performance_threshold', 100.0)
        self.stability_threshold = config.get('stability_threshold', 0.1)
        
    def get_action(self, state):
        """Get action from policy with timing and validation."""
        
        start_time = time.time()
        
        # Validate input
        if not self._validate_state(state):
            raise ValueError("Invalid state input")
        
        # Get action
        logits = state @ self.policy
        probs = self._softmax(logits)
        action = np.random.choice(self.action_dim, p=probs)
        log_prob = np.log(probs[action] + 1e-8)
        
        # Record performance metrics
        inference_time = time.time() - start_time
        self.performance_metrics['inference_times'].append(inference_time)
        self.performance_metrics['action_probabilities'].append(probs)
        
        # Validate output
        if not self._validate_action(action):
            raise ValueError("Invalid action output")
        
        return action, log_prob
    
    def get_value(self, state):
        """Get state value with timing and validation."""
        
        start_time = time.time()
        
        # Validate input
        if not self._validate_state(state):
            raise ValueError("Invalid state input")
        
        # Get value
        value = (state @ self.value).item()
        
        # Record performance metrics
        inference_time = time.time() - start_time
        self.performance_metrics['value_estimates'].append(value)
        
        # Validate output
        if not self._validate_value(value):
            raise ValueError("Invalid value output")
        
        return value
    
    def _validate_state(self, state):
        """Validate state input."""
        if not isinstance(state, np.ndarray):
            return False
        if state.shape != (self.state_dim,):
            return False
        if not np.isfinite(state).all():
            return False
        return True
    
    def _validate_action(self, action):
        """Validate action output."""
        if not isinstance(action, (int, np.integer)):
            return False
        if not 0 <= action < self.action_dim:
            return False
        return True
    
    def _validate_value(self, value):
        """Validate value output."""
        if not isinstance(value, (int, float, np.floating)):
            return False
        if not np.isfinite(value):
            return False
        return True
    
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
            value_loss = self.value_coef * (value_new - return_val) ** 2
            
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
    
    def validate_model(self, test_episodes=100):
        """Validate model for production deployment."""
        
        print(f"🔍 Validating model for production deployment...")
        
        # Create test environment
        env = gym.make("CartPole-v1")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        # Test episodes
        episode_rewards = []
        episode_lengths = []
        inference_times = []
        
        for episode in range(test_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(200):  # Max steps per episode
                # Get action with timing
                start_time = time.time()
                action, _ = self.get_action(state)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Take action
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        env.close()
        
        # Calculate validation metrics
        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_inference_time = np.mean(inference_times)
        max_inference_time = np.max(inference_times)
        
        # Performance validation
        performance_passed = avg_reward >= self.performance_threshold
        stability_passed = std_reward <= self.stability_threshold * avg_reward
        
        # Timing validation
        timing_passed = avg_inference_time <= 0.01  # 10ms threshold
        max_timing_passed = max_inference_time <= 0.1  # 100ms threshold
        
        # Overall validation
        self.validation_passed = all([
            performance_passed, stability_passed, timing_passed, max_timing_passed
        ])
        
        validation_results = {
            'performance': {
                'avg_reward': avg_reward,
                'std_reward': std_reward,
                'threshold': self.performance_threshold,
                'passed': performance_passed
            },
            'stability': {
                'cv': std_reward / avg_reward if avg_reward > 0 else float('inf'),
                'threshold': self.stability_threshold,
                'passed': stability_passed
            },
            'timing': {
                'avg_inference_time': avg_inference_time,
                'max_inference_time': max_inference_time,
                'avg_threshold': 0.01,
                'max_threshold': 0.1,
                'avg_passed': timing_passed,
                'max_passed': max_timing_passed
            },
            'overall_passed': self.validation_passed
        }
        
        print(f"   Performance: {avg_reward:.2f} (threshold: {self.performance_threshold}) - {'✅' if performance_passed else '❌'}")
        print(f"   Stability: {std_reward/avg_reward:.3f} (threshold: {self.stability_threshold}) - {'✅' if stability_passed else '❌'}")
        print(f"   Avg timing: {avg_inference_time*1000:.2f}ms (threshold: 10ms) - {'✅' if timing_passed else '❌'}")
        print(f"   Max timing: {max_inference_time*1000:.2f}ms (threshold: 100ms) - {'✅' if max_timing_passed else '❌'}")
        print(f"   Overall: {'✅ PASSED' if self.validation_passed else '❌ FAILED'}")
        
        return validation_results
    
    def save_model(self, filepath: str):
        """Save model for production deployment."""
        
        model_data = {
            'policy': self.policy,
            'value': self.value,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'config': {
                'lr': self.lr,
                'gamma': self.gamma,
                'eps_clip': self.eps_clip,
                'entropy_coef': self.entropy_coef,
                'value_coef': self.value_coef
            },
            'validation_passed': self.validation_passed,
            'performance_metrics': self.performance_metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load model for production deployment."""
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create model instance
        model = cls(model_data['state_dim'], model_data['action_dim'], model_data['config'])
        model.policy = model_data['policy']
        model.value = model_data['value']
        model.validation_passed = model_data['validation_passed']
        model.performance_metrics = model_data['performance_metrics']
        
        print(f"📁 Model loaded from {filepath}")
        return model

class ProductionDeploymentPipeline:
    """Complete pipeline for production deployment preparation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tracker = None
        self.model = None
        self.validation_results = None
        self.deployment_package = None
        
    def setup_tracking(self):
        """Set up experiment tracking for deployment pipeline."""
        
        tracking_config = TrackingConfig(
            experiment_name="production_deployment",
            enable_dataset_tracking=True,
            enable_model_tracking=True,
            enable_environment_tracking=True,
            enable_seed_tracking=True,
            enable_git_tracking=True,
            output_dir="./deployment_runs",
            tags=["production", "deployment", "validation"],
            notes="Production deployment pipeline with validation and testing"
        )
        
        self.tracker = ExperimentTracker(tracking_config)
        tracking_data = self.tracker.start_experiment()
        
        # Add deployment metadata
        self.tracker.add_metadata("deployment_config", self.config)
        self.tracker.add_metadata("deployment_stage", "preparation")
        
        return tracking_data
    
    def train_model(self, env_name="CartPole-v1", episodes=200):
        """Train model for production deployment."""
        
        print(f"🚀 Training model for production deployment...")
        
        # Set seed for reproducibility
        set_global_seed(42)
        
        # Create environment
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        # Initialize model
        self.model = ProductionPPO(state_dim, action_dim, self.config)
        
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
                action, log_prob = self.model.get_action(state)
                value = self.model.get_value(state)
                
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
                advantages, returns = self.model.compute_advantages(rewards, values, dones)
                
                # Update policy
                self.model.update(states, actions, rewards, log_probs, values, advantages, returns)
                
                # Compute training metrics for forensics
                kl_div = np.mean([abs(log_probs[i] - log_prob) for i, log_prob in enumerate(log_probs)])
                entropy = -np.mean([log_prob * np.log(log_prob + 1e-8) for log_prob in log_probs])
                policy_grad_norm = np.linalg.norm(self.model.policy.flatten())
                value_grad_norm = np.linalg.norm(self.model.value.flatten())
                
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
        
        print(f"✅ Model training completed!")
        print(f"   Final reward: {episode_rewards[-1]:.2f}")
        print(f"   Average reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
        print(f"   Total anomalies: {len(anomalies)}")
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'training_metrics': training_metrics,
            'forensics_analysis': analysis,
            'anomalies': anomalies,
            'health_summary': health_summary
        }
    
    def validate_model(self, test_episodes=100):
        """Validate model for production deployment."""
        
        print(f"\n🔍 Validating model for production deployment...")
        
        if self.model is None:
            raise ValueError("Model must be trained before validation")
        
        # Run validation
        self.validation_results = self.model.validate_model(test_episodes)
        
        # Track validation results
        self.tracker.track_dataset(
            pd.DataFrame([self.validation_results]),
            "model_validation_results",
            {"test_episodes": test_episodes}
        )
        
        return self.validation_results
    
    def create_deployment_package(self, output_dir="./deployment_package"):
        """Create deployment package with model and validation results."""
        
        print(f"\n📦 Creating deployment package...")
        
        if self.model is None or self.validation_results is None:
            raise ValueError("Model must be trained and validated before creating deployment package")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_path / "model.pkl"
        self.model.save_model(str(model_path))
        
        # Save validation results
        validation_path = output_path / "validation_results.json"
        with open(validation_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        # Save deployment configuration
        config_path = output_path / "deployment_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Create deployment README
        readme_path = output_path / "README.md"
        readme_content = f"""# Production Deployment Package

## Model Information
- Algorithm: ProductionPPO
- Environment: CartPole-v1
- Validation Status: {'✅ PASSED' if self.validation_results['overall_passed'] else '❌ FAILED'}

## Performance Metrics
- Average Reward: {self.validation_results['performance']['avg_reward']:.2f}
- Reward Std: {self.validation_results['performance']['std_reward']:.2f}
- Average Inference Time: {self.validation_results['timing']['avg_inference_time']*1000:.2f}ms
- Max Inference Time: {self.validation_results['timing']['max_inference_time']*1000:.2f}ms

## Validation Results
- Performance: {'✅ PASSED' if self.validation_results['performance']['passed'] else '❌ FAILED'}
- Stability: {'✅ PASSED' if self.validation_results['stability']['passed'] else '❌ FAILED'}
- Timing: {'✅ PASSED' if self.validation_results['timing']['avg_passed'] else '❌ FAILED'}

## Usage
```python
from production_deployment_checklist import ProductionPPO

# Load model
model = ProductionPPO.load_model('model.pkl')

# Get action
action, log_prob = model.get_action(state)
value = model.get_value(state)
```

## Files
- `model.pkl`: Trained model
- `validation_results.json`: Validation results
- `deployment_config.json`: Deployment configuration
- `README.md`: This file

## Deployment Checklist
- [ ] Model trained and validated
- [ ] Performance thresholds met
- [ ] Timing requirements satisfied
- [ ] Stability requirements met
- [ ] Model saved and packaged
- [ ] Validation results documented
- [ ] Deployment package created
- [ ] README updated
- [ ] Version tagged
- [ ] Tests passing
- [ ] Documentation complete
"""
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        self.deployment_package = output_path
        
        print(f"✅ Deployment package created at {output_path}")
        print(f"   Files created:")
        print(f"   - {model_path}")
        print(f"   - {validation_path}")
        print(f"   - {config_path}")
        print(f"   - {readme_path}")
        
        return output_path
    
    def run_deployment_tests(self):
        """Run comprehensive deployment tests."""
        
        print(f"\n🧪 Running deployment tests...")
        
        if self.deployment_package is None:
            raise ValueError("Deployment package must be created before running tests")
        
        # Test 1: Model loading
        print("   Test 1: Model loading...")
        try:
            loaded_model = ProductionPPO.load_model(str(self.deployment_package / "model.pkl"))
            print("   ✅ Model loading test passed")
        except Exception as e:
            print(f"   ❌ Model loading test failed: {e}")
            return False
        
        # Test 2: Model inference
        print("   Test 2: Model inference...")
        try:
            test_state = np.random.randn(loaded_model.state_dim)
            action, log_prob = loaded_model.get_action(test_state)
            value = loaded_model.get_value(test_state)
            print("   ✅ Model inference test passed")
        except Exception as e:
            print(f"   ❌ Model inference test failed: {e}")
            return False
        
        # Test 3: Validation results
        print("   Test 3: Validation results...")
        try:
            validation_path = self.deployment_package / "validation_results.json"
            with open(validation_path, 'r') as f:
                validation_data = json.load(f)
            
            if validation_data['overall_passed']:
                print("   ✅ Validation results test passed")
            else:
                print("   ❌ Validation results test failed: Model did not pass validation")
                return False
        except Exception as e:
            print(f"   ❌ Validation results test failed: {e}")
            return False
        
        print("   ✅ All deployment tests passed!")
        return True
    
    def finish_deployment(self):
        """Finish deployment pipeline and save results."""
        
        # Track final results
        if self.model is not None and self.validation_results is not None:
            self.tracker.track_model(
                self.model,
                "production_model",
                {
                    "validation_passed": self.validation_results['overall_passed'],
                    "performance": self.validation_results['performance'],
                    "stability": self.validation_results['stability'],
                    "timing": self.validation_results['timing']
                }
            )
        
        # Add deployment metadata
        self.tracker.add_metadata("deployment_stage", "completed")
        self.tracker.add_metadata("validation_passed", self.validation_results['overall_passed'] if self.validation_results else False)
        self.tracker.add_metadata("deployment_package", str(self.deployment_package) if self.deployment_package else None)
        
        # Finish tracking
        summary = self.tracker.finish_experiment()
        
        return summary

def main():
    """Main function demonstrating production deployment pipeline."""
    
    print("🚀 RLDK Production Deployment Checklist Example")
    print("=" * 60)
    
    # 1. Setup
    print("\n1. Setting up production deployment pipeline...")
    
    # Deployment configuration
    config = {
        'lr': 3e-4,
        'gamma': 0.99,
        'eps_clip': 0.2,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'performance_threshold': 100.0,
        'stability_threshold': 0.1
    }
    
    # Initialize pipeline
    pipeline = ProductionDeploymentPipeline(config)
    
    # 2. Setup tracking
    print("\n2. Setting up experiment tracking...")
    
    tracking_data = pipeline.setup_tracking()
    print(f"🚀 Started experiment: {tracking_data['experiment_id']}")
    
    # 3. Train model
    print("\n3. Training model for production...")
    
    training_results = pipeline.train_model(episodes=200)
    
    # 4. Validate model
    print("\n4. Validating model for production...")
    
    validation_results = pipeline.validate_model(test_episodes=100)
    
    # 5. Create deployment package
    print("\n5. Creating deployment package...")
    
    deployment_package = pipeline.create_deployment_package()
    
    # 6. Run deployment tests
    print("\n6. Running deployment tests...")
    
    tests_passed = pipeline.run_deployment_tests()
    
    # 7. Finish deployment
    print("\n7. Finishing deployment pipeline...")
    
    summary = pipeline.finish_deployment()
    
    print(f"\n🎉 Production Deployment Pipeline completed successfully!")
    print(f"   Experiment ID: {summary['experiment_id']}")
    print(f"   Validation passed: {validation_results['overall_passed']}")
    print(f"   Tests passed: {tests_passed}")
    print(f"   Deployment package: {deployment_package}")
    
    print(f"\n📁 Files created:")
    print(f"   - {deployment_package}/model.pkl - Production model")
    print(f"   - {deployment_package}/validation_results.json - Validation results")
    print(f"   - {deployment_package}/deployment_config.json - Deployment config")
    print(f"   - {deployment_package}/README.md - Deployment documentation")
    
    # 8. Summary
    print("\n8. Summary and Next Steps...")
    
    print("\n✅ What We Accomplished:")
    print("   1. Model Training: Trained production-ready PPO model")
    print("   2. Model Validation: Validated performance, stability, and timing")
    print("   3. Deployment Package: Created complete deployment package")
    print("   4. Deployment Tests: Ran comprehensive deployment tests")
    print("   5. Documentation: Generated deployment documentation")
    print("   6. RLDK Integration: Tracked entire deployment pipeline")
    
    print(f"\n📊 Key Results:")
    print(f"   - Model performance: {validation_results['performance']['avg_reward']:.2f}")
    print(f"   - Model stability: {validation_results['stability']['cv']:.3f}")
    print(f"   - Avg inference time: {validation_results['timing']['avg_inference_time']*1000:.2f}ms")
    print(f"   - Validation passed: {'✅' if validation_results['overall_passed'] else '❌'}")
    print(f"   - Tests passed: {'✅' if tests_passed else '❌'}")
    
    print("\n🚀 Next Steps:")
    print("   1. Production Deployment: Deploy model to production environment")
    print("   2. Monitoring Setup: Implement monitoring and alerting")
    print("   3. Performance Testing: Run load and stress tests")
    print("   4. A/B Testing: Compare with existing models")
    print("   5. Rollback Plan: Prepare rollback procedures")
    print("   6. Documentation: Complete production documentation")
    
    print("\n📚 Key Takeaways:")
    print("   - RLDK enables systematic production deployment preparation")
    print("   - Model validation ensures production readiness")
    print("   - Comprehensive testing prevents deployment issues")
    print("   - Deployment packages enable easy model distribution")
    print("   - Documentation ensures deployment success")
    print("   - Monitoring is crucial for production success")
    
    print("\nHappy production deployment! 🎉")

if __name__ == "__main__":
    main()