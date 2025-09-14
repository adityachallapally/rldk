#!/usr/bin/env python3
"""
Demonstration of error conditions that would be hidden by silent error handling.

This script creates realistic error scenarios that commonly occur during PPO training
and shows how silent error handling hides important debugging information.
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add the src directory to the path so we can import RLDK modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

class ErrorConditionSimulator:
    """Simulates common error conditions in PPO training."""

    def __init__(self, error_probability: float = 0.3):
        self.error_probability = error_probability
        self.step = 0
        self.error_count = 0

    def generate_metrics_with_errors(self) -> Dict[str, Any]:
        """Generate metrics with potential error conditions."""
        self.step += 1

        # Base metrics
        metrics = {
            'ppo/rewards/mean': 0.5 + self.step * 0.01,
            'ppo/rewards/std': 0.2,
            'ppo/policy/kl_mean': 0.05,
            'ppo/policy/entropy': 2.0,
            'ppo/policy/clipfrac': 0.1,
            'ppo/val/value_loss': 0.3,
            'learning_rate': 1e-5,
            'grad_norm': 0.5,
        }

        # Introduce error conditions based on step and probability
        if np.random.random() < self.error_probability:
            self.error_count += 1
            error_type = self._select_error_type()
            metrics = self._apply_error(metrics, error_type)

        return metrics

    def _select_error_type(self) -> str:
        """Select a random error type."""
        error_types = [
            "negative_kl",
            "nan_reward",
            "inf_value_loss",
            "missing_entropy",
            "invalid_clipfrac",
            "zero_grad_norm"
        ]
        return np.random.choice(error_types)

    def _apply_error(self, metrics: Dict[str, Any], error_type: str) -> Dict[str, Any]:
        """Apply a specific error condition to metrics."""
        if error_type == "negative_kl":
            metrics['ppo/policy/kl_mean'] = -0.1  # Invalid negative KL
        elif error_type == "nan_reward":
            metrics['ppo/rewards/mean'] = float('nan')  # NaN reward
        elif error_type == "inf_value_loss":
            metrics['ppo/val/value_loss'] = float('inf')  # Infinite value loss
        elif error_type == "missing_entropy":
            del metrics['ppo/policy/entropy']  # Missing metric
        elif error_type == "invalid_clipfrac":
            metrics['ppo/policy/clipfrac'] = 1.5  # Invalid clip fraction > 1
        elif error_type == "zero_grad_norm":
            metrics['grad_norm'] = 0.0  # Zero gradient norm (suspicious)

        return metrics

class SilentErrorHandler:
    """Handles errors silently (current TRL behavior)."""

    def __init__(self, name: str = "SilentHandler"):
        self.name = name
        self.processed_metrics = []
        self.silent_errors = []
        self.error_types = {}

    def process_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Process metrics with silent error handling."""
        try:
            result = self._extract_metrics_silently(metrics)
            if result:
                self.processed_metrics.append(result)
                return True
            return False
        except Exception as e:
            # Silent error handling - this is the problem!
            self.silent_errors.append(str(e))
            return False

    def _extract_metrics_silently(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract metrics with silent error handling."""
        try:
            result = {}

            # Extract reward metrics
            if 'ppo/rewards/mean' in metrics:
                reward = metrics['ppo/rewards/mean']
                if np.isnan(reward):
                    raise ValueError(f"NaN reward: {reward}")
                if np.isinf(reward):
                    raise ValueError(f"Infinite reward: {reward}")
                result['reward_mean'] = reward

            # Extract KL divergence
            if 'ppo/policy/kl_mean' in metrics:
                kl = metrics['ppo/policy/kl_mean']
                if kl < 0:
                    raise ValueError(f"Negative KL divergence: {kl}")
                if np.isnan(kl):
                    raise ValueError(f"NaN KL divergence: {kl}")
                result['kl_mean'] = kl

            # Extract entropy
            if 'ppo/policy/entropy' not in metrics:
                raise KeyError("Missing entropy metric")
            entropy = metrics['ppo/policy/entropy']
            if np.isnan(entropy):
                raise ValueError(f"NaN entropy: {entropy}")
            result['entropy'] = entropy

            # Extract value loss
            if 'ppo/val/value_loss' in metrics:
                value_loss = metrics['ppo/val/value_loss']
                if np.isinf(value_loss):
                    raise ValueError(f"Infinite value loss: {value_loss}")
                if np.isnan(value_loss):
                    raise ValueError(f"NaN value loss: {value_loss}")
                result['value_loss'] = value_loss

            # Extract clip fraction
            if 'ppo/policy/clipfrac' in metrics:
                clipfrac = metrics['ppo/policy/clipfrac']
                if clipfrac > 1.0:
                    raise ValueError(f"Invalid clip fraction: {clipfrac}")
                if np.isnan(clipfrac):
                    raise ValueError(f"NaN clip fraction: {clipfrac}")
                result['clipfrac'] = clipfrac

            # Extract gradient norm
            if 'grad_norm' in metrics:
                grad_norm = metrics['grad_norm']
                if grad_norm == 0.0:
                    raise ValueError(f"Zero gradient norm: {grad_norm}")
                if np.isnan(grad_norm):
                    raise ValueError(f"NaN gradient norm: {grad_norm}")
                result['grad_norm'] = grad_norm

            return result

        except Exception:
            # Silent error handling - this hides the error!
            return None

class VerboseErrorHandler:
    """Handles errors with proper logging (improved behavior)."""

    def __init__(self, name: str = "VerboseHandler"):
        self.name = name
        self.processed_metrics = []
        self.logged_errors = []
        self.error_types = {}

    def process_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Process metrics with verbose error handling."""
        try:
            result = self._extract_metrics_verbosely(metrics)
            if result:
                self.processed_metrics.append(result)
                return True
            return False
        except Exception as e:
            # Proper error logging - this helps with debugging!
            error_msg = f"Metric processing failed: {e}"
            self.logged_errors.append(error_msg)
            self._categorize_error(str(e))
            warnings.warn(f"{self.name}: {error_msg}", stacklevel=2)
            print(f"⚠️  {self.name}: {error_msg}")
            return False

    def _extract_metrics_verbosely(self, metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract metrics with verbose error handling."""
        try:
            result = {}

            # Extract reward metrics
            if 'ppo/rewards/mean' in metrics:
                reward = metrics['ppo/rewards/mean']
                if np.isnan(reward):
                    raise ValueError(f"NaN reward detected: {reward} (check reward function and data)")
                if np.isinf(reward):
                    raise ValueError(f"Infinite reward detected: {reward} (check reward scaling)")
                result['reward_mean'] = reward

            # Extract KL divergence
            if 'ppo/policy/kl_mean' in metrics:
                kl = metrics['ppo/policy/kl_mean']
                if kl < 0:
                    raise ValueError(f"Negative KL divergence: {kl} (training may be unstable, check learning rate)")
                if np.isnan(kl):
                    raise ValueError(f"NaN KL divergence: {kl} (check policy network)")
                result['kl_mean'] = kl

            # Extract entropy
            if 'ppo/policy/entropy' not in metrics:
                raise KeyError("Missing entropy metric (check PPO configuration)")
            entropy = metrics['ppo/policy/entropy']
            if np.isnan(entropy):
                raise ValueError(f"NaN entropy: {entropy} (check policy network)")
            result['entropy'] = entropy

            # Extract value loss
            if 'ppo/val/value_loss' in metrics:
                value_loss = metrics['ppo/val/value_loss']
                if np.isinf(value_loss):
                    raise ValueError(f"Infinite value loss: {value_loss} (check value function learning rate)")
                if np.isnan(value_loss):
                    raise ValueError(f"NaN value loss: {value_loss} (check value network)")
                result['value_loss'] = value_loss

            # Extract clip fraction
            if 'ppo/policy/clipfrac' in metrics:
                clipfrac = metrics['ppo/policy/clipfrac']
                if clipfrac > 1.0:
                    raise ValueError(f"Invalid clip fraction: {clipfrac} (should be <= 1.0)")
                if np.isnan(clipfrac):
                    raise ValueError(f"NaN clip fraction: {clipfrac} (check PPO computation)")
                result['clipfrac'] = clipfrac

            # Extract gradient norm
            if 'grad_norm' in metrics:
                grad_norm = metrics['grad_norm']
                if grad_norm == 0.0:
                    raise ValueError(f"Zero gradient norm: {grad_norm} (check if gradients are being computed)")
                if np.isnan(grad_norm):
                    raise ValueError(f"NaN gradient norm: {grad_norm} (check gradient computation)")
                result['grad_norm'] = grad_norm

            return result

        except Exception as e:
            # Proper error logging with context
            raise Exception(f"Failed to extract metrics: {e}") from e

    def _categorize_error(self, error_msg: str):
        """Categorize error types for analysis."""
        if "NaN" in error_msg:
            self.error_types["NaN"] = self.error_types.get("NaN", 0) + 1
        elif "Infinite" in error_msg or "inf" in error_msg:
            self.error_types["Infinite"] = self.error_types.get("Infinite", 0) + 1
        elif "Negative" in error_msg:
            self.error_types["Negative"] = self.error_types.get("Negative", 0) + 1
        elif "Missing" in error_msg:
            self.error_types["Missing"] = self.error_types.get("Missing", 0) + 1
        elif "Invalid" in error_msg:
            self.error_types["Invalid"] = self.error_types.get("Invalid", 0) + 1
        elif "Zero" in error_msg:
            self.error_types["Zero"] = self.error_types.get("Zero", 0) + 1

def demonstrate_error_conditions():
    """Demonstrate various error conditions and their handling."""

    print("🔍 Error Conditions Demonstration")
    print("=" * 80)

    # Create error simulator with high error probability
    simulator = ErrorConditionSimulator(error_probability=0.4)

    # Create handlers
    silent_handler = SilentErrorHandler("SilentHandler")
    verbose_handler = VerboseErrorHandler("VerboseHandler")

    print("Running 30 training steps with error conditions...")
    print("(40% probability of error per step)")
    print()

    for step in range(30):
        # Generate metrics with potential errors
        metrics = simulator.generate_metrics_with_errors()

        # Process with both handlers
        silent_success = silent_handler.process_metrics(metrics)
        verbose_success = verbose_handler.process_metrics(metrics)

        # Show step results
        status = "✅" if silent_success else "❌"
        print(f"Step {step + 1:2d}: {status} Silent: {silent_success}, Verbose: {verbose_success}")

    print("\n📊 Results Summary:")
    print(f"  - Total error conditions: {simulator.error_count}")
    print(f"  - Silent handler processed: {len(silent_handler.processed_metrics)}")
    print(f"  - Silent handler errors: {len(silent_handler.silent_errors)}")
    print(f"  - Verbose handler processed: {len(verbose_handler.processed_metrics)}")
    print(f"  - Verbose handler errors: {len(verbose_handler.logged_errors)}")

    print("\n🔍 Error Type Analysis (Verbose Handler):")
    for error_type, count in verbose_handler.error_types.items():
        print(f"  - {error_type}: {count} occurrences")

    return silent_handler, verbose_handler

def analyze_debugging_value():
    """Analyze the debugging value of verbose error handling."""

    print("\n🔧 Debugging Value Analysis")
    print("-" * 60)

    # Simulate different training scenarios
    scenarios = [
        {
            "name": "Learning Rate Too High",
            "error_type": "negative_kl",
            "description": "High learning rate causes negative KL divergence",
            "silent_impact": "Training continues with invalid metrics, model may diverge",
            "verbose_benefit": "User sees 'Negative KL divergence' warning, can reduce learning rate"
        },
        {
            "name": "Reward Function Issues",
            "error_type": "nan_reward",
            "description": "Reward function returns NaN values",
            "silent_impact": "Gradient updates become invalid, model stops learning",
            "verbose_benefit": "User sees 'NaN reward' warning, can check reward function"
        },
        {
            "name": "Value Function Instability",
            "error_type": "inf_value_loss",
            "description": "Value function loss becomes infinite",
            "silent_impact": "Value function training fails, PPO becomes unstable",
            "verbose_benefit": "User sees 'Infinite value loss' warning, can adjust value learning rate"
        },
        {
            "name": "Configuration Error",
            "error_type": "missing_entropy",
            "description": "PPO configuration missing entropy metric",
            "silent_impact": "Incomplete monitoring, may miss important signals",
            "verbose_benefit": "User sees 'Missing entropy metric' warning, can fix configuration"
        },
        {
            "name": "Gradient Issues",
            "error_type": "zero_grad_norm",
            "description": "Gradient norm becomes zero",
            "silent_impact": "Model may not be learning, training appears stuck",
            "verbose_benefit": "User sees 'Zero gradient norm' warning, can check gradient computation"
        }
    ]

    for scenario in scenarios:
        print(f"\n🎯 Scenario: {scenario['name']}")
        print(f"   Error Type: {scenario['error_type']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Silent Impact: {scenario['silent_impact']}")
        print(f"   Verbose Benefit: {scenario['verbose_benefit']}")

def demonstrate_training_impact():
    """Demonstrate the impact on training success."""

    print("\n📈 Training Impact Analysis")
    print("-" * 60)

    # Simulate training with different error rates
    error_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    training_steps = 100

    print("Simulating training with different error rates...")
    print(f"Training steps: {training_steps}")
    print()

    for error_rate in error_rates:
        simulator = ErrorConditionSimulator(error_probability=error_rate)
        silent_handler = SilentErrorHandler()
        verbose_handler = VerboseErrorHandler()

        # Run training simulation
        for step in range(training_steps):
            metrics = simulator.generate_metrics_with_errors()
            silent_handler.process_metrics(metrics)
            verbose_handler.process_metrics(metrics)

        # Calculate success rates
        silent_success_rate = len(silent_handler.processed_metrics) / training_steps
        verbose_success_rate = len(verbose_handler.processed_metrics) / training_steps

        print(f"Error Rate: {error_rate*100:3.0f}% | "
              f"Silent Success: {silent_success_rate:5.1%} | "
              f"Verbose Success: {verbose_success_rate:5.1%} | "
              f"Hidden Errors: {len(silent_handler.silent_errors):2d} | "
              f"Logged Errors: {len(verbose_handler.logged_errors):2d}")

def generate_implementation_guide():
    """Generate a practical implementation guide."""

    print("\n🛠️ Implementation Guide")
    print("-" * 60)

    print("To fix silent error handling in TRL metric extractors:")
    print()

    print("1️⃣ Identify Silent Error Blocks:")
    print("   Search for 'except Exception: pass' in TRL codebase")
    print("   Common locations:")
    print("   - trl/callbacks.py (metric extraction)")
    print("   - trl/trainer/ppo_trainer.py (PPO metrics)")
    print("   - trl/trainer/ppo_trainer.py (logging)")
    print()

    print("2️⃣ Replace with Proper Error Handling:")
    print("   Current:")
    print("     except Exception: pass")
    print("   ")
    print("   Improved:")
    print("     except Exception as e:")
    print("         logger.warning(f'Metric extraction failed: {e}')")
    print("         logger.debug(f'Failed to extract metrics from: {logs}')")
    print()

    print("3️⃣ Add Context to Error Messages:")
    print("   - Include step number and phase")
    print("   - Provide actionable suggestions")
    print("   - Log relevant state information")
    print()

    print("4️⃣ Test Error Handling:")
    print("   - Create test cases with error conditions")
    print("   - Verify warnings are logged correctly")
    print("   - Ensure training continues despite errors")
    print()

    print("5️⃣ Update Documentation:")
    print("   - Document common error conditions")
    print("   - Provide troubleshooting guide")
    print("   - Include examples of error messages")

def main():
    """Main demonstration function."""

    print("🚀 TRL Silent Error Handling - Error Conditions Demo")
    print("=" * 80)
    print("This script demonstrates how silent error handling in TRL metric")
    print("extractors hides important debugging information during PPO training.")
    print()

    # Set random seed for reproducible results
    np.random.seed(42)

    # Run demonstrations
    silent_handler, verbose_handler = demonstrate_error_conditions()

    analyze_debugging_value()

    demonstrate_training_impact()

    generate_implementation_guide()

    # Final summary
    print("\n🎯 Summary and Value Assessment")
    print("-" * 60)

    print("✅ Demonstrated Value:")
    print("  1. Error Visibility: Silent handler hid errors, verbose handler logged them")
    print("  2. Debugging Support: Verbose errors provide actionable information")
    print("  3. Training Monitoring: Complete visibility into metric extraction")
    print("  4. Issue Resolution: Clear error messages guide users to solutions")
    print("  5. Training Stability: Early detection of configuration problems")

    print("\n📊 Key Results:")
    print(f"  - Silent handler: {len(silent_handler.processed_metrics)} successful, {len(silent_handler.silent_errors)} hidden errors")
    print(f"  - Verbose handler: {len(verbose_handler.processed_metrics)} successful, {len(verbose_handler.logged_errors)} logged errors")
    print(f"  - Error types detected: {list(verbose_handler.error_types.keys())}")

    print("\n💡 Conclusion:")
    print("Fixing silent error handling in TRL metric extractors is valuable because:")
    print("- It provides visibility into training issues that would otherwise be hidden")
    print("- It helps users debug PPO training problems more effectively")
    print("- It improves the overall reliability and usability of the TRL library")
    print("- It follows best practices for error handling in ML libraries")
    print("- It enables better monitoring and debugging of RL training runs")

    print("\n🎯 Recommendation:")
    print("Implement proper error logging in TRL metric extractors to replace")
    print("silent 'except Exception: pass' blocks with informative warnings")
    print("that help users understand and resolve training issues.")

if __name__ == "__main__":
    main()
