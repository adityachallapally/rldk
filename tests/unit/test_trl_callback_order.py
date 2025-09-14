"""Simple test to verify TRL callback order and behavior.

This test ensures that metrics are stored only after logs carry real values,
and that the callback order (on_step_end -> on_log) is respected.
"""

import tempfile
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, Mock


# Mock the transformers classes
class MockTrainingArguments:
    def __init__(self, output_dir, learning_rate=0.001, seed=42):
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.seed = seed

    def to_dict(self):
        return {
            "output_dir": self.output_dir,
            "learning_rate": self.learning_rate,
            "seed": self.seed
        }

class MockTrainerState:
    def __init__(self, global_step=1, epoch=0.0, log_history=None):
        self.global_step = global_step
        self.epoch = epoch
        self.log_history = log_history or []

class MockTrainerControl:
    def __init__(self):
        pass

# Simplified RLDKMetrics for testing
@dataclass
class RLDKMetrics:
    """Container for RLDK metrics collected during training."""

    # Training metrics
    step: int = 0
    epoch: float = 0.0
    learning_rate: float = 0.0
    loss: float = 0.0
    grad_norm: float = 0.0

    # PPO-specific metrics
    reward_mean: float = 0.0
    kl_mean: float = 0.0
    entropy_mean: float = 0.0
    clip_frac: float = 0.0
    value_loss: float = 0.0
    policy_loss: float = 0.0

    # Resource metrics
    gpu_memory_used: float = 0.0
    cpu_memory_used: float = 0.0

    # Timing metrics
    step_time: float = 0.0
    wall_time: float = 0.0

    # Training health indicators
    training_stability_score: float = 1.0
    convergence_indicator: float = 0.0

    # Metadata
    run_id: str = ""
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

# Simplified RLDKCallback for testing
class RLDKCallback:
    """Simplified RLDK callback for testing."""

    def __init__(self, output_dir, log_interval=10, enable_jsonl_logging=False, enable_resource_monitoring=False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.enable_jsonl_logging = enable_jsonl_logging
        self.enable_resource_monitoring = enable_resource_monitoring

        # Metrics storage
        self.metrics_history: List[RLDKMetrics] = []
        self.current_metrics = RLDKMetrics()
        self.step_start_time = time.time()
        self.run_start_time = time.time()

        # Generate run ID
        self.run_id = f"test_run_{int(time.time())}"
        self.current_metrics.run_id = self.run_id

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.run_start_time = time.time()
        self.current_metrics.run_id = self.run_id

        # Initialize metrics from args
        self.current_metrics.learning_rate = args.learning_rate
        self.current_metrics.seed = args.seed

    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each training step."""
        self.step_start_time = time.time()
        self.current_metrics.step = state.global_step
        self.current_metrics.epoch = state.epoch

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step.

        NOTE: This method is called BEFORE on_log and should only collect resource snapshots.
        Do not update metrics from logs here as on_log will be called after this with the
        actual concrete values from the training step.
        """
        # Calculate step timing
        step_time = time.time() - self.step_start_time
        self.current_metrics.step_time = step_time
        self.current_metrics.wall_time = time.time() - self.run_start_time

        # Monitor resources if enabled - this is safe to do here as it's just resource snapshots
        if self.enable_resource_monitoring:
            self._monitor_resources()

        # NOTE: We deliberately do NOT update metrics from state.log_history here because:
        # 1. on_log fires after on_step_end and contains the concrete values
        # 2. We want to ensure metrics are stored only after logs carry real values
        # 3. The actual metric updates and history appending happens in on_log

    def on_log(self, args, state, control, logs: Dict[str, float], **kwargs):
        """Called when logs are generated.

        NOTE: This method is called AFTER on_step_end and contains concrete values from the
        training step. This is where we update metrics from logs, append to history, write
        JSONL if enabled, run alert checks, and perform detailed logging.
        """
        # Step 1: Update current_metrics from the logs (concrete values)
        if 'train_loss' in logs:
            self.current_metrics.loss = logs['train_loss']
        if 'learning_rate' in logs:
            self.current_metrics.learning_rate = logs['learning_rate']
        if 'grad_norm' in logs:
            self.current_metrics.grad_norm = logs['grad_norm']

        # PPO-specific metrics from logs
        if 'ppo/rewards/mean' in logs:
            self.current_metrics.reward_mean = logs['ppo/rewards/mean']
        if 'ppo/policy/kl_mean' in logs:
            self.current_metrics.kl_mean = logs['ppo/policy/kl_mean']
        if 'ppo/policy/entropy' in logs:
            self.current_metrics.entropy_mean = logs['ppo/policy/entropy']
        if 'ppo/policy/clipfrac' in logs:
            self.current_metrics.clip_frac = logs['ppo/policy/clipfrac']
        if 'ppo/val/value_loss' in logs:
            self.current_metrics.value_loss = logs['ppo/val/value_loss']
        if 'ppo/policy/policy_loss' in logs:
            self.current_metrics.policy_loss = logs['ppo/policy/policy_loss']

        # Step 2: Analyze logs and update derived metrics BEFORE storing
        self._analyze_logs(logs, state)

        # Step 3: Append a copy of current_metrics to history (only after logs carry real values)
        self.metrics_history.append(RLDKMetrics(**self.current_metrics.to_dict()))

        # Step 4: Write JSONL if enabled (simplified for testing)
        if self.enable_jsonl_logging:
            self._log_jsonl_event(state, logs)

        # Step 5: Run alert checks (simplified for testing)
        self._check_alerts()

        # Step 6: Any detailed logging
        if state.global_step % self.log_interval == 0:
            self._log_detailed_metrics()

    def _analyze_logs(self, logs, state):
        """Analyze logs and update derived metrics (simplified for testing)."""
        # Simulate updating derived metrics like training_stability_score
        if len(self.metrics_history) > 0:
            self.current_metrics.training_stability_score = 0.95  # Mock value
            self.current_metrics.convergence_indicator = 0.1  # Mock value

    def _monitor_resources(self):
        """Monitor resource usage (simplified for testing)."""
        self.current_metrics.gpu_memory_used = 1.0  # Mock value
        self.current_metrics.cpu_memory_used = 2.0  # Mock value

    def _log_jsonl_event(self, state, logs):
        """Log JSONL event (simplified for testing)."""
        pass  # Simplified for testing

    def _check_alerts(self):
        """Check for alerts (simplified for testing)."""
        pass  # Simplified for testing

    def _log_detailed_metrics(self):
        """Log detailed metrics (simplified for testing)."""
        pass  # Simplified for testing


class TestTRLCallbackOrder(unittest.TestCase):
    """Test TRL callback order and metric storage behavior."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test output
        self.temp_dir = tempfile.mkdtemp()

        # Create a mock TrainerState with log_history
        self.trainer_state = MockTrainerState(
            global_step=1,
            epoch=0.0,
            log_history=[
                {"step": 0, "train_loss": 0.5, "learning_rate": 0.001, "grad_norm": 1.0}
            ]
        )

        # Create training arguments
        self.training_args = MockTrainingArguments(
            output_dir=self.temp_dir,
            learning_rate=0.001,
            seed=42
        )

        # Create trainer control
        self.trainer_control = MockTrainerControl()

        # Initialize the callback
        self.callback = RLDKCallback(
            output_dir=self.temp_dir,
            log_interval=1,
            enable_jsonl_logging=False,  # Disable for testing
            enable_resource_monitoring=False  # Disable for testing
        )

        # Initialize training
        self.callback.on_train_begin(
            args=self.training_args,
            state=self.trainer_state,
            control=self.trainer_control
        )

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_callback_order_and_metric_storage(self):
        """Test that metrics are stored only after on_log with real values."""
        # Verify initial state
        initial_history_length = len(self.callback.metrics_history)
        self.assertEqual(initial_history_length, 0, "History should start empty")

        # Simulate on_step_begin
        self.callback.on_step_begin(
            args=self.training_args,
            state=self.trainer_state,
            control=self.trainer_control
        )

        # Simulate on_step_end call
        # This should NOT append to history, only collect resource snapshots
        self.callback.on_step_end(
            args=self.training_args,
            state=self.trainer_state,
            control=self.trainer_control
        )

        # Verify history length hasn't increased after on_step_end
        history_after_step_end = len(self.callback.metrics_history)
        self.assertEqual(history_after_step_end, initial_history_length,
                        "History length should not increase after on_step_end")

        # Verify that current_metrics still has default values (not updated from logs)
        self.assertEqual(self.callback.current_metrics.loss, 0.0,
                        "Loss should still be default value after on_step_end")
        self.assertEqual(self.callback.current_metrics.learning_rate, 0.001,
                        "Learning rate should be from args, not logs")

        # Simulate on_log call with concrete values
        logs = {
            "train_loss": 0.75,
            "learning_rate": 0.0009,
            "grad_norm": 1.5,
            "ppo/rewards/mean": 0.8,
            "ppo/policy/kl_mean": 0.05
        }

        self.callback.on_log(
            args=self.training_args,
            state=self.trainer_state,
            control=self.trainer_control,
            logs=logs
        )

        # Verify history length increased after on_log
        history_after_log = len(self.callback.metrics_history)
        self.assertEqual(history_after_log, initial_history_length + 1,
                        "History length should increase by 1 after on_log")

        # Verify that stored metrics contain the values from logs
        stored_metrics = self.callback.metrics_history[-1]
        self.assertEqual(stored_metrics.loss, 0.75,
                        f"Stored loss should equal log value (0.75), got {stored_metrics.loss}")
        self.assertEqual(stored_metrics.learning_rate, 0.0009,
                        f"Stored learning rate should equal log value (0.0009), got {stored_metrics.learning_rate}")
        self.assertEqual(stored_metrics.grad_norm, 1.5,
                        f"Stored grad norm should equal log value (1.5), got {stored_metrics.grad_norm}")
        self.assertEqual(stored_metrics.reward_mean, 0.8,
                        f"Stored reward mean should equal log value (0.8), got {stored_metrics.reward_mean}")
        self.assertEqual(stored_metrics.kl_mean, 0.05,
                        f"Stored KL mean should equal log value (0.05), got {stored_metrics.kl_mean}")

    def test_multiple_steps_behavior(self):
        """Test behavior across multiple training steps."""
        # Simulate multiple steps
        for step in range(1, 4):
            # Update state for this step
            self.trainer_state.global_step = step
            self.trainer_state.epoch = step / 10.0

            # Simulate on_step_begin
            self.callback.on_step_begin(
                args=self.training_args,
                state=self.trainer_state,
                control=self.trainer_control
            )

            # Simulate on_step_end
            self.callback.on_step_end(
                args=self.training_args,
                state=self.trainer_state,
                control=self.trainer_control
            )

            # Verify no metrics stored yet
            self.assertEqual(len(self.callback.metrics_history), step - 1,
                            f"History length should be {step - 1} after on_step_end for step {step}")

            # Simulate on_log with step-specific values
            logs = {
                "train_loss": 0.5 + step * 0.1,  # Increasing loss
                "learning_rate": 0.001 - step * 0.0001,  # Decreasing LR
                "grad_norm": 1.0 + step * 0.2,  # Increasing grad norm
                "ppo/rewards/mean": 0.5 + step * 0.1,  # Increasing reward
            }

            self.callback.on_log(
                args=self.training_args,
                state=self.trainer_state,
                control=self.trainer_control,
                logs=logs
            )

            # Verify metrics stored with correct values
            self.assertEqual(len(self.callback.metrics_history), step,
                            f"History length should be {step} after on_log for step {step}")

            stored_metrics = self.callback.metrics_history[-1]
            self.assertEqual(stored_metrics.step, step,
                            f"Stored step should be {step}, got {stored_metrics.step}")
            self.assertEqual(stored_metrics.loss, logs["train_loss"],
                            f"Stored loss should equal log value for step {step}")
            self.assertEqual(stored_metrics.learning_rate, logs["learning_rate"],
                            f"Stored learning rate should equal log value for step {step}")
            self.assertEqual(stored_metrics.grad_norm, logs["grad_norm"],
                            f"Stored grad norm should equal log value for step {step}")
            self.assertEqual(stored_metrics.reward_mean, logs["ppo/rewards/mean"],
                            f"Stored reward mean should equal log value for step {step}")

    def test_on_step_end_only_collects_resources(self):
        """Test that on_step_end only collects resource snapshots, not log values."""
        # Set up some log history in state
        self.trainer_state.log_history = [
            {"step": 0, "train_loss": 0.8, "learning_rate": 0.002, "grad_norm": 2.0}
        ]

        # Simulate on_step_begin
        self.callback.on_step_begin(
            args=self.training_args,
            state=self.trainer_state,
            control=self.trainer_control
        )

        # Call on_step_end
        self.callback.on_step_end(
            args=self.training_args,
            state=self.trainer_state,
            control=self.trainer_control
        )

        # Verify that current_metrics were NOT updated from state.log_history
        # (This is the key behavior we want to ensure)
        self.assertEqual(self.callback.current_metrics.loss, 0.0,
                        "Loss should not be updated from state.log_history in on_step_end")
        self.assertEqual(self.callback.current_metrics.learning_rate, 0.001,
                        "Learning rate should be from args, not from state.log_history")
        self.assertEqual(self.callback.current_metrics.grad_norm, 0.0,
                        "Grad norm should not be updated from state.log_history in on_step_end")

        # Verify timing metrics were updated (these are safe to update in on_step_end)
        self.assertGreater(self.callback.current_metrics.step_time, 0,
                          "Step time should be calculated in on_step_end")
        self.assertGreater(self.callback.current_metrics.wall_time, 0,
                          "Wall time should be calculated in on_step_end")

    def test_on_log_updates_from_concrete_logs(self):
        """Test that on_log properly updates metrics from concrete log values."""
        # Call on_log with specific log values
        logs = {
            "train_loss": 1.25,
            "learning_rate": 0.0005,
            "grad_norm": 3.0,
            "ppo/rewards/mean": 1.2,
            "ppo/policy/kl_mean": 0.15,
            "ppo/policy/entropy": 0.8,
            "ppo/policy/clipfrac": 0.1
        }

        self.callback.on_log(
            args=self.training_args,
            state=self.trainer_state,
            control=self.trainer_control,
            logs=logs
        )

        # Verify all metrics were updated from logs
        stored_metrics = self.callback.metrics_history[-1]
        self.assertEqual(stored_metrics.loss, 1.25)
        self.assertEqual(stored_metrics.learning_rate, 0.0005)
        self.assertEqual(stored_metrics.grad_norm, 3.0)
        self.assertEqual(stored_metrics.reward_mean, 1.2)
        self.assertEqual(stored_metrics.kl_mean, 0.15)
        self.assertEqual(stored_metrics.entropy_mean, 0.8)
        self.assertEqual(stored_metrics.clip_frac, 0.1)

    def test_derived_metrics_included_in_stored_metrics(self):
        """Test that derived metrics calculated by _analyze_logs are included in stored metrics."""
        # First, add some metrics to history so _analyze_logs has data to work with
        logs1 = {"train_loss": 1.0}
        self.callback.on_log(
            args=self.training_args,
            state=self.trainer_state,
            control=self.trainer_control,
            logs=logs1
        )

        # Now test that derived metrics are included in the second step
        logs2 = {"train_loss": 1.5}
        self.callback.on_log(
            args=self.training_args,
            state=self.trainer_state,
            control=self.trainer_control,
            logs=logs2
        )

        # Verify that the stored metrics include the derived metrics
        stored_metrics = self.callback.metrics_history[-1]
        self.assertEqual(stored_metrics.training_stability_score, 0.95,
                        "Training stability score should be included in stored metrics")
        self.assertEqual(stored_metrics.convergence_indicator, 0.1,
                        "Convergence indicator should be included in stored metrics")

    def test_derived_metrics_timing_verification(self):
        """Test that derived metrics are calculated and stored in the same step, not delayed."""
        # Test with multiple steps to verify timing
        for step in range(1, 4):
            # Update state
            self.trainer_state.global_step = step

            # Call on_step_begin to set the step number
            self.callback.on_step_begin(
                args=self.training_args,
                state=self.trainer_state,
                control=self.trainer_control
            )

            # Call on_log with step-specific data
            logs = {"train_loss": 0.5 + step * 0.1}
            self.callback.on_log(
                args=self.training_args,
                state=self.trainer_state,
                control=self.trainer_control,
                logs=logs
            )

            # Verify that derived metrics are immediately available in the stored metrics
            stored_metrics = self.callback.metrics_history[-1]

            # For steps after the first one, derived metrics should be calculated
            if step > 1:
                self.assertEqual(stored_metrics.training_stability_score, 0.95,
                               f"Training stability score should be calculated in step {step}")
                self.assertEqual(stored_metrics.convergence_indicator, 0.1,
                               f"Convergence indicator should be calculated in step {step}")

            # Verify the step number is correct (no delay)
            self.assertEqual(stored_metrics.step, step,
                           f"Step number should be {step}, not delayed")


if __name__ == "__main__":
    unittest.main()
