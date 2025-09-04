"""RLDK callbacks for TRL training loops."""

import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

try:
    from trl import PPOTrainer
    from trl.trainer.ppo_trainer import PPOTrainer as PPOTrainerClass
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    PPOTrainer = None
    PPOTrainerClass = None

# Import Event schema for proper JSONL emission
try:
    from ...io.event_schema import Event, create_event_from_row
    EVENT_SCHEMA_AVAILABLE = True
except ImportError:
    EVENT_SCHEMA_AVAILABLE = False
    Event = None
    create_event_from_row = None


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
    reward_std: float = 0.0
    reward_min: float = 0.0
    reward_max: float = 0.0
    kl_mean: float = 0.0
    kl_std: float = 0.0
    entropy_mean: float = 0.0
    clip_frac: float = 0.0
    value_loss: float = 0.0
    policy_loss: float = 0.0
    value_policy_loss: float = 0.0
    policy_grad_norm: float = 0.0
    value_grad_norm: float = 0.0
    value_mean: float = 0.0
    value_std: float = 0.0
    rollout_length_mean: float = 0.0
    rollout_length_std: float = 0.0
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    
    # PPO internal state metrics
    kl_coef: float = 1.0
    target_kl: float = 0.1
    advantage_normalized: bool = True
    clip_range: float = 0.2
    clip_range_value: float = 0.2
    batch_size: int = 0
    global_step: int = 0
    
    # Model and dataset information
    model_type: str = "unknown"
    vocab_size: int = 0
    tokenizer_vocab_size: int = 0
    dataset_size: int = 0
    
    # PPO rollout metrics
    rollout_buffer_size: int = 0
    rollout_buffer_pos: int = 0
    rollout_mean_reward: float = 0.0
    rollout_std_reward: float = 0.0
    
    # PPO policy metrics
    policy_total_params: int = 0
    policy_trainable_params: int = 0
    policy_lr: float = 0.0
    
    # PPO value metrics
    value_total_params: int = 0
    value_trainable_params: int = 0
    value_lr: float = 0.0
    
    # Resource metrics
    gpu_memory_used: float = 0.0
    gpu_memory_allocated: float = 0.0
    cpu_memory_used: float = 0.0
    
    # Timing metrics
    step_time: float = 0.0
    wall_time: float = 0.0
    
    # Token metrics
    tokens_in: int = 0
    tokens_out: int = 0
    
    # Training health indicators
    training_stability_score: float = 1.0
    convergence_indicator: float = 0.0
    
    # Metadata
    phase: str = "train"
    run_id: str = ""
    git_sha: str = ""
    seed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }


class RLDKCallback(TrainerCallback):
    """RLDK callback for real-time training monitoring and analysis."""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        log_interval: int = 10,
        alert_thresholds: Optional[Dict[str, float]] = None,
        enable_checkpoint_analysis: bool = True,
        enable_resource_monitoring: bool = True,
        run_id: Optional[str] = None,
        enable_jsonl_logging: bool = True,
        jsonl_log_interval: int = 1,
    ):
        """Initialize RLDK callback.
        
        Args:
            output_dir: Directory to save RLDK logs and analysis
            log_interval: Steps between detailed logging
            alert_thresholds: Thresholds for triggering alerts
            enable_checkpoint_analysis: Whether to analyze checkpoints
            enable_resource_monitoring: Whether to monitor resource usage
            run_id: Unique identifier for this training run
        """
        if not TRL_AVAILABLE:
            raise ImportError(
                "TRL is required for RLDKCallback. Install with: pip install trl"
            )
            
        self.output_dir = Path(output_dir) if output_dir else Path("./rldk_logs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate log intervals
        if log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if jsonl_log_interval <= 0:
            raise ValueError("jsonl_log_interval must be positive")
        
        self.log_interval = log_interval
        self.enable_checkpoint_analysis = enable_checkpoint_analysis
        self.enable_resource_monitoring = enable_resource_monitoring
        self.enable_jsonl_logging = enable_jsonl_logging
        self.jsonl_log_interval = jsonl_log_interval
        
        # Default alert thresholds
        self.alert_thresholds = {
            "kl_divergence": 0.1,
            "clip_fraction": 0.2,
            "gradient_norm": 1.0,
            "reward_std": 0.5,
            "loss_spike": 2.0,
            "memory_usage": 0.9,
        }
        if alert_thresholds:
            self.alert_thresholds.update(alert_thresholds)
        
        # Metrics storage
        self.metrics_history: List[RLDKMetrics] = []
        self.current_metrics = RLDKMetrics()
        self.step_start_time = time.time()
        self.run_start_time = time.time()
        
        # Generate run ID if not provided
        self.run_id = run_id or f"rldk_run_{int(time.time())}"
        self.current_metrics.run_id = self.run_id
        
        # JSONL logging setup (after run_id is initialized)
        self.jsonl_file = None
        if self.enable_jsonl_logging:
            self._setup_jsonl_logging()
        
        # Alert system
        self.alerts: List[Dict[str, Any]] = []
        
        print(f"🚀 RLDK Callback initialized - Run ID: {self.run_id}")
        print(f"📊 Output directory: {self.output_dir}")
        print(f"⚠️  Alert thresholds: {self.alert_thresholds}")
        if self.enable_jsonl_logging:
            print(f"📝 JSONL logging enabled - interval: {self.jsonl_log_interval}")
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of training."""
        print("🎯 RLDK: Training started")
        self.run_start_time = time.time()
        self.current_metrics.run_id = self.run_id
        
        # Initialize metrics
        self.current_metrics.learning_rate = args.learning_rate
        self.current_metrics.seed = args.seed
        
        # Save training configuration
        config_path = self.output_dir / f"{self.run_id}_config.json"
        with open(config_path, "w") as f:
            json.dump(args.to_dict(), f, indent=2)
    
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of each training step."""
        self.step_start_time = time.time()
        self.current_metrics.step = state.global_step
        self.current_metrics.epoch = state.epoch
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each training step."""
        # Calculate step timing
        step_time = time.time() - self.step_start_time
        self.current_metrics.step_time = step_time
        self.current_metrics.wall_time = time.time() - self.run_start_time
        
        # Update learning rate
        if hasattr(state, 'log_history') and state.log_history:
            latest_log = state.log_history[-1]
            self.current_metrics.learning_rate = latest_log.get('learning_rate', self.current_metrics.learning_rate)
            self.current_metrics.loss = latest_log.get('train_loss', self.current_metrics.loss)
            self.current_metrics.grad_norm = latest_log.get('grad_norm', self.current_metrics.grad_norm)
        
        # Monitor resources if enabled
        if self.enable_resource_monitoring:
            self._monitor_resources()
        
        # Collect PPO-specific metrics if available
        self._collect_ppo_metrics(kwargs)
        
        # Note: We don't store metrics here because on_log is called after on_step_end
        # and contains the actual logged values. We'll store metrics in on_log instead.
        
        # Log detailed metrics at intervals
        if state.global_step % self.log_interval == 0:
            self._log_detailed_metrics()
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, float], **kwargs):
        """Called when logs are generated."""
        # Update current metrics with log data
        if 'train_loss' in logs:
            self.current_metrics.loss = logs['train_loss']
        if 'learning_rate' in logs:
            self.current_metrics.learning_rate = logs['learning_rate']
        if 'grad_norm' in logs:
            self.current_metrics.grad_norm = logs['grad_norm']
        
        # PPO-specific metrics from logs
        if 'ppo/rewards/mean' in logs:
            self.current_metrics.reward_mean = logs['ppo/rewards/mean']
        if 'ppo/rewards/std' in logs:
            self.current_metrics.reward_std = logs['ppo/rewards/std']
        if 'ppo/rewards/min' in logs:
            self.current_metrics.reward_min = logs['ppo/rewards/min']
        if 'ppo/rewards/max' in logs:
            self.current_metrics.reward_max = logs['ppo/rewards/max']
            
        if 'ppo/policy/kl_mean' in logs:
            self.current_metrics.kl_mean = logs['ppo/policy/kl_mean']
        if 'ppo/policy/kl_std' in logs:
            self.current_metrics.kl_std = logs['ppo/policy/kl_std']
        if 'ppo/policy/entropy' in logs:
            self.current_metrics.entropy_mean = logs['ppo/policy/entropy']
        if 'ppo/policy/clipfrac' in logs:
            self.current_metrics.clip_frac = logs['ppo/policy/clipfrac']
        # Policy loss - handle both keys independently
        if 'ppo/policy/policy_loss' in logs:
            self.current_metrics.policy_loss = logs['ppo/policy/policy_loss']
        if 'ppo/val/policy_loss' in logs:
            self.current_metrics.value_policy_loss = logs['ppo/val/policy_loss']
            # For backward compatibility, also set policy_loss if not already set
            if 'ppo/policy/policy_loss' not in logs:
                self.current_metrics.policy_loss = logs['ppo/val/policy_loss']
        if 'ppo/policy/grad_norm' in logs:
            self.current_metrics.policy_grad_norm = logs['ppo/policy/grad_norm']
            
        if 'ppo/val/value_loss' in logs:
            self.current_metrics.value_loss = logs['ppo/val/value_loss']
        if 'ppo/val/grad_norm' in logs:
            self.current_metrics.value_grad_norm = logs['ppo/val/grad_norm']
        if 'ppo/val/mean' in logs:
            self.current_metrics.value_mean = logs['ppo/val/mean']
        if 'ppo/val/std' in logs:
            self.current_metrics.value_std = logs['ppo/val/std']
            
        # PPO rollout metrics
        if 'ppo/rollout/length_mean' in logs:
            self.current_metrics.rollout_length_mean = logs['ppo/rollout/length_mean']
        if 'ppo/rollout/length_std' in logs:
            self.current_metrics.rollout_length_std = logs['ppo/rollout/length_std']
            
        # PPO advantage metrics
        if 'ppo/advantages/mean' in logs:
            self.current_metrics.advantage_mean = logs['ppo/advantages/mean']
        if 'ppo/advantages/std' in logs:
            self.current_metrics.advantage_std = logs['ppo/advantages/std']
        
        # Real-time analysis
        self._analyze_logs(logs, state)
        
        # Store metrics AFTER log values are applied
        self.metrics_history.append(RLDKMetrics(**self.current_metrics.to_dict()))
        
        # Log JSONL event after each call to _log_metrics
        if self.enable_jsonl_logging:
            self._log_jsonl_event(state, logs)
        
        # Check for alerts AFTER metrics are stored
        self._check_alerts()
    
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called when a checkpoint is saved."""
        if self.enable_checkpoint_analysis:
            self._analyze_checkpoint(kwargs.get('model'), state)
        
        # Save metrics history
        self._save_metrics_history()
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training."""
        print("🏁 RLDK: Training completed")
        
        # Close JSONL file
        self._close_jsonl_file()
        
        # Final analysis
        self._generate_final_report()
        
        # Save all data
        self._save_metrics_history()
        self._save_alerts()
        
        print(f"📁 RLDK data saved to: {self.output_dir}")
    
    def _monitor_resources(self):
        """Monitor GPU and CPU resource usage."""
        try:
            if torch.cuda.is_available():
                self.current_metrics.gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                self.current_metrics.gpu_memory_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            # CPU memory monitoring (simplified)
            import psutil
            process = psutil.Process()
            self.current_metrics.cpu_memory_used = process.memory_info().rss / 1024**3  # GB
        except Exception as e:
            warnings.warn(f"Resource monitoring failed: {e}")
    
    def _collect_ppo_metrics(self, kwargs: Dict[str, Any]):
        """Collect PPO-specific metrics from trainer."""
        # Try to extract PPO metrics from trainer if available
        trainer = kwargs.get('trainer')
        if not trainer:
            return
            
        try:
            # Check if this is a PPO trainer
            if not TRL_AVAILABLE or not isinstance(trainer, PPOTrainer):
                return
                
            # Extract PPO-specific metrics from trainer's internal state
            self._extract_ppo_internal_metrics(trainer)
            
        except Exception as e:
            warnings.warn(f"Failed to collect PPO metrics: {e}")
    
    def _extract_ppo_internal_metrics(self, trainer):
        """Extract PPO metrics from trainer's internal state."""
        try:
            # Extract KL coefficient if available
            if hasattr(trainer, 'config') and hasattr(trainer.config, 'kl_coef'):
                self.current_metrics.kl_coef = trainer.config.kl_coef
            
            # Extract advantage statistics if available
            if hasattr(trainer, 'config') and hasattr(trainer.config, 'advantage_normalization'):
                self.current_metrics.advantage_normalized = trainer.config.advantage_normalization
            
            # Extract rollout statistics if available
            if hasattr(trainer, 'config') and hasattr(trainer.config, 'target_kl'):
                self.current_metrics.target_kl = trainer.config.target_kl
            
            # Extract learning rate schedule information
            if hasattr(trainer, 'config') and hasattr(trainer.config, 'learning_rate'):
                self.current_metrics.learning_rate = trainer.config.learning_rate
            
            # Extract batch size information
            if hasattr(trainer, 'config') and hasattr(trainer.config, 'batch_size'):
                self.current_metrics.batch_size = trainer.config.batch_size
            
            # Extract PPO-specific hyperparameters
            if hasattr(trainer, 'config') and hasattr(trainer.config, 'cliprange'):
                self.current_metrics.clip_range = trainer.config.cliprange
            
            if hasattr(trainer, 'config') and hasattr(trainer.config, 'cliprange_value'):
                self.current_metrics.clip_range_value = trainer.config.cliprange_value
            
            # Extract model information if available
            if hasattr(trainer, 'model'):
                model = trainer.model
                if hasattr(model, 'config'):
                    self.current_metrics.model_type = getattr(model.config, 'model_type', 'unknown')
                    self.current_metrics.vocab_size = getattr(model.config, 'vocab_size', 0)
            
            # Extract tokenizer information if available
            if hasattr(trainer, 'tokenizer'):
                tokenizer = trainer.tokenizer
                if hasattr(tokenizer, 'vocab_size'):
                    self.current_metrics.tokenizer_vocab_size = tokenizer.vocab_size
            
            # Extract dataset information if available
            if hasattr(trainer, 'train_dataset'):
                dataset = trainer.train_dataset
                if hasattr(dataset, '__len__'):
                    self.current_metrics.dataset_size = len(dataset)
            
            # Extract training step information
            if hasattr(trainer, 'state') and hasattr(trainer.state, 'global_step'):
                self.current_metrics.global_step = trainer.state.global_step
            
            # Extract PPO-specific internal state metrics
            self._extract_ppo_rollout_metrics(trainer)
            self._extract_ppo_policy_metrics(trainer)
            self._extract_ppo_value_metrics(trainer)
            
        except Exception as e:
            warnings.warn(f"Failed to extract PPO internal metrics: {e}")
    
    def _extract_ppo_rollout_metrics(self, trainer):
        """Extract PPO rollout-specific metrics."""
        try:
            # Try to access rollout buffer if available
            if hasattr(trainer, 'rollout_buffer'):
                buffer = trainer.rollout_buffer
                if hasattr(buffer, 'size'):
                    self.current_metrics.rollout_buffer_size = buffer.size
                if hasattr(buffer, 'pos'):
                    self.current_metrics.rollout_buffer_pos = buffer.pos
            
            # Try to access rollout statistics if available
            if hasattr(trainer, 'rollout_stats'):
                stats = trainer.rollout_stats
                if hasattr(stats, 'mean_reward'):
                    self.current_metrics.rollout_mean_reward = stats.mean_reward
                if hasattr(stats, 'std_reward'):
                    self.current_metrics.rollout_std_reward = stats.std_reward
                    
        except Exception as e:
            # Rollout metrics extraction is optional, don't warn
            pass
    
    def _extract_ppo_policy_metrics(self, trainer):
        """Extract PPO policy-specific metrics."""
        try:
            # Try to access policy network if available
            if hasattr(trainer, 'policy'):
                policy = trainer.policy
                if hasattr(policy, 'parameters'):
                    total_params = sum(p.numel() for p in policy.parameters())
                    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
                    self.current_metrics.policy_total_params = total_params
                    self.current_metrics.policy_trainable_params = trainable_params
            
            # Try to access policy optimizer if available
            if hasattr(trainer, 'policy_optimizer'):
                optimizer = trainer.policy_optimizer
                if hasattr(optimizer, 'param_groups') and optimizer.param_groups:
                    self.current_metrics.policy_lr = optimizer.param_groups[0].get('lr', 0.0)
                    
        except Exception as e:
            # Policy metrics extraction is optional, don't warn
            pass
    
    def _extract_ppo_value_metrics(self, trainer):
        """Extract PPO value function-specific metrics."""
        try:
            # Try to access value network if available
            if hasattr(trainer, 'value'):
                value = trainer.value
                if hasattr(value, 'parameters'):
                    total_params = sum(p.numel() for p in value.parameters())
                    trainable_params = sum(p.numel() for p in value.parameters() if p.requires_grad)
                    self.current_metrics.value_total_params = total_params
                    self.current_metrics.value_trainable_params = trainable_params
            
            # Try to access value optimizer if available
            if hasattr(trainer, 'value_optimizer'):
                optimizer = trainer.value_optimizer
                if hasattr(optimizer, 'param_groups') and optimizer.param_groups:
                    self.current_metrics.value_lr = optimizer.param_groups[0].get('lr', 0.0)
                    
        except Exception as e:
            # Value metrics extraction is optional, don't warn
            pass
    
    def _check_alerts(self):
        """Check for training issues and generate alerts."""
        current = self.current_metrics
        
        # KL divergence alert
        if current.kl_mean > self.alert_thresholds["kl_divergence"]:
            self._add_alert("high_kl_divergence", 
                          f"KL divergence {current.kl_mean:.4f} exceeds threshold {self.alert_thresholds['kl_divergence']}")
        
        # Clip fraction alert
        if current.clip_frac > self.alert_thresholds["clip_fraction"]:
            self._add_alert("high_clip_fraction",
                          f"Clip fraction {current.clip_frac:.4f} exceeds threshold {self.alert_thresholds['clip_fraction']}")
        
        # Gradient norm alert
        if current.grad_norm > self.alert_thresholds["gradient_norm"]:
            self._add_alert("high_gradient_norm",
                          f"Gradient norm {current.grad_norm:.4f} exceeds threshold {self.alert_thresholds['gradient_norm']}")
        
        # Memory usage alert
        if current.gpu_memory_used > self.alert_thresholds["memory_usage"] * 24:  # Assuming 24GB GPU
            self._add_alert("high_memory_usage",
                          f"GPU memory usage {current.gpu_memory_used:.2f}GB is high")
    
    def _add_alert(self, alert_type: str, message: str):
        """Add an alert to the alert list."""
        alert = {
            "type": alert_type,
            "message": message,
            "step": self.current_metrics.step,
            "timestamp": time.time(),
            "severity": "warning"
        }
        self.alerts.append(alert)
        print(f"⚠️  RLDK Alert: {message}")
    
    def _analyze_logs(self, logs: Dict[str, float], state: TrainerState):
        """Analyze logs for training health indicators."""
        # Calculate training stability score
        if len(self.metrics_history) > 10:
            recent_losses = [m.loss for m in self.metrics_history[-10:] if m.loss > 0]
            if recent_losses:
                loss_std = np.std(recent_losses)
                self.current_metrics.training_stability_score = max(0, 1 - loss_std)
        
        # Convergence indicator (simplified)
        if len(self.metrics_history) > 50:
            recent_rewards = [m.reward_mean for m in self.metrics_history[-50:] if m.reward_mean != 0]
            if recent_rewards:
                reward_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
                self.current_metrics.convergence_indicator = reward_trend
    
    def _analyze_checkpoint(self, model, state: TrainerState):
        """Analyze model checkpoint for health indicators."""
        if model is None:
            return
        
        try:
            # Basic checkpoint analysis
            checkpoint_path = self.output_dir / f"{self.run_id}_checkpoint_{state.global_step}.json"
            
            checkpoint_info = {
                "step": state.global_step,
                "epoch": state.epoch,
                "timestamp": time.time(),
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            }
            
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_info, f, indent=2)
                
        except Exception as e:
            warnings.warn(f"Checkpoint analysis failed: {e}")
    
    def _log_detailed_metrics(self):
        """Log detailed metrics at intervals."""
        current = self.current_metrics
        print(f"📊 RLDK Step {current.step}: "
              f"Loss={current.loss:.4f}, "
              f"Reward={current.reward_mean:.4f}, "
              f"KL={current.kl_mean:.4f}, "
              f"ClipFrac={current.clip_frac:.4f}, "
              f"Stability={current.training_stability_score:.3f}")
    
    def _save_metrics_history(self):
        """Save metrics history to file."""
        if not self.metrics_history:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([m.to_dict() for m in self.metrics_history])
        
        # Save as CSV and JSON (aggregates only)
        csv_path = self.output_dir / f"{self.run_id}_metrics.csv"
        json_path = self.output_dir / f"{self.run_id}_metrics.json"
        
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient='records', indent=2)
        
        # Note: JSONL file is handled separately and should not be modified here
        # The JSONL file contains per-step events and is written in real-time
    
    def _save_alerts(self):
        """Save alerts to file."""
        if not self.alerts:
            return
        
        alerts_path = self.output_dir / f"{self.run_id}_alerts.json"
        with open(alerts_path, "w") as f:
            json.dump(self.alerts, f, indent=2)
    
    def save_metrics_history(self):
        """Public method to save metrics history."""
        self._save_metrics_history()
        self._save_alerts()
        print(f"📊 Metrics history saved to {self.output_dir}")
    
    def _generate_final_report(self):
        """Generate final training report."""
        if not self.metrics_history:
            return
        
        # Calculate summary statistics
        df = pd.DataFrame([m.to_dict() for m in self.metrics_history])
        
        report = {
            "run_id": self.run_id,
            "total_steps": len(self.metrics_history),
            "total_time": self.metrics_history[-1].wall_time if self.metrics_history else 0,
            "final_loss": self.metrics_history[-1].loss if self.metrics_history else 0,
            "final_reward": self.metrics_history[-1].reward_mean if self.metrics_history else 0,
            "total_alerts": len(self.alerts),
            "training_stability": np.mean([m.training_stability_score for m in self.metrics_history]),
            "convergence_indicator": self.metrics_history[-1].convergence_indicator if self.metrics_history else 0,
        }
        
        report_path = self.output_dir / f"{self.run_id}_final_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Final Report: {report}")
    
    def _setup_jsonl_logging(self):
        """Setup JSONL logging file."""
        if not EVENT_SCHEMA_AVAILABLE:
            warnings.warn("Event schema not available, JSONL logging disabled")
            self.enable_jsonl_logging = False
            return
        
        # Safety check: ensure run_id is available
        if not hasattr(self, 'run_id') or self.run_id is None:
            warnings.warn("run_id not available, JSONL logging disabled")
            self.enable_jsonl_logging = False
            return
        
        jsonl_path = self.output_dir / f"{self.run_id}_events.jsonl"
        self.jsonl_file = open(jsonl_path, "w")
        print(f"📝 JSONL events will be written to: {jsonl_path}")
    
    def _log_jsonl_event(self, state: TrainerState, logs: Dict[str, float]):
        """Log a standardized JSONL event with Event schema structure."""
        if not self.jsonl_file or not EVENT_SCHEMA_AVAILABLE:
            return
        
        try:
            # Create event data with required fields for Event schema
            event_data = {
                "step": state.global_step,
                "timestamp": time.time(),
                "phase": "train",
                "wall_time": self.current_metrics.wall_time,
                "reward_mean": self.current_metrics.reward_mean,
                "reward_std": self.current_metrics.reward_std,
                "kl_mean": self.current_metrics.kl_mean,
                "kl_std": self.current_metrics.kl_std,
                "entropy_mean": self.current_metrics.entropy_mean,
                "clip_frac": self.current_metrics.clip_frac,
                "grad_norm": self.current_metrics.grad_norm,
                "lr": self.current_metrics.learning_rate,
                "loss": self.current_metrics.loss,
                "value_loss": self.current_metrics.value_loss,
                "policy_loss": self.current_metrics.policy_loss,
                "tokens_in": self.current_metrics.tokens_in,
                "tokens_out": self.current_metrics.tokens_out,
                "seed": self.current_metrics.seed,
                "run_id": self.current_metrics.run_id,
                "git_sha": self.current_metrics.git_sha,
                "epoch": self.current_metrics.epoch,
                "step_time": self.current_metrics.step_time,
                "gpu_memory_used": self.current_metrics.gpu_memory_used,
                "gpu_memory_allocated": self.current_metrics.gpu_memory_allocated,
                "cpu_memory_used": self.current_metrics.cpu_memory_used,
                "training_stability_score": self.current_metrics.training_stability_score,
                "convergence_indicator": self.current_metrics.convergence_indicator,
            }
            
            # Create Event object using the schema
            event = create_event_from_row(event_data, self.run_id, self.current_metrics.git_sha)
            
            # Write JSONL line with proper formatting
            json_line = event.to_json()
            self.jsonl_file.write(json_line + "\n")
            self.jsonl_file.flush()  # Ensure immediate write
            
        except Exception as e:
            warnings.warn(f"Failed to log JSONL event: {e}")
    
    def _emit_jsonl_event(self, state: TrainerState, logs: Dict[str, float]):
        """Emit a standardized JSONL event compatible with Event schema and TRLAdapter."""
        # Call the new _log_jsonl_event function for consistency
        self._log_jsonl_event(state, logs)
    
    def _close_jsonl_file(self):
        """Close the JSONL file."""
        if self.jsonl_file:
            self.jsonl_file.close()
            self.jsonl_file = None


class RLDKMonitor(RLDKCallback):
    """Simplified RLDK monitor for easy integration."""
    
    def __init__(self, **kwargs):
        """Initialize with sensible defaults."""
        super().__init__(
            log_interval=kwargs.get('log_interval', 10),
            enable_checkpoint_analysis=kwargs.get('enable_checkpoint_analysis', True),
            enable_resource_monitoring=kwargs.get('enable_resource_monitoring', True),
            **kwargs
        )