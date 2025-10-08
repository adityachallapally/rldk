"""RLDK callbacks for TRL training loops."""

import json
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd
try:
    from transformers import (
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )
    TRAINER_API_AVAILABLE = True
except ImportError:  # pragma: no cover - transformers Trainer APIs are optional in some envs
    TRAINER_API_AVAILABLE = False

    class _TrainerAPINotAvailableStub:
        """Minimal stub that raises a helpful error when trainer APIs are missing."""

        _ERROR = (
            "Transformers trainer callbacks are required for the TRL integrations. "
            "Install the full transformers package with trainer extras, for example: "
            "pip install 'transformers[torch]'"
        )

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401 - short helper
            raise ImportError(self._ERROR)

    class TrainerCallback(_TrainerAPINotAvailableStub):  # type: ignore[override]
        pass

    class TrainerControl(_TrainerAPINotAvailableStub):  # type: ignore[override]
        pass

    class TrainerState(_TrainerAPINotAvailableStub):  # type: ignore[override]
        pass

    class TrainingArguments(_TrainerAPINotAvailableStub):  # type: ignore[override]
        pass

try:
    from trl import PPOTrainer
    from trl.trainer.ppo_trainer import PPOTrainer as PPOTrainerClass
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    PPOTrainer = None
    PPOTrainerClass = None

# Import Event schema for proper JSONL emission
from ...emit import EventWriter
from ...monitor.presets import FIELD_MAP_PRESETS

try:
    from ...io.event_schema import Event, create_event_from_row
    EVENT_SCHEMA_AVAILABLE = True
except ImportError:
    EVENT_SCHEMA_AVAILABLE = False
    Event = None
    create_event_from_row = None


class EventWriterCallback(TrainerCallback):
    """Lightweight callback that mirrors TRL logs into ``EventWriter`` JSONL files."""

    _SOURCE = "trl"

    def __init__(
        self,
        event_log_path: Union[str, Path],
        *,
        run_id: Optional[str] = None,
        tags: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not TRAINER_API_AVAILABLE:
            raise ImportError(
                "Transformers trainer callbacks are required for EventWriterCallback. "
                "Install with: pip install 'transformers[torch]'"
            )

        self.event_log_path = Path(event_log_path)
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._provided_tags = dict(tags) if tags else None
        self._writer: Optional[EventWriter] = None
        self._trl_field_map: Dict[str, str] = dict(FIELD_MAP_PRESETS.get("trl", {}))
        self._metric_field_map: Dict[str, str] = dict(FIELD_MAP_PRESETS.get("grpo", {}))
        self._token_normalizer = {
            "rewards": "reward",
            "advantages": "advantage",
            "vals": "value",
            "val": "value",
            "values": "value",
            "kl_divergence": "kl",
            "clipfrac": "clip_frac",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_writer(self) -> EventWriter:
        if self._writer is None:
            self._writer = EventWriter(self.event_log_path)
        return self._writer

    def _normalise_token(self, token: str) -> str:
        token = token.replace("-", "_")
        return self._token_normalizer.get(token, token)

    def _candidate_metric_keys(self, key: str) -> List[str]:
        sanitized = key.strip().replace("-", "_")
        parts = [part for part in sanitized.split("/") if part]
        candidates: List[str] = []

        for start in range(len(parts)):
            suffix = parts[start:]
            direct = "_".join(suffix)
            if direct and direct not in candidates:
                candidates.append(direct)

            normalized_suffix = [self._normalise_token(part) for part in suffix]
            normalized = "_".join(normalized_suffix)
            if normalized and normalized not in candidates:
                candidates.append(normalized)

        if sanitized and sanitized not in candidates:
            candidates.append(sanitized)

        return candidates

    def _canonical_metric_name(self, key: str) -> str:
        for candidate in self._candidate_metric_keys(key):
            mapped = self._metric_field_map.get(candidate)
            if mapped:
                return mapped
        return key.replace("/", "_")

    # ------------------------------------------------------------------
    # Trainer callback hooks
    # ------------------------------------------------------------------
    def on_train_begin(self, args, state, control, **kwargs):  # type: ignore[override]
        self._ensure_writer()
        if self.run_id is None:
            self.run_id = getattr(args, "run_name", None)
        return control

    def on_train_end(self, args, state, control, **kwargs):  # type: ignore[override]
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs:
            return control

        writer = self._ensure_writer()
        step = getattr(state, "global_step", 0) or 0
        timestamp = time.time()

        context_tags: Dict[str, Any] = {}
        context_meta: Dict[str, Any] = {}
        context_run_id: Optional[str] = None
        context_step: Optional[int] = None
        context_time: Optional[Any] = None

        for raw_key, raw_value in list(logs.items()):
            mapped_field = self._trl_field_map.get(raw_key)
            if mapped_field == "step":
                try:
                    context_step = int(raw_value)  # type: ignore[arg-type]
                except Exception:
                    continue
            elif mapped_field == "time":
                context_time = raw_value
            elif mapped_field == "run_id":
                context_run_id = str(raw_value)
            elif mapped_field == "tags" and isinstance(raw_value, dict):
                context_tags.update(raw_value)
            elif mapped_field == "meta" and isinstance(raw_value, dict):
                context_meta.update(raw_value)

        if context_step is not None:
            step = context_step
        event_time = context_time if context_time is not None else timestamp
        if isinstance(event_time, (int, float)):
            event_time = datetime.fromtimestamp(event_time, tz=timezone.utc)
        run_identifier = context_run_id or self.run_id

        base_tags = dict(self._provided_tags) if self._provided_tags else {}
        if context_tags:
            base_tags.update(context_tags)

        for raw_key, raw_value in logs.items():
            if self._trl_field_map.get(raw_key) in {"step", "time", "run_id", "tags", "meta"}:
                continue

            if isinstance(raw_value, (int, float)):
                numeric_value = float(raw_value)
            else:
                try:
                    numeric_value = float(raw_value)
                except (TypeError, ValueError):
                    continue

            canonical_name = self._canonical_metric_name(str(raw_key))
            meta_payload = {"source": self._SOURCE, "raw_name": raw_key}
            if context_meta:
                meta_payload.update(context_meta)

            writer.log(
                step=step,
                name=canonical_name,
                value=numeric_value,
                time=event_time,
                run_id=run_identifier,
                tags=base_tags or None,
                meta=meta_payload,
            )

        return control


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
        if not TRAINER_API_AVAILABLE:
            raise ImportError(
                "Transformers trainer callbacks are required for RLDKCallback. "
                "Install with: pip install 'transformers[torch]'"
            )

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

        # Collect PPO-specific metrics if available - this is also safe as it's internal state
        self._collect_ppo_metrics(kwargs)

        # NOTE: We deliberately do NOT update metrics from state.log_history here because:
        # 1. on_log fires after on_step_end and contains the concrete values
        # 2. We want to ensure metrics are stored only after logs carry real values
        # 3. The actual metric updates and history appending happens in on_log

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, float], **kwargs):
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

        # Step 2: Analyze logs and update derived metrics BEFORE storing
        self._analyze_logs(logs, state)

        # Step 3: Append a copy of current_metrics to history (only after logs carry real values)
        self.metrics_history.append(RLDKMetrics(**self.current_metrics.to_dict()))

        # Step 4: Write JSONL if enabled
        if (self.enable_jsonl_logging and
            self.jsonl_log_interval > 0 and
            state.global_step % self.jsonl_log_interval == 0):
            self._log_jsonl_event(state, logs)

        # Step 5: Run alert checks
        self._check_alerts()

        # Step 6: Log detailed metrics at intervals
        if state.global_step % self.log_interval == 0:
            self._log_detailed_metrics()

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
            import torch  # Lazy import to avoid CLI hang
            if torch.cuda.is_available():
                self.current_metrics.gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                self.current_metrics.gpu_memory_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB

            # CPU memory monitoring (simplified)
            import psutil
            process = psutil.Process()
            self.current_metrics.cpu_memory_used = process.memory_info().rss / 1024**3  # GB
        except Exception as e:
            warnings.warn(f"Resource monitoring failed: {e}", stacklevel=2)

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
            warnings.warn(f"Failed to collect PPO metrics: {e}", stacklevel=2)

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
            warnings.warn(f"Failed to extract PPO internal metrics: {e}", stacklevel=2)

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

        except Exception:
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

        except Exception:
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

        except Exception:
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
            warnings.warn(f"Checkpoint analysis failed: {e}", stacklevel=2)

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
        pd.DataFrame([m.to_dict() for m in self.metrics_history])

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
            warnings.warn("Event schema not available, JSONL logging disabled", stacklevel=2)
            self.enable_jsonl_logging = False
            return

        # Safety check: ensure run_id is available
        if not hasattr(self, 'run_id') or self.run_id is None:
            warnings.warn("run_id not available, JSONL logging disabled", stacklevel=2)
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

            # Write JSONL line with proper formatting (single line, not pretty-printed)
            json_line = json.dumps(event.to_dict())
            self.jsonl_file.write(json_line + "\n")
            self.jsonl_file.flush()  # Ensure immediate write

        except Exception as e:
            warnings.warn(f"Failed to log JSONL event: {e}", stacklevel=2)

    def _emit_jsonl_event(self, state: TrainerState, logs: Dict[str, float]):
        """Emit a standardized JSONL event compatible with Event schema and TRLAdapter."""
        # Always emit when called directly - interval checking is handled in on_log
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
