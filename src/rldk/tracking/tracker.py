"""
Main experiment tracker that coordinates all tracking components.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from .config import TrackingConfig
from .dataset_tracker import DatasetTracker
from .environment_tracker import EnvironmentTracker
from .git_tracker import GitTracker
from .model_tracker import ModelTracker
from .seed_tracker import SeedTracker


logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Main experiment tracker that coordinates all tracking components."""

    def __init__(self, config: TrackingConfig):
        self.config = config
        self.experiment_id = config.experiment_id
        self.experiment_name = config.experiment_name
        self.output_dir = config.output_dir

        # Initialize tracking components with config for performance optimization
        self.dataset_tracker = DatasetTracker(config.dataset_checksum_algorithm, config) if config.enable_dataset_tracking else None
        self.model_tracker = ModelTracker(config.model_fingerprint_algorithm, config) if config.enable_model_tracking else None
        self.environment_tracker = EnvironmentTracker(config) if config.enable_environment_tracking else None
        self.seed_tracker = SeedTracker() if config.enable_seed_tracking else None
        self.git_tracker = GitTracker(config.git_repo_path, config) if config.enable_git_tracking else None

        # Storage for tracking data
        self.tracking_data: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "config": self._serialize_config(),
            "datasets": {},
            "models": {},
            "environment": {},
            "seeds": {},
            "git": {},
            "metrics": [],
            "metadata": config.metadata.copy(),
            "tags": config.tags.copy(),
            "notes": config.notes
        }

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazily created wandb run reused across saves for this tracker instance
        self._wandb_run = None

    async def start_experiment_async(self, progress_callback=None) -> Dict[str, Any]:
        """Async version of start_experiment with progress indicators and graceful degradation."""
        if progress_callback:
            progress_callback("Starting experiment initialization...")

        print(f"Starting experiment: {self.experiment_name} (ID: {self.experiment_id})")

        # Initialize components concurrently with individual error handling
        tasks = []
        task_names = []

        if self.environment_tracker:
            if progress_callback:
                progress_callback("Scheduling environment capture...")
            tasks.append(self._capture_environment_safe())
            task_names.append("environment")

        if self.git_tracker:
            if progress_callback:
                progress_callback("Scheduling Git state capture...")
            tasks.append(self._capture_git_safe())
            task_names.append("git")

        if self.seed_tracker:
            if progress_callback:
                progress_callback("Scheduling seed capture...")
            tasks.append(self._capture_seeds_safe())
            task_names.append("seeds")

        if tasks:
            if progress_callback:
                progress_callback(f"Running {len(tasks)} initialization tasks concurrently...")

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, (task_name, result) in enumerate(zip(task_names, results)):
                if not isinstance(result, Exception):
                    has_error = (isinstance(result, dict) and result.get("error")) or (hasattr(result, 'get') and result.get("error"))
                    if not has_error:
                        self.tracking_data[task_name] = result
                        if progress_callback:
                            progress_callback(f"✓ {task_name} capture completed")
                    else:
                        error_msg = result.get("error", "Unknown error") if hasattr(result, 'get') else str(result)
                        self.tracking_data[task_name] = {"error": error_msg}
                        if progress_callback:
                            progress_callback(f"⚠ {task_name} capture failed: {error_msg}")
                else:
                    error_msg = str(result)
                    self.tracking_data[task_name] = {"error": error_msg}
                    if progress_callback:
                        progress_callback(f"⚠ {task_name} capture failed: {error_msg}")

        # Save initial state
        self._save_tracking_data()

        if progress_callback:
            progress_callback("Experiment initialization complete!")

        return self.tracking_data

    async def _capture_environment_safe(self) -> Dict[str, Any]:
        """Safely capture environment with timeout and error handling."""
        try:
            return await self.environment_tracker.capture_environment_async(
                capture_conda=self.config.capture_conda_env,
                capture_pip=self.config.capture_pip_freeze,
                capture_system=self.config.capture_system_info,
                timeout=self.config.environment_timeout
            )
        except Exception as e:
            return {"error": f"Environment capture failed: {str(e)}"}

    async def _capture_git_safe(self) -> Dict[str, Any]:
        """Safely capture Git state with timeout and error handling."""
        try:
            return await self.git_tracker.capture_git_state_async(
                capture_commit=True,
                capture_diff=self.config.capture_git_diff,
                capture_status=self.config.capture_git_status,
                capture_remote=True,
                timeout=self.config.git_timeout
            )
        except Exception as e:
            return {"error": f"Git capture failed: {str(e)}"}

    async def _capture_seeds_safe(self) -> Dict[str, Any]:
        """Safely capture seed state with error handling."""
        try:
            return await asyncio.get_running_loop().run_in_executor(
                None, 
                lambda: self.seed_tracker.capture_seeds(
                    track_python=self.config.track_python_seed,
                    track_numpy=self.config.track_numpy_seed,
                    track_torch=self.config.track_torch_seed,
                    track_cuda=self.config.track_cuda_seed
                )
            )
        except Exception as e:
            return {"error": f"Seed capture failed: {str(e)}"}

    def start_experiment(self) -> Dict[str, Any]:
        """
        Synchronous version of start_experiment for backward compatibility.
        """
        if self.config.enable_async_init:
            # Use async version if enabled
            try:
                asyncio.get_running_loop()
                return self._start_experiment_sync()
            except RuntimeError:
                return asyncio.run(self.start_experiment_async())
        else:
            print(f"Starting experiment: {self.experiment_name} (ID: {self.experiment_id})")

            # Capture environment state
            if self.environment_tracker:
                print("Capturing environment state...")
                try:
                    self.tracking_data["environment"] = self.environment_tracker.capture_environment(
                        capture_conda=self.config.capture_conda_env,
                        capture_pip=self.config.capture_pip_freeze,
                        capture_system=self.config.capture_system_info
                    )
                except Exception as e:
                    self.tracking_data["environment"] = {"error": f"Environment capture failed: {str(e)}"}

            # Capture Git state
            if self.git_tracker:
                print("Capturing Git state...")
                try:
                    self.tracking_data["git"] = self.git_tracker.capture_git_state(
                        capture_commit=True,
                        capture_diff=self.config.capture_git_diff,
                        capture_status=self.config.capture_git_status,
                        capture_remote=True
                    )
                except Exception as e:
                    self.tracking_data["git"] = {"error": f"Git capture failed: {str(e)}"}

            # Capture initial seed state
            if self.seed_tracker:
                print("Capturing seed state...")
                try:
                    self.tracking_data["seeds"] = self.seed_tracker.capture_seeds(
                        track_python=self.config.track_python_seed,
                        track_numpy=self.config.track_numpy_seed,
                        track_torch=self.config.track_torch_seed,
                        track_cuda=self.config.track_cuda_seed
                    )
                except Exception as e:
                    self.tracking_data["seeds"] = {"error": f"Seed capture failed: {str(e)}"}

            # Save initial state
            self._save_tracking_data()

            return self.tracking_data

    def _start_experiment_sync(self) -> Dict[str, Any]:
        """Synchronous fallback when already in async context."""
        print(f"Starting experiment: {self.experiment_name} (ID: {self.experiment_id})")

        # Capture environment state synchronously
        if self.environment_tracker:
            print("Capturing environment state...")
            try:
                self.tracking_data["environment"] = self.environment_tracker.capture_environment(
                    capture_conda=self.config.capture_conda_env,
                    capture_pip=self.config.capture_pip_freeze,
                    capture_system=self.config.capture_system_info
                )
            except Exception as e:
                self.tracking_data["environment"] = {"error": f"Environment capture failed: {str(e)}"}

        # Capture Git state synchronously
        if self.git_tracker:
            print("Capturing Git state...")
            try:
                self.tracking_data["git"] = self.git_tracker.capture_git_state(
                    capture_commit=True,
                    capture_diff=self.config.capture_git_diff,
                    capture_status=self.config.capture_git_status,
                    capture_remote=True
                )
            except Exception as e:
                self.tracking_data["git"] = {"error": f"Git capture failed: {str(e)}"}

        # Capture seed state
        if self.seed_tracker:
            print("Capturing seed state...")
            try:
                self.tracking_data["seeds"] = self.seed_tracker.capture_seeds(
                    track_python=self.config.track_python_seed,
                    track_numpy=self.config.track_numpy_seed,
                    track_torch=self.config.track_torch_seed,
                    track_cuda=self.config.track_cuda_seed
                )
            except Exception as e:
                self.tracking_data["seeds"] = {"error": f"Seed capture failed: {str(e)}"}

        # Save initial state
        self._save_tracking_data()

        return self.tracking_data

    async def track_dataset_async(
        self,
        dataset: Any,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """Async version of track_dataset with timeout and progress."""
        if not self.dataset_tracker:
            raise RuntimeError("Dataset tracking is not enabled")

        if progress_callback:
            progress_callback(f"Starting dataset tracking for {name}")

        tracking_info = await self.dataset_tracker.track_dataset_async(
            dataset, name, metadata, timeout, progress_callback
        )

        if not tracking_info.get("error"):
            self.tracking_data["datasets"][name] = tracking_info
            self._save_tracking_data()

        return tracking_info

    def track_dataset(
        self,
        dataset: Any,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Track a dataset."""
        if not self.dataset_tracker:
            raise RuntimeError("Dataset tracking is not enabled")

        print(f"Tracking dataset: {name}")
        tracking_info = self.dataset_tracker.track_dataset(dataset, name, metadata)
        self.tracking_data["datasets"][name] = tracking_info
        self._save_tracking_data()

        return tracking_info

    async def track_model_async(
        self,
        model: Any,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        save_architecture: Optional[bool] = None,
        save_weights: Optional[bool] = None,
        timeout: Optional[int] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """Async version of track_model with timeout and progress."""
        if not self.model_tracker:
            raise RuntimeError("Model tracking is not enabled")

        if progress_callback:
            progress_callback(f"Starting model tracking for {name}")

        # Use config defaults if not specified
        if save_architecture is None:
            save_architecture = self.config.save_model_architecture
        if save_weights is None:
            save_weights = self.config.save_model_weights

        tracking_info = await self.model_tracker.track_model_async(
            model, name, metadata, save_architecture, save_weights, timeout, progress_callback
        )

        if not tracking_info.get("error"):
            # Save model files if requested and tracking was successful
            if save_architecture:
                try:
                    arch_path = self.model_tracker.save_model_architecture(
                        model, self.output_dir, name
                    )
                    tracking_info["architecture_file"] = str(arch_path)
                except Exception as e:
                    tracking_info["architecture_file_error"] = str(e)

            if save_weights:
                try:
                    weights_path = self.model_tracker.save_model_weights(
                        model, self.output_dir, name
                    )
                    tracking_info["weights_file"] = str(weights_path)
                except Exception as e:
                    tracking_info["weights_file_error"] = str(e)

            self.tracking_data["models"][name] = tracking_info
            self._save_tracking_data()

        return tracking_info

    def track_model(
        self,
        model: Any,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        save_architecture: Optional[bool] = None,
        save_weights: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Synchronous version of track_model for backward compatibility.
        """
        if self.config.enable_async_init:
            # Use async version if enabled
            try:
                asyncio.get_running_loop()
                return self._track_model_sync(model, name, metadata, save_architecture, save_weights)
            except RuntimeError:
                return asyncio.run(
                    self.track_model_async(model, name, metadata, save_architecture, save_weights)
                )
        else:
            if not self.model_tracker:
                raise RuntimeError("Model tracking is not enabled")

            print(f"Tracking model: {name}")

            # Use config defaults if not specified
            if save_architecture is None:
                save_architecture = self.config.save_model_architecture
            if save_weights is None:
                save_weights = self.config.save_model_weights

            tracking_info = self.model_tracker.track_model(
                model, name, metadata, save_architecture, save_weights
            )

            # Save model files if requested
            if save_architecture:
                try:
                    arch_path = self.model_tracker.save_model_architecture(
                        model, self.output_dir, name
                    )
                    tracking_info["architecture_file"] = str(arch_path)
                except Exception as e:
                    tracking_info["architecture_file_error"] = str(e)

            if save_weights:
                try:
                    weights_path = self.model_tracker.save_model_weights(
                        model, self.output_dir, name
                    )
                    tracking_info["weights_file"] = str(weights_path)
                except Exception as e:
                    tracking_info["weights_file_error"] = str(e)

            self.tracking_data["models"][name] = tracking_info
            self._save_tracking_data()

            return tracking_info

    def _track_model_sync(
        self,
        model: Any,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        save_architecture: Optional[bool] = None,
        save_weights: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Synchronous fallback implementation for track_model when event loop is running.
        """
        if not self.model_tracker:
            raise RuntimeError("Model tracking is not enabled")

        print(f"Tracking model: {name}")

        # Use config defaults if not specified
        if save_architecture is None:
            save_architecture = self.config.save_model_architecture
        if save_weights is None:
            save_weights = self.config.save_model_weights

        tracking_info = self.model_tracker.track_model(
            model, name, metadata, save_architecture, save_weights
        )

        # Save model files if requested
        if save_architecture:
            try:
                arch_path = self.model_tracker.save_model_architecture(
                    model, self.output_dir, name
                )
                tracking_info["architecture_file"] = str(arch_path)
            except Exception as e:
                tracking_info["architecture_file_error"] = str(e)

        if save_weights:
            try:
                weights_path = self.model_tracker.save_model_weights(
                    model, self.output_dir, name
                )
                tracking_info["weights_file"] = str(weights_path)
            except Exception as e:
                tracking_info["weights_file_error"] = str(e)

        self.tracking_data["models"][name] = tracking_info
        self._save_tracking_data()

        return tracking_info

    def track_tokenizer(
        self,
        tokenizer: Any,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Track a tokenizer."""
        if not self.model_tracker:
            raise RuntimeError("Model tracking is not enabled")

        print(f"Tracking tokenizer: {name}")
        tracking_info = self.model_tracker.track_tokenizer(tokenizer, name, metadata)

        # Store tokenizer info in models section
        if "tokenizers" not in self.tracking_data["models"]:
            self.tracking_data["models"]["tokenizers"] = {}
        self.tracking_data["models"]["tokenizers"][name] = tracking_info
        self._save_tracking_data()

        return tracking_info

    def set_seeds(self, seed: int) -> Dict[str, Any]:
        """Set seeds for reproducibility."""
        if not self.seed_tracker:
            raise RuntimeError("Seed tracking is not enabled")

        print(f"Setting seeds to: {seed}")
        seed_info = self.seed_tracker.set_seeds(seed)
        self.tracking_data["seeds"] = seed_info
        self._save_tracking_data()

        return seed_info

    def create_reproducible_environment(self, seed: int) -> Dict[str, Any]:
        """Create a fully reproducible environment."""
        if not self.seed_tracker:
            raise RuntimeError("Seed tracking is not enabled")

        print(f"Creating reproducible environment with seed: {seed}")
        seed_info = self.seed_tracker.create_reproducible_environment(seed)
        self.tracking_data["seeds"] = seed_info
        self._save_tracking_data()

        return seed_info

    def _append_metric_entry(
        self,
        name: str,
        value: Any,
        step: Optional[int] = None,
        timestamp: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Create a metric entry and append it to tracking data without persisting."""
        if not name or not str(name).strip():
            raise ValueError("Metric name must be a non-empty string")

        if isinstance(timestamp, datetime):
            metric_timestamp = timestamp.isoformat()
        elif timestamp is None:
            metric_timestamp = datetime.now().isoformat()
        else:
            metric_timestamp = str(timestamp)

        entry: Dict[str, Any] = {
            "name": str(name),
            "value": value,
            "timestamp": metric_timestamp,
        }

        if step is not None:
            entry["step"] = int(step)

        self.tracking_data.setdefault("metrics", []).append(entry)
        return entry

    def log_metric(
        self,
        name: str,
        value: Any,
        *,
        step: Optional[int] = None,
        timestamp: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Log a single metric value and persist tracking data."""

        entry = self._append_metric_entry(name, value, step=step, timestamp=timestamp)
        self._save_tracking_data()
        return entry

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        step: Optional[int] = None,
        timestamp: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Log multiple metrics at once and persist tracking data."""

        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping of metric names to values")

        if not metrics:
            return {"metrics": []}

        entries = [
            self._append_metric_entry(name, value, step=step, timestamp=timestamp)
            for name, value in metrics.items()
        ]
        self._save_tracking_data()
        return {"metrics": entries}

    def add_metadata(self, key: str, value: Any) -> None:
        """Add custom metadata to the experiment."""
        self.tracking_data["metadata"][key] = value
        self._save_tracking_data()

    def add_tag(self, tag: str) -> None:
        """Add a tag to the experiment."""
        if tag not in self.tracking_data["tags"]:
            self.tracking_data["tags"].append(tag)
            self._save_tracking_data()

    def set_notes(self, notes: str) -> None:
        """Set notes for the experiment."""
        self.tracking_data["notes"] = notes
        self._save_tracking_data()

    def get_tracking_summary(self) -> Dict[str, Any]:
        """Get a summary of all tracking data."""
        summary = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "timestamp": self.tracking_data["timestamp"],
            "output_dir": str(self.output_dir),
            "datasets_tracked": len(self.tracking_data["datasets"]),
            "models_tracked": len(self.tracking_data["models"]),
            "has_environment": bool(self.tracking_data["environment"]),
            "has_git": bool(self.tracking_data["git"]),
            "has_seeds": bool(self.tracking_data["seeds"]),
            "tags": self.tracking_data["tags"],
            "metadata_keys": list(self.tracking_data["metadata"].keys())
        }

        # Add component summaries
        if self.dataset_tracker:
            summary["dataset_summary"] = self.dataset_tracker.get_tracking_summary()

        if self.model_tracker:
            summary["model_summary"] = self.model_tracker.get_tracking_summary()

        if self.environment_tracker:
            summary["environment_summary"] = self.environment_tracker.get_tracking_summary()

        if self.seed_tracker:
            summary["seed_summary"] = self.seed_tracker.get_tracking_summary()

        if self.git_tracker:
            summary["git_summary"] = self.git_tracker.get_tracking_summary()

        return summary

    def _save_tracking_data(self) -> None:
        """Save tracking data to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.config.save_to_json:
            json_path = self.output_dir / f"{self.experiment_name}_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(self.tracking_data, f, indent=2, default=str)

            # Also save a latest version
            latest_json_path = self.output_dir / f"{self.experiment_name}_latest.json"
            with open(latest_json_path, 'w') as f:
                json.dump(self.tracking_data, f, indent=2, default=str)

            # Save canonical filename for easy access by external tools
            canonical_json_path = self.output_dir / "experiment.json"
            with open(canonical_json_path, 'w') as f:
                json.dump(self.tracking_data, f, indent=2, default=str)

        if self.config.save_to_yaml:
            yaml_path = self.output_dir / f"{self.experiment_name}_{timestamp}.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump(self.tracking_data, f, default_flow_style=False)

            # Also save a latest version
            latest_yaml_path = self.output_dir / f"{self.experiment_name}_latest.yaml"
            with open(latest_yaml_path, 'w') as f:
                yaml.dump(self.tracking_data, f, default_flow_style=False)

            # Save canonical filename for easy access by external tools
            canonical_yaml_path = self.output_dir / "experiment.yaml"
            with open(canonical_yaml_path, 'w') as f:
                yaml.dump(self.tracking_data, f, default_flow_style=False)

        # Save to Weights & Biases if enabled
        if self.config.save_to_wandb:
            self._save_to_wandb()

    def _save_to_wandb(self) -> None:
        """Save tracking data to Weights & Biases."""
        flag_value = os.environ.get("RLDK_NON_INTERACTIVE", "").strip().lower()
        non_interactive_flag = flag_value in {"1", "true", "yes", "on"}
        stdin_isatty = getattr(sys.stdin, "isatty", lambda: False)
        is_non_interactive = non_interactive_flag or not stdin_isatty()

        if is_non_interactive:
            os.environ.setdefault("WANDB_MODE", "offline")
            os.environ.setdefault("WANDB_SILENT", "true")

        try:
            import wandb
        except ImportError:
            logger.warning("wandb not available, skipping wandb logging")
            return
        except Exception as import_error:
            logger.warning("Failed to import wandb: %s", import_error)
            return

        if self._wandb_run is None:
            try:
                wandb_run = wandb.init(
                    project=self.config.wandb_project or "rldk-experiments",
                    name=self.experiment_name,
                    id=self.experiment_id,
                    tags=self.tracking_data["tags"],
                    notes=self.tracking_data["notes"],
                    config=self.tracking_data["config"],
                    reinit=True,
                )
            except Exception as init_error:
                logger.warning("Failed to initialize wandb run: %s", init_error)
                return

            if wandb_run is None:
                logger.warning("Skipping wandb logging because initialization did not succeed")
                return

            self._wandb_run = wandb_run

        run = self._wandb_run

        try:
            run.log(self.tracking_data["metadata"])

            summary = self.get_tracking_summary()
            run.summary.update(summary)
        except Exception as log_error:
            logger.warning("Failed to save to wandb: %s", log_error)

    def _serialize_config(self) -> Dict[str, Any]:
        """Serialize config to dictionary."""
        config_dict = {}
        for key, value in self.config.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict

    def finish_experiment(self) -> Dict[str, Any]:
        """Finish the experiment and save final state."""
        print(f"Finishing experiment: {self.experiment_name}")

        # Update timestamp
        self.tracking_data["finished_at"] = datetime.now().isoformat()

        # Save final state
        self._save_tracking_data()

        # Print summary
        summary = self.get_tracking_summary()
        print("\n" + "="*50)
        print("EXPERIMENT TRACKING SUMMARY")
        print("="*50)
        print(f"Experiment: {summary['experiment_name']}")
        print(f"ID: {summary['experiment_id']}")
        print(f"Output Directory: {summary['output_dir']}")
        print(f"Datasets Tracked: {summary['datasets_tracked']}")
        print(f"Models Tracked: {summary['models_tracked']}")
        print(f"Environment Captured: {summary['has_environment']}")
        print(f"Git State Captured: {summary['has_git']}")
        print(f"Seeds Tracked: {summary['has_seeds']}")
        print(f"Tags: {', '.join(summary['tags']) if summary['tags'] else 'None'}")
        print("="*50)

        return summary
