"""
Main experiment tracker that coordinates all tracking components.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Callable

import yaml

from .config import TrackingConfig
from .dataset_tracker import DatasetTracker
from .environment_tracker import EnvironmentTracker
from .git_tracker import GitTracker
from .model_tracker import ModelTracker
from .seed_tracker import SeedTracker
from ..config import settings
from ..utils.runtime import with_timeout


class ExperimentTracker:
    """Main experiment tracker that coordinates all tracking components."""

    def __init__(self, config: TrackingConfig, progress_callback: Optional[Callable[[str], None]] = None):
        self.config = config
        self.experiment_id = config.experiment_id
        self.experiment_name = config.experiment_name
        self.output_dir = config.output_dir
        self.progress_callback = progress_callback
        self._settings = settings
        self._initialized = False

        # Initialize tracking components lazily
        self.dataset_tracker = None
        self.model_tracker = None
        self.environment_tracker = None
        self.seed_tracker = None
        self.git_tracker = None

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
            "metadata": config.metadata.copy(),
            "tags": config.tags.copy(),
            "notes": config.notes
        }

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def initialize_async(self) -> None:
        """Initialize tracking components asynchronously."""
        if self._initialized:
            return

        start_time = time.time()
        self._log_progress("Initializing experiment tracker...")

        try:
            # Initialize components in parallel
            tasks = []
            
            if self.config.enable_dataset_tracking:
                tasks.append(self._init_dataset_tracker())
            
            if self.config.enable_model_tracking:
                tasks.append(self._init_model_tracker())
            
            if self.config.enable_environment_tracking:
                tasks.append(self._init_environment_tracker())
            
            if self.config.enable_seed_tracking:
                tasks.append(self._init_seed_tracker())
            
            if self.config.enable_git_tracking:
                tasks.append(self._init_git_tracker())

            # Wait for all components to initialize
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                # Check for exceptions and log them
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        component_names = ["dataset_tracker", "model_tracker", "environment_tracker", "seed_tracker", "git_tracker"]
                        if i < len(component_names):
                            print(f"Warning: Failed to initialize {component_names[i]}: {result}")
                        else:
                            print(f"Warning: Failed to initialize component {i}: {result}")

            self._initialized = True
            init_time = time.time() - start_time
            self._log_progress(f"Tracker initialized in {init_time:.2f} seconds")

        except Exception as e:
            self._log_progress(f"Initialization failed: {e}")
            raise

    def _log_progress(self, message: str) -> None:
        """Log progress message."""
        if self.progress_callback:
            self.progress_callback(message)
        else:
            print(f"[Tracker] {message}")

    async def _init_dataset_tracker(self) -> None:
        """Initialize dataset tracker asynchronously."""
        self._log_progress("Initializing dataset tracker...")
        self.dataset_tracker = DatasetTracker(self.config.dataset_checksum_algorithm)

    async def _init_model_tracker(self) -> None:
        """Initialize model tracker asynchronously."""
        self._log_progress("Initializing model tracker...")
        self.model_tracker = ModelTracker(self.config.model_fingerprint_algorithm)

    async def _init_environment_tracker(self) -> None:
        """Initialize environment tracker asynchronously."""
        self._log_progress("Initializing environment tracker...")
        self.environment_tracker = EnvironmentTracker()

    async def _init_seed_tracker(self) -> None:
        """Initialize seed tracker asynchronously."""
        self._log_progress("Initializing seed tracker...")
        self.seed_tracker = SeedTracker()

    async def _init_git_tracker(self) -> None:
        """Initialize git tracker asynchronously."""
        self._log_progress("Initializing git tracker...")
        self.git_tracker = GitTracker(self.config.git_repo_path)

    def initialize_sync(self) -> None:
        """Initialize tracking components synchronously (fallback)."""
        if self._initialized:
            return

        start_time = time.time()
        self._log_progress("Initializing experiment tracker (sync mode)...")

        try:
            if self.config.enable_dataset_tracking:
                self._log_progress("Initializing dataset tracker...")
                self.dataset_tracker = DatasetTracker(self.config.dataset_checksum_algorithm)
            
            if self.config.enable_model_tracking:
                self._log_progress("Initializing model tracker...")
                self.model_tracker = ModelTracker(self.config.model_fingerprint_algorithm)
            
            if self.config.enable_environment_tracking:
                self._log_progress("Initializing environment tracker...")
                self.environment_tracker = EnvironmentTracker()
            
            if self.config.enable_seed_tracking:
                self._log_progress("Initializing seed tracker...")
                self.seed_tracker = SeedTracker()
            
            if self.config.enable_git_tracking:
                self._log_progress("Initializing git tracker...")
                self.git_tracker = GitTracker(self.config.git_repo_path)

            self._initialized = True
            init_time = time.time() - start_time
            self._log_progress(f"Tracker initialized in {init_time:.2f} seconds")

        except Exception as e:
            self._log_progress(f"Initialization failed: {e}")
            raise

    def start_experiment(self) -> Dict[str, Any]:
        """Start the experiment and capture initial state."""
        print(f"Starting experiment: {self.experiment_name} (ID: {self.experiment_id})")

        # Initialize components if not already done
        if not self._initialized:
            if self._settings.enable_async_init:
                try:
                    # Check if we're already in an event loop
                    try:
                        loop = asyncio.get_running_loop()
                        # We're in an event loop, create a task
                        loop.create_task(self.initialize_async())
                    except RuntimeError:
                        # No event loop running, use asyncio.run
                        asyncio.run(self.initialize_async())
                except Exception as e:
                    print(f"Async initialization failed, falling back to sync: {e}")
                    self.initialize_sync()
            else:
                self.initialize_sync()

        # Capture environment state
        if self.environment_tracker:
            print("Capturing environment state...")
            self.tracking_data["environment"] = self.environment_tracker.capture_environment(
                capture_conda=self.config.capture_conda_env,
                capture_pip=self.config.capture_pip_freeze,
                capture_system=self.config.capture_system_info
            )

        # Capture Git state
        if self.git_tracker:
            print("Capturing Git state...")
            self.tracking_data["git"] = self.git_tracker.capture_git_state(
                capture_commit=True,
                capture_diff=self.config.capture_git_diff,
                capture_status=self.config.capture_git_status,
                capture_remote=True
            )

        # Capture initial seed state
        if self.seed_tracker:
            print("Capturing seed state...")
            self.tracking_data["seeds"] = self.seed_tracker.capture_seeds(
                track_python=self.config.track_python_seed,
                track_numpy=self.config.track_numpy_seed,
                track_torch=self.config.track_torch_seed,
                track_cuda=self.config.track_cuda_seed
            )

        # Save initial state
        self._save_tracking_data()

        return self.tracking_data

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

    def track_model(
        self,
        model: Any,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        save_architecture: Optional[bool] = None,
        save_weights: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Track a model."""
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
            arch_path = self.model_tracker.save_model_architecture(
                model, self.output_dir, name
            )
            tracking_info["architecture_file"] = str(arch_path)

        if save_weights:
            weights_path = self.model_tracker.save_model_weights(
                model, self.output_dir, name
            )
            tracking_info["weights_file"] = str(weights_path)

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

        if self.config.save_to_yaml:
            yaml_path = self.output_dir / f"{self.experiment_name}_{timestamp}.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump(self.tracking_data, f, default_flow_style=False)

            # Also save a latest version
            latest_yaml_path = self.output_dir / f"{self.experiment_name}_latest.yaml"
            with open(latest_yaml_path, 'w') as f:
                yaml.dump(self.tracking_data, f, default_flow_style=False)

        # Save to Weights & Biases if enabled
        if self.config.save_to_wandb:
            self._save_to_wandb()

    def _save_to_wandb(self) -> None:
        """Save tracking data to Weights & Biases."""
        try:
            import wandb

            # Initialize wandb if not already done
            if not wandb.run:
                wandb.init(
                    project=self.config.wandb_project or "rldk-experiments",
                    name=self.experiment_name,
                    id=self.experiment_id,
                    tags=self.tracking_data["tags"],
                    notes=self.tracking_data["notes"],
                    config=self.tracking_data["config"]
                )

            # Log tracking data
            wandb.log(self.tracking_data["metadata"])

            # Log summaries
            summary = self.get_tracking_summary()
            wandb.summary.update(summary)

        except ImportError:
            print("Warning: wandb not available, skipping wandb logging")
        except Exception as e:
            print(f"Warning: Failed to save to wandb: {e}")

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
