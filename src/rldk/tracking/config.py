"""
Configuration classes for the tracking system.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import settings


@dataclass
class TrackingConfig:
    """Configuration for experiment tracking."""

    # Base configuration
    experiment_name: str
    experiment_id: Optional[str] = None
    output_dir: Path = field(default_factory=lambda: settings.runs_dir)

    # Tracking components to enable
    enable_dataset_tracking: bool = True
    enable_model_tracking: bool = True
    enable_environment_tracking: bool = True
    enable_seed_tracking: bool = True
    enable_git_tracking: bool = True

    # Dataset tracking options
    dataset_checksum_algorithm: str = "sha256"
    dataset_cache_dir: Optional[Path] = None

    # Model tracking options
    model_fingerprint_algorithm: str = "sha256"
    save_model_architecture: bool = True
    save_model_weights: bool = False  # Usually too large

    # Environment tracking options
    capture_conda_env: bool = True
    capture_pip_freeze: bool = True
    capture_system_info: bool = True

    # Seed tracking options
    track_numpy_seed: bool = True
    track_torch_seed: bool = True
    track_python_seed: bool = True
    track_cuda_seed: bool = True

    # Git tracking options
    git_repo_path: Optional[Path] = None
    capture_git_diff: bool = True
    capture_git_status: bool = True

    # Output options
    save_to_json: bool = True
    save_to_yaml: bool = True
    save_to_wandb: bool = field(default_factory=lambda: settings.wandb_enabled)
    wandb_project: Optional[str] = field(default_factory=lambda: settings.wandb_project)

    # Additional metadata
    tags: List[str] = field(default_factory=lambda: settings.wandb_tags.copy())
    notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.experiment_id is None:
            import uuid
            self.experiment_id = str(uuid.uuid4())

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set default dataset cache dir
        if self.dataset_cache_dir is None:
            self.dataset_cache_dir = self.output_dir / "dataset_cache"

        # Set default git repo path
        if self.git_repo_path is None:
            self.git_repo_path = Path.cwd()
