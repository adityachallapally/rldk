"""Adapter for Weights & Biases (wandb) runs."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .base import BaseAdapter

logger = logging.getLogger(__name__)


class WandBAdapter(BaseAdapter):
    """Adapter for Weights & Biases runs."""

    def __init__(self, source: str):
        super().__init__(source)
        if not WANDB_AVAILABLE:
            raise ImportError("wandb package is required for WandBAdapter")

        # Parse wandb://entity/project/run_id format
        try:
            self.entity, self.project, self.run_id = self._parse_wandb_uri(source)
        except ValueError:
            self.entity = None
            self.project = None
            self.run_id = None

    def _parse_wandb_uri(self, uri: str) -> tuple[str, str, str]:
        """Parse wandb://entity/project/run_id format."""
        if not uri.startswith("wandb://"):
            raise ValueError("WandB URI must start with 'wandb://'")

        # Remove wandb:// prefix
        path = uri[8:]
        parts = path.split("/")

        if len(parts) != 3:
            raise ValueError(
                "WandB URI must be in format: wandb://entity/project/run_id"
            )

        return parts[0], parts[1], parts[2]

    def can_handle(self) -> bool:
        """Check if source is a valid wandb URI."""
        return (
            self.entity is not None
            and self.project is not None
            and self.run_id is not None
        )

    def load(self) -> pd.DataFrame:
        """Load wandb run data and convert to standard format."""
        if not self.can_handle():
            raise ValueError(f"Cannot handle source: {self.source}")

        try:
            # Initialize wandb API
            api = wandb.Api()

            # Get the run
            run = api.run(f"{self.entity}/{self.project}/{self.run_id}")

            # Get run history
            history = run.history()

            if history.empty:
                raise ValueError(f"No history data found for run {self.run_id}")

            # Convert to standard format
            metrics = self._convert_wandb_history(history, run)

            return pd.DataFrame(metrics)

        except Exception as e:
            # Try to fall back to local export if W&B API fails
            if "authentication" in str(e).lower() or "credentials" in str(e).lower():
                return self._load_from_local_export()
            raise ValueError(f"Failed to load wandb run {self.run_id}: {e}")

    def _load_from_local_export(self) -> pd.DataFrame:
        """Fallback to local W&B export files."""
        # Look for common local export patterns
        local_patterns = [
            f"wandb-export-{self.run_id}.jsonl",
            f"{self.run_id}-export.jsonl",
            f"wandb-{self.run_id}.jsonl",
        ]

        for pattern in local_patterns:
            if Path(pattern).exists():
                try:
                    return pd.read_json(pattern, lines=True)
                except Exception as e:
                    logger.debug(f"Failed to read local export file '{pattern}': {e}")
                    continue

        raise ValueError(
            f"No local export found for run {self.run_id}. Please export from W&B or check credentials."
        )

    def _convert_wandb_history(
        self, history: pd.DataFrame, run
    ) -> List[Dict[str, Any]]:
        """Convert wandb history to standard metrics format."""
        metrics = []

        # Get run metadata
        run_metadata = {
            "run_id": run.id,
            "git_sha": run.commit,
            "seed": self._extract_seed_from_config(run.config),
        }

        for idx, row in history.iterrows():
            metric = {
                "step": row.get("_step", idx),
                "phase": "train",  # Default phase
                "reward_mean": row.get("reward_mean") or row.get("reward"),
                "reward_std": row.get("reward_std"),
                "kl_mean": row.get("kl_mean") or row.get("kl_div") or row.get("kl"),
                "entropy_mean": row.get("entropy_mean") or row.get("entropy"),
                "clip_frac": row.get("clip_frac") or row.get("clipped_ratio"),
                "grad_norm": row.get("grad_norm") or row.get("gradient_norm"),
                "lr": row.get("lr") or row.get("learning_rate"),
                "loss": row.get("loss") or row.get("total_loss"),
                "tokens_in": row.get("tokens_in") or row.get("input_tokens"),
                "tokens_out": row.get("tokens_out") or row.get("output_tokens"),
                # _runtime is in seconds from W&B
                "wall_time": row.get("_runtime") if "_runtime" in row else None,
                "seed": run_metadata["seed"],
                "run_id": run_metadata["run_id"],
                "git_sha": run_metadata["git_sha"],
            }

            # Clean up None values
            metric = {k: v for k, v in metric.items() if v is not None}
            metrics.append(metric)

        return metrics

    def _extract_seed_from_config(self, config: Dict[str, Any]) -> Optional[int]:
        """Extract random seed from wandb run config."""
        seed_keys = ["seed", "random_seed", "rng_seed", "torch_seed"]

        for key in seed_keys:
            if key in config:
                try:
                    return int(config[key])
                except (ValueError, TypeError):
                    continue

        return None

    def get_metadata(self) -> dict:
        """Get metadata about the wandb run."""
        try:
            api = wandb.Api()
            run = api.run(f"{self.entity}/{self.project}/{self.run_id}")

            return {
                "run_id": run.id,
                "name": run.name,
                "state": run.state,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "config": run.config,
                "tags": run.tags,
                "git_sha": run.commit,
            }
        except Exception:
            return {}
