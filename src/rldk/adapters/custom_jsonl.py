"""Custom adapter for our JSONL training logs."""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .base import BaseAdapter


class CustomJSONLAdapter(BaseAdapter):
    """Adapter for our custom JSONL training logs.

    .. deprecated:: 0.1.0
        Use :class:`FlexibleDataAdapter` instead for better field resolution
        and support for multiple formats. This adapter will be removed in a future version.
    """

    def can_handle(self) -> bool:
        """Check if source contains our custom JSONL logs."""
        if not self.source.exists():
            return False

        if self.source.is_file():
            return self._is_custom_jsonl_file(self.source)
        elif self.source.is_dir():
            # Check directory contents for at least one file with the custom schema
            for jsonl_file in self.source.glob("*.jsonl"):
                if self._is_custom_jsonl_file(jsonl_file):
                    return True
            return False

        return False

    def _is_custom_jsonl_file(self, file_path: Path) -> bool:
        """Check if a file contains our custom JSONL logs."""
        try:
            if file_path.suffix == ".jsonl":
                with open(file_path) as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        # Check for our custom schema - be very specific to avoid misclassifying TRL/OpenRLHF
                        # Only check for required keys if data is a dict
                        if isinstance(data, dict):
                            # Check for explicit custom format indicators (not standard format)
                            # These are the unique identifiers of our custom format
                            custom_indicators = [
                                "global_step", "reward_scalar", "kl_to_ref"  # Unique to custom format
                            ]

                            # Must have at least one of the unique custom indicators
                            has_custom_indicators = any(key in data for key in custom_indicators)

                            # Also check that it doesn't look like standard TRL/OpenRLHF format
                            # Standard formats typically have nested metrics or different structure
                            is_standard_format = (
                                "reward" in data and isinstance(data["reward"], dict) or  # Nested metrics
                                "metrics" in data and isinstance(data["metrics"], dict) or  # Nested metrics
                                ("step" in data and "reward_mean" in data and "kl_mean" in data and
                                 "entropy_mean" in data and "clip_frac" in data and "grad_norm" in data)  # Full standard schema
                            )

                            # Only classify as custom if it has custom indicators AND doesn't look like standard format
                            # This is already restrictive enough - requires explicit custom field names
                            return has_custom_indicators and not is_standard_format
                        else:
                            # For non-dict data, it's not our custom format
                            return False
        except (OSError, json.JSONDecodeError, UnicodeDecodeError, TypeError) as e:
            # Log the specific error for debugging but don't fail the check
            print(f"Warning: Error checking if file {file_path} is custom JSONL format: {e}")
            return False
        return False

    def load(self) -> pd.DataFrame:
        """Load our custom JSONL logs and convert to standard format."""
        import warnings
        warnings.warn(
            "CustomJSONLAdapter is deprecated. Use FlexibleDataAdapter instead for better "
            "field resolution and support for multiple formats.",
            DeprecationWarning,
            stacklevel=2
        )

        if not self.can_handle():
            raise ValueError(f"Cannot handle source: {self.source}")

        metrics = []

        if self.source.is_file():
            metrics = self._parse_file(self.source)
        elif self.source.is_dir():
            # Find and parse all JSONL log files
            jsonl_files = list(self.source.glob("*.jsonl"))
            for jsonl_file in jsonl_files:
                metrics.extend(self._parse_file(jsonl_file))

        if not metrics:
            raise ValueError(f"No valid custom JSONL metrics found in {self.source}")

        # Convert to DataFrame
        df = pd.DataFrame(metrics)

        # Ensure required columns exist for the standard format
        required_cols = [
            "step",
            "phase",
            "reward_mean",
            "reward_std",
            "kl_mean",
            "entropy_mean",
            "clip_frac",
            "grad_norm",
            "lr",
            "loss",
            "tokens_in",
            "tokens_out",
            "wall_time",
            "seed",
            "run_id",
            "git_sha",
        ]

        for col in required_cols:
            if col not in df.columns:
                df[col] = None

        return df[required_cols]

    def _parse_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse a single custom JSONL log file."""
        metrics = []

        try:
            if file_path.suffix == ".jsonl":
                with open(file_path) as f:
                    for line_num, line in enumerate(f, 1):  # Start line numbering from 1
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            metric = self._extract_custom_metric(data, line_num)
                            if metric:
                                metrics.append(metric)
                        except json.JSONDecodeError as e:
                            print(f"Warning: JSON decode error in {file_path} at line {line_num}: {e}")
                            continue
        except (OSError, UnicodeDecodeError) as e:
            print(f"Error parsing {file_path}: {e}")
            # Re-raise the exception with context
            raise RuntimeError(f"Failed to parse custom JSONL file {file_path}: {e}") from e

        return metrics

    def _extract_custom_metric(
        self, data: Dict[str, Any], line_num: int
    ) -> Dict[str, Any]:
        """Extract metric from our custom JSONL format."""
        try:
            # Helper function to safely get values with null handling
            def safe_get(key, default=None):
                """Get value from data, handling null values properly."""
                value = data.get(key)
                return value if value is not None else default

            # Map our custom schema to the expected format
            # Handle various possible field names for better compatibility
            # Use 'in' checks to avoid skipping valid zeros, but handle null values
            step = safe_get("global_step") if "global_step" in data and data["global_step"] is not None else safe_get("step", line_num)
            reward_scalar = safe_get("reward_scalar") if "reward_scalar" in data and data["reward_scalar"] is not None else safe_get("reward_mean", 0.0)
            kl_value = safe_get("kl_to_ref") if "kl_to_ref" in data and data["kl_to_ref"] is not None else safe_get("kl_mean", 0.0)
            loss_value = safe_get("loss", 0.0)

            # Extract RNG seed from various possible locations
            # Fix operator precedence by using proper conditional logic
            seed = 42  # Default fallback
            if "rng.python" in data and data["rng.python"] is not None:
                seed = data["rng.python"]
            elif "seed" in data and data["seed"] is not None:
                seed = data["seed"]
            elif "rng" in data and isinstance(data["rng"], dict) and "python" in data["rng"] and data["rng"]["python"] is not None:
                seed = data["rng"]["python"]

            # Extract additional metrics if available
            # Use 'in' checks to avoid skipping valid zeros, but handle null values
            entropy_value = safe_get("entropy") if "entropy" in data and data["entropy"] is not None else safe_get("entropy_mean", 0.0)
            clip_frac_value = safe_get("clip_frac", 0.0)
            grad_norm_value = safe_get("grad_norm", 0.0)
            lr_value = safe_get("lr") if "lr" in data and data["lr"] is not None else safe_get("learning_rate", 0.0)
            reward_std_value = safe_get("reward_std", 0.0)

            # Extract data slice information if available
            tokens_in = safe_get("tokens_in", 0)
            tokens_out = safe_get("tokens_out", 0)

            # Extract wall time if available
            # Use 'in' check to avoid skipping valid zeros, but handle null values
            wall_time = safe_get("wall_time") if "wall_time" in data and data["wall_time"] is not None else safe_get("timestamp", 0.0)

            metric = {
                "step": int(step),
                "phase": safe_get("phase", "train"),
                "reward_mean": float(reward_scalar),
                "reward_std": float(reward_std_value),
                "kl_mean": float(kl_value),
                "entropy_mean": float(entropy_value),
                "clip_frac": float(clip_frac_value),
                "grad_norm": float(grad_norm_value),
                "lr": float(lr_value),
                "loss": float(loss_value),
                "tokens_in": int(tokens_in),
                "tokens_out": int(tokens_out),
                "wall_time": float(wall_time),
                "seed": int(seed),
                "run_id": safe_get("run_id", f"custom_{line_num}"),
                "git_sha": safe_get("git_sha", "unknown"),
            }

            return metric
        except Exception as e:
            print(f"Error extracting metric from line {line_num}: {e}")
            return None
