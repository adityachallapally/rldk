"""File naming conventions for RL Debug Kit outputs."""

from datetime import datetime
from pathlib import Path
from typing import Optional


class FileNamingConventions:
    """Standardized file naming conventions for all RL Debug Kit outputs."""

    # Report types and their standard filenames
    REPORT_TYPES = {
        "drift_card": "drift_card",
        "determinism_card": "determinism_card",
        "reward_card": "reward_card",
        "ppo_scan": "ppo_scan",
        "ckpt_diff": "ckpt_diff",
        "reward_drift": "reward_drift",
        "run_comparison": "run_comparison",
        "eval_summary": "eval_summary",
        "env_audit": "env_audit",
        "replay_comparison": "replay_comparison",
        "tracking_data": "tracking_data",
        "reward_health_summary": "reward_health_summary",
        "golden_master_summary": "golden_master_summary",
    }

    # File extensions for different content types
    EXTENSIONS = {
        "json": ".json",
        "jsonl": ".jsonl",
        "csv": ".csv",
        "markdown": ".md",
        "png": ".png",
        "pdf": ".pdf",
        "txt": ".txt",
    }

    @classmethod
    def get_filename(
        cls,
        report_type: str,
        extension: str = "json",
        timestamp: bool = False,
        run_id: Optional[str] = None,
        suffix: Optional[str] = None
    ) -> str:
        """
        Generate standardized filename for report.

        Args:
            report_type: Type of report (e.g., "drift_card", "determinism_card")
            extension: File extension (e.g., "json", "csv", "md")
            timestamp: Whether to include timestamp in filename
            run_id: Optional run ID to include in filename
            suffix: Optional suffix to append before extension

        Returns:
            Standardized filename
        """
        if report_type not in cls.REPORT_TYPES:
            raise ValueError(f"Unknown report type: {report_type}")

        base_name = cls.REPORT_TYPES[report_type]

        # Add run_id if provided
        if run_id:
            base_name = f"{base_name}_{run_id}"

        # Add timestamp if requested
        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{base_name}_{timestamp_str}"

        # Add suffix if provided
        if suffix:
            base_name = f"{base_name}_{suffix}"

        # Add extension
        if extension not in cls.EXTENSIONS:
            raise ValueError(f"Unknown extension: {extension}")

        return f"{base_name}{cls.EXTENSIONS[extension]}"

    @classmethod
    def get_output_directory(
        cls,
        base_dir: str = "rldk_reports",
        run_id: Optional[str] = None,
        timestamp: bool = False
    ) -> Path:
        """
        Generate standardized output directory path.

        Args:
            base_dir: Base directory name
            run_id: Optional run ID to include in path
            timestamp: Whether to include timestamp in path

        Returns:
            Path object for output directory
        """
        path_parts = [base_dir]

        if run_id:
            path_parts.append(run_id)

        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            path_parts.append(timestamp_str)

        return Path(*path_parts)

    @classmethod
    def get_metrics_filename(
        cls,
        run_id: str,
        phase: str = "train",
        timestamp: bool = False
    ) -> str:
        """
        Generate filename for metrics data.

        Args:
            run_id: Run identifier
            phase: Training phase (train, eval, etc.)
            timestamp: Whether to include timestamp

        Returns:
            Standardized metrics filename
        """
        base_name = f"metrics_{run_id}_{phase}"

        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{base_name}_{timestamp_str}"

        return f"{base_name}.jsonl"

    @classmethod
    def get_checkpoint_filename(
        cls,
        run_id: str,
        step: int,
        timestamp: bool = False
    ) -> str:
        """
        Generate filename for checkpoint data.

        Args:
            run_id: Run identifier
            step: Training step
            timestamp: Whether to include timestamp

        Returns:
            Standardized checkpoint filename
        """
        base_name = f"checkpoint_{run_id}_step_{step:06d}"

        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{base_name}_{timestamp_str}"

        return f"{base_name}.pt"

    @classmethod
    def get_plot_filename(
        cls,
        plot_type: str,
        run_id: Optional[str] = None,
        timestamp: bool = False
    ) -> str:
        """
        Generate filename for plot files.

        Args:
            plot_type: Type of plot (e.g., "calibration", "drift", "metrics")
            run_id: Optional run identifier
            timestamp: Whether to include timestamp

        Returns:
            Standardized plot filename
        """
        base_name = f"plot_{plot_type}"

        if run_id:
            base_name = f"{base_name}_{run_id}"

        if timestamp:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{base_name}_{timestamp_str}"

        return f"{base_name}.png"

    @classmethod
    def validate_filename(cls, filename: str) -> bool:
        """
        Validate that filename follows naming conventions.

        Args:
            filename: Filename to validate

        Returns:
            True if filename is valid, False otherwise
        """
        try:
            path = Path(filename)
            name = path.stem
            ext = path.suffix

            # Check if extension is valid
            if ext not in cls.EXTENSIONS.values():
                return False

            # Check if name contains valid report type
            valid_prefixes = list(cls.REPORT_TYPES.values()) + [
                "metrics", "checkpoint", "plot", "data"
            ]

            return any(name.startswith(prefix) for prefix in valid_prefixes)

        except Exception:
            return False


# Convenience functions for common use cases
def get_drift_card_filename(run_id: Optional[str] = None) -> str:
    """Get standardized drift card filename."""
    return FileNamingConventions.get_filename(
        "drift_card", "json", run_id=run_id
    )


def get_determinism_card_filename(run_id: Optional[str] = None) -> str:
    """Get standardized determinism card filename."""
    return FileNamingConventions.get_filename(
        "determinism_card", "json", run_id=run_id
    )


def get_reward_card_filename(run_id: Optional[str] = None) -> str:
    """Get standardized reward card filename."""
    return FileNamingConventions.get_filename(
        "reward_card", "json", run_id=run_id
    )


def get_metrics_filename(run_id: str, phase: str = "train") -> str:
    """Get standardized metrics filename."""
    return FileNamingConventions.get_metrics_filename(run_id, phase)


def get_plot_filename(plot_type: str, run_id: Optional[str] = None) -> str:
    """Get standardized plot filename."""
    return FileNamingConventions.get_plot_filename(plot_type, run_id)


def get_output_directory(run_id: Optional[str] = None) -> Path:
    """Get standardized output directory."""
    return FileNamingConventions.get_output_directory(run_id=run_id)
