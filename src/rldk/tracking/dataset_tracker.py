"""
Dataset tracking for versioning and checksums.
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import Dataset as TorchDataset


class DatasetTracker:
    """Tracks dataset versioning, checksums, and metadata."""

    def __init__(self, algorithm: str = "sha256"):
        self.algorithm = algorithm
        self.tracked_datasets: Dict[str, Dict[str, Any]] = {}

    def track_dataset(
        self,
        dataset: Union[Dataset, DatasetDict, TorchDataset, np.ndarray, pd.DataFrame, str, Path],
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Track a dataset and compute its fingerprint.

        Args:
            dataset: The dataset to track
            name: Name identifier for the dataset
            metadata: Additional metadata to store

        Returns:
            Dictionary containing tracking information
        """
        tracking_info = {
            "name": name,
            "type": type(dataset).__name__,
            "algorithm": self.algorithm,
            "metadata": metadata or {},
            "timestamp": self._get_timestamp()
        }

        # Compute checksum based on dataset type
        if isinstance(dataset, (str, Path)):
            tracking_info.update(self._track_file_dataset(dataset))
        elif isinstance(dataset, (Dataset, DatasetDict)):
            tracking_info.update(self._track_huggingface_dataset(dataset))
        elif isinstance(dataset, TorchDataset):
            tracking_info.update(self._track_torch_dataset(dataset))
        elif isinstance(dataset, np.ndarray):
            tracking_info.update(self._track_numpy_dataset(dataset))
        elif isinstance(dataset, pd.DataFrame):
            tracking_info.update(self._track_pandas_dataset(dataset))
        else:
            tracking_info.update(self._track_generic_dataset(dataset))

        self.tracked_datasets[name] = tracking_info
        return tracking_info

    def _track_file_dataset(self, dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """Track a dataset stored as files."""
        path = Path(dataset_path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")

        info = {
            "path": str(path.absolute()),
            "size_bytes": self._get_file_size(path),
            "file_type": path.suffix
        }

        if path.is_file():
            info["checksum"] = self._compute_file_checksum(path)
        elif path.is_dir():
            info["checksum"] = self._compute_directory_checksum(path)
            info["file_count"] = len(list(path.rglob("*")))

        return info

    def _track_huggingface_dataset(self, dataset: Union[Dataset, DatasetDict]) -> Dict[str, Any]:
        """Track a Hugging Face dataset."""
        info = {
            "num_rows": len(dataset) if isinstance(dataset, Dataset) else sum(len(split) for split in dataset.values()),
            "features": list(dataset.features.keys()) if isinstance(dataset, Dataset) else list(dataset.column_names.keys()),
        }

        if isinstance(dataset, DatasetDict):
            info["splits"] = list(dataset.keys())
            info["split_sizes"] = {split: len(ds) for split, ds in dataset.items()}

        # Compute checksum from a sample of the data
        info["checksum"] = self._compute_dataset_checksum(dataset)

        return info

    def _track_torch_dataset(self, dataset: TorchDataset) -> Dict[str, Any]:
        """Track a PyTorch dataset."""
        info = {
            "num_samples": len(dataset),
            "checksum": self._compute_torch_dataset_checksum(dataset)
        }

        # Try to get additional info if available
        if hasattr(dataset, 'data') and hasattr(dataset, 'targets'):
            info["data_shape"] = getattr(dataset.data, 'shape', None)
            info["targets_shape"] = getattr(dataset.targets, 'shape', None)

        return info

    def _track_numpy_dataset(self, dataset: np.ndarray) -> Dict[str, Any]:
        """Track a NumPy array dataset."""
        return {
            "shape": dataset.shape,
            "dtype": str(dataset.dtype),
            "size_bytes": dataset.nbytes,
            "checksum": self._compute_numpy_checksum(dataset)
        }

    def _track_pandas_dataset(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Track a Pandas DataFrame dataset."""
        return {
            "shape": dataset.shape,
            "columns": list(dataset.columns),
            "dtypes": dataset.dtypes.to_dict(),
            "size_bytes": dataset.memory_usage(deep=True).sum(),
            "checksum": self._compute_pandas_checksum(dataset)
        }

    def _track_generic_dataset(self, dataset: Any) -> Dict[str, Any]:
        """Track a generic dataset by serializing it."""
        try:
            serialized = pickle.dumps(dataset)
            return {
                "size_bytes": len(serialized),
                "checksum": hashlib.sha256(serialized).hexdigest()
            }
        except Exception as e:
            return {
                "error": f"Could not serialize dataset: {str(e)}",
                "checksum": "unknown"
            }

    def _compute_file_checksum(self, file_path: Path) -> str:
        """Compute checksum of a single file."""
        hash_obj = hashlib.new(self.algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()

    def _compute_directory_checksum(self, dir_path: Path) -> str:
        """Compute checksum of a directory by hashing sorted list of relative paths plus file contents."""
        hash_obj = hashlib.new(self.algorithm)

        # Get all files and sort by relative path for deterministic ordering
        all_files = []
        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(dir_path)
                all_files.append((str(rel_path), file_path))

        # Sort by relative path for consistent ordering
        all_files.sort(key=lambda x: x[0])

        for rel_path_str, file_path in all_files:
            # Include relative path in hash
            hash_obj.update(rel_path_str.encode())

            # Include file content
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)

        return hash_obj.hexdigest()

    def _compute_dataset_checksum(self, dataset: Union[Dataset, DatasetDict]) -> str:
        """Compute checksum of a Hugging Face dataset."""
        hash_obj = hashlib.new(self.algorithm)

        if isinstance(dataset, DatasetDict):
            # Hash each split
            for split_name, split_dataset in dataset.items():
                hash_obj.update(split_name.encode())
                hash_obj.update(self._compute_dataset_checksum(split_dataset).encode())
        else:
            # Sample a subset for large datasets using deterministic sampling by stride
            sample_size = min(1000, len(dataset))
            if len(dataset) > sample_size:
                # Use deterministic sampling: take every nth element by stride
                # Ensure we get exactly sample_size items
                step = max(1, len(dataset) // sample_size)
                sample_indices = []
                for i in range(0, len(dataset), step):
                    if len(sample_indices) >= sample_size:
                        break
                    sample_indices.append(i)
                # If we still need more samples, fill from the end
                while len(sample_indices) < sample_size and len(sample_indices) < len(dataset):
                    sample_indices.append(len(dataset) - 1 - len(sample_indices))
            else:
                # Use all indices if dataset is small
                sample_indices = list(range(len(dataset)))

            for idx in sample_indices:
                sample = dataset[idx]
                # Convert to JSON string for consistent hashing
                sample_str = json.dumps(sample, sort_keys=True, default=str)
                hash_obj.update(sample_str.encode())

        return hash_obj.hexdigest()

    def _compute_torch_dataset_checksum(self, dataset: TorchDataset) -> str:
        """Compute checksum of a PyTorch dataset."""
        hash_obj = hashlib.new(self.algorithm)

        # Sample a subset for large datasets using deterministic sampling by stride
        sample_size = min(100, len(dataset))
        if len(dataset) > sample_size:
            # Use deterministic sampling: take every nth element by stride
            # Ensure we get exactly sample_size items
            step = max(1, len(dataset) // sample_size)
            sample_indices = []
            for i in range(0, len(dataset), step):
                if len(sample_indices) >= sample_size:
                    break
                sample_indices.append(i)
            # If we still need more samples, fill from the end
            while len(sample_indices) < sample_size and len(sample_indices) < len(dataset):
                sample_indices.append(len(dataset) - 1 - len(sample_indices))
        else:
            # Use all indices if dataset is small
            sample_indices = list(range(len(dataset)))

        for idx in sample_indices:
            sample = dataset[idx]
            if isinstance(sample, (tuple, list)):
                # Handle tuple/list samples
                for item in sample:
                    if isinstance(item, torch.Tensor):
                        hash_obj.update(item.detach().cpu().numpy().tobytes())
                    else:
                        hash_obj.update(str(item).encode())
            elif isinstance(sample, torch.Tensor):
                hash_obj.update(sample.detach().cpu().numpy().tobytes())
            else:
                hash_obj.update(str(sample).encode())

        return hash_obj.hexdigest()

    def _compute_numpy_checksum(self, dataset: np.ndarray) -> str:
        """Compute checksum of a NumPy array."""
        # For large arrays, sample a subset deterministically by stride
        if dataset.size > 1000000:  # 1M elements
            flat = dataset.flatten()
            sample_size = min(100000, len(flat))
            # Use deterministic sampling: take every nth element by stride
            # Ensure we get exactly sample_size items
            step = max(1, len(flat) // sample_size)
            sample_indices = []
            for i in range(0, len(flat), step):
                if len(sample_indices) >= sample_size:
                    break
                sample_indices.append(i)
            # If we still need more samples, fill from the end
            while len(sample_indices) < sample_size and len(sample_indices) < len(flat):
                sample_indices.append(len(flat) - 1 - len(sample_indices))
            sample = flat[sample_indices]
        else:
            sample = dataset.flatten()

        return hashlib.sha256(sample.tobytes()).hexdigest()

    def _compute_pandas_checksum(self, dataset: pd.DataFrame) -> str:
        """Compute checksum of a Pandas DataFrame."""
        # For large DataFrames, sample a subset deterministically by stride
        if len(dataset) > 10000:
            sample_size = min(1000, len(dataset))
            # Use deterministic sampling: take every nth element by stride
            # Ensure we get exactly sample_size items
            step = max(1, len(dataset) // sample_size)
            sample_indices = []
            for i in range(0, len(dataset), step):
                if len(sample_indices) >= sample_size:
                    break
                sample_indices.append(i)
            # If we still need more samples, fill from the end
            while len(sample_indices) < sample_size and len(sample_indices) < len(dataset):
                sample_indices.append(len(dataset) - 1 - len(sample_indices))
            sample_df = dataset.iloc[sample_indices]
        else:
            sample_df = dataset

        # Use deterministic JSON serialization instead of to_string()
        # This ensures consistent representation across pandas versions
        try:
            # Convert to dict with sorted columns for deterministic ordering
            df_dict = sample_df.to_dict('records')
            df_str = json.dumps(df_dict, sort_keys=True, default=str)
        except Exception:
            # Fallback to to_string() if JSON serialization fails
            df_str = sample_df.to_string()

        return hashlib.sha256(df_str.encode()).hexdigest()

    def _get_file_size(self, path: Path) -> int:
        """Get total size of file or directory."""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return 0

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

    def get_tracking_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked datasets."""
        return {
            "total_datasets": len(self.tracked_datasets),
            "datasets": self.tracked_datasets,
            "algorithm": self.algorithm
        }
