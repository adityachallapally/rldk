"""
Dataset tracking for versioning and checksums.
"""

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import Dataset as TorchDataset

from ..config import settings
from ..utils.runtime import with_timeout


class DatasetTracker:
    """Tracks dataset versioning, checksums, and metadata."""

    def __init__(self, algorithm: str = "sha256"):
        self.algorithm = algorithm
        self.tracked_datasets: Dict[str, Dict[str, Any]] = {}
        self._settings = settings
        self._cache_dir = self._settings.get_cache_dir() / "dataset_checksums"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

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
        # Check cache first
        cache_key = self._get_cache_key(dataset, name)
        cached_info = self._load_from_cache(cache_key)
        if cached_info:
            # Update timestamp and metadata
            cached_info["timestamp"] = self._get_timestamp()
            if metadata:
                cached_info["metadata"].update(metadata)
            self.tracked_datasets[name] = cached_info
            return cached_info

        tracking_info = {
            "name": name,
            "type": type(dataset).__name__,
            "algorithm": self.algorithm,
            "metadata": metadata or {},
            "timestamp": self._get_timestamp()
        }

        # Compute checksum based on dataset type
        try:
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
        except Exception as e:
            # If tracking fails, create a minimal tracking info
            tracking_info.update({
                "error": f"Failed to track dataset: {str(e)}",
                "checksum": "unknown"
            })

        # Save to cache
        self._save_to_cache(cache_key, tracking_info)
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
        """Compute checksum of a single file with streaming."""
        hash_obj = hashlib.new(self.algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):  # Larger chunk size for better performance
                hash_obj.update(chunk)
        return hash_obj.hexdigest()

    def _compute_file_checksum_streaming(self, file_path: Path, chunk_size: int = 8192) -> str:
        """Compute checksum of a large file with streaming and progress tracking."""
        hash_obj = hashlib.new(self.algorithm)
        file_size = file_path.stat().st_size
        
        # For very large files, use streaming with progress
        if file_size > 100 * 1024 * 1024:  # 100MB
            with open(file_path, 'rb') as f:
                bytes_read = 0
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    hash_obj.update(chunk)
                    bytes_read += len(chunk)
                    # Optional: Add progress callback here
        else:
            # For smaller files, use regular method
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
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

    @with_timeout(30.0)  # 30 second timeout for dataset checksum
    def _compute_dataset_checksum(self, dataset: Union[Dataset, DatasetDict]) -> str:
        """Compute checksum of a Hugging Face dataset with intelligent sampling."""
        hash_obj = hashlib.new(self.algorithm)

        if isinstance(dataset, DatasetDict):
            # Hash each split
            for split_name, split_dataset in dataset.items():
                hash_obj.update(split_name.encode())
                hash_obj.update(self._compute_dataset_checksum(split_dataset).encode())
        else:
            # Use configurable sample size from settings
            sample_size = min(self._settings.dataset_sample_size, len(dataset))
            
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

            # Process samples in batches to avoid memory issues
            batch_size = min(100, len(sample_indices))
            for i in range(0, len(sample_indices), batch_size):
                batch_indices = sample_indices[i:i + batch_size]
                for idx in batch_indices:
                    try:
                        sample = dataset[idx]
                        # Convert to JSON string for consistent hashing
                        sample_str = json.dumps(sample, sort_keys=True, default=str)
                        hash_obj.update(sample_str.encode())
                    except Exception as e:
                        # If sampling fails, use a fallback hash
                        hash_obj.update(f"error_{idx}_{str(e)}".encode())

        return hash_obj.hexdigest()

    @with_timeout(30.0)  # 30 second timeout for PyTorch dataset checksum
    def _compute_torch_dataset_checksum(self, dataset: TorchDataset) -> str:
        """Compute checksum of a PyTorch dataset with intelligent sampling."""
        hash_obj = hashlib.new(self.algorithm)

        # Use configurable sample size from settings
        sample_size = min(self._settings.dataset_sample_size // 10, len(dataset))  # Smaller sample for PyTorch datasets
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

        # Process samples in batches to avoid memory issues
        batch_size = min(50, len(sample_indices))
        for i in range(0, len(sample_indices), batch_size):
            batch_indices = sample_indices[i:i + batch_size]
            for idx in batch_indices:
                try:
                    sample = dataset[idx]
                    if isinstance(sample, (tuple, list)):
                        # Handle tuple/list samples
                        for item in sample:
                            if isinstance(item, torch.Tensor):
                                # For large tensors, sample a subset
                                if item.numel() > 10000:
                                    flat_item = item.detach().cpu().flatten()
                                    step = max(1, len(flat_item) // 1000)
                                    sampled = flat_item[::step][:1000]
                                    hash_obj.update(sampled.numpy().tobytes())
                                else:
                                    hash_obj.update(item.detach().cpu().numpy().tobytes())
                            else:
                                hash_obj.update(str(item).encode())
                    elif isinstance(sample, torch.Tensor):
                        # For large tensors, sample a subset
                        if sample.numel() > 10000:
                            flat_sample = sample.detach().cpu().flatten()
                            step = max(1, len(flat_sample) // 1000)
                            sampled = flat_sample[::step][:1000]
                            hash_obj.update(sampled.numpy().tobytes())
                        else:
                            hash_obj.update(sample.detach().cpu().numpy().tobytes())
                    else:
                        hash_obj.update(str(sample).encode())
                except Exception as e:
                    # If sampling fails, use a fallback hash
                    hash_obj.update(f"error_{idx}_{str(e)}".encode())

        return hash_obj.hexdigest()

    @with_timeout(30.0)  # 30 second timeout for NumPy checksum
    def _compute_numpy_checksum(self, dataset: np.ndarray) -> str:
        """Compute checksum of a NumPy array with intelligent sampling."""
        # Use configurable sample size from settings
        max_elements = self._settings.dataset_sample_size * 100  # 100x for NumPy arrays
        
        if dataset.size > max_elements:
            flat = dataset.flatten()
            sample_size = min(max_elements, len(flat))
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

    @with_timeout(30.0)  # 30 second timeout for Pandas checksum
    def _compute_pandas_checksum(self, dataset: pd.DataFrame) -> str:
        """Compute checksum of a Pandas DataFrame with intelligent sampling."""
        # Use configurable sample size from settings
        sample_size = min(self._settings.dataset_sample_size, len(dataset))
        
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

    def _get_cache_key(self, dataset: Any, name: str) -> str:
        """Generate cache key for dataset."""
        # Create a simple hash based on dataset type and name
        key_data = f"{type(dataset).__name__}_{name}_{self.algorithm}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load tracking info from cache."""
        cache_file = self._cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _save_to_cache(self, cache_key: str, tracking_info: Dict[str, Any]) -> None:
        """Save tracking info to cache."""
        cache_file = self._cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(tracking_info, f, indent=2, default=str)
        except Exception:
            pass  # Cache failures shouldn't break the main functionality

    def get_tracking_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked datasets."""
        return {
            "total_datasets": len(self.tracked_datasets),
            "datasets": self.tracked_datasets,
            "algorithm": self.algorithm,
            "cache_dir": str(self._cache_dir)
        }
