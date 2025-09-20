"""
Dataset tracking for versioning and checksums.
"""

import asyncio
import hashlib
import json
import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import Dataset as TorchDataset

from .cache import TrackingCache, run_with_timeout_and_progress


class DatasetTracker:
    """Tracks dataset versioning, checksums, and metadata."""

    def __init__(self, algorithm: str = "sha256", config=None):
        self.algorithm = algorithm
        self.config = config
        self.tracked_datasets: Dict[str, Dict[str, Any]] = {}
        self._cache = TrackingCache(
            cache_dir=config.dataset_cache_dir if config else None,
            ttl=config.cache_timeout if config else 3600,
            max_memory_mb=int((config.max_memory_gb if config else 2.0) * 1024 * 0.25)  # 25% of total memory for dataset cache
        ) if config else None

    async def track_dataset_async(
        self,
        dataset: Union[Dataset, DatasetDict, TorchDataset, np.ndarray, pd.DataFrame, str, Path],
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Async version of track_dataset with timeout and progress indicators.
        """
        timeout = timeout or (self.config.tracking_timeout if self.config else 30)

        if self._cache:
            cache_key = f"dataset_{name}_{self._get_dataset_cache_key(dataset)}"
            cached_result = await self._cache.get_async(cache_key)
            if cached_result:
                if progress_callback:
                    progress_callback(f"Using cached result for dataset {name}")
                return cached_result

        if progress_callback:
            progress_callback(f"Starting dataset tracking for {name}")

        try:
            result = await run_with_timeout_and_progress(
                self._track_dataset_internal(dataset, name, metadata, progress_callback),
                timeout=timeout,
                progress_callback=progress_callback,
                error_message=f"Dataset tracking for {name} timed out"
            )

            if self._cache and not result.get("error"):
                cache_key = f"dataset_{name}_{self._get_dataset_cache_key(dataset)}"
                await self._cache.set_async(cache_key, result)

            return result

        except Exception as e:
            return {
                "name": name,
                "error": f"Dataset tracking failed: {str(e)}",
                "checksum": "error"
            }

    def track_dataset(
        self,
        dataset: Union[Dataset, DatasetDict, TorchDataset, np.ndarray, pd.DataFrame, str, Path],
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synchronous version of track_dataset for backward compatibility.
        """
        try:
            asyncio.get_running_loop()
            raise RuntimeError("Cannot run async method from within async context")
        except RuntimeError:
            return asyncio.run(
                self.track_dataset_async(dataset, name, metadata)
            )

    async def _track_dataset_internal(
        self,
        dataset: Union[Dataset, DatasetDict, TorchDataset, np.ndarray, pd.DataFrame, str, Path],
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Internal async dataset tracking implementation.
        """
        tracking_info = {
            "name": name,
            "type": type(dataset).__name__,
            "algorithm": self.algorithm,
            "metadata": metadata or {},
            "timestamp": self._get_timestamp()
        }

        if progress_callback:
            progress_callback(f"Analyzing dataset type: {type(dataset).__name__}")

        # Compute checksum based on dataset type
        if isinstance(dataset, (str, Path)):
            tracking_info.update(await self._track_file_dataset_async(dataset, progress_callback))
        elif isinstance(dataset, (Dataset, DatasetDict)):
            tracking_info.update(await self._track_huggingface_dataset_async(dataset, progress_callback))
        elif isinstance(dataset, TorchDataset):
            tracking_info.update(await self._track_torch_dataset_async(dataset, progress_callback))
        elif isinstance(dataset, np.ndarray):
            tracking_info.update(await self._track_numpy_dataset_async(dataset, progress_callback))
        elif isinstance(dataset, pd.DataFrame):
            tracking_info.update(await self._track_pandas_dataset_async(dataset, progress_callback))
        else:
            tracking_info.update(await self._track_generic_dataset_async(dataset, progress_callback))

        self.tracked_datasets[name] = tracking_info
        return tracking_info

    async def _track_file_dataset_async(self, dataset_path: Union[str, Path], progress_callback=None) -> Dict[str, Any]:
        """Async version of file dataset tracking."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self._track_file_dataset, dataset_path
        )

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

    async def _track_huggingface_dataset_async(self, dataset: Union[Dataset, DatasetDict], progress_callback=None) -> Dict[str, Any]:
        """Async version of Hugging Face dataset tracking."""
        info = {
            "num_rows": len(dataset) if isinstance(dataset, Dataset) else sum(len(split) for split in dataset.values()),
            "features": list(dataset.features.keys()) if isinstance(dataset, Dataset) else list(dataset.column_names.keys()),
        }

        if isinstance(dataset, DatasetDict):
            info["splits"] = list(dataset.keys())
            info["split_sizes"] = {split: len(ds) for split, ds in dataset.items()}

        if progress_callback:
            progress_callback("Computing dataset checksum...")

        # Compute checksum with multiprocessing for large datasets
        info["checksum"] = await self._compute_dataset_checksum_async(dataset, progress_callback)

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

    async def _track_torch_dataset_async(self, dataset: TorchDataset, progress_callback=None) -> Dict[str, Any]:
        """Async version of PyTorch dataset tracking."""
        info = {
            "num_samples": len(dataset),
        }

        if progress_callback:
            progress_callback("Computing PyTorch dataset checksum...")

        info["checksum"] = await self._compute_torch_dataset_checksum_async(dataset, progress_callback)

        # Try to get additional info if available
        if hasattr(dataset, 'data') and hasattr(dataset, 'targets'):
            info["data_shape"] = getattr(dataset.data, 'shape', None)
            info["targets_shape"] = getattr(dataset.targets, 'shape', None)

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

    async def _track_numpy_dataset_async(self, dataset: np.ndarray, progress_callback=None) -> Dict[str, Any]:
        """Async version of NumPy dataset tracking."""
        if progress_callback:
            progress_callback("Computing NumPy array checksum...")

        return {
            "shape": dataset.shape,
            "dtype": str(dataset.dtype),
            "size_bytes": dataset.nbytes,
            "checksum": await self._compute_numpy_checksum_async(dataset, progress_callback)
        }

    def _track_numpy_dataset(self, dataset: np.ndarray) -> Dict[str, Any]:
        """Track a NumPy array dataset."""
        return {
            "shape": dataset.shape,
            "dtype": str(dataset.dtype),
            "size_bytes": dataset.nbytes,
            "checksum": self._compute_numpy_checksum(dataset)
        }

    async def _track_pandas_dataset_async(self, dataset: pd.DataFrame, progress_callback=None) -> Dict[str, Any]:
        """Async version of Pandas dataset tracking."""
        if progress_callback:
            progress_callback("Computing Pandas DataFrame checksum...")

        return {
            "shape": dataset.shape,
            "columns": list(dataset.columns),
            "dtypes": dataset.dtypes.to_dict(),
            "size_bytes": dataset.memory_usage(deep=True).sum(),
            "checksum": await self._compute_pandas_checksum_async(dataset, progress_callback)
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

    async def _track_generic_dataset_async(self, dataset: Any, progress_callback=None) -> Dict[str, Any]:
        """Async version of generic dataset tracking."""
        if progress_callback:
            progress_callback("Computing generic dataset checksum...")

        return await asyncio.get_running_loop().run_in_executor(
            None, self._track_generic_dataset, dataset
        )

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

    async def _compute_dataset_checksum_async(self, dataset: Union[Dataset, DatasetDict], progress_callback=None) -> str:
        """Async version of dataset checksum computation with multiprocessing."""
        if isinstance(dataset, DatasetDict):
            # Hash each split
            hash_obj = hashlib.new(self.algorithm)
            for split_name, split_dataset in dataset.items():
                if progress_callback:
                    progress_callback(f"Processing split: {split_name}")
                hash_obj.update(split_name.encode())
                split_checksum = await self._compute_dataset_checksum_async(split_dataset, progress_callback)
                hash_obj.update(split_checksum.encode())
            return hash_obj.hexdigest()

        sample_size = self.config.dataset_sample_size if self.config else 1000
        dataset_size = len(dataset)

        if dataset_size > sample_size:
            if progress_callback:
                progress_callback(f"Large dataset detected ({dataset_size} samples), using intelligent sampling")

            return await self._compute_large_dataset_checksum_mp(dataset, sample_size, progress_callback)
        else:
            return await asyncio.get_running_loop().run_in_executor(
                None, self._compute_dataset_checksum, dataset
            )

    async def _compute_large_dataset_checksum_mp(self, dataset, sample_size: int, progress_callback=None) -> str:
        """Compute checksum for large datasets using multiprocessing with resource management."""
        import psutil
        import os
        
        try:
            step = max(1, len(dataset) // sample_size)
            sample_indices = list(range(0, len(dataset), step))[:sample_size]

            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            max_workers_by_memory = max(1, int(available_memory_gb / 0.5))  # 0.5GB per worker
            max_workers_by_cpu = min(mp.cpu_count(), 4)
            num_workers = min(max_workers_by_memory, max_workers_by_cpu)
            
            if available_memory_gb < 1.0:
                if progress_callback:
                    progress_callback(f"Low memory ({available_memory_gb:.1f}GB), using single-threaded processing")
                return await asyncio.get_running_loop().run_in_executor(
                    None, self._compute_dataset_checksum, dataset
                )

            chunk_size = max(1, len(sample_indices) // num_workers)
            index_chunks = [sample_indices[i:i + chunk_size] for i in range(0, len(sample_indices), chunk_size)]

            if progress_callback:
                progress_callback(f"Processing {len(sample_indices)} samples using {num_workers} workers (Memory: {available_memory_gb:.1f}GB)")

            executor = None
            try:
                executor = ProcessPoolExecutor(max_workers=num_workers)
                tasks = []
                
                for chunk in index_chunks:
                    chunk_samples = []
                    for idx in chunk:
                        try:
                            sample = dataset[idx]
                            chunk_samples.append(json.dumps(sample, sort_keys=True, default=str))
                        except Exception:
                            continue  # Skip problematic samples

                    if chunk_samples:
                        task = asyncio.get_running_loop().run_in_executor(
                            executor, self._hash_sample_chunk, chunk_samples, self.algorithm
                        )
                        tasks.append(task)

                timeout = 60  # 60 seconds timeout
                chunk_hashes = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)

                final_hash = hashlib.new(self.algorithm)
                for chunk_hash in chunk_hashes:
                    final_hash.update(chunk_hash.encode())

                return final_hash.hexdigest()
                
            finally:
                if executor:
                    executor.shutdown(wait=True)

        except (asyncio.TimeoutError, MemoryError, OSError) as e:
            if progress_callback:
                progress_callback(f"Multiprocessing failed ({type(e).__name__}), falling back to single-threaded")
            # Fallback to single-threaded processing
            return await asyncio.get_running_loop().run_in_executor(
                None, self._compute_dataset_checksum, dataset
            )
        except Exception as e:
            if progress_callback:
                progress_callback(f"Multiprocessing failed, falling back to single-threaded: {str(e)}")
            # Fallback to single-threaded processing
            return await asyncio.get_running_loop().run_in_executor(
                None, self._compute_dataset_checksum, dataset
            )

    @staticmethod
    def _hash_sample_chunk(samples: list, algorithm: str) -> str:
        """Hash a chunk of samples (used in multiprocessing)."""
        hash_obj = hashlib.new(algorithm)
        for sample_str in samples:
            hash_obj.update(sample_str.encode())
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

    async def _compute_torch_dataset_checksum_async(self, dataset: TorchDataset, progress_callback=None) -> str:
        """Async version of PyTorch dataset checksum computation."""
        sample_size = min(100, len(dataset))
        if len(dataset) > sample_size:
            if progress_callback:
                progress_callback(f"Large PyTorch dataset, sampling {sample_size} items")

        return await asyncio.get_running_loop().run_in_executor(
            None, self._compute_torch_dataset_checksum, dataset
        )

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

    async def _compute_numpy_checksum_async(self, dataset: np.ndarray, progress_callback=None) -> str:
        """Async version of NumPy checksum computation."""
        if dataset.size > 1000000:  # 1M elements
            if progress_callback:
                progress_callback(f"Large NumPy array ({dataset.size} elements), using sampling")

        return await asyncio.get_running_loop().run_in_executor(
            None, self._compute_numpy_checksum, dataset
        )

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

    async def _compute_pandas_checksum_async(self, dataset: pd.DataFrame, progress_callback=None) -> str:
        """Async version of Pandas checksum computation."""
        if len(dataset) > 10000:
            if progress_callback:
                progress_callback(f"Large DataFrame ({len(dataset)} rows), using sampling")

        return await asyncio.get_running_loop().run_in_executor(
            None, self._compute_pandas_checksum, dataset
        )

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

    def _get_dataset_cache_key(self, dataset) -> str:
        """Generate a deterministic cache key for a dataset."""
        try:
            if hasattr(dataset, '__len__') and hasattr(dataset, '__getitem__'):
                return f"{type(dataset).__name__}_{len(dataset)}"
            elif hasattr(dataset, 'shape'):
                return f"{type(dataset).__name__}_{dataset.shape}"
            elif isinstance(dataset, (str, Path)):
                path = Path(dataset)
                if path.exists():
                    return f"file_{path.name}_{path.stat().st_mtime}"
                else:
                    return f"file_{path.name}"
            else:
                # Fallback to type and string representation hash
                import hashlib
                content = f"{type(dataset).__name__}_{str(dataset)}"
                return hashlib.md5(content.encode()).hexdigest()[:16]
        except Exception:
            import hashlib
            content = f"{type(dataset).__name__}_{hash(str(type(dataset)))}"
            return hashlib.md5(content.encode()).hexdigest()[:16]
