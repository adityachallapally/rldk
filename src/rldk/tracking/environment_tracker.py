"""
Environment tracking for capturing system state and dependencies.
"""

import asyncio
import hashlib
import json
import platform
import subprocess
import sys
from typing import Any, Dict, Optional

try:
    from importlib import metadata
except ImportError:
    metadata = None
import numpy as np
import torch

from .cache import TrackingCache


class EnvironmentTracker:
    """Tracks environment state including dependencies and system info."""

    def __init__(self, config=None):
        self.config = config
        self.tracking_info: Dict[str, Any] = {}
        self._cache = TrackingCache(
            cache_dir=config.dataset_cache_dir / "environment" if config and config.dataset_cache_dir else None,
            ttl=config.cache_timeout if config else 3600,
            max_memory_mb=int((config.max_memory_gb if config else 2.0) * 1024 * 0.1)  # 10% of total memory for env cache
        ) if config else None

    async def capture_environment_async(
        self,
        capture_conda: bool = True,
        capture_pip: bool = True,
        capture_system: bool = True,
        timeout: Optional[int] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Async version of capture_environment with caching and timeout.
        """
        timeout = timeout or (self.config.environment_timeout if self.config else 30)

        if self._cache:
            cached_result = await self._cache.get_async("environment")
            if cached_result:
                if progress_callback:
                    progress_callback("Using cached environment information")
                return cached_result

        if progress_callback:
            progress_callback("Starting environment capture...")

        try:
            tasks = []
            task_names = []

            if capture_conda:
                tasks.append(self._capture_conda_environment_async())
                task_names.append("conda")
            if capture_pip:
                tasks.append(self._capture_pip_environment_async())
                task_names.append("pip")
            if capture_system:
                tasks.append(self._capture_system_info_async())
                task_names.append("system")

            tasks.append(self._capture_ml_frameworks_async())
            task_names.append("ml_frameworks")

            if progress_callback:
                progress_callback(f"Running {len(tasks)} environment capture tasks...")

            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )

            env_info = {
                "timestamp": self._get_timestamp(),
                "python_version": sys.version,
                "python_executable": sys.executable
            }

            for i, (task_name, result) in enumerate(zip(task_names, results)):
                if not isinstance(result, Exception):
                    env_info[task_name] = result
                else:
                    env_info[task_name] = {"error": str(result)}
                    if progress_callback:
                        progress_callback(f"Warning: {task_name} capture failed: {str(result)}")

            # Compute environment fingerprint
            env_info["environment_checksum"] = self._compute_environment_checksum(env_info)

            if self._cache:
                await self._cache.set_async("environment", env_info)

            self.tracking_info = env_info
            return env_info

        except asyncio.TimeoutError:
            return {
                "timestamp": self._get_timestamp(),
                "error": f"Environment capture timed out after {timeout}s",
                "python_version": sys.version,
                "python_executable": sys.executable
            }

    def capture_environment(
        self,
        capture_conda: bool = True,
        capture_pip: bool = True,
        capture_system: bool = True
    ) -> Dict[str, Any]:
        """
        Synchronous version of capture_environment for backward compatibility.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return self._capture_environment_sync(capture_conda, capture_pip, capture_system)
            else:
                return loop.run_until_complete(
                    self.capture_environment_async(capture_conda, capture_pip, capture_system)
                )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                self.capture_environment_async(capture_conda, capture_pip, capture_system)
            )

    async def _capture_conda_environment_async(self) -> Dict[str, Any]:
        """Async version of conda environment capture."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self._capture_conda_environment
        )

    def _capture_conda_environment(self) -> Dict[str, Any]:
        """Capture conda environment information (lightweight version)."""
        conda_info = {}

        try:
            result = subprocess.run(
                ["conda", "--version"],
                capture_output=True,
                text=True,
                timeout=5  # Very short timeout
            )
            if result.returncode == 0:
                conda_info["version"] = result.stdout.strip()
                conda_info["available"] = True
            else:
                conda_info["available"] = False
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            conda_info["available"] = False

        conda_info["note"] = "Lightweight mode - detailed conda info skipped for performance"

        return conda_info

    async def _capture_pip_environment_async(self) -> Dict[str, Any]:
        """Async version of pip environment capture."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self._capture_pip_environment
        )

    def _capture_pip_environment(self) -> Dict[str, Any]:
        """Capture pip environment information (lightweight version)."""
        pip_info = {}

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=10  # Reduced timeout
            )
            if result.returncode == 0:
                pip_info["freeze"] = result.stdout.strip().split('\n')
            else:
                pip_info["freeze"] = f"pip freeze failed: {result.stderr}"
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pip_info["freeze"] = "pip freeze failed"

        pip_info["note"] = "Lightweight mode - full package metadata skipped for performance"

        return pip_info

    async def _capture_system_info_async(self) -> Dict[str, Any]:
        """Async version of system info capture."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self._capture_system_info
        )

    def _capture_system_info(self) -> Dict[str, Any]:
        """Capture system information (lightweight version)."""
        system_info = {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
        }

        system_info["note"] = "Lightweight mode - detailed system metrics skipped for performance"

        return system_info

    async def _capture_ml_frameworks_async(self) -> Dict[str, Any]:
        """Async version of ML frameworks capture."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self._capture_ml_frameworks
        )

    def _capture_ml_frameworks(self) -> Dict[str, Any]:
        """Capture ML framework versions and configurations."""
        frameworks = {}

        # PyTorch
        try:
            frameworks["torch"] = {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None
            }

            if torch.cuda.is_available():
                frameworks["torch"]["device_properties"] = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    frameworks["torch"]["device_properties"].append({
                        "name": props.name,
                        "major": props.major,
                        "minor": props.minor,
                        "total_memory": props.total_memory,
                        "multi_processor_count": props.multi_processor_count
                    })
        except Exception as e:
            frameworks["torch"] = f"Error getting torch info: {str(e)}"

        # NumPy
        try:
            frameworks["numpy"] = {
                "version": np.__version__,
                "blas_info": np.show_config() if hasattr(np, 'show_config') else "not available"
            }
        except Exception as e:
            frameworks["numpy"] = f"Error getting numpy info: {str(e)}"

        # Transformers
        try:
            import transformers
            frameworks["transformers"] = {
                "version": transformers.__version__
            }
        except ImportError:
            frameworks["transformers"] = "not installed"
        except Exception as e:
            frameworks["transformers"] = f"Error getting transformers info: {str(e)}"

        # Datasets
        try:
            import datasets
            frameworks["datasets"] = {
                "version": datasets.__version__
            }
        except ImportError:
            frameworks["datasets"] = "not installed"
        except Exception as e:
            frameworks["datasets"] = f"Error getting datasets info: {str(e)}"

        # Scikit-learn
        try:
            import sklearn
            frameworks["sklearn"] = {
                "version": sklearn.__version__
            }
        except ImportError:
            frameworks["sklearn"] = "not installed"
        except Exception as e:
            frameworks["sklearn"] = f"Error getting sklearn info: {str(e)}"

        return frameworks

    def _compute_environment_checksum(self, env_info: Dict[str, Any]) -> str:
        """Compute checksum of environment information."""
        # Create a simplified version for hashing
        hash_info = {
            "python_version": env_info.get("python_version"),
            "system": env_info.get("system", {}).get("platform"),
            "pip_packages": env_info.get("pip", {}).get("freeze", []),
            "ml_frameworks": {
                name: info.get("version") if isinstance(info, dict) else str(info)
                for name, info in env_info.get("ml_frameworks", {}).items()
            }
        }

        # Convert to JSON string and hash
        json_str = json.dumps(hash_info, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

    def _capture_environment_sync(
        self,
        capture_conda: bool = True,
        capture_pip: bool = True,
        capture_system: bool = True
    ) -> Dict[str, Any]:
        """
        Synchronous fallback when already in async context.
        """
        if self._cache:
            cached_result = self._cache.get("environment")
            if cached_result:
                return cached_result

        env_info = {
            "timestamp": self._get_timestamp(),
            "python_version": sys.version,
            "python_executable": sys.executable
        }

        if capture_conda:
            env_info["conda"] = self._capture_conda_environment()

        if capture_pip:
            env_info["pip"] = self._capture_pip_environment()

        if capture_system:
            env_info["system"] = self._capture_system_info()

        # Capture ML framework versions
        env_info["ml_frameworks"] = self._capture_ml_frameworks()

        # Compute environment fingerprint
        env_info["environment_checksum"] = self._compute_environment_checksum(env_info)

        if self._cache:
            self._cache.set("environment", env_info)

        self.tracking_info = env_info
        return env_info

    def get_tracking_summary(self) -> Dict[str, Any]:
        """Get summary of environment tracking."""
        return self.tracking_info
