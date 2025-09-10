"""
Environment tracking for capturing system state and dependencies.
"""

import subprocess
import sys
import platform
import json
import hashlib
import site
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import numpy as np

# Use importlib.metadata instead of deprecated pkg_resources
try:
    from importlib.metadata import distributions, version
except ImportError:
    # Fallback for Python < 3.8
    from importlib_metadata import distributions, version


class EnvironmentTracker:
    """Tracks environment state including dependencies and system info."""
    
    def __init__(self):
        self.tracking_info: Dict[str, Any] = {}
    
    def capture_environment(
        self,
        capture_conda: bool = True,
        capture_pip: bool = True,
        capture_system: bool = True
    ) -> Dict[str, Any]:
        """
        Capture comprehensive environment information.
        
        Args:
            capture_conda: Whether to capture conda environment
            capture_pip: Whether to capture pip freeze
            capture_system: Whether to capture system information
            
        Returns:
            Dictionary containing environment information
        """
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
        
        self.tracking_info = env_info
        return env_info
    
    def _capture_conda_environment(self) -> Dict[str, Any]:
        """Capture conda environment information."""
        conda_info = {}
        
        try:
            # Get conda info
            result = subprocess.run(
                ["conda", "info", "--json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                conda_info["info"] = json.loads(result.stdout)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            conda_info["info"] = "conda not available"
        
        try:
            # Get current environment
            result = subprocess.run(
                ["conda", "list", "--json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                conda_info["packages"] = json.loads(result.stdout)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            conda_info["packages"] = "conda list failed"
        
        try:
            # Get environment name
            result = subprocess.run(
                ["conda", "info", "--envs"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                # Parse active environment from output
                lines = result.stdout.strip().split('\n')
                active_env = None
                for line in lines:
                    if line.startswith('*'):
                        active_env = line.split()[0]
                        break
                conda_info["active_environment"] = active_env
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            conda_info["active_environment"] = "unknown"
        
        return conda_info
    
    def _capture_pip_environment(self) -> Dict[str, Any]:
        """Capture pip environment information."""
        pip_info = {}
        
        try:
            # Get pip freeze output
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                pip_info["freeze"] = result.stdout.strip().split('\n')
            else:
                pip_info["freeze"] = f"pip freeze failed: {result.stderr}"
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pip_info["freeze"] = "pip freeze failed"
        
        try:
            # Get pip list with versions
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                pip_info["list"] = json.loads(result.stdout)
            else:
                pip_info["list"] = f"pip list failed: {result.stderr}"
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pip_info["list"] = "pip list failed"
        
        # Get installed packages using importlib.metadata
        try:
            installed_packages = []
            # Get site-packages directories
            site_packages_dirs = site.getsitepackages() + [site.getusersitepackages()]
            
            for dist in distributions():
                # Get the package location properly
                try:
                    # Use the distribution's origin to get the location
                    if hasattr(dist, 'origin') and dist.origin:
                        origin_path = Path(dist.origin)
                        # Find which site-packages directory this package is in
                        for site_dir in site_packages_dirs:
                            if str(origin_path).startswith(site_dir):
                                # Get the package name from the path
                                relative_path = origin_path.relative_to(site_dir)
                                package_name = relative_path.parts[0]
                                
                                # Handle .dist-info directories (metadata only packages)
                                if package_name.endswith('.dist-info'):
                                    # For metadata packages, use the site-packages directory
                                    location = site_dir
                                else:
                                    # For regular packages, use the package directory
                                    location = str(Path(site_dir) / package_name)
                                break
                        else:
                            # Fallback to parent directory
                            location = str(origin_path.parent)
                    elif hasattr(dist, 'files') and dist.files:
                        # Fallback: get from first file in distribution
                        first_file = dist.files[0]
                        if hasattr(first_file, 'locate'):
                            file_path = Path(first_file.locate())
                            # Find which site-packages directory this package is in
                            for site_dir in site_packages_dirs:
                                if str(file_path).startswith(site_dir):
                                    relative_path = file_path.relative_to(site_dir)
                                    package_name = relative_path.parts[0]
                                    
                                    # Handle .dist-info directories (metadata only packages)
                                    if package_name.endswith('.dist-info'):
                                        # For metadata packages, use the site-packages directory
                                        location = site_dir
                                    else:
                                        # For regular packages, use the package directory
                                        location = str(Path(site_dir) / package_name)
                                    break
                            else:
                                location = str(file_path.parent)
                        else:
                            location = "unknown"
                    else:
                        location = "unknown"
                except Exception:
                    location = "unknown"
                
                installed_packages.append({
                    "name": dist.metadata["Name"],
                    "version": dist.version,
                    "location": location
                })
            pip_info["installed_packages"] = installed_packages
        except Exception as e:
            pip_info["installed_packages"] = f"Error getting installed packages: {str(e)}"
        
        return pip_info
    
    def _capture_system_info(self) -> Dict[str, Any]:
        """Capture system information."""
        system_info = {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "hostname": platform.node(),
            "python_implementation": platform.python_implementation(),
            "python_compiler": platform.python_compiler(),
        }
        
        # Get CPU info
        try:
            import psutil
            system_info["cpu"] = {
                "count": psutil.cpu_count(),
                "count_logical": psutil.cpu_count(logical=True),
                "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                "usage_percent": psutil.cpu_percent(interval=1)
            }
        except ImportError:
            system_info["cpu"] = "psutil not available"
        
        # Get memory info
        try:
            import psutil
            memory = psutil.virtual_memory()
            system_info["memory"] = {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used,
                "free": memory.free
            }
        except ImportError:
            system_info["memory"] = "psutil not available"
        
        # Get disk info
        try:
            import psutil
            disk = psutil.disk_usage('/')
            system_info["disk"] = {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100
            }
        except ImportError:
            system_info["disk"] = "psutil not available"
        
        return system_info
    
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
    
    def get_tracking_summary(self) -> Dict[str, Any]:
        """Get summary of environment tracking."""
        return self.tracking_info