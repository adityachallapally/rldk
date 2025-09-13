"""
Seed tracking for reproducibility across all components.
"""

import random
import os
import hashlib
import json
from typing import Dict, Any, Optional, List
import numpy as np
import torch


class SeedTracker:
    """Tracks random seeds across all components for reproducibility."""
    
    def __init__(self):
        self.seed_info: Dict[str, Any] = {}
        self.original_states: Dict[str, Any] = {}
    
    def capture_seeds(
        self,
        track_python: bool = True,
        track_numpy: bool = True,
        track_torch: bool = True,
        track_cuda: bool = True
    ) -> Dict[str, Any]:
        """
        Capture current seed states from all components.
        
        Args:
            track_python: Whether to track Python random seed
            track_numpy: Whether to track NumPy random seed
            track_torch: Whether to track PyTorch random seed
            track_cuda: Whether to track CUDA random seed
            
        Returns:
            Dictionary containing seed information
        """
        seed_info = {
            "timestamp": self._get_timestamp(),
            "seeds": {}
        }
        
        if track_python:
            seed_info["seeds"]["python"] = self._capture_python_seed()
        
        if track_numpy:
            seed_info["seeds"]["numpy"] = self._capture_numpy_seed()
        
        if track_torch:
            seed_info["seeds"]["torch"] = self._capture_torch_seed()
        
        if track_cuda:
            seed_info["seeds"]["cuda"] = self._capture_cuda_seed()
        
        # Compute seed fingerprint
        seed_info["seed_checksum"] = self._compute_seed_checksum(seed_info["seeds"])
        
        self.seed_info = seed_info
        return seed_info
    
    def set_seeds(
        self,
        seed: int,
        track_python: bool = True,
        track_numpy: bool = True,
        track_torch: bool = True,
        track_cuda: bool = True
    ) -> Dict[str, Any]:
        """
        Set seeds for all components and track the operation.
        
        Args:
            seed: The seed value to set
            track_python: Whether to set Python random seed
            track_numpy: Whether to set NumPy random seed
            track_torch: Whether to set PyTorch random seed
            track_cuda: Whether to set CUDA random seed
            
        Returns:
            Dictionary containing seed setting information
        """
        seed_info = {
            "timestamp": self._get_timestamp(),
            "set_seed": seed,
            "seeds": {}
        }
        
        if track_python:
            seed_info["seeds"]["python"] = self._set_python_seed(seed)
        
        if track_numpy:
            seed_info["seeds"]["numpy"] = self._set_numpy_seed(seed)
        
        if track_torch:
            seed_info["seeds"]["torch"] = self._set_torch_seed(seed)
        
        if track_cuda:
            seed_info["seeds"]["cuda"] = self._set_cuda_seed(seed)
        
        # Compute seed fingerprint
        seed_info["seed_checksum"] = self._compute_seed_checksum(seed_info["seeds"])
        
        self.seed_info = seed_info
        return seed_info
    
    def _capture_python_seed(self) -> Dict[str, Any]:
        """Capture Python random seed state."""
        return {
            "seed": random.getstate()[1][0] if random.getstate()[0] == 3 else None,
            "state": random.getstate()
        }
    
    def _capture_numpy_seed(self) -> Dict[str, Any]:
        """Capture NumPy random seed state."""
        rng = np.random.get_state()
        return {
            "seed": rng[1][0] if len(rng) > 1 and len(rng[1]) > 0 else None,
            "state": rng
        }
    
    def _capture_torch_seed(self) -> Dict[str, Any]:
        """Capture PyTorch random seed state."""
        return {
            "seed": torch.initial_seed(),
            "state": torch.get_rng_state().tolist()
        }
    
    def _capture_cuda_seed(self) -> Dict[str, Any]:
        """Capture CUDA random seed state."""
        if torch.cuda.is_available():
            return {
                "seed": torch.cuda.initial_seed(),
                "state": torch.cuda.get_rng_state().tolist(),
                "device_count": torch.cuda.device_count()
            }
        else:
            return {
                "available": False,
                "message": "CUDA not available"
            }
    
    def _set_python_seed(self, seed: int) -> Dict[str, Any]:
        """Set Python random seed."""
        random.seed(seed)
        return {
            "seed": seed,
            "state": random.getstate()
        }
    
    def _set_numpy_seed(self, seed: int) -> Dict[str, Any]:
        """Set NumPy random seed."""
        np.random.seed(seed)
        return {
            "seed": seed,
            "state": np.random.get_state()
        }
    
    def _set_torch_seed(self, seed: int) -> Dict[str, Any]:
        """Set PyTorch random seed."""
        torch.manual_seed(seed)
        return {
            "seed": seed,
            "state": torch.get_rng_state().tolist()
        }
    
    def _set_cuda_seed(self, seed: int) -> Dict[str, Any]:
        """Set CUDA random seed."""
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # Set for all devices
            return {
                "seed": seed,
                "state": torch.cuda.get_rng_state().tolist(),
                "device_count": torch.cuda.device_count()
            }
        else:
            return {
                "available": False,
                "message": "CUDA not available"
            }
    
    def save_seed_state(self, output_path: str) -> str:
        """Save current seed state to file."""
        # Get numpy state and convert to JSON-serializable format
        np_state = np.random.get_state()
        np_state_serializable = (np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4])
        
        seed_state = {
            "timestamp": self._get_timestamp(),
            "seed_info": self.seed_info,
            "python_state": random.getstate(),
            "numpy_state": np_state_serializable,
            "torch_state": torch.get_rng_state().tolist(),
            "cuda_state": torch.cuda.get_rng_state().tolist() if torch.cuda.is_available() else None
        }
        
        with open(output_path, 'w') as f:
            json.dump(seed_state, f, indent=2, default=str)
        
        return output_path
    
    def load_seed_state(self, input_path: str) -> Dict[str, Any]:
        """Load seed state from file and restore it."""
        with open(input_path, 'r') as f:
            seed_state = json.load(f)
        
        # Restore Python state - convert list back to proper tuple structure
        python_state = seed_state["python_state"]
        if isinstance(python_state, list):
            # Convert back to proper Python random state format
            python_state = (python_state[0], tuple(python_state[1]), python_state[2])
        random.setstate(python_state)
        
        # Restore numpy state - need to handle the tuple structure properly
        np_state = seed_state["numpy_state"]
        if isinstance(np_state, list):
            # Convert back to proper numpy state format
            np_state = (np_state[0], np.array(np_state[1], dtype=np.uint32), np_state[2], np_state[3], np_state[4])
        np.random.set_state(np_state)
        
        # Restore torch RNG state with uint8 dtype
        torch.set_rng_state(torch.tensor(seed_state["torch_state"], dtype=torch.uint8))
        
        # Restore CUDA RNG state with uint8 dtype if available
        if seed_state["cuda_state"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(torch.tensor(seed_state["cuda_state"], dtype=torch.uint8))
        
        self.seed_info = seed_state["seed_info"]
        return seed_state
    
    def _compute_seed_checksum(self, seeds: Dict[str, Any]) -> str:
        """Compute checksum of seed information."""
        # Create a simplified version for hashing
        hash_info = {}
        for component, info in seeds.items():
            if isinstance(info, dict):
                hash_info[component] = {
                    "seed": info.get("seed"),
                    "available": info.get("available", True)
                }
            else:
                hash_info[component] = str(info)
        
        # Convert to JSON string and hash
        json_str = json.dumps(hash_info, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_tracking_summary(self) -> Dict[str, Any]:
        """Get summary of seed tracking."""
        return self.seed_info
    
    def create_reproducible_environment(self, seed: int) -> Dict[str, Any]:
        """
        Create a fully reproducible environment by setting all seeds and deterministic flags.
        
        Args:
            seed: The seed value to use
            
        Returns:
            Dictionary containing the seed configuration
        """
        # Set environment variable for additional reproducibility
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Set all seeds
        seed_info = self.set_seeds(seed)
        
        # Set deterministic cuDNN flags for reproducibility
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Turn on deterministic algorithms with warn_only=True
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        seed_info["reproducibility_settings"] = {
            "PYTHONHASHSEED": str(seed),
            "torch_deterministic": True,
            "cudnn_deterministic": torch.backends.cudnn.deterministic if torch.cuda.is_available() else None,
            "cudnn_benchmark": torch.backends.cudnn.benchmark if torch.cuda.is_available() else None
        }
        
        return seed_info