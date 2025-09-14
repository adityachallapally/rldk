"""
Model tracking for architecture fingerprinting and versioning.
"""

import hashlib
import json
import time
import random
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache

from ..config import settings
from ..utils.runtime import with_timeout


def _import_torch():
    import torch
    import torch.nn as nn
    return torch, nn

def _import_transformers():
    from transformers import PreTrainedModel, PreTrainedTokenizer
    return PreTrainedModel, PreTrainedTokenizer

def _import_numpy():
    import numpy as np
    return np


class ModelTracker:
    """Tracks model architecture, weights, and metadata."""

    def __init__(self, algorithm: str = "sha256"):
        self.algorithm = algorithm
        self.tracked_models: Dict[str, Dict[str, Any]] = {}
        self._settings = settings
        self._cache_dir = self._settings.get_cache_dir() / "model_fingerprints"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def track_model(
        self,
        model,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        save_architecture: bool = True,
        save_weights: bool = False
    ) -> Dict[str, Any]:
        """
        Track a model and compute its fingerprint.

        Args:
            model: The model to track
            name: Name identifier for the model
            metadata: Additional metadata to store
            save_architecture: Whether to save model architecture
            save_weights: Whether to save model weights (usually too large)

        Returns:
            Dictionary containing tracking information
        """
        # Check cache first
        cache_key = self._get_cache_key(model, name)
        cached_info = self._load_from_cache(cache_key)
        if cached_info:
            # Update timestamp and metadata
            cached_info["timestamp"] = self._get_timestamp()
            if metadata:
                cached_info["metadata"].update(metadata)
            self.tracked_models[name] = cached_info
            return cached_info

        tracking_info = {
            "name": name,
            "type": type(model).__name__,
            "algorithm": self.algorithm,
            "metadata": metadata or {},
            "timestamp": self._get_timestamp(),
            "save_architecture": save_architecture,
            "save_weights": save_weights
        }

        try:
            # Get model architecture info (lazy analysis)
            tracking_info.update(self._get_model_architecture_info_lazy(model))

            # Compute architecture fingerprint with timeout
            tracking_info["architecture_checksum"] = self._compute_architecture_checksum_lazy(model)

            # Compute weights fingerprint if requested and model is not too large
            if save_weights and self._should_compute_weights_checksum(model):
                tracking_info["weights_checksum"] = self._compute_weights_checksum_lazy(model)
                tracking_info["weights_size_bytes"] = self._get_model_size(model)

            # Handle special cases for different model types
            PreTrainedModel, _ = _import_transformers()
            if isinstance(model, PreTrainedModel):
                tracking_info.update(self._get_pretrained_model_info_lazy(model))

        except Exception as e:
            # If tracking fails, create a minimal tracking info
            tracking_info.update({
                "error": f"Failed to track model: {str(e)}",
                "architecture_checksum": "unknown"
            })

        # Save to cache
        self._save_to_cache(cache_key, tracking_info)
        self.tracked_models[name] = tracking_info
        return tracking_info

    def _should_compute_weights_checksum(self, model) -> bool:
        """Check if model is small enough to compute weights checksum."""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            return total_params <= self._settings.model_fingerprint_limit
        except Exception:
            return False

    @with_timeout(10.0)  # 10 second timeout for architecture analysis
    def _get_model_architecture_info_lazy(self, model) -> Dict[str, Any]:
        """Extract architecture information from a model with lazy analysis."""
        info = {
            "num_parameters": 0,
            "num_trainable_parameters": 0,
            "num_layers": 0,
            "device": "unknown"
        }

        try:
            # Get basic parameter counts
            info["num_parameters"] = sum(p.numel() for p in model.parameters())
            info["num_trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Get device info
            if list(model.parameters()):
                info["device"] = next(model.parameters()).device.type
            else:
                info["device"] = "cpu"

            # Get layer count (limit for large models)
            if info["num_parameters"] <= self._settings.model_fingerprint_limit:
                info["num_layers"] = len(list(model.modules()))
                
                # Get layer information for smaller models
                layer_info = []
                for name, module in model.named_modules():
                    if len(list(module.children())) == 0:  # Leaf modules only
                        layer_info.append({
                            "name": name,
                            "type": type(module).__name__,
                            "parameters": sum(p.numel() for p in module.parameters())
                        })
                info["layers"] = layer_info
            else:
                # For large models, just get basic info
                info["num_layers"] = "too_large_to_count"
                info["layers"] = []

            # Get model structure as string (limit size)
            if info["num_parameters"] <= self._settings.model_fingerprint_limit // 10:
                info["structure"] = str(model)
            else:
                info["structure"] = f"Model too large to display ({info['num_parameters']} parameters)"

        except Exception as e:
            info["error"] = f"Failed to analyze architecture: {str(e)}"

        return info

    @with_timeout(5.0)  # 5 second timeout for pretrained model info
    def _get_pretrained_model_info_lazy(self, model) -> Dict[str, Any]:
        """Get additional info for pre-trained models with lazy analysis."""
        info = {}

        try:
            # Try to get config info
            if hasattr(model, 'config'):
                config = model.config
                info["config"] = {
                    "model_type": getattr(config, 'model_type', None),
                    "hidden_size": getattr(config, 'hidden_size', None),
                    "num_attention_heads": getattr(config, 'num_attention_heads', None),
                    "num_hidden_layers": getattr(config, 'num_hidden_layers', None),
                    "vocab_size": getattr(config, 'vocab_size', None),
                    "max_position_embeddings": getattr(config, 'max_position_embeddings', None),
                }

            # Try to get model name/identifier
            if hasattr(model, 'name_or_path'):
                info["model_name"] = model.name_or_path

        except Exception as e:
            info["error"] = f"Failed to get pretrained model info: {str(e)}"

        return info

    @with_timeout(15.0)  # 15 second timeout for architecture checksum
    def _compute_architecture_checksum_lazy(self, model) -> str:
        """Compute checksum of model architecture with lazy analysis."""
        hash_obj = hashlib.new(self.algorithm)

        try:
            # Hash the model structure (limit size for large models)
            total_params = sum(p.numel() for p in model.parameters())
            if total_params <= self._settings.model_fingerprint_limit:
                structure_str = str(model)
            else:
                # For large models, use a simplified structure
                structure_str = f"Large model with {total_params} parameters"
            hash_obj.update(structure_str.encode())

            # Hash parameter shapes and types (limit for large models)
            param_count = 0
            for name, param in model.named_parameters():
                if param_count >= 1000:  # Limit parameter info for very large models
                    hash_obj.update(f"truncated_{param_count}_more_params".encode())
                    break
                param_info = f"{name}:{param.shape}:{param.dtype}"
                hash_obj.update(param_info.encode())
                param_count += 1

            # Hash layer information (limit for large models)
            layer_count = 0
            for name, module in model.named_modules():
                if layer_count >= 1000:  # Limit layer info for very large models
                    hash_obj.update(f"truncated_{layer_count}_more_layers".encode())
                    break
                if len(list(module.children())) == 0:  # Leaf modules only
                    module_info = f"{name}:{type(module).__name__}"
                    hash_obj.update(module_info.encode())
                layer_count += 1

        except Exception as e:
            # If checksum computation fails, use a fallback
            hash_obj.update(f"error_{str(e)}".encode())

        return hash_obj.hexdigest()

    @with_timeout(30.0)  # 30 second timeout for weights checksum
    def _compute_weights_checksum_lazy(self, model) -> str:
        """Compute checksum of model weights with lazy analysis."""
        hash_obj = hashlib.new(self.algorithm)
        _import_numpy()

        try:
            # Get parameters in a fixed order (sorted by name)
            named_params = sorted(model.named_parameters(), key=lambda x: x[0])
            total_params = sum(p.numel() for _, p in named_params)

            # Use configurable limit from settings
            if total_params > self._settings.model_fingerprint_limit:
                # For very large models, sample tensors by fixed stride per parameter
                # Use streaming approach to avoid memory pressure
                sample_size = 10000  # Fixed sample size for large models

                for name, param in named_params:
                    if param.numel() > 0:
                        flat_param = param.detach().cpu().flatten()
                        if len(flat_param) > sample_size:
                            # Use fixed stride sampling: take every nth element
                            # Ensure we get exactly sample_size items
                            step = max(1, len(flat_param) // sample_size)
                            sample_indices = []
                            for i in range(0, len(flat_param), step):
                                if len(sample_indices) >= sample_size:
                                    break
                                sample_indices.append(i)
                            # If we still need more samples, fill from the end
                            # Fill remaining slots with random indices if needed
                            if len(sample_indices) < sample_size:
                                remaining_needed = sample_size - len(sample_indices)
                                available_indices = [i for i in range(len(flat_param)) if i not in sample_indices]
                                if available_indices:
                                    additional_indices = random.sample(available_indices, min(remaining_needed, len(available_indices)))
                                    sample_indices.extend(additional_indices)

                            # Hash the sampled weights directly to avoid memory accumulation
                            sampled_weights = flat_param[sample_indices]
                            hash_obj.update(sampled_weights.numpy().tobytes())
                        else:
                            # Hash the entire parameter if it's small
                            hash_obj.update(flat_param.numpy().tobytes())
            else:
                # For smaller models, hash all weights in fixed parameter order
                for name, param in named_params:
                    if param.numel() > 0:
                        hash_obj.update(param.detach().cpu().numpy().tobytes())

        except Exception as e:
            # If weights checksum computation fails, use a fallback
            hash_obj.update(f"error_{str(e)}".encode())

        return hash_obj.hexdigest()

    def _get_model_size(self, model) -> int:
        """Get total size of model parameters in bytes."""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size

    def track_tokenizer(
        self,
        tokenizer,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Track a tokenizer."""
        tracking_info = {
            "name": name,
            "type": type(tokenizer).__name__,
            "algorithm": self.algorithm,
            "metadata": metadata or {},
            "timestamp": self._get_timestamp()
        }

        # Get tokenizer info
        tracking_info.update({
            "vocab_size": getattr(tokenizer, 'vocab_size', None),
            "model_max_length": getattr(tokenizer, 'model_max_length', None),
            "padding_side": getattr(tokenizer, 'padding_side', None),
            "truncation_side": getattr(tokenizer, 'truncation_side', None),
        })

        # Compute checksum
        try:
            # Serialize tokenizer config
            config_str = json.dumps(tokenizer.init_kwargs, sort_keys=True, default=str)
            tracking_info["checksum"] = hashlib.sha256(config_str.encode()).hexdigest()
        except Exception as e:
            tracking_info["checksum"] = f"error: {str(e)}"

        return tracking_info

    def save_model_architecture(
        self,
        model,
        output_path: Path,
        name: str
    ) -> Path:
        """Save model architecture to file."""
        arch_path = output_path / f"{name}_architecture.txt"

        with open(arch_path, 'w') as f:
            f.write(f"Model: {name}\n")
            f.write(f"Type: {type(model).__name__}\n")
            f.write(f"Timestamp: {self._get_timestamp()}\n")
            f.write("=" * 50 + "\n\n")
            f.write(str(model))

        return arch_path

    def save_model_weights(
        self,
        model,
        output_path: Path,
        name: str
    ) -> Path:
        """Save model weights to file."""
        torch, _ = _import_torch()
        weights_path = output_path / f"{name}_weights.pt"
        torch.save(model.state_dict(), weights_path)
        return weights_path

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

    def _get_cache_key(self, model, name: str) -> str:
        """Generate cache key for model."""
        # Create a simple hash based on model type and name
        key_data = f"{type(model).__name__}_{name}_{self.algorithm}"
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
        """Get summary of all tracked models."""
        return {
            "total_models": len(self.tracked_models),
            "models": self.tracked_models,
            "algorithm": self.algorithm,
            "cache_dir": str(self._cache_dir)
        }
