"""
Model tracking for architecture fingerprinting and versioning.
"""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .cache import TrackingCache, run_with_timeout_and_progress


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

    def __init__(self, algorithm: str = "sha256", config=None):
        self.algorithm = algorithm
        self.config = config
        self.tracked_models: Dict[str, Dict[str, Any]] = {}
        self._cache = TrackingCache(
            cache_dir=config.dataset_cache_dir / "models" if config and config.dataset_cache_dir else None,
            ttl=config.cache_timeout if config else 3600,
            max_memory_mb=int((config.max_memory_gb if config else 2.0) * 1024 * 0.25)  # 25% of total memory for model cache
        ) if config else None

    async def track_model_async(
        self,
        model,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        save_architecture: bool = True,
        save_weights: bool = False,
        timeout: Optional[int] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Async version of track_model with size limits and timeout.
        """
        timeout = timeout or (self.config.tracking_timeout if self.config else 30)

        if self._cache:
            cache_key = f"model_{name}_{self._get_model_cache_key(model)}"
            cached_result = await self._cache.get_async(cache_key)
            if cached_result:
                if progress_callback:
                    progress_callback(f"Using cached result for model {name}")
                return cached_result

        if progress_callback:
            progress_callback(f"Starting model tracking for {name}")

        try:
            num_params = sum(p.numel() for p in model.parameters())
            if self.config and num_params > self.config.model_fingerprint_limit:
                if progress_callback:
                    progress_callback(f"Large model detected ({num_params} parameters), using lightweight fingerprinting")
                return await self._track_large_model_lightweight(model, name, metadata, progress_callback)
        except Exception:
            pass

        try:
            result = await run_with_timeout_and_progress(
                self._track_model_internal(model, name, metadata, save_architecture, save_weights, progress_callback),
                timeout=timeout,
                progress_callback=progress_callback,
                error_message=f"Model tracking for {name} timed out"
            )

            if self._cache and not result.get("error"):
                cache_key = f"model_{name}_{self._get_model_cache_key(model)}"
                await self._cache.set_async(cache_key, result)

            return result

        except Exception as e:
            return {
                "name": name,
                "error": f"Model tracking failed: {str(e)}",
                "architecture_checksum": "error"
            }

    def track_model(
        self,
        model,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        save_architecture: bool = True,
        save_weights: bool = False
    ) -> Dict[str, Any]:
        """
        Synchronous version of track_model for backward compatibility.
        """
        try:
            asyncio.get_running_loop()
            raise RuntimeError("Cannot run async method from within async context")
        except RuntimeError:
            return asyncio.run(
                self.track_model_async(model, name, metadata, save_architecture, save_weights)
            )

    async def _track_model_internal(
        self,
        model,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        save_architecture: bool = True,
        save_weights: bool = False,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Internal async model tracking implementation.
        """
        tracking_info = {
            "name": name,
            "type": type(model).__name__,
            "algorithm": self.algorithm,
            "metadata": metadata or {},
            "timestamp": self._get_timestamp(),
            "save_architecture": save_architecture,
            "save_weights": save_weights
        }

        if progress_callback:
            progress_callback("Analyzing model architecture...")

        # Get model architecture info
        tracking_info.update(await self._get_model_architecture_info_async(model, progress_callback))

        if progress_callback:
            progress_callback("Computing architecture fingerprint...")

        # Compute architecture fingerprint
        tracking_info["architecture_checksum"] = await self._compute_architecture_checksum_async(model, progress_callback)

        # Compute weights fingerprint if requested
        if save_weights:
            if progress_callback:
                progress_callback("Computing weights fingerprint...")
            tracking_info["weights_checksum"] = await self._compute_weights_checksum_async(model, progress_callback)
            tracking_info["weights_size_bytes"] = self._get_model_size(model)

        # Handle special cases for different model types
        try:
            PreTrainedModel, _ = _import_transformers()
            if isinstance(model, PreTrainedModel):
                if progress_callback:
                    progress_callback("Extracting pre-trained model info...")
                tracking_info.update(self._get_pretrained_model_info(model))
        except ImportError:
            pass  # transformers not available

        self.tracked_models[name] = tracking_info
        return tracking_info

    async def _track_large_model_lightweight(
        self,
        model,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Lightweight tracking for large models that exceed size limits.
        """
        tracking_info = {
            "name": name,
            "type": type(model).__name__,
            "algorithm": self.algorithm,
            "metadata": metadata or {},
            "timestamp": self._get_timestamp(),
            "lightweight_mode": True
        }

        if progress_callback:
            progress_callback("Using lightweight mode for large model...")

        try:
            num_params = sum(p.numel() for p in model.parameters())
            tracking_info["num_parameters"] = num_params
            tracking_info["num_trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Use model class name and parameter count as lightweight fingerprint
            lightweight_fingerprint = f"{type(model).__name__}_{num_params}"
            tracking_info["architecture_checksum"] = hashlib.sha256(lightweight_fingerprint.encode()).hexdigest()

        except Exception as e:
            tracking_info["error"] = f"Lightweight tracking failed: {str(e)}"
            tracking_info["architecture_checksum"] = "error"

        try:
            PreTrainedModel, _ = _import_transformers()
            if isinstance(model, PreTrainedModel):
                tracking_info.update(self._get_pretrained_model_info(model))
        except ImportError:
            pass

        self.tracked_models[name] = tracking_info
        return tracking_info

    async def _get_model_architecture_info_async(self, model, progress_callback=None) -> Dict[str, Any]:
        """Async version of model architecture info extraction."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self._get_model_architecture_info, model
        )

    def _get_model_architecture_info(self, model) -> Dict[str, Any]:
        """Extract architecture information from a model."""
        info = {
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "num_layers": len(list(model.modules())),
            "device": next(model.parameters()).device.type if list(model.parameters()) else "cpu"
        }

        # Get layer information
        layer_info = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_info.append({
                    "name": name,
                    "type": type(module).__name__,
                    "parameters": sum(p.numel() for p in module.parameters())
                })

        info["layers"] = layer_info

        # Get model structure as string
        info["structure"] = str(model)

        return info

    def _get_pretrained_model_info(self, model) -> Dict[str, Any]:
        """Get additional info for pre-trained models."""
        info = {}

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

        return info

    async def _compute_architecture_checksum_async(self, model, progress_callback=None) -> str:
        """Async version of architecture checksum computation."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self._compute_architecture_checksum, model
        )

    def _compute_architecture_checksum(self, model) -> str:
        """Compute checksum of model architecture."""
        hash_obj = hashlib.new(self.algorithm)

        # Hash the model structure
        structure_str = str(model)
        hash_obj.update(structure_str.encode())

        # Hash parameter shapes and types
        for name, param in model.named_parameters():
            param_info = f"{name}:{param.shape}:{param.dtype}"
            hash_obj.update(param_info.encode())

        # Hash layer information
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                module_info = f"{name}:{type(module).__name__}"
                hash_obj.update(module_info.encode())

        return hash_obj.hexdigest()

    async def _compute_weights_checksum_async(self, model, progress_callback=None) -> str:
        """Async version of weights checksum computation."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self._compute_weights_checksum, model
        )

    def _compute_weights_checksum(self, model) -> str:
        """Compute checksum of model weights."""
        hash_obj = hashlib.new(self.algorithm)
        _import_numpy()

        # Get parameters in a fixed order (sorted by name)
        named_params = sorted(model.named_parameters(), key=lambda x: x[0])
        total_params = sum(p.numel() for _, p in named_params)

        if total_params > 100000000:  # 100M parameters
            # For very large models, sample tensors by fixed stride per parameter
            # Use streaming approach to avoid memory pressure

            for name, param in named_params:
                if param.numel() > 0:
                    flat_param = param.detach().cpu().flatten()
                    if len(flat_param) > 10000:
                        # Use fixed stride sampling: take every nth element
                        # Ensure we get exactly 10000 items
                        step = max(1, len(flat_param) // 10000)
                        sample_indices = []
                        for i in range(0, len(flat_param), step):
                            if len(sample_indices) >= 10000:
                                break
                            sample_indices.append(i)
                        # If we still need more samples, fill from the end
                        while len(sample_indices) < 10000 and len(sample_indices) < len(flat_param):
                            sample_indices.append(len(flat_param) - 1 - len(sample_indices))

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

    def get_tracking_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked models."""
        return {
            "total_models": len(self.tracked_models),
            "models": self.tracked_models,
            "algorithm": self.algorithm
        }

    def _get_model_cache_key(self, model) -> str:
        """Generate a deterministic cache key for a model."""
        try:
            import hashlib
            
            if hasattr(model, 'parameters'):
                param_count = sum(p.numel() for p in model.parameters())
                model_type = type(model).__name__
                return f"{model_type}_{param_count}"
            elif hasattr(model, 'get_config'):
                config_str = str(model.get_config())
                return hashlib.md5(config_str.encode()).hexdigest()[:16]
            elif hasattr(model, '__dict__'):
                model_type = type(model).__name__
                attr_hash = hashlib.md5(str(sorted(model.__dict__.keys())).encode()).hexdigest()[:8]
                return f"{model_type}_{attr_hash}"
            else:
                return f"{type(model).__name__}_{id(model)}"
        except Exception:
            return f"{type(model).__name__}_{id(model)}"
