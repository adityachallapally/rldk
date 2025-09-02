"""
Model tracking for architecture fingerprinting and versioning.
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np


class ModelTracker:
    """Tracks model architecture, weights, and metadata."""
    
    def __init__(self, algorithm: str = "sha256"):
        self.algorithm = algorithm
        self.tracked_models: Dict[str, Dict[str, Any]] = {}
    
    def track_model(
        self,
        model: Union[nn.Module, PreTrainedModel],
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
        tracking_info = {
            "name": name,
            "type": type(model).__name__,
            "algorithm": self.algorithm,
            "metadata": metadata or {},
            "timestamp": self._get_timestamp(),
            "save_architecture": save_architecture,
            "save_weights": save_weights
        }
        
        # Get model architecture info
        tracking_info.update(self._get_model_architecture_info(model))
        
        # Compute architecture fingerprint
        tracking_info["architecture_checksum"] = self._compute_architecture_checksum(model)
        
        # Compute weights fingerprint if requested
        if save_weights:
            tracking_info["weights_checksum"] = self._compute_weights_checksum(model)
            tracking_info["weights_size_bytes"] = self._get_model_size(model)
        
        # Handle special cases for different model types
        if isinstance(model, PreTrainedModel):
            tracking_info.update(self._get_pretrained_model_info(model))
        
        self.tracked_models[name] = tracking_info
        return tracking_info
    
    def _get_model_architecture_info(self, model: nn.Module) -> Dict[str, Any]:
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
    
    def _get_pretrained_model_info(self, model: PreTrainedModel) -> Dict[str, Any]:
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
    
    def _compute_architecture_checksum(self, model: nn.Module) -> str:
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
    
    def _compute_weights_checksum(self, model: nn.Module) -> str:
        """Compute checksum of model weights."""
        hash_obj = hashlib.new(self.algorithm)
        
        # For large models, sample a subset of weights deterministically
        total_params = sum(p.numel() for p in model.parameters())
        
        if total_params > 100000000:  # 100M parameters
            # Sample weights for very large models using deterministic sampling
            sample_size = min(1000000, total_params)  # 1M parameters max
            sampled_weights = []
            
            for param in model.parameters():
                if param.numel() > 0:
                    flat_param = param.detach().cpu().flatten()
                    if len(flat_param) > 10000:
                        # Use deterministic sampling: take every nth element
                        step = len(flat_param) // 10000
                        sample_indices = list(range(0, len(flat_param), step))[:10000]
                        sampled_weights.append(flat_param[sample_indices])
                    else:
                        sampled_weights.append(flat_param)
            
            # Concatenate and hash
            if sampled_weights:
                all_weights = torch.cat(sampled_weights)
                hash_obj.update(all_weights.numpy().tobytes())
        else:
            # Hash all weights for smaller models
            for param in model.parameters():
                if param.numel() > 0:
                    hash_obj.update(param.detach().cpu().numpy().tobytes())
        
        return hash_obj.hexdigest()
    
    def _get_model_size(self, model: nn.Module) -> int:
        """Get total size of model parameters in bytes."""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size
    
    def track_tokenizer(
        self,
        tokenizer: PreTrainedTokenizer,
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
        model: nn.Module,
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
        model: nn.Module,
        output_path: Path,
        name: str
    ) -> Path:
        """Save model weights to file."""
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