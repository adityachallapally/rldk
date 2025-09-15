"""Utility functions for TRL integration."""

from typing import Optional, Union
import torch
import torch.nn as nn

from packaging import version
from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM

try:
    from trl import AutoModelForCausalLMWithValueHead
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    AutoModelForCausalLMWithValueHead = None


def fix_generation_config(
    model: "AutoModelForCausalLMWithValueHead",
    tokenizer: AutoTokenizer,
    generation_config: Optional[GenerationConfig] = None
) -> "AutoModelForCausalLMWithValueHead":
    """Fix missing generation_config and base_model_prefix attributes on TRL models.

    This is a common issue with TRL 0.23.0+ where AutoModelForCausalLMWithValueHead
    doesn't have a generation_config or base_model_prefix attribute by default, 
    causing AttributeError when PPOTrainer tries to access them.

    Args:
        model: The TRL model to fix
        tokenizer: The tokenizer used with the model
        generation_config: Optional custom generation config. If None, creates a default one.

    Returns:
        The model with generation_config and base_model_prefix attributes set

    Raises:
        ImportError: If TRL is not available
        AttributeError: If model doesn't have required attributes
    """
    if not TRL_AVAILABLE:
        raise ImportError("TRL is required for this function. Install with: pip install trl")

    if not isinstance(model, AutoModelForCausalLMWithValueHead):
        raise AttributeError("Model must be an AutoModelForCausalLMWithValueHead instance")

    # Create generation config if not provided
    if generation_config is None:
        generation_config = GenerationConfig(
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=getattr(tokenizer, 'bos_token_id', None),
            max_length=512,  # Default max length
            do_sample=True,  # Enable sampling for generation
            temperature=1.0,  # Default temperature
            top_p=1.0,  # Default top_p
        )

    # Set the generation_config attribute
    model.generation_config = generation_config

    # Fix missing base_model_prefix attribute (required by PPOTrainer)
    if not hasattr(model, 'base_model_prefix'):
        # The AutoModelForCausalLMWithValueHead wraps a pretrained_model that has the base_model_prefix
        if hasattr(model, 'pretrained_model') and hasattr(model.pretrained_model, 'base_model_prefix'):
            model.base_model_prefix = model.pretrained_model.base_model_prefix
        else:
            # Fallback: try to infer from the model name or use a default
            model_name = getattr(model, 'name_or_path', '').lower()
            if 'gpt2' in model_name or 'gpt' in model_name:
                model.base_model_prefix = "transformer"
            elif 'llama' in model_name:
                model.base_model_prefix = "model"
            else:
                model.base_model_prefix = "transformer"  # Default fallback

    # Fix missing base model attribute (required by PPOTrainer)
    # The PPOTrainer expects the model to have an attribute with the name of base_model_prefix
    if hasattr(model, 'base_model_prefix') and hasattr(model, 'pretrained_model'):
        base_model_prefix = model.base_model_prefix
        if not hasattr(model, base_model_prefix):
            # Add the base model attribute by referencing the pretrained_model's attribute
            if hasattr(model.pretrained_model, base_model_prefix):
                setattr(model, base_model_prefix, getattr(model.pretrained_model, base_model_prefix))

    # Fix missing gradient checkpointing attribute (required by PPOTrainer)
    if not hasattr(model, 'is_gradient_checkpointing'):
        if hasattr(model, 'pretrained_model') and hasattr(model.pretrained_model, 'is_gradient_checkpointing'):
            model.is_gradient_checkpointing = model.pretrained_model.is_gradient_checkpointing
        else:
            model.is_gradient_checkpointing = False  # Default to False

    return model


def create_simple_value_model(
    base_model: Union[AutoModelForCausalLM, "AutoModelForCausalLMWithValueHead"],
    hidden_size: Optional[int] = None
) -> nn.Module:
    """Create a simple value model compatible with TRL's PPOTrainer.
    
    This function creates a value model that has all the required attributes
    (base_model_prefix, transformer, score method) for TRL compatibility.
    
    Args:
        base_model: The base model to create a value head for
        hidden_size: Optional hidden size. If None, inferred from base_model.config
        
    Returns:
        A value model with proper TRL-compatible interface
        
    Raises:
        ImportError: If TRL is not available
        AttributeError: If base_model doesn't have required attributes
    """
    if not TRL_AVAILABLE:
        raise ImportError("TRL is required for this function. Install with: pip install trl")
    
    if not hasattr(base_model, 'config'):
        raise AttributeError("base_model must have a config attribute")
    
    # Get hidden size
    if hidden_size is None:
        hidden_size = getattr(base_model.config, 'hidden_size', 768)
    
    class ValueModel(nn.Module):
        """Simple value model for PPO training."""
        
        def __init__(self, base_model, hidden_size):
            super().__init__()
            self.base_model = base_model
            self.value_head = nn.Linear(hidden_size, 1)
            
            # Required attributes for TRL compatibility
            self.base_model_prefix = getattr(base_model, 'base_model_prefix', 'transformer')
            
            # Set transformer attribute if it exists
            if hasattr(base_model, 'transformer'):
                self.transformer = base_model.transformer
            elif hasattr(base_model, 'model'):
                self.transformer = base_model.model
            else:
                # Fallback: use the base model itself
                self.transformer = base_model
        
        def forward(self, input_ids, attention_mask=None, **kwargs):
            """Forward pass for value estimation."""
            # Get hidden states from base model
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            
            # Extract hidden states
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states[-1]
            else:
                # Fallback: assume outputs is a tuple
                hidden_states = outputs[0]
            
            # Pool hidden states (mean pooling) with safety check for division by zero
            if attention_mask is not None:
                mask_sum = attention_mask.sum(dim=1, keepdim=True)
                pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / torch.clamp(mask_sum, min=1e-8)
            else:
                pooled = hidden_states.mean(dim=1)
            
            # Get value estimate
            values = self.value_head(pooled).squeeze(-1)
            return values
        
        def score(self, input_ids, attention_mask=None, **kwargs):
            """Score method required by TRL."""
            return self.forward(input_ids, attention_mask, **kwargs)
    
    return ValueModel(base_model, hidden_size)


def create_simple_reward_model(
    base_model: Union[AutoModelForCausalLM, "AutoModelForCausalLMWithValueHead"],
    hidden_size: Optional[int] = None
) -> nn.Module:
    """Create a simple reward model compatible with TRL's get_reward() function.
    
    This function creates a reward model that works with TRL's reward computation
    and has proper tensor shapes and interfaces.
    
    Args:
        base_model: The base model to create a reward head for
        hidden_size: Optional hidden size. If None, inferred from base_model.config
        
    Returns:
        A reward model with proper TRL-compatible interface
        
    Raises:
        ImportError: If TRL is not available
        AttributeError: If base_model doesn't have required attributes
    """
    if not TRL_AVAILABLE:
        raise ImportError("TRL is required for this function. Install with: pip install trl")
    
    if not hasattr(base_model, 'config'):
        raise AttributeError("base_model must have a config attribute")
    
    # Get hidden size
    if hidden_size is None:
        hidden_size = getattr(base_model.config, 'hidden_size', 768)
    
    class RewardModel(nn.Module):
        """Simple reward model for PPO training."""
        
        def __init__(self, base_model, hidden_size):
            super().__init__()
            self.base_model = base_model
            self.reward_head = nn.Linear(hidden_size, 1)
            
            # Required attributes for TRL compatibility
            self.base_model_prefix = getattr(base_model, 'base_model_prefix', 'transformer')
        
        def forward(self, input_ids, attention_mask=None, **kwargs):
            """Forward pass for reward computation."""
            # Get hidden states from base model
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            
            # Extract hidden states
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states[-1]
            else:
                # Fallback: assume outputs is a tuple
                hidden_states = outputs[0]
            
            # Pool hidden states (mean pooling) with safety check for division by zero
            if attention_mask is not None:
                mask_sum = attention_mask.sum(dim=1, keepdim=True)
                pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / torch.clamp(mask_sum, min=1e-8)
            else:
                pooled = hidden_states.mean(dim=1)
            
            # Get reward
            reward = self.reward_head(pooled)
            return reward
        
        def get_reward(self, input_ids, attention_mask=None, **kwargs):
            """Get reward method compatible with TRL's get_reward() function."""
            return self.forward(input_ids, attention_mask, **kwargs)
    
    return RewardModel(base_model, hidden_size)


def prepare_models_for_ppo(
    model_name: str,
    tokenizer: Optional[AutoTokenizer] = None,
    generation_config: Optional[GenerationConfig] = None,
    use_separate_value_model: bool = True
) -> tuple["AutoModelForCausalLMWithValueHead", AutoModelForCausalLM, nn.Module, nn.Module, AutoTokenizer]:
    """Prepare all required models for PPO training with proper generation_config.

    This function creates and configures all the models needed for PPO training,
    ensuring they have the required generation_config attribute to avoid AttributeError.
    
    The function now returns models compatible with current TRL API (0.23.0+) that requires
    separate reward_model and value_model parameters.

    Args:
        model_name: Name or path of the base model
        tokenizer: Optional tokenizer. If None, will be loaded from model_name
        generation_config: Optional custom generation config
        use_separate_value_model: If True, creates separate value model. If False, uses policy model as value model.

    Returns:
        Tuple of (policy_model, ref_model, value_model, reward_model, tokenizer)

    Raises:
        ImportError: If TRL is not available
    """
    if not TRL_AVAILABLE:
        raise ImportError("TRL is required for this function. Install with: pip install trl")

    # Load tokenizer if not provided
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Create policy model with value head
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    
    # Create reference model (frozen base model)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create base model for reward and value models
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create reward model
    reward_model = create_simple_reward_model(base_model)
    
    # Create value model
    if use_separate_value_model:
        value_model = create_simple_value_model(base_model)
    else:
        # Use policy model as value model (standard approach in some cases)
        value_model = policy_model

    # Fix generation_config for policy model
    policy_model = fix_generation_config(policy_model, tokenizer, generation_config)

    return policy_model, ref_model, value_model, reward_model, tokenizer


def check_trl_compatibility() -> dict:
    """Check TRL version compatibility and common issues.

    Returns:
        Dictionary with compatibility information and warnings
    """
    if not TRL_AVAILABLE:
        return {
            "trl_available": False,
            "version": None,
            "warnings": ["TRL is not installed. Install with: pip install trl"],
            "recommendations": ["Install TRL: pip install trl>=0.7.0"]
        }

    try:
        import trl
        trl_version_str = trl.__version__

        warnings_list = []
        recommendations = []

        # Check for known issues using proper semantic versioning
        trl_version_obj = version.parse(trl_version_str)

        if trl_version_obj >= version.parse("0.23.0"):
            warnings_list.append(
                "TRL 0.23.0+ has known issues with AutoModelForCausalLMWithValueHead.generation_config. "
                "Use fix_generation_config() utility function."
            )
            recommendations.append(
                "Use prepare_models_for_ppo() or fix_generation_config() to avoid AttributeError"
            )

        if trl_version_obj < version.parse("0.7.0"):
            warnings_list.append("TRL version is quite old. Consider upgrading to 0.7.0+")
            recommendations.append("Upgrade TRL: pip install --upgrade trl")

        # Check for very recent versions that might have breaking changes
        if trl_version_obj >= version.parse("0.25.0"):
            warnings_list.append(
                "Using a very recent TRL version. Some features may not be fully tested. "
                "Report any issues if they occur."
            )

        # Check for specific problematic versions
        if version.parse("0.20.0") <= trl_version_obj < version.parse("0.22.0"):
            warnings_list.append(
                "TRL versions 0.20.0-0.21.x have known stability issues. "
                "Consider upgrading to 0.22.0+ or downgrading to 0.19.x"
            )
            recommendations.append("Upgrade TRL: pip install --upgrade trl>=0.22.0")

        return {
            "trl_available": True,
            "version": trl_version_str,
            "warnings": warnings_list,
            "recommendations": recommendations
        }

    except Exception as e:
        return {
            "trl_available": True,
            "version": "unknown",
            "warnings": [f"Could not determine TRL version: {e}"],
            "recommendations": ["Check TRL installation"]
        }


def validate_ppo_setup(
    policy_model: "AutoModelForCausalLMWithValueHead",
    ref_model: AutoModelForCausalLM,
    value_model: nn.Module,
    reward_model: nn.Module,
    tokenizer: AutoTokenizer
) -> dict:
    """Validate PPO setup for common issues.

    Args:
        policy_model: Main PPO policy model
        ref_model: Reference model
        value_model: Value model
        reward_model: Reward model
        tokenizer: Tokenizer

    Returns:
        Dictionary with validation results
    """
    issues = []
    warnings = []

    # Check generation_config attribute for policy model
    if not hasattr(policy_model, 'generation_config'):
        issues.append("policy_model missing generation_config attribute")
    elif policy_model.generation_config is None:
        warnings.append("policy_model has None generation_config")

    # Check tokenizer compatibility
    if not hasattr(tokenizer, 'eos_token_id') or tokenizer.eos_token_id is None:
        issues.append("Tokenizer missing eos_token_id")

    if not hasattr(tokenizer, 'pad_token_id') or tokenizer.pad_token_id is None:
        warnings.append("Tokenizer missing pad_token_id")

    # Check model types
    if not isinstance(policy_model, AutoModelForCausalLMWithValueHead):
        issues.append("policy_model is not an AutoModelForCausalLMWithValueHead instance")
    
    if not isinstance(ref_model, AutoModelForCausalLM):
        issues.append("ref_model is not an AutoModelForCausalLM instance")
    
    # Check value model has required methods
    if not hasattr(value_model, 'score'):
        issues.append("value_model missing score method")
    
    if not hasattr(value_model, 'forward'):
        issues.append("value_model missing forward method")
    
    # Check reward model has required methods
    if not hasattr(reward_model, 'get_reward'):
        warnings.append("reward_model missing get_reward method (will use forward)")
    
    if not hasattr(reward_model, 'forward'):
        issues.append("reward_model missing forward method")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "recommendations": [
            "Use fix_generation_config() to fix generation_config issues",
            "Ensure policy_model is AutoModelForCausalLMWithValueHead instance",
            "Ensure ref_model is AutoModelForCausalLM instance",
            "Check value_model has score and forward methods",
            "Check reward_model has forward method",
            "Check tokenizer has required token IDs"
        ] if issues else []
    }
