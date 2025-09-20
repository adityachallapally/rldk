"""Utility functions for TRL integration."""

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple
import warnings

from packaging import version
from packaging.version import InvalidVersion
from transformers import AutoTokenizer, GenerationConfig

try:
    from trl import AutoModelForCausalLMWithValueHead
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    AutoModelForCausalLMWithValueHead = None

if TYPE_CHECKING:  # pragma: no cover - imports for type checkers only
    from datasets import Dataset
    from trl import PPOConfig, PPOTrainer


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


def prepare_models_for_ppo(
    model_name: str,
    tokenizer: Optional[AutoTokenizer] = None,
    generation_config: Optional[GenerationConfig] = None
) -> Tuple[
    "AutoModelForCausalLMWithValueHead",
    "AutoModelForCausalLMWithValueHead",
    "AutoModelForCausalLMWithValueHead",
    AutoTokenizer,
]:
    """Prepare policy, reference, and reward models for PPO training.

    The models are returned with a consistent generation configuration to avoid
    TRL AttributeErrors. The value model required by newer TRL releases can be
    created via :func:`_prepare_value_model` or the :func:`create_ppo_trainer`
    factory.

    Returns:
        Tuple of (model, ref_model, reward_model, tokenizer).
    """

    if not TRL_AVAILABLE:
        raise ImportError("TRL is required for this function. Install with: pip install trl")

    # Load tokenizer if not provided
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Create models - use same model for policy and reward heads
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

    # Fix generation_config for all models
    model = fix_generation_config(model, tokenizer, generation_config)
    ref_model = fix_generation_config(ref_model, tokenizer, generation_config)
    reward_model = fix_generation_config(reward_model, tokenizer, generation_config)

    return model, ref_model, reward_model, tokenizer


def _prepare_value_model(
    model_name: str,
    tokenizer: AutoTokenizer,
    generation_config: Optional[GenerationConfig] = None,
) -> "AutoModelForCausalLMWithValueHead":
    """Prepare a value model that matches TRL's value head expectations."""

    value_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    return fix_generation_config(value_model, tokenizer, generation_config)


def _ensure_generation_config(
    candidate: Optional["AutoModelForCausalLMWithValueHead"],
    tokenizer: Optional[AutoTokenizer],
    generation_config: Optional[GenerationConfig],
    *,
    allow_missing_tokenizer: bool = False,
) -> Optional["AutoModelForCausalLMWithValueHead"]:
    """Apply :func:`fix_generation_config` when possible."""

    if candidate is None or not isinstance(candidate, AutoModelForCausalLMWithValueHead):
        return candidate

    if tokenizer is None:
        if allow_missing_tokenizer:
            return candidate
        raise ValueError(
            "Tokenizer is required when providing custom TRL models. "
            "Pass a tokenizer or allow prepare_models_for_ppo() to create one."
        )

    return fix_generation_config(candidate, tokenizer, generation_config)


def _collect_model_names(
    models: Iterable[Optional["AutoModelForCausalLMWithValueHead"]]
) -> List[str]:
    """Collect potential model identifiers for tokenizer inference."""

    names: List[str] = []
    for model in models:
        if model is None:
            continue

        for attr in ("name_or_path", "model_name"):
            value = getattr(model, attr, None)
            if value:
                names.append(value)

        pretrained = getattr(model, "pretrained_model", None)
        if pretrained is not None:
            for attr in ("name_or_path", "model_name", "_name_or_path"):
                value = getattr(pretrained, attr, None)
                if value:
                    names.append(value)

        config = getattr(model, "config", None)
        if config is not None:
            for attr in ("name_or_path", "model_type", "_name_or_path"):
                value = getattr(config, attr, None)
                if value:
                    names.append(value)

    return names


def _infer_tokenizer(
    tokenizer: Optional[AutoTokenizer],
    model_name: str,
    *models: Optional["AutoModelForCausalLMWithValueHead"],
) -> Optional[AutoTokenizer]:
    """Infer a tokenizer from provided models or model name."""

    if tokenizer is not None:
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    candidates = []
    seen = set()
    for candidate in _collect_model_names(models):
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    if model_name and model_name not in seen:
        candidates.append(model_name)

    for candidate in candidates:
        try:
            inferred = AutoTokenizer.from_pretrained(candidate)
        except Exception:
            continue

        if getattr(inferred, "pad_token", None) is None and getattr(inferred, "eos_token", None) is not None:
            inferred.pad_token = inferred.eos_token
        if getattr(inferred, "pad_token_id", None) is None and getattr(inferred, "eos_token_id", None) is not None:
            inferred.pad_token_id = inferred.eos_token_id
        return inferred

    return None


def _auto_fix_generation_configs(
    models: Iterable[Tuple[str, Optional["AutoModelForCausalLMWithValueHead"]]],
    tokenizer: Optional[AutoTokenizer],
    generation_config: Optional[GenerationConfig],
) -> List[str]:
    """Ensure TRL models have the expected generation configuration."""

    issues: List[str] = []

    if tokenizer is None:
        issues.append(
            "Unable to infer a tokenizer for automatic generation_config fixes; "
            "pass a tokenizer explicitly to silence this warning."
        )
        return issues

    seen: set[int] = set()
    for name, candidate in models:
        if candidate is None or not isinstance(candidate, AutoModelForCausalLMWithValueHead):
            continue

        candidate_id = id(candidate)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)

        try:
            fix_generation_config(candidate, tokenizer, generation_config)
        except Exception as exc:  # pragma: no cover - defensive branch
            issues.append(f"{name}: auto-fix failed with error: {exc}")
            continue

        if not hasattr(candidate, "generation_config") or candidate.generation_config is None:
            issues.append(f"{name}: missing generation_config after auto-fix")
            continue

        base_prefix = getattr(candidate, "base_model_prefix", None)
        if base_prefix and not hasattr(candidate, base_prefix):
            issues.append(
                f"{name}: missing base model attribute '{base_prefix}' after auto-fix"
            )

    return issues


def create_ppo_trainer(
    model_name: str,
    ppo_config: "PPOConfig",
    train_dataset: "Dataset",
    callbacks: Optional[List[Any]] = None,
    *,
    tokenizer: Optional[AutoTokenizer] = None,
    generation_config: Optional[GenerationConfig] = None,
    model: Optional["AutoModelForCausalLMWithValueHead"] = None,
    ref_model: Optional["AutoModelForCausalLMWithValueHead"] = None,
    reward_model: Optional[Any] = None,
    value_model: Optional["AutoModelForCausalLMWithValueHead"] = None,
    validate_setup: bool = True,
    **ppo_kwargs: Any,
) -> "PPOTrainer":
    """Create a :class:`trl.PPOTrainer` with version-aware defaults."""

    if not TRL_AVAILABLE:
        raise ImportError("TRL is required for this function. Install with: pip install trl")

    # Local import to avoid import errors when TRL is missing at module import time
    from trl import PPOTrainer  # type: ignore

    compatibility = check_trl_compatibility()
    trl_version: Optional[version.Version]
    try:
        trl_version = (
            version.parse(compatibility["version"]) if compatibility.get("version") else None
        )
    except (InvalidVersion, TypeError, ValueError):
        trl_version = None

    use_new_api = trl_version is None or trl_version >= version.parse("0.23.0")

    tokenizer = _infer_tokenizer(
        tokenizer,
        model_name,
        model,
        ref_model,
        reward_model if isinstance(reward_model, AutoModelForCausalLMWithValueHead) else None,
        value_model,
    )

    callbacks_list: List[Any] = list(callbacks) if callbacks else []

    required_components_missing = (
        tokenizer is None
        or any(component is None for component in (model, ref_model, reward_model))
    )
    components_missing = required_components_missing or (use_new_api and value_model is None)

    if components_missing:
        (
            default_model,
            default_ref_model,
            default_reward_model,
            default_tokenizer,
        ) = prepare_models_for_ppo(
            model_name,
            tokenizer=tokenizer,
            generation_config=generation_config,
        )

        model = model or default_model
        ref_model = ref_model or default_ref_model
        reward_model = reward_model or default_reward_model
        tokenizer = tokenizer or default_tokenizer

        if use_new_api and value_model is None:
            value_model = _prepare_value_model(
                model_name,
                tokenizer,
                generation_config,
            )
    else:
        model = _ensure_generation_config(
            model,
            tokenizer,
            generation_config,
            allow_missing_tokenizer=use_new_api,
        )
        ref_model = _ensure_generation_config(
            ref_model,
            tokenizer,
            generation_config,
            allow_missing_tokenizer=use_new_api,
        )
        if isinstance(reward_model, AutoModelForCausalLMWithValueHead):
            reward_model = _ensure_generation_config(
                reward_model,
                tokenizer,
                generation_config,
                allow_missing_tokenizer=use_new_api,
            )
        if value_model is not None:
            value_model = _ensure_generation_config(
                value_model,
                tokenizer,
                generation_config,
                allow_missing_tokenizer=use_new_api,
            )

    tokenizer = _infer_tokenizer(
        tokenizer,
        model_name,
        model,
        ref_model,
        reward_model if isinstance(reward_model, AutoModelForCausalLMWithValueHead) else None,
        value_model,
    )

    if tokenizer is None:
        raise ValueError("Tokenizer could not be prepared for PPO training")

    if use_new_api and value_model is None:
        value_model = _prepare_value_model(
            model_name,
            tokenizer,
            generation_config,
        )

    if use_new_api:
        autofix_issues = _auto_fix_generation_configs(
            (
                ("model", model),
                ("ref_model", ref_model),
                ("reward_model", reward_model),
                ("value_model", value_model),
            ),
            tokenizer,
            generation_config,
        )
        if autofix_issues:
            warning_message = (
                "Automatic TRL generation_config fixes encountered issues: "
                + "; ".join(sorted(set(autofix_issues)))
            )
            warnings.warn(warning_message, RuntimeWarning, stacklevel=2)

    if validate_setup:
        validation = validate_ppo_setup(
            model,
            ref_model,
            reward_model,
            value_model,
            tokenizer,
            require_value_model=use_new_api,
        )
        if not validation["valid"]:
            issues = "; ".join(validation["issues"])
            raise ValueError(f"Invalid PPO setup detected: {issues}")

    trainer_kwargs: Dict[str, Any] = {
        "args": ppo_config,
        "model": model,
        "ref_model": ref_model,
        "processing_class": tokenizer,
        "train_dataset": train_dataset,
        "callbacks": callbacks_list,
    }
    trainer_kwargs.update(ppo_kwargs)

    if use_new_api:
        trainer_kwargs.update({
            "reward_model": reward_model,
            "value_model": value_model,
        })

    trainer = PPOTrainer(**trainer_kwargs)

    # Ensure reward/value models remain accessible even for legacy TRL versions
    if not use_new_api:
        if reward_model is not None and not hasattr(trainer, "reward_model"):
            setattr(trainer, "reward_model", reward_model)
        if value_model is not None and not hasattr(trainer, "value_model"):
            setattr(trainer, "value_model", value_model)

    return trainer


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
            "recommendations": ["Install TRL: pip install trl"],
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
                "TRL 0.23.0+ requires generation_config attributes on value-head models. "
                "RLDK automatically applies this fix when creating PPO trainers."
            )
            recommendations.append(
                "If you bypass RLDK utilities, call fix_generation_config() on custom TRL models"
            )

        if trl_version_obj < version.parse("0.23.0"):
            warnings_list.append("TRL version is quite old. Consider upgrading to 0.23.0+")
            recommendations.append("Upgrade TRL: pip install --upgrade trl>=0.23.0")

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
    model: Optional["AutoModelForCausalLMWithValueHead"],
    ref_model: Optional["AutoModelForCausalLMWithValueHead"],
    reward_model: Optional[Any],
    value_model: Optional["AutoModelForCausalLMWithValueHead"],
    tokenizer: Optional[AutoTokenizer],
    *,
    require_value_model: bool = True,
) -> Dict[str, Any]:
    """Validate PPO setup for common issues."""

    issues: List[str] = []
    warnings: List[str] = []

    # Helper to validate TRL value-head models
    def _check_value_head(name: str, model_obj: Optional[Any], required: bool = True) -> None:
        if model_obj is None:
            message = f"{name} is missing"
            if required:
                issues.append(message)
            else:
                warnings.append(f"{message} (not required for this TRL version)")
            return

        if not isinstance(model_obj, AutoModelForCausalLMWithValueHead):
            issues.append(f"{name} is not an AutoModelForCausalLMWithValueHead instance")
            return

        if not hasattr(model_obj, "generation_config"):
            issues.append(f"{name} missing generation_config attribute")
        elif model_obj.generation_config is None:
            warnings.append(f"{name} has None generation_config")

    _check_value_head("model", model)
    _check_value_head("ref_model", ref_model)
    _check_value_head("value_model", value_model, required=require_value_model)

    if reward_model is None:
        issues.append("reward_model is missing")
    elif isinstance(reward_model, AutoModelForCausalLMWithValueHead):
        if not hasattr(reward_model, "generation_config"):
            issues.append("reward_model missing generation_config attribute")
        elif reward_model.generation_config is None:
            warnings.append("reward_model has None generation_config")
    elif not hasattr(reward_model, "forward"):
        warnings.append("reward_model does not define a forward method")

    if tokenizer is None:
        issues.append("Tokenizer is missing")
    else:
        if not hasattr(tokenizer, "eos_token_id") or tokenizer.eos_token_id is None:
            issues.append("Tokenizer missing eos_token_id")
        if not hasattr(tokenizer, "pad_token_id") or tokenizer.pad_token_id is None:
            warnings.append("Tokenizer missing pad_token_id")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "recommendations": [
            "Use fix_generation_config() to fix generation_config issues",
            "Ensure PPO components include policy, reference, reward, and value models",
            "Verify tokenizer has required token IDs",
        ] if issues else [],
    }
