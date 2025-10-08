"""Utility functions for TRL integration."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import warnings

from packaging import version
from packaging.version import InvalidVersion
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PreTrainedModel

from .callbacks import EventWriterCallback

try:  # pragma: no cover - torch is an optional dependency for some integrations
    import torch
except ImportError:  # pragma: no cover - keep working when torch is absent
    torch = None  # type: ignore[assignment]

try:
    from trl import AutoModelForCausalLMWithValueHead
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    AutoModelForCausalLMWithValueHead = None

if TYPE_CHECKING:  # pragma: no cover - imports for type checkers only
    from datasets import Dataset
    from trl import GRPOConfig, PPOConfig, PPOTrainer


def _accelerator_available() -> bool:
    """Return ``True`` when a CUDA or MPS accelerator is available."""

    if torch is None:
        return False

    try:
        if torch.cuda.is_available():
            return True
    except Exception:  # pragma: no cover - defensive fallback
        pass

    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True
    except Exception:  # pragma: no cover - defensive fallback
        pass

    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return True
    except Exception:  # pragma: no cover - defensive fallback
        pass

    return False


def _bf16_supported() -> bool:
    """Return ``True`` if the current hardware stack supports bfloat16 precision."""

    try:
        from transformers.utils import is_torch_bf16_cpu_available, is_torch_bf16_gpu_available
    except ImportError:  # pragma: no cover - transformers < 4.32
        pass
    else:
        for checker in (is_torch_bf16_gpu_available, is_torch_bf16_cpu_available):
            try:
                if checker():
                    return True
            except Exception:  # pragma: no cover - defensive fallback
                continue

    if torch is None:
        return False

    try:
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()  # type: ignore[attr-defined]
            return major >= 8
    except Exception:  # pragma: no cover - defensive fallback
        pass

    return False


def _apply_precision_fallbacks(config_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Disable unsupported half-precision settings when running without accelerator support."""

    updated_kwargs = dict(config_kwargs)

    accelerator_available = _accelerator_available()

    if not accelerator_available:
        if updated_kwargs.get("fp16"):
            warnings.warn(
                "Disabling fp16 because no GPU/MPS accelerator is available.",
                RuntimeWarning,
                stacklevel=2,
            )
        updated_kwargs["fp16"] = False

        if torch is not None:
            dtype = updated_kwargs.get("torch_dtype")
            if dtype in {"float16", "half", getattr(torch, "float16", "float16")}:
                updated_kwargs["torch_dtype"] = getattr(torch, "float32", None)

    if not _bf16_supported():
        original_bf16 = updated_kwargs.pop("bf16", None)
        if original_bf16:
            warnings.warn(
                "Disabling bf16 because the current hardware does not support it.",
                RuntimeWarning,
                stacklevel=2,
            )
        updated_kwargs["bf16"] = False

        if torch is not None:
            dtype = updated_kwargs.get("torch_dtype")
            if dtype in {"bfloat16", "bf16", getattr(torch, "bfloat16", "bfloat16")}:
                updated_kwargs["torch_dtype"] = getattr(torch, "float32", None)

    return updated_kwargs


def create_grpo_config(**config_kwargs: Any) -> "GRPOConfig":
    """Create a :class:`trl.GRPOConfig` with safe CPU defaults.

    When running on machines without accelerator support, TRL's default GRPO configuration
    enables ``bf16`` which immediately raises ``ValueError`` during initialization. This
    helper mirrors the standard constructor while disabling unsupported precision modes so
    training can proceed on CPU-only hosts.

    Args:
        **config_kwargs: Keyword arguments forwarded to :class:`trl.GRPOConfig`.

    Returns:
        A ``GRPOConfig`` instance with precision flags adjusted for the current hardware.

    Raises:
        ImportError: If TRL or ``GRPOConfig`` are unavailable in the environment.
    """

    if not TRL_AVAILABLE:
        raise ImportError("TRL is required for create_grpo_config. Install with: pip install trl")

    try:
        from trl import GRPOConfig  # type: ignore
    except ImportError as exc:  # pragma: no cover - older TRL releases
        raise ImportError(
            "GRPOConfig is not available in the installed TRL version. "
            "Upgrade with: pip install --upgrade trl"
        ) from exc

    safe_kwargs = _apply_precision_fallbacks(config_kwargs)
    return GRPOConfig(**safe_kwargs)


def _build_generation_config(
    tokenizer: AutoTokenizer,
    generation_config: Optional[GenerationConfig],
) -> GenerationConfig:
    """Return a usable :class:`~transformers.GenerationConfig` instance."""

    if generation_config is not None:
        return generation_config

    return GenerationConfig(
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=getattr(tokenizer, "bos_token_id", None),
        max_length=512,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
    )


def _ensure_value_head_score(
    model: "AutoModelForCausalLMWithValueHead",
) -> "AutoModelForCausalLMWithValueHead":
    """Ensure TRL value-head models expose a ``score`` attribute."""

    if hasattr(model, "v_head") and not hasattr(model, "score"):
        model.score = model.v_head

    return model


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

    if AutoModelForCausalLMWithValueHead is None:
        raise ImportError(
            "AutoModelForCausalLMWithValueHead is unavailable despite TRL being installed."
            " Upgrade TRL to a version that provides value head models."
        )

    if not isinstance(model, AutoModelForCausalLMWithValueHead):
        raise AttributeError("Model must be an AutoModelForCausalLMWithValueHead instance")

    # Create generation config if not provided
    generation_config = _build_generation_config(tokenizer, generation_config)

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

    # Ensure TRL compatibility helpers can track emitted warnings.
    if not hasattr(model, 'warnings_issued'):
        model.warnings_issued = {}

    return _ensure_value_head_score(model)


def prepare_models_for_ppo(
    model_name: str,
    tokenizer: Optional[AutoTokenizer] = None,
    generation_config: Optional[GenerationConfig] = None
) -> Tuple[
    PreTrainedModel,
    PreTrainedModel,
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

    # Create models - load policy/reference as base causal LMs to preserve weights
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

    # Ensure policy/ref models expose a usable generation_config
    shared_generation_config = _build_generation_config(tokenizer, generation_config)
    for candidate in (model, ref_model):
        if getattr(candidate, "generation_config", None) is None:
            candidate.generation_config = shared_generation_config

    # Fix generation_config for TRL models
    reward_model = fix_generation_config(reward_model, tokenizer, generation_config)

    return model, ref_model, reward_model, tokenizer


def _prepare_value_model(
    model_name: str,
    tokenizer: AutoTokenizer,
    generation_config: Optional[GenerationConfig] = None,
) -> "AutoModelForCausalLMWithValueHead":
    """Prepare a value model that matches TRL's value head expectations."""

    value_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    value_model = fix_generation_config(value_model, tokenizer, generation_config)
    return _ensure_value_head_score(value_model)


def _ensure_generation_config(
    candidate: Optional["AutoModelForCausalLMWithValueHead"],
    tokenizer: Optional[AutoTokenizer],
    generation_config: Optional[GenerationConfig],
) -> Optional["AutoModelForCausalLMWithValueHead"]:
    """Apply :func:`fix_generation_config` when possible."""

    if candidate is None or not isinstance(candidate, AutoModelForCausalLMWithValueHead):
        return candidate

    if tokenizer is None:
        raise ValueError(
            "Tokenizer is required when providing custom TRL models. "
            "Pass a tokenizer or allow prepare_models_for_ppo() to create one."
        )

    return fix_generation_config(candidate, tokenizer, generation_config)


def _wrap_policy_for_legacy_trl(
    candidate: Optional[Union[PreTrainedModel, "AutoModelForCausalLMWithValueHead"]],
    *,
    model_name: str,
    tokenizer: AutoTokenizer,
    generation_config: Optional[GenerationConfig],
) -> "AutoModelForCausalLMWithValueHead":
    """Wrap policy models with TRL value heads without losing pretrained weights."""

    if AutoModelForCausalLMWithValueHead is None:
        raise ImportError(
            "AutoModelForCausalLMWithValueHead is unavailable. Upgrade TRL to wrap policy models."
        )

    if isinstance(candidate, AutoModelForCausalLMWithValueHead):
        return fix_generation_config(candidate, tokenizer, generation_config)

    if candidate is not None:
        wrapped = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model=candidate)
    else:
        wrapped = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

    return fix_generation_config(wrapped, tokenizer, generation_config)


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
    event_log_path: Optional[Union[str, Path]] = None,
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
        model = _ensure_generation_config(model, tokenizer, generation_config)
        ref_model = _ensure_generation_config(ref_model, tokenizer, generation_config)
        if isinstance(reward_model, AutoModelForCausalLMWithValueHead):
            reward_model = _ensure_generation_config(
                reward_model,
                tokenizer,
                generation_config,
            )
        if value_model is not None:
            value_model = _ensure_generation_config(value_model, tokenizer, generation_config)

    if tokenizer is None:
        raise ValueError("Tokenizer could not be prepared for PPO training")

    if not use_new_api:
        model = _wrap_policy_for_legacy_trl(
            model,
            model_name=model_name,
            tokenizer=tokenizer,
            generation_config=generation_config,
        )
        ref_model = _wrap_policy_for_legacy_trl(
            ref_model,
            model_name=model_name,
            tokenizer=tokenizer,
            generation_config=generation_config,
        )

    if use_new_api and value_model is None:
        value_model = _prepare_value_model(
            model_name,
            tokenizer,
            generation_config,
        )

    if validate_setup:
        validation = validate_ppo_setup(
            model,
            ref_model,
            reward_model,
            value_model,
            tokenizer,
            require_value_model=use_new_api,
            allow_base_models=use_new_api,
        )
        if not validation["valid"]:
            issues = "; ".join(validation["issues"])
            raise ValueError(f"Invalid PPO setup detected: {issues}")

    if event_log_path is not None:
        already_has_writer = any(
            isinstance(cb, EventWriterCallback) for cb in callbacks_list
        )
        if not already_has_writer:
            callbacks_list.append(
                EventWriterCallback(
                    event_log_path,
                    run_id=getattr(ppo_config, "run_name", None),
                )
            )

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
                "TRL 0.23.0+ has known issues with AutoModelForCausalLMWithValueHead.generation_config. "
                "Use fix_generation_config() utility function."
            )
            recommendations.append(
                "Use prepare_models_for_ppo() or fix_generation_config() to avoid AttributeError"
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
    allow_base_models: bool = False,
) -> Dict[str, Any]:
    """Validate PPO setup for common issues."""

    issues: List[str] = []
    warnings: List[str] = []

    # Helper to validate TRL value-head models
    def _check_value_head(
        name: str,
        model_obj: Optional[Any],
        *,
        required: bool = True,
        allow_plain: bool = False,
    ) -> None:
        if model_obj is None:
            message = f"{name} is missing"
            if required:
                issues.append(message)
            else:
                warnings.append(f"{message} (not required for this TRL version)")
            return

        if not isinstance(model_obj, AutoModelForCausalLMWithValueHead):
            if allow_plain and name in {"model", "ref_model"}:
                if hasattr(model_obj, "forward"):
                    if getattr(model_obj, "generation_config", None) is None:
                        warnings.append(f"{name} missing generation_config attribute")
                    return
            issues.append(f"{name} is not an AutoModelForCausalLMWithValueHead instance")
            return

        if not hasattr(model_obj, "generation_config"):
            issues.append(f"{name} missing generation_config attribute")
        elif model_obj.generation_config is None:
            warnings.append(f"{name} has None generation_config")

    _check_value_head("model", model, allow_plain=allow_base_models)
    _check_value_head("ref_model", ref_model, allow_plain=allow_base_models)
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


def tokenize_text_column(
    dataset: "Dataset",
    tokenizer: AutoTokenizer,
    *,
    text_column: str,
    padding: Union[bool, str] = True,
    truncation: Union[bool, str] = True,
    max_length: Optional[int] = None,
    add_special_tokens: bool = True,
    keep_original: bool = False,
    desc: Optional[str] = None,
) -> "Dataset":
    """Add ``input_ids`` and ``attention_mask`` columns to a dataset.

    The helper relies on :meth:`datasets.Dataset.map` to apply the tokenizer and, by default,
    removes the source text column so the resulting dataset only contains tensor-friendly fields.
    Set ``keep_original=True`` to retain the raw strings alongside the tokenized payload.

    Args:
        dataset: The dataset whose records contain ``text_column`` entries.
        tokenizer: Tokenizer used to encode the strings.
        text_column: Name of the column that should be tokenized.
        padding: Padding strategy forwarded to the tokenizer.
        truncation: Truncation strategy forwarded to the tokenizer.
        max_length: Optional maximum sequence length.
        add_special_tokens: Whether to include special tokens during tokenization.
        keep_original: When ``True`` the raw ``text_column`` is retained in the mapped dataset.
        desc: Optional description forwarded to :meth:`datasets.Dataset.map`.

    Returns:
        A dataset instance that includes ``input_ids`` and ``attention_mask`` columns.

    Raises:
        KeyError: If ``text_column`` is missing from the dataset records.
    """

    tokenization_kwargs: Dict[str, Any] = {
        "return_tensors": None,
        "padding": padding,
        "truncation": truncation,
        "add_special_tokens": add_special_tokens,
    }

    if max_length is not None:
        tokenization_kwargs["max_length"] = max_length

    def _tokenize(batch: Dict[str, Any]) -> Dict[str, Any]:
        if text_column not in batch:
            raise KeyError(f"Column '{text_column}' not found in batch")
        texts = batch[text_column]
        return tokenizer(texts, **tokenization_kwargs)

    remove_columns: Optional[List[str]] = None
    if not keep_original:
        remove_columns = [text_column]

    return dataset.map(
        _tokenize,
        batched=True,
        remove_columns=remove_columns,
        desc=desc or f"Tokenizing '{text_column}' column",
    )

