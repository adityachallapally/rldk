"""Field resolver utility for flexible data adapter schema mapping."""

import difflib
import logging
from typing import Any, Dict, List, Optional, Set

from ..utils.error_handling import AdapterError


class FieldResolver:
    """Resolves field names using synonyms and provides helpful error messages."""

    # Canonical field names and their synonyms
    FIELD_SYNONYMS = {
        "step": [
            "global_step", "step", "iteration", "iter", "timestep", "step_id",
            "epoch", "batch", "update", "training_step"
        ],
        "reward": [
            "reward_scalar", "reward", "score", "return", "r", "reward_mean",
            "avg_reward", "mean_reward", "total_reward", "cumulative_reward"
        ],
        "kl": [
            "kl_to_ref", "kl", "kl_ref", "kl_value", "kl_mean",
            "kl_divergence", "kl_loss", "kl_penalty", "kl_regularization"
        ],
        "entropy": [
            "entropy", "entropy_mean", "avg_entropy", "mean_entropy",
            "policy_entropy", "action_entropy"
        ],
        "loss": [
            "loss", "total_loss", "policy_loss", "value_loss", "actor_loss",
            "critic_loss", "combined_loss", "training_loss"
        ],
        "phase": [
            "phase", "stage", "mode", "train_phase", "training_phase"
        ],
        "wall_time": [
            "wall_time", "timestamp", "time", "elapsed_time", "duration",
            "wall_time_ms", "timestamp_ms", "time_ms"
        ],
        "seed": [
            "seed", "random_seed", "rng_seed", "rng.python", "rng.python_seed"
        ],
        "run_id": [
            "run_id", "experiment_id", "run_name", "experiment_name", "job_id"
        ],
        "git_sha": [
            "git_sha", "commit_hash", "commit_sha", "version", "git_version"
        ],
        "lr": [
            "lr", "learning_rate", "lr_value", "current_lr", "optimizer_lr"
        ],
        "grad_norm": [
            "grad_norm", "gradient_norm", "grad_norm_value", "gradient_magnitude"
        ],
        "clip_frac": [
            "clip_frac", "clipped_ratio", "clipping_fraction", "clip_ratio"
        ],
        "tokens_in": [
            "tokens_in", "input_tokens", "input_length", "prompt_length"
        ],
        "tokens_out": [
            "tokens_out", "output_tokens", "output_length", "response_length"
        ]
    }

    def __init__(self, allow_dot_paths: bool = True):
        """Initialize the field resolver.

        Args:
            allow_dot_paths: Whether to support nested field access with dot notation
        """
        self.allow_dot_paths = allow_dot_paths
        self.logger = logging.getLogger(self.__class__.__name__)

        # Create reverse mapping from synonym to canonical name
        self._synonym_to_canonical = {}
        for canonical, synonyms in self.FIELD_SYNONYMS.items():
            for synonym in synonyms:
                self._synonym_to_canonical[synonym] = canonical

    def resolve_field(
        self,
        canonical_name: str,
        available_headers: List[str],
        field_map: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """Resolve a canonical field name to an actual header name.

        Args:
            canonical_name: The canonical field name (e.g., 'step', 'reward')
            available_headers: List of available column/field names
            field_map: Optional explicit mapping from canonical to actual names

        Returns:
            The resolved field name if found, None otherwise
        """
        if field_map and canonical_name in field_map:
            mapped_name = field_map[canonical_name]
            if mapped_name in available_headers:
                return mapped_name
            elif self.allow_dot_paths and self._check_nested_field(mapped_name, available_headers):
                return mapped_name
            else:
                self.logger.warning(f"Field map specifies '{mapped_name}' for '{canonical_name}' but it's not found in headers")
                return None

        # Try synonyms in order of preference
        synonyms = self.FIELD_SYNONYMS.get(canonical_name, [canonical_name])
        for synonym in synonyms:
            if synonym in available_headers:
                return synonym
            elif self.allow_dot_paths and self._check_nested_field(synonym, available_headers):
                return synonym

        return None

    def _check_nested_field(self, field_path: str, available_headers: List[str]) -> bool:
        """Check if a nested field path exists in the data structure.

        Args:
            field_path: Dot-separated field path (e.g., 'metrics.reward')
            available_headers: List of available top-level field names

        Returns:
            True if the nested field path is valid
        """
        if not self.allow_dot_paths or '.' not in field_path:
            return False

        # For now, we'll assume nested fields are valid if the top-level key exists
        # The actual validation will happen during data extraction
        top_level_key = field_path.split('.')[0]
        return top_level_key in available_headers

    def get_missing_fields(
        self,
        required_fields: List[str],
        available_headers: List[str],
        field_map: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """Get list of required fields that cannot be resolved.

        Args:
            required_fields: List of canonical field names that are required
            available_headers: List of available column/field names
            field_map: Optional explicit mapping from canonical to actual names

        Returns:
            List of canonical field names that could not be resolved
        """
        missing = []
        for field in required_fields:
            if not self.resolve_field(field, available_headers, field_map):
                missing.append(field)
        return missing

    def get_suggestions(
        self,
        canonical_name: str,
        available_headers: List[str],
        max_suggestions: int = 3
    ) -> List[str]:
        """Get suggestions for field names based on approximate matching.

        Args:
            canonical_name: The canonical field name that couldn't be resolved
            available_headers: List of available column/field names
            max_suggestions: Maximum number of suggestions to return

        Returns:
            List of suggested field names sorted by similarity
        """
        if not available_headers:
            return []

        # Get all synonyms for the canonical field
        synonyms = self.FIELD_SYNONYMS.get(canonical_name, [canonical_name])

        # Find approximate matches
        suggestions = []
        normalized_headers = {
            header: ''.join(ch for ch in header.lower() if ch.isalnum())
            for header in available_headers
        }

        for synonym in synonyms:
            matches = difflib.get_close_matches(
                synonym, available_headers, n=max_suggestions, cutoff=0.6
            )
            suggestions.extend(matches)

            normalized = ''.join(ch for ch in synonym.lower() if ch.isalnum())
            if normalized:
                normalized_matches = difflib.get_close_matches(
                    normalized,
                    list(normalized_headers.values()),
                    n=max_suggestions,
                    cutoff=0.6,
                )
                for match in normalized_matches:
                    for header, normalized_header in normalized_headers.items():
                        if normalized_header == match:
                            suggestions.append(header)

        # Also try matching against the canonical name itself
        canonical_matches = difflib.get_close_matches(
            canonical_name, available_headers, n=max_suggestions, cutoff=0.6
        )
        suggestions.extend(canonical_matches)

        normalized_canonical = ''.join(ch for ch in canonical_name.lower() if ch.isalnum())
        if normalized_canonical:
            normalized_canonical_matches = difflib.get_close_matches(
                normalized_canonical,
                list(normalized_headers.values()),
                n=max_suggestions,
                cutoff=0.6,
            )
            for match in normalized_canonical_matches:
                for header, normalized_header in normalized_headers.items():
                    if normalized_header == match:
                        suggestions.append(header)

        # Remove duplicates and limit results
        unique_suggestions = list(dict.fromkeys(suggestions))
        return unique_suggestions[:max_suggestions]

    def create_field_map_suggestion(
        self,
        missing_fields: List[str],
        available_headers: List[str]
    ) -> Dict[str, str]:
        """Create a field map suggestion for missing fields.

        Args:
            missing_fields: List of canonical field names that are missing
            available_headers: List of available column/field names

        Returns:
            Dictionary mapping canonical names to suggested actual names
        """
        suggestion = {}
        for field in missing_fields:
            suggestions = self.get_suggestions(field, available_headers)
            if suggestions:
                suggestion[field] = suggestions[0]
        return suggestion

    def validate_field_map(
        self,
        field_map: Dict[str, str],
        available_headers: List[str]
    ) -> Dict[str, Any]:
        """Validate a field map against available headers.

        Args:
            field_map: Dictionary mapping canonical names to actual names
            available_headers: List of available column/field names

        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "missing_fields": [],
            "invalid_mappings": [],
            "warnings": []
        }

        for canonical, actual in field_map.items():
            if actual in available_headers:
                continue

            if self.allow_dot_paths and '.' in actual:
                results["warnings"].append(
                    f"Field '{actual}' for '{canonical}' uses nested path - validation will happen during data extraction"
                )
                continue

            if self.allow_dot_paths and self._check_nested_field(actual, available_headers):
                results["warnings"].append(
                    f"Field '{actual}' for '{canonical}' uses nested path - validation will happen during data extraction"
                )
                continue

            results["invalid_mappings"].append(canonical)
            results["valid"] = False

        return results

    def get_canonical_fields(self) -> Set[str]:
        """Get the set of all canonical field names.

        Returns:
            Set of canonical field names
        """
        return set(self.FIELD_SYNONYMS.keys())

    def get_synonyms(self, canonical_name: str) -> List[str]:
        """Get all synonyms for a canonical field name.

        Args:
            canonical_name: The canonical field name

        Returns:
            List of synonyms for the field
        """
        return self.FIELD_SYNONYMS.get(canonical_name, [canonical_name])

    def log_resolution_summary(
        self,
        resolved_fields: Dict[str, str],
        missing_fields: List[str],
        total_records: int
    ) -> None:
        """Log a summary of field resolution results.

        Args:
            resolved_fields: Dictionary of resolved field mappings
            missing_fields: List of fields that could not be resolved
            total_records: Total number of records processed
        """
        self.logger.info(f"Loaded {total_records} records")

        if resolved_fields:
            self.logger.info("Field mappings resolved:")
            for canonical, actual in resolved_fields.items():
                self.logger.info(f"  {canonical} -> {actual}")

        if missing_fields:
            self.logger.warning(f"Missing fields: {', '.join(missing_fields)}")
        else:
            self.logger.info("All required fields found")


class SchemaError(AdapterError):
    """Error raised when schema validation fails."""

    def __init__(
        self,
        message: str,
        missing_fields: List[str],
        available_headers: List[str],
        field_resolver: FieldResolver,
        suggestion: Optional[str] = None
    ):
        """Initialize schema error with helpful suggestions.

        Args:
            message: Error message
            missing_fields: List of missing canonical field names
            available_headers: List of available headers
            field_resolver: Field resolver instance for generating suggestions
            suggestion: Optional custom suggestion
        """
        self.missing_fields = missing_fields
        self.available_headers = available_headers
        self.field_resolver = field_resolver

        # Generate suggestions for missing fields
        suggestions = []
        field_map_suggestion = {}

        for field in missing_fields:
            field_suggestions = field_resolver.get_suggestions(field, available_headers)
            if field_suggestions:
                suggestions.append(f"  {field}: {', '.join(field_suggestions)}")
                field_map_suggestion[field] = field_suggestions[0]

        # Create detailed error message
        detailed_message = message
        if suggestions:
            detailed_message += "\n\nFound similar fields:\n" + "\n".join(suggestions)

        if field_map_suggestion:
            field_map_str = ", ".join([f'"{k}": "{v}"' for k, v in field_map_suggestion.items()])
            detailed_message += f"\n\nTry this field_map: {{{field_map_str}}}"

        if not suggestion:
            suggestion = "Check field names and provide a field_map if needed"

        super().__init__(
            detailed_message,
            suggestion=suggestion,
            error_code="SCHEMA_VALIDATION_FAILED",
            details={
                "missing_fields": missing_fields,
                "available_headers": available_headers,
                "field_map_suggestion": field_map_suggestion
            }
        )
