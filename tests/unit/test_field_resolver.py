"""Tests for field resolver utility."""

import pytest

from rldk.adapters.field_resolver import FieldResolver, SchemaError


class TestFieldResolver:
    """Test field resolver functionality."""

    def test_init(self):
        """Test field resolver initialization."""
        resolver = FieldResolver()
        assert resolver.allow_dot_paths is True

        resolver_no_dots = FieldResolver(allow_dot_paths=False)
        assert resolver_no_dots.allow_dot_paths is False

    def test_resolve_field_exact_match(self):
        """Test resolving fields with exact matches."""
        resolver = FieldResolver()
        headers = ["step", "reward", "kl", "entropy"]

        # Test exact matches
        assert resolver.resolve_field("step", headers) == "step"
        assert resolver.resolve_field("reward", headers) == "reward"
        assert resolver.resolve_field("kl", headers) == "kl"
        assert resolver.resolve_field("entropy", headers) == "entropy"

    def test_resolve_field_synonyms(self):
        """Test resolving fields using synonyms."""
        resolver = FieldResolver()
        headers = ["global_step", "reward_scalar", "kl_to_ref", "entropy_mean"]

        # Test synonym resolution
        assert resolver.resolve_field("step", headers) == "global_step"
        assert resolver.resolve_field("reward", headers) == "reward_scalar"
        assert resolver.resolve_field("kl", headers) == "kl_to_ref"
        assert resolver.resolve_field("entropy", headers) == "entropy_mean"

    def test_resolve_field_with_field_map(self):
        """Test resolving fields with explicit field mapping."""
        resolver = FieldResolver()
        headers = ["iteration", "score", "kl_divergence", "policy_entropy"]
        field_map = {
            "step": "iteration",
            "reward": "score",
            "kl": "kl_divergence",
            "entropy": "policy_entropy"
        }

        # Test field map resolution
        assert resolver.resolve_field("step", headers, field_map) == "iteration"
        assert resolver.resolve_field("reward", headers, field_map) == "score"
        assert resolver.resolve_field("kl", headers, field_map) == "kl_divergence"
        assert resolver.resolve_field("entropy", headers, field_map) == "policy_entropy"

    def test_resolve_field_nested_paths(self):
        """Test resolving fields with nested dot paths."""
        resolver = FieldResolver(allow_dot_paths=True)
        headers = ["metrics", "data", "config"]

        # Test nested path resolution
        assert resolver.resolve_field("reward", headers, {"reward": "metrics.reward"}) == "metrics.reward"
        assert resolver.resolve_field("kl", headers, {"kl": "data.kl_value"}) == "data.kl_value"

    def test_resolve_field_no_match(self):
        """Test resolving fields when no match is found."""
        resolver = FieldResolver()
        headers = ["unrelated_field1", "unrelated_field2"]

        # Test no match
        assert resolver.resolve_field("step", headers) is None
        assert resolver.resolve_field("reward", headers) is None

    def test_get_missing_fields(self):
        """Test getting missing required fields."""
        resolver = FieldResolver()
        headers = ["step", "reward"]
        required_fields = ["step", "reward", "kl", "entropy"]

        missing = resolver.get_missing_fields(required_fields, headers)
        assert set(missing) == {"kl", "entropy"}

    def test_get_missing_fields_with_field_map(self):
        """Test getting missing fields with field mapping."""
        resolver = FieldResolver()
        headers = ["iteration", "score"]
        required_fields = ["step", "reward", "kl"]
        field_map = {"step": "iteration", "reward": "score"}

        missing = resolver.get_missing_fields(required_fields, headers, field_map)
        assert missing == ["kl"]

    def test_get_suggestions(self):
        """Test getting field name suggestions."""
        resolver = FieldResolver()
        headers = ["global_step", "reward_scalar", "kl_divergence", "entropy_value"]

        # Test suggestions for step
        suggestions = resolver.get_suggestions("step", headers)
        assert "global_step" in suggestions

        # Test suggestions for reward
        suggestions = resolver.get_suggestions("reward", headers)
        assert "reward_scalar" in suggestions

        # Test suggestions for kl
        suggestions = resolver.get_suggestions("kl", headers)
        assert "kl_divergence" in suggestions

    def test_get_suggestions_empty_headers(self):
        """Test getting suggestions with empty headers."""
        resolver = FieldResolver()
        suggestions = resolver.get_suggestions("step", [])
        assert suggestions == []

    def test_create_field_map_suggestion(self):
        """Test creating field map suggestions."""
        resolver = FieldResolver()
        headers = ["global_step", "reward_scalar", "kl_divergence"]
        missing_fields = ["step", "reward", "kl"]

        suggestion = resolver.create_field_map_suggestion(missing_fields, headers)
        assert suggestion["step"] == "global_step"
        assert suggestion["reward"] == "reward_scalar"
        assert suggestion["kl"] == "kl_divergence"

    def test_validate_field_map(self):
        """Test validating field map against headers."""
        resolver = FieldResolver()
        headers = ["step", "reward", "kl"]
        field_map = {
            "step": "step",
            "reward": "reward",
            "kl": "kl",
            "entropy": "missing_field"
        }

        result = resolver.validate_field_map(field_map, headers)
        assert result["valid"] is False
        assert "entropy" in result["invalid_mappings"]
        assert "step" not in result["invalid_mappings"]

    def test_validate_field_map_nested_paths(self):
        """Test validating field map with nested paths."""
        resolver = FieldResolver(allow_dot_paths=True)
        headers = ["metrics", "data"]
        field_map = {
            "reward": "metrics.reward",
            "kl": "data.kl_value",
            "entropy": "missing.field"
        }

        result = resolver.validate_field_map(field_map, headers)
        # Nested paths should generate warnings but not invalid mappings
        assert result["valid"] is True
        assert len(result["warnings"]) > 0

    def test_get_canonical_fields(self):
        """Test getting canonical field names."""
        resolver = FieldResolver()
        canonical_fields = resolver.get_canonical_fields()

        expected_fields = {
            "step", "reward", "kl", "entropy", "loss", "phase", "wall_time",
            "seed", "run_id", "git_sha", "lr", "grad_norm", "clip_frac",
            "tokens_in", "tokens_out"
        }
        assert canonical_fields == expected_fields

    def test_get_synonyms(self):
        """Test getting synonyms for canonical fields."""
        resolver = FieldResolver()

        # Test step synonyms
        step_synonyms = resolver.get_synonyms("step")
        assert "global_step" in step_synonyms
        assert "iteration" in step_synonyms
        assert "step" in step_synonyms

        # Test reward synonyms
        reward_synonyms = resolver.get_synonyms("reward")
        assert "reward_scalar" in reward_synonyms
        assert "score" in reward_synonyms
        assert "reward" in reward_synonyms

        # Test unknown field
        unknown_synonyms = resolver.get_synonyms("unknown_field")
        assert unknown_synonyms == ["unknown_field"]

    def test_check_nested_field(self):
        """Test checking nested field paths."""
        resolver = FieldResolver(allow_dot_paths=True)
        headers = ["metrics", "data", "config"]

        # Test valid nested paths
        assert resolver._check_nested_field("metrics.reward", headers) is True
        assert resolver._check_nested_field("data.kl_value", headers) is True

        # Test invalid nested paths
        assert resolver._check_nested_field("missing.field", headers) is False
        assert resolver._check_nested_field("simple_field", headers) is False

        # Test with dot paths disabled
        resolver_no_dots = FieldResolver(allow_dot_paths=False)
        assert resolver_no_dots._check_nested_field("metrics.reward", headers) is False


class TestSchemaError:
    """Test schema error functionality."""

    def test_schema_error_creation(self):
        """Test creating schema error with suggestions."""
        resolver = FieldResolver()
        missing_fields = ["step", "reward"]
        available_headers = ["global_step", "reward_scalar", "kl_value"]

        error = SchemaError(
            "Missing required fields",
            missing_fields,
            available_headers,
            resolver
        )

        assert "step" in error.missing_fields
        assert "reward" in error.missing_fields
        assert "global_step" in error.available_headers
        assert "reward_scalar" in error.available_headers

        # Check that suggestions are included in the message
        assert "global_step" in str(error)
        assert "reward_scalar" in str(error)

    def test_schema_error_with_field_map_suggestion(self):
        """Test schema error with field map suggestion."""
        resolver = FieldResolver()
        missing_fields = ["step", "reward"]
        available_headers = ["global_step", "reward_scalar"]

        error = SchemaError(
            "Missing required fields",
            missing_fields,
            available_headers,
            resolver
        )

        # Check that field map suggestion is included
        assert '{"step": "global_step", "reward": "reward_scalar"}' in str(error)

    def test_schema_error_no_suggestions(self):
        """Test schema error when no suggestions are available."""
        resolver = FieldResolver()
        missing_fields = ["step", "reward"]
        available_headers = ["completely_unrelated_field"]

        error = SchemaError(
            "Missing required fields",
            missing_fields,
            available_headers,
            resolver
        )

        # Should still create error but without field map suggestion
        assert "step" in error.missing_fields
        assert "reward" in error.missing_fields
        assert "completely_unrelated_field" in error.available_headers


class TestFieldResolverIntegration:
    """Integration tests for field resolver."""

    def test_real_world_scenario_1(self):
        """Test scenario with TRL-style data."""
        resolver = FieldResolver()
        headers = ["step", "phase", "reward_mean", "kl_mean", "entropy_mean", "loss", "lr"]
        required_fields = ["step", "reward", "kl", "entropy"]

        # Should resolve all fields
        missing = resolver.get_missing_fields(required_fields, headers)
        assert missing == []

        # Test individual resolutions
        assert resolver.resolve_field("step", headers) == "step"
        assert resolver.resolve_field("reward", headers) == "reward_mean"
        assert resolver.resolve_field("kl", headers) == "kl_mean"
        assert resolver.resolve_field("entropy", headers) == "entropy_mean"

    def test_real_world_scenario_2(self):
        """Test scenario with custom JSONL-style data."""
        resolver = FieldResolver()
        headers = ["global_step", "reward_scalar", "kl_to_ref", "entropy", "loss", "learning_rate"]
        required_fields = ["step", "reward", "kl", "entropy"]

        # Should resolve all fields
        missing = resolver.get_missing_fields(required_fields, headers)
        assert missing == []

        # Test individual resolutions
        assert resolver.resolve_field("step", headers) == "global_step"
        assert resolver.resolve_field("reward", headers) == "reward_scalar"
        assert resolver.resolve_field("kl", headers) == "kl_to_ref"
        assert resolver.resolve_field("entropy", headers) == "entropy"

    def test_real_world_scenario_3(self):
        """Test scenario with nested data structure."""
        resolver = FieldResolver(allow_dot_paths=True)
        headers = ["metrics", "data", "config", "step"]
        field_map = {
            "reward": "metrics.reward",
            "kl": "metrics.kl_divergence",
            "entropy": "data.entropy_value"
        }
        required_fields = ["step", "reward", "kl", "entropy"]

        # Should resolve all fields including nested ones
        missing = resolver.get_missing_fields(required_fields, headers, field_map)
        assert missing == []

        # Test individual resolutions
        assert resolver.resolve_field("step", headers, field_map) == "step"
        assert resolver.resolve_field("reward", headers, field_map) == "metrics.reward"
        assert resolver.resolve_field("kl", headers, field_map) == "metrics.kl_divergence"
        assert resolver.resolve_field("entropy", headers, field_map) == "data.entropy_value"

    def test_missing_required_fields_scenario(self):
        """Test scenario where required fields are missing."""
        resolver = FieldResolver()
        headers = ["unrelated_field1", "unrelated_field2", "some_metric"]
        required_fields = ["step", "reward", "kl"]

        # Should identify missing fields
        missing = resolver.get_missing_fields(required_fields, headers)
        assert set(missing) == {"step", "reward", "kl"}

        # Should provide suggestions
        step_suggestions = resolver.get_suggestions("step", headers)
        reward_suggestions = resolver.get_suggestions("reward", headers)
        kl_suggestions = resolver.get_suggestions("kl", headers)

        # Suggestions should be empty since no similar fields exist
        assert step_suggestions == []
        assert reward_suggestions == []
        assert kl_suggestions == []

    def test_partial_field_match_scenario(self):
        """Test scenario with partial field matches."""
        resolver = FieldResolver()
        headers = ["step_count", "reward_value", "kl_div", "entropy_measure"]
        required_fields = ["step", "reward", "kl", "entropy"]

        # Should identify missing fields
        missing = resolver.get_missing_fields(required_fields, headers)
        assert set(missing) == {"step", "reward", "kl", "entropy"}

        # Should provide suggestions based on similarity
        step_suggestions = resolver.get_suggestions("step", headers)
        reward_suggestions = resolver.get_suggestions("reward", headers)
        kl_suggestions = resolver.get_suggestions("kl", headers)
        entropy_suggestions = resolver.get_suggestions("entropy", headers)

        # Should find similar fields
        assert "step_count" in step_suggestions
        assert "reward_value" in reward_suggestions
        assert "kl_div" in kl_suggestions
        assert "entropy_measure" in entropy_suggestions
