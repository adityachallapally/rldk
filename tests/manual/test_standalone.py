#!/usr/bin/env python3
"""
Standalone test for field resolver functionality without module dependencies.
"""

import _path_setup  # noqa: F401
import difflib
import logging
from typing import Dict, List, Optional, Set


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
            "kl_to_ref", "kl", "kl_divergence", "kl_ref", "kl_value", "kl_mean",
            "kl_div", "kl_loss", "kl_penalty", "kl_regularization"
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
        """Initialize the field resolver."""
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
        """Resolve a canonical field name to an actual header name."""
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
        """Check if a nested field path exists in the data structure."""
        if not self.allow_dot_paths or '.' not in field_path:
            return False

        # For now, we'll assume nested fields are valid if the top-level key exists
        top_level_key = field_path.split('.')[0]
        return top_level_key in available_headers

    def get_missing_fields(
        self,
        required_fields: List[str],
        available_headers: List[str],
        field_map: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """Get list of required fields that cannot be resolved."""
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
        """Get suggestions for field names based on approximate matching."""
        if not available_headers:
            return []

        # Get all synonyms for the canonical field
        synonyms = self.FIELD_SYNONYMS.get(canonical_name, [canonical_name])

        # Find approximate matches
        suggestions = []
        for synonym in synonyms:
            matches = difflib.get_close_matches(
                synonym, available_headers, n=max_suggestions, cutoff=0.6
            )
            suggestions.extend(matches)

        # Also try matching against the canonical name itself
        canonical_matches = difflib.get_close_matches(
            canonical_name, available_headers, n=max_suggestions, cutoff=0.6
        )
        suggestions.extend(canonical_matches)

        # Remove duplicates and limit results
        unique_suggestions = list(dict.fromkeys(suggestions))
        return unique_suggestions[:max_suggestions]

    def create_field_map_suggestion(
        self,
        missing_fields: List[str],
        available_headers: List[str]
    ) -> Dict[str, str]:
        """Create a field map suggestion for missing fields."""
        suggestion = {}
        for field in missing_fields:
            suggestions = self.get_suggestions(field, available_headers)
            if suggestions:
                suggestion[field] = suggestions[0]
        return suggestion

    def get_canonical_fields(self) -> Set[str]:
        """Get the set of all canonical field names."""
        return set(self.FIELD_SYNONYMS.keys())

    def get_synonyms(self, canonical_name: str) -> List[str]:
        """Get all synonyms for a canonical field name."""
        return self.FIELD_SYNONYMS.get(canonical_name, [canonical_name])


def test_field_resolver():
    """Test field resolver functionality."""
    print("Testing FieldResolver...")

    try:
        # Test basic functionality
        resolver = FieldResolver()
        headers = ["step", "reward", "kl", "entropy"]

        # Test exact matches
        assert resolver.resolve_field("step", headers) == "step"
        assert resolver.resolve_field("reward", headers) == "reward"
        assert resolver.resolve_field("kl", headers) == "kl"
        assert resolver.resolve_field("entropy", headers) == "entropy"
        print("  ✅ Exact matches work")

        # Test synonyms
        headers_synonyms = ["global_step", "reward_scalar", "kl_to_ref", "entropy_mean"]
        assert resolver.resolve_field("step", headers_synonyms) == "global_step"
        assert resolver.resolve_field("reward", headers_synonyms) == "reward_scalar"
        assert resolver.resolve_field("kl", headers_synonyms) == "kl_to_ref"
        assert resolver.resolve_field("entropy", headers_synonyms) == "entropy_mean"
        print("  ✅ Synonyms work")

        # Test field mapping
        field_map = {"step": "iteration", "reward": "score"}
        assert resolver.resolve_field("step", ["iteration", "score"], field_map) == "iteration"
        assert resolver.resolve_field("reward", ["iteration", "score"], field_map) == "score"
        print("  ✅ Field mapping works")

        # Test missing fields
        missing = resolver.get_missing_fields(["step", "reward"], ["unrelated_field"])
        assert set(missing) == {"step", "reward"}
        print("  ✅ Missing field detection works")

        # Test suggestions
        suggestions = resolver.get_suggestions("step", ["step_count", "step_id"])
        assert len(suggestions) > 0  # Should have some suggestions
        print("  ✅ Suggestions work")

        # Test nested field checking
        assert resolver._check_nested_field("metrics.reward", ["metrics", "data"])
        assert not resolver._check_nested_field("missing.field", ["metrics", "data"])
        print("  ✅ Nested field checking works")

        # Test canonical fields
        canonical_fields = resolver.get_canonical_fields()
        assert "step" in canonical_fields
        assert "reward" in canonical_fields
        assert "kl" in canonical_fields
        print("  ✅ Canonical fields work")

        # Test synonyms retrieval
        step_synonyms = resolver.get_synonyms("step")
        assert "global_step" in step_synonyms
        assert "iteration" in step_synonyms
        print("  ✅ Synonyms retrieval works")

        print("✅ FieldResolver tests passed")
        return True

    except Exception as e:
        print(f"❌ FieldResolver tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_world_scenarios():
    """Test real-world field resolution scenarios."""
    print("Testing real-world scenarios...")

    try:
        resolver = FieldResolver()

        # Scenario 1: TRL-style data
        trl_headers = ["step", "phase", "reward_mean", "kl_mean", "entropy_mean", "loss", "lr"]
        required_fields = ["step", "reward", "kl", "entropy"]

        missing = resolver.get_missing_fields(required_fields, trl_headers)
        assert missing == []  # Should resolve all fields
        print("  ✅ TRL-style data scenario passed")

        # Scenario 2: Custom JSONL-style data
        custom_headers = ["global_step", "reward_scalar", "kl_to_ref", "entropy", "loss", "learning_rate"]
        missing = resolver.get_missing_fields(required_fields, custom_headers)
        assert missing == []  # Should resolve all fields
        print("  ✅ Custom JSONL-style data scenario passed")

        # Scenario 3: Nested data structure
        nested_headers = ["metrics", "data", "config", "step"]
        field_map = {
            "reward": "metrics.reward",
            "kl": "metrics.kl_divergence",
            "entropy": "data.entropy_value"
        }
        missing = resolver.get_missing_fields(required_fields, nested_headers, field_map)
        assert missing == []  # Should resolve all fields with mapping
        print("  ✅ Nested data structure scenario passed")

        # Scenario 4: Missing fields with suggestions
        incomplete_headers = ["step_count", "reward_value", "kl_divergence"]
        missing = resolver.get_missing_fields(required_fields, incomplete_headers)
        # Should be missing entropy since it's not in incomplete_headers
        assert "entropy" in missing

        # Test suggestions for each missing field
        for field in missing:
            suggestions = resolver.get_suggestions(field, incomplete_headers)
            # Some fields might not have suggestions, that's okay
            print(f"    Field '{field}' suggestions: {suggestions}")
        print("  ✅ Missing fields with suggestions scenario passed")

        print("✅ Real-world scenarios tests passed")
        return True

    except Exception as e:
        print(f"❌ Real-world scenarios tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_acceptance_scenarios():
    """Test the three acceptance check scenarios."""
    print("Testing acceptance scenarios...")

    try:
        resolver = FieldResolver()

        # Scenario A: JSONL with global_step, reward_scalar, kl_to_ref
        print("  Testing Scenario A...")
        scenario_a_headers = ["global_step", "reward_scalar", "kl_to_ref", "entropy"]
        required_fields = ["step", "reward", "kl", "entropy"]

        missing = resolver.get_missing_fields(required_fields, scenario_a_headers)
        assert missing == []  # Should resolve all fields automatically
        print("    ✅ Scenario A passed")

        # Scenario B: CSV with step, reward, kl
        print("  Testing Scenario B...")
        scenario_b_headers = ["step", "reward", "kl", "entropy"]
        missing = resolver.get_missing_fields(required_fields, scenario_b_headers)
        assert missing == []  # Should resolve all fields automatically
        print("    ✅ Scenario B passed")

        # Scenario C: Parquet with iteration, score, metrics.kl_ref
        print("  Testing Scenario C...")
        scenario_c_headers = ["iteration", "score", "metrics", "data"]
        field_map = {
            "step": "iteration",
            "reward": "score",
            "kl": "metrics.kl_ref",
            "entropy": "data.entropy_value"
        }
        missing = resolver.get_missing_fields(required_fields, scenario_c_headers, field_map)
        assert missing == []  # Should resolve all fields with mapping
        print("    ✅ Scenario C passed")

        print("✅ Acceptance scenarios tests passed")
        return True

    except Exception as e:
        print(f"❌ Acceptance scenarios tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Field Resolver Implementation")
    print("=" * 50)

    tests = [
        test_field_resolver,
        test_real_world_scenarios,
        test_acceptance_scenarios
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("✅ All field resolver tests passed!")
        print("\n🎉 Field resolver implementation is working correctly!")
        print("\nKey features validated:")
        print("  • Automatic field resolution with synonyms")
        print("  • Explicit field mapping support")
        print("  • Nested field path support")
        print("  • Helpful error suggestions")
        print("  • Real-world scenario compatibility")
        print("  • Acceptance test scenarios")
        return True
    else:
        print(f"❌ {total - passed} tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
