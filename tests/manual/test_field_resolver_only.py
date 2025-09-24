#!/usr/bin/env python3
"""
Simple test script to validate field resolver implementation without pandas dependency.
"""

import _path_setup  # noqa: F401

def test_field_resolver():
    """Test field resolver functionality."""
    print("Testing FieldResolver...")

    try:
        from rldk.adapters.field_resolver import FieldResolver

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
        assert "step_count" in suggestions
        assert "step_id" in suggestions
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

def test_schema_error():
    """Test schema error functionality."""
    print("Testing SchemaError...")

    try:
        from rldk.adapters.field_resolver import FieldResolver, SchemaError

        resolver = FieldResolver()
        missing_fields = ["step", "reward"]
        available_headers = ["step_count", "reward_value", "kl_divergence"]

        error = SchemaError(
            "Missing required fields",
            missing_fields,
            available_headers,
            resolver
        )

        # Check error properties
        assert "step" in error.missing_fields
        assert "reward" in error.missing_fields
        assert "step_count" in error.available_headers
        assert "reward_value" in error.available_headers

        # Check error message contains suggestions
        error_message = str(error)
        assert "step_count" in error_message
        assert "reward_value" in error_message
        assert "field_map" in error_message.lower()

        print("✅ SchemaError tests passed")
        return True

    except Exception as e:
        print(f"❌ SchemaError tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_world_scenarios():
    """Test real-world field resolution scenarios."""
    print("Testing real-world scenarios...")

    try:
        from rldk.adapters.field_resolver import FieldResolver

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
        assert set(missing) == {"step", "reward", "kl", "entropy"}

        # Test suggestions for each missing field
        for field in missing:
            suggestions = resolver.get_suggestions(field, incomplete_headers)
            assert len(suggestions) > 0  # Should have suggestions
        print("  ✅ Missing fields with suggestions scenario passed")

        print("✅ Real-world scenarios tests passed")
        return True

    except Exception as e:
        print(f"❌ Real-world scenarios tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Field Resolver Implementation")
    print("=" * 50)

    tests = [
        test_field_resolver,
        test_schema_error,
        test_real_world_scenarios
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
        return True
    else:
        print(f"❌ {total - passed} tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
