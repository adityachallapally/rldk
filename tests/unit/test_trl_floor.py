"""Test to ensure TRL version meets minimum requirements."""

def test_trl_version_floor():
    """Test that TRL version is >= 0.23.0."""
    try:
        import trl
        version_parts = trl.__version__.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        assert (major, minor) >= (0, 23), f"TRL version {trl.__version__} is below required minimum 0.23.0"
        print(f"✅ TRL version {trl.__version__} meets minimum requirement (>= 0.23.0)")
        return True
    except ImportError:
        print("❌ TRL is not installed")
        return False
    except (ValueError, IndexError) as e:
        print(f"❌ Could not parse TRL version: {e}")
        return False


def test_trl_imports():
    """Test that basic TRL imports work with the required version."""
    try:
        from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
        # These imports should work with TRL 0.23+
        assert PPOTrainer is not None
        assert PPOConfig is not None
        assert AutoModelForCausalLMWithValueHead is not None
        print("✅ Required TRL imports successful")
        return True
    except ImportError as e:
        print(f"❌ Required TRL imports failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing TRL version requirements...")
    print("=" * 50)

    version_ok = test_trl_version_floor()
    imports_ok = test_trl_imports()

    if version_ok and imports_ok:
        print("\n🎉 All TRL tests passed!")
        exit(0)
    else:
        print("\n❌ Some TRL tests failed.")
        exit(1)
