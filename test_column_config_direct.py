#!/usr/bin/env python3
"""Direct test of column configuration functionality."""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'rldk', 'evals'))

def test_column_config():
    """Test column configuration functionality."""
    print("Testing column configuration...")
    
    # Import the column config module directly
    from column_config import ColumnConfig, detect_columns, suggest_columns, get_evaluation_kwargs
    
    config = ColumnConfig()
    
    # Test getting default config
    throughput_config = config.get_config("throughput")
    assert "primary_column" in throughput_config
    assert "alternative_columns" in throughput_config
    assert "fallback_metrics" in throughput_config
    print("✓ Default config retrieval works")
    
    # Test setting primary column
    config.set_primary_column("throughput", "my_events")
    updated_config = config.get_config("throughput")
    assert updated_config["primary_column"] == "my_events"
    print("✓ Primary column setting works")
    
    # Test adding alternative column
    config.add_alternative_column("throughput", "custom_logs")
    updated_config = config.get_config("throughput")
    assert "custom_logs" in updated_config["alternative_columns"]
    print("✓ Alternative column addition works")

def test_detect_columns():
    """Test column detection functionality."""
    print("\nTesting column detection...")
    
    from column_config import detect_columns
    
    data_columns = ["response", "logs", "toxicity_score", "bias_score", "random_column"]
    
    detected = detect_columns(data_columns)
    
    assert "throughput" in detected
    assert "toxicity" in detected
    assert "bias" in detected
    print("✓ All metrics detected")
    
    # Check throughput detection
    assert detected["throughput"]["primary_found"] == False  # "events" not in data_columns
    assert "logs" in detected["throughput"]["alternatives_found"]
    print("✓ Throughput column detection works")
    
    # Check toxicity detection
    assert detected["toxicity"]["primary_found"] == False  # "output" not in data_columns
    assert "response" in detected["toxicity"]["alternatives_found"]
    assert "toxicity_score" in detected["toxicity"]["fallbacks_found"]
    print("✓ Toxicity column detection works")
    
    # Check bias detection
    assert detected["bias"]["primary_found"] == False  # "output" not in data_columns
    assert "response" in detected["bias"]["alternatives_found"]
    assert "bias_score" in detected["bias"]["fallbacks_found"]
    print("✓ Bias column detection works")

def test_suggest_columns():
    """Test column suggestion functionality."""
    print("\nTesting column suggestions...")
    
    from column_config import suggest_columns
    
    data_columns = ["response", "logs", "toxicity_score", "bias_score", "random_column"]
    
    suggestions = suggest_columns(data_columns)
    
    assert "throughput" in suggestions
    assert "toxicity" in suggestions
    assert "bias" in suggestions
    print("✓ All metric suggestions generated")
    
    # Check that suggestions are provided
    assert len(suggestions["throughput"]) > 0
    assert len(suggestions["toxicity"]) > 0
    assert len(suggestions["bias"]) > 0
    print("✓ Suggestions are non-empty")
    
    # Print suggestions for verification
    print("\nColumn suggestions:")
    for metric, suggestions_list in suggestions.items():
        print(f"  {metric}: {suggestions_list}")

def test_evaluation_kwargs():
    """Test evaluation kwargs generation."""
    print("\nTesting evaluation kwargs...")
    
    from column_config import get_evaluation_kwargs
    
    kwargs = get_evaluation_kwargs("throughput")
    
    assert "log_column" in kwargs
    assert "alternative_columns" in kwargs
    assert "fallback_to_other_metrics" in kwargs
    print("✓ Default evaluation kwargs generated")
    
    # Test with custom config
    custom_config = {
        "primary_column": "my_events",
        "alternative_columns": ["custom_logs"]
    }
    
    kwargs = get_evaluation_kwargs("throughput", custom_config)
    assert kwargs["log_column"] == "my_events"
    assert kwargs["alternative_columns"] == ["custom_logs"]
    print("✓ Custom evaluation kwargs work")

def main():
    """Run all tests."""
    print("Running robust column handling tests...\n")
    
    try:
        test_column_config()
        test_detect_columns()
        test_suggest_columns()
        test_evaluation_kwargs()
        
        print("\n🎉 All tests passed! Robust column handling is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)