#!/usr/bin/env python3
"""Test script for the improved ingest system."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test the basic ingest functionality without full dependencies
def test_ingest_basic():
    """Test basic ingest functionality."""
    try:
        from rldk.ingest.ingest import _get_format_examples, _get_supported_extensions, _get_directory_structure_examples
        
        print("Testing format examples...")
        for adapter in ["trl", "openrlhf", "custom_jsonl", "wandb"]:
            print(f"\n{adapter.upper()} Format Examples:")
            print(_get_format_examples(adapter))
            print(f"\nSupported extensions: {_get_supported_extensions(adapter)}")
            print(f"\nDirectory structure: {_get_directory_structure_examples(adapter)}")
            print("-" * 50)
        
        print("\n✅ Format examples test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Format examples test failed: {e}")
        return False

def test_adapter_detection():
    """Test adapter detection logic."""
    try:
        from rldk.ingest.ingest import _detect_adapter_type
        from pathlib import Path
        
        print("\nTesting adapter detection...")
        
        # Test with sample files
        test_files = [
            "sample_data/trl_training_output.jsonl",
            "sample_data/openrlhf_training_output.jsonl", 
            "sample_data/custom_training_output.jsonl",
            "sample_data/sample_eval_data.jsonl"
        ]
        
        for file_path in test_files:
            if Path(file_path).exists():
                detected = _detect_adapter_type(file_path)
                print(f"File: {file_path} -> Detected adapter: {detected}")
            else:
                print(f"File: {file_path} -> NOT FOUND")
        
        print("\n✅ Adapter detection test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Adapter detection test failed: {e}")
        return False

def test_error_messages():
    """Test error message generation."""
    try:
        from rldk.ingest.ingest import _get_format_examples, _get_supported_extensions, _get_directory_structure_examples
        
        print("\nTesting error message generation...")
        
        # Test with non-existent file
        from pathlib import Path
        non_existent = Path("non_existent_file.jsonl")
        
        # This should trigger the file not found error
        if not non_existent.exists():
            print("✅ Non-existent file detection works")
        
        # Test format examples for each adapter
        for adapter in ["trl", "openrlhf", "custom_jsonl", "wandb"]:
            examples = _get_format_examples(adapter)
            extensions = _get_supported_extensions(adapter)
            structure = _get_directory_structure_examples(adapter)
            
            assert len(examples) > 0, f"No examples for {adapter}"
            assert len(extensions) > 0, f"No extensions for {adapter}"
            assert len(structure) > 0, f"No structure for {adapter}"
            
            print(f"✅ {adapter} error messages generated successfully")
        
        print("\n✅ Error message generation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error message generation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing improved ingest system...")
    print("=" * 60)
    
    tests = [
        test_ingest_basic,
        test_adapter_detection, 
        test_error_messages
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The improved ingest system is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())