#!/usr/bin/env python3
"""
Validation script for TRL callback JSONL implementation.

This script validates that the JSONL event emission works correctly
without requiring full dependency installation.
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def validate_imports():
    """Validate that all required imports work."""
    print("🔍 Validating imports...")
    
    try:
        from rldk.integrations.trl.callbacks import RLDKCallback
        print("✅ RLDKCallback import successful")
        
        from rldk.io.event_schema import Event, create_event_from_row
        print("✅ Event schema import successful")
        
        from rldk.adapters.trl import TRLAdapter
        print("✅ TRLAdapter import successful")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def validate_jsonl_emission():
    """Validate JSONL event emission."""
    print("\n📝 Validating JSONL event emission...")
    
    try:
        from rldk.integrations.trl.callbacks import RLDKCallback
        from rldk.io.event_schema import Event, create_event_from_row
        from rldk.adapters.trl import TRLAdapter
        from unittest.mock import Mock
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Create callback
            callback = RLDKCallback(
                output_dir=output_dir,
                enable_jsonl_logging=True,
                jsonl_log_interval=1,
                run_id="validation_test"
            )
            
            # Check JSONL file creation
            jsonl_files = list(output_dir.glob("*_events.jsonl"))
            if len(jsonl_files) != 1:
                print(f"❌ Expected 1 JSONL file, found {len(jsonl_files)}")
                return False
            
            print("✅ JSONL file creation successful")
            
            # Mock training step
            state = Mock()
            state.global_step = 1
            state.epoch = 0.1
            
            logs = {
                'train_loss': 0.5,
                'learning_rate': 0.001,
                'ppo/rewards/mean': 0.8,
                'ppo/policy/kl_mean': 0.05,
            }
            
            # Update metrics
            callback.current_metrics.step = 1
            callback.current_metrics.loss = 0.5
            callback.current_metrics.learning_rate = 0.001
            callback.current_metrics.reward_mean = 0.8
            callback.current_metrics.kl_mean = 0.05
            callback.current_metrics.wall_time = 10.0
            
            # Emit event
            callback._emit_jsonl_event(state, logs)
            callback._close_jsonl_file()
            
            # Validate event structure
            with open(jsonl_files[0], 'r') as f:
                lines = f.readlines()
                if len(lines) != 1:
                    print(f"❌ Expected 1 event, found {len(lines)}")
                    return False
                
                event_data = json.loads(lines[0])
                
                # Check required fields
                required_fields = ["step", "wall_time", "metrics", "rng", "data_slice", "model_info", "notes"]
                for field in required_fields:
                    if field not in event_data:
                        print(f"❌ Missing required field: {field}")
                        return False
                
                # Check specific values
                if event_data["step"] != 1:
                    print(f"❌ Expected step 1, got {event_data['step']}")
                    return False
                
                if event_data["metrics"]["loss"] != 0.5:
                    print(f"❌ Expected loss 0.5, got {event_data['metrics']['loss']}")
                    return False
                
                print("✅ JSONL event structure validation successful")
            
            # Test TRLAdapter compatibility
            adapter = TRLAdapter(jsonl_files[0])
            if not adapter.can_handle():
                print("❌ TRLAdapter cannot handle generated file")
                return False
            
            df = adapter.load()
            if len(df) != 1:
                print(f"❌ Expected 1 row in DataFrame, got {len(df)}")
                return False
            
            print("✅ TRLAdapter compatibility validation successful")
            
            return True
            
    except Exception as e:
        print(f"❌ JSONL emission validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main validation function."""
    print("🚀 TRL Callback JSONL Implementation Validation")
    print("=" * 50)
    
    # Validate imports
    if not validate_imports():
        print("\n❌ Import validation failed")
        return False
    
    # Validate JSONL emission
    if not validate_jsonl_emission():
        print("\n❌ JSONL emission validation failed")
        return False
    
    print("\n🎉 All validations passed!")
    print("✅ JSONL implementation is working correctly")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)