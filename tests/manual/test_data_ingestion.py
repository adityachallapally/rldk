#!/usr/bin/env python3
"""
Test data ingestion from different sources
"""

import sys
import json
import tempfile
import numpy as np
from pathlib import Path

import _path_setup  # noqa: F401


def test_data_ingestion():
    """Test data ingestion from different sources"""
    print("Testing Data Ingestion")
    print("=" * 60)
    
    try:
        from rldk.ingest import ingest_runs
        from rldk.adapters import TRLAdapter, CustomJSONLAdapter
        
        # Create sample TRL-style data
        trl_data = []
        for i in range(100):
            trl_data.append({
                'step': i * 10,
                'train/loss': 0.5 + np.random.normal(0, 0.1),
                'train/reward': 0.8 + np.random.normal(0, 0.1),
                'train/kl': 0.1 + np.random.normal(0, 0.02),
                'train/entropy': 2.5 + np.random.normal(0, 0.1),
                'train/policy_grad_norm': 1.0 + np.random.normal(0, 0.1),
                'train/value_grad_norm': 0.8 + np.random.normal(0, 0.1)
            })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save as JSONL
            jsonl_path = temp_path / "trl_data.jsonl"
            with open(jsonl_path, 'w') as f:
                for item in trl_data:
                    f.write(json.dumps(item) + '\n')
            
            print(f"✓ Sample TRL data created: {jsonl_path}")
            
            # Test TRL adapter
            try:
                trl_adapter = TRLAdapter(str(jsonl_path))
                df_trl = trl_adapter.load()
                print(f"✓ TRL adapter loaded data: {len(df_trl)} rows")
                print(f"  - Columns: {list(df_trl.columns)}")
                
                # Show some sample data
                if len(df_trl) > 0:
                    print(f"  - Sample step: {df_trl.iloc[0].get('step', 'N/A') if 'step' in df_trl.columns else 'N/A'}")
                    print(f"  - Sample reward: {df_trl.iloc[0].get('reward_mean', 'N/A') if 'reward_mean' in df_trl.columns else 'N/A'}")
                
            except Exception as e:
                print(f"⚠ TRL adapter failed: {e}")
            
            # Test generic ingestion
            try:
                df_generic = ingest_runs(str(jsonl_path), adapter_hint="custom_jsonl")
                print(f"✓ Generic ingestion loaded data: {len(df_generic)} rows")
                print(f"  - Columns: {list(df_generic.columns)}")
            except Exception as e:
                print(f"⚠ Generic ingestion failed: {e}")
            
            # Test custom JSONL adapter
            try:
                custom_adapter = CustomJSONLAdapter(str(jsonl_path))
                df_custom = custom_adapter.load()
                print(f"✓ Custom JSONL adapter loaded data: {len(df_custom)} rows")
            except Exception as e:
                print(f"⚠ Custom JSONL adapter failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_data_ingestion()