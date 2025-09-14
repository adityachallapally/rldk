#!/usr/bin/env python3
"""Simple test to verify RLDK imports work correctly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import rldk
    print("✅ RLDK imports successfully")
    print(f"   RLDK version: {getattr(rldk, '__version__', 'unknown')}")
    
    from rldk.tracking import ExperimentTracker
    from rldk.forensics import ComprehensivePPOForensics
    from rldk.determinism import check
    print("✅ Key RLDK components import successfully")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)

print("✅ All imports successful")
