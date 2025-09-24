#!/usr/bin/env python3
"""Basic import test for network monitoring components."""

import os
import sys
from unittest.mock import MagicMock, patch

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'rldk', 'integrations', 'openrlhf'))

def test_imports():
    """Test basic imports without external dependencies."""
    try:
        # Test importing the basic classes
        from network_monitor import NetworkMetrics
        print("✅ NetworkMetrics imported successfully")

        # Test creating an instance
        metrics = NetworkMetrics(
            bandwidth_mbps=100.0,
            latency_ms=5.0,
            timestamp=1234567890.0
        )
        print("✅ NetworkMetrics instance created successfully")

        # Test to_dict method
        metrics_dict = metrics.to_dict()
        print("✅ to_dict method works")
        print(f"   Bandwidth: {metrics_dict['bandwidth_mbps']}")
        print(f"   Latency: {metrics_dict['latency_ms']}")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_distributed_imports():
    """Test distributed module imports."""
    try:
        # Mock psutil for testing
        # Create a mock psutil module
        mock_psutil = MagicMock()
        mock_net_io = MagicMock()
        mock_net_io.bytes_sent = 1000
        mock_net_io.bytes_recv = 2000
        mock_psutil.net_io_counters.return_value = mock_net_io

        with patch.dict(sys.modules, {'psutil': mock_psutil}):
            # Now try to import the distributed module
            from distributed import NetworkMonitor
            print("✅ NetworkMonitor imported successfully")

            # Test creating an instance
            NetworkMonitor()
            print("✅ NetworkMonitor instance created successfully")

        return True

    except ImportError as e:
        print(f"❌ Distributed import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Distributed error: {e}")
        return False

def test_dashboard_imports():
    """Test dashboard module imports."""
    try:
        # Mock required modules
        # Mock Flask
        mock_flask = MagicMock()
        # Mock plotly
        mock_plotly = MagicMock()
        # Mock numpy
        mock_numpy = MagicMock()
        # Mock pandas
        mock_pandas = MagicMock()
        patched_modules = {
            'flask': mock_flask,
            'plotly': mock_plotly,
            'numpy': mock_numpy,
            'pandas': mock_pandas,
        }

        with patch.dict(sys.modules, patched_modules):
            # Now try to import the dashboard module
            from dashboard import OpenRLHFDashboard
            print("✅ OpenRLHFDashboard imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Dashboard import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        return False

def main():
    """Run all import tests."""
    print("🧪 Testing basic imports...")
    print("=" * 50)

    success = True

    # Test basic network monitor imports
    if not test_imports():
        success = False

    # Test distributed imports
    if not test_distributed_imports():
        success = False

    # Test dashboard imports
    if not test_dashboard_imports():
        success = False

    print("\n" + "=" * 50)
    if success:
        print("✅ All basic imports successful!")
        print("📝 Note: External dependencies (psutil, numpy, pandas, etc.) need to be installed for full functionality")
    else:
        print("❌ Some imports failed")

    return success

if __name__ == "__main__":
    main()
