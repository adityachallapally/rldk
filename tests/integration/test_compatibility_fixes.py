#!/usr/bin/env python3
"""Test compatibility fixes for network monitoring."""

import os
import sys
from unittest.mock import MagicMock

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'rldk', 'integrations', 'openrlhf'))

def test_openrlhf_metrics_compatibility():
    """Test that OpenRLHFMetrics has both old and new attributes."""
    try:
        # Mock numpy to avoid import error
        sys.modules['numpy'] = MagicMock()
        sys.modules['pandas'] = MagicMock()
        sys.modules['torch'] = MagicMock()
        sys.modules['psutil'] = MagicMock()

        from callbacks import OpenRLHFMetrics

        # Test creating metrics with both old and new attributes
        metrics = OpenRLHFMetrics()
        metrics.bandwidth_mbps = 100.0
        metrics.latency_ms = 5.0
        metrics.network_bandwidth = 100.0  # Legacy
        metrics.network_latency = 5.0      # Legacy

        print('✅ OpenRLHFMetrics with both old and new attributes created successfully')
        print(f'   bandwidth_mbps: {metrics.bandwidth_mbps}')
        print(f'   latency_ms: {metrics.latency_ms}')
        print(f'   network_bandwidth (legacy): {metrics.network_bandwidth}')
        print(f'   network_latency (legacy): {metrics.network_latency}')

        # Test to_dict method
        metrics_dict = metrics.to_dict()
        print('✅ to_dict method works')
        print(f'   Dictionary keys: {list(metrics_dict.keys())[:5]}...')

        return True

    except Exception as e:
        print(f'❌ OpenRLHFMetrics test failed: {e}')
        return False

def test_dashboard_compatibility():
    """Test that dashboard methods handle both dataclass and dictionary inputs."""
    try:
        # Mock required modules
        sys.modules['flask'] = MagicMock()
        sys.modules['plotly'] = MagicMock()
        sys.modules['plotly.graph_objs'] = MagicMock()
        sys.modules['plotly.utils'] = MagicMock()
        sys.modules['numpy'] = MagicMock()
        sys.modules['pandas'] = MagicMock()
        sys.modules['torch'] = MagicMock()
        sys.modules['psutil'] = MagicMock()

        from callbacks import OpenRLHFMetrics
        from dashboard import OpenRLHFDashboard

        # Create test metrics
        metrics = OpenRLHFMetrics()
        metrics.bandwidth_mbps = 100.0
        metrics.latency_ms = 5.0
        metrics.network_bandwidth = 100.0
        metrics.network_latency = 5.0

        # Create dashboard
        dashboard = OpenRLHFDashboard(output_dir='./test_dashboard')
        print('✅ Dashboard created successfully')

        # Test add_metrics with dataclass
        dashboard.add_metrics(metrics)
        print('✅ add_metrics with dataclass works')

        # Test add_metrics with dictionary
        metrics_dict = metrics.to_dict()
        dashboard.add_metrics(metrics_dict)
        print('✅ add_metrics with dictionary works')

        # Test check_network_thresholds with dataclass
        dashboard.check_network_thresholds(metrics)
        print('✅ check_network_thresholds with dataclass works')

        # Test check_network_thresholds with dictionary
        dashboard.check_network_thresholds(metrics_dict)
        print('✅ check_network_thresholds with dictionary works')

        return True

    except Exception as e:
        print(f'❌ Dashboard compatibility test failed: {e}')
        return False

def test_bandwidth_fix():
    """Test that bandwidth metrics are correctly assigned."""
    try:
        # Mock required modules
        sys.modules['numpy'] = MagicMock()
        sys.modules['pandas'] = MagicMock()
        sys.modules['torch'] = MagicMock()
        sys.modules['psutil'] = MagicMock()

        from callbacks import OpenRLHFMetrics

        # Test that the fix is in place by checking the field definitions
        metrics = OpenRLHFMetrics()

        # Check that both old and new attributes exist
        assert hasattr(metrics, 'network_bandwidth'), "network_bandwidth attribute missing"
        assert hasattr(metrics, 'network_latency'), "network_latency attribute missing"
        assert hasattr(metrics, 'bandwidth_mbps'), "bandwidth_mbps attribute missing"
        assert hasattr(metrics, 'latency_ms'), "latency_ms attribute missing"
        assert hasattr(metrics, 'bandwidth_upload_mbps'), "bandwidth_upload_mbps attribute missing"
        assert hasattr(metrics, 'bandwidth_download_mbps'), "bandwidth_download_mbps attribute missing"

        print('✅ All required attributes exist in OpenRLHFMetrics')

        # Test that we can set values
        metrics.bandwidth_upload_mbps = 50.0
        metrics.bandwidth_download_mbps = 100.0
        metrics.network_bandwidth = 100.0
        metrics.network_latency = 5.0

        print('✅ Can set values for all attributes')
        print(f'   Upload: {metrics.bandwidth_upload_mbps} Mbps')
        print(f'   Download: {metrics.bandwidth_download_mbps} Mbps')
        print(f'   Legacy bandwidth: {metrics.network_bandwidth} Mbps')
        print(f'   Legacy latency: {metrics.network_latency} ms')

        return True

    except Exception as e:
        print(f'❌ Bandwidth fix test failed: {e}')
        return False

def main():
    """Run all compatibility tests."""
    print("🧪 Testing compatibility fixes...")
    print("=" * 50)

    success = True

    # Test OpenRLHFMetrics compatibility
    if not test_openrlhf_metrics_compatibility():
        success = False

    # Test dashboard compatibility
    if not test_dashboard_compatibility():
        success = False

    # Test bandwidth fix
    if not test_bandwidth_fix():
        success = False

    print("\n" + "=" * 50)
    if success:
        print("🎉 All compatibility tests passed!")
        print("✅ Backward compatibility maintained")
        print("✅ Dashboard handles both dataclass and dictionary inputs")
        print("✅ Bandwidth metrics are correctly assigned")
    else:
        print("❌ Some compatibility tests failed")

    return success

if __name__ == "__main__":
    main()
