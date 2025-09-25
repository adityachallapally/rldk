"""Integration tests for OpenRLHF network monitoring."""

import importlib
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))


@contextmanager
def _mock_openrlhf_modules():
    """Temporarily mock OpenRLHF modules during imports."""

    def _create_psutil_mock() -> MagicMock:
        psutil_mock = MagicMock()
        psutil_mock.__version__ = "0.0.0"
        psutil_mock.PROCFS_PATH = '/proc'
        psutil_mock.cpu_percent.return_value = 0.0
        psutil_mock.cpu_count.return_value = 1
        psutil_mock.cpu_times.return_value = MagicMock()
        psutil_mock.virtual_memory.return_value = MagicMock(total=0, available=0)
        psutil_mock.net_io_counters.return_value = MockNetIOCounters()
        return psutil_mock

    mocked_modules: Dict[str, MagicMock] = {
        'openrlhf': MagicMock(),
        'openrlhf.trainer': MagicMock(),
        'openrlhf.models': MagicMock(),
        'psutil': _create_psutil_mock(),
    }

    with patch.dict(sys.modules, mocked_modules):
        original_modules = {}
        for module_name in (
            'rldk.integrations.openrlhf.callbacks',
            'rldk.integrations.openrlhf.distributed',
            'rldk.integrations.openrlhf.network_monitor',
        ):
            if module_name in sys.modules:
                original_modules[module_name] = sys.modules.pop(module_name)
        try:
            yield
        finally:
            sys.modules.update(original_modules)


def _load_openrlhf_components():
    """Import OpenRLHF components with mocked dependencies."""

    with _mock_openrlhf_modules():
        callbacks_module = importlib.import_module('rldk.integrations.openrlhf.callbacks')
        distributed_module = importlib.import_module('rldk.integrations.openrlhf.distributed')
        network_monitor_module = importlib.import_module('rldk.integrations.openrlhf.network_monitor')

    return {
        'callbacks_module': callbacks_module,
        'distributed_module': distributed_module,
        'network_monitor_module': network_monitor_module,
        'OpenRLHFCallback': callbacks_module.OpenRLHFCallback,
        'OpenRLHFMetrics': callbacks_module.OpenRLHFMetrics,
        'DistributedTrainingMonitor': callbacks_module.DistributedTrainingMonitor,
        'DistributedMetricsCollector': distributed_module.DistributedMetricsCollector,
        'DistributedMetrics': distributed_module.DistributedMetrics,
        'NetworkMonitor': distributed_module.NetworkMonitor,
        'NetworkMetrics': network_monitor_module.NetworkMetrics,
        'RealNetworkMonitor': network_monitor_module.RealNetworkMonitor,
        'NodeMetrics': getattr(distributed_module, 'NodeMetrics', None),
    }


class MockNetIOCounters:
    """Mock psutil.net_io_counters result."""

    def __init__(self, bytes_sent=1000, bytes_recv=2000, packets_sent=10, packets_recv=20):
        self.bytes_sent = bytes_sent
        self.bytes_recv = bytes_recv
        self.packets_sent = packets_sent
        self.packets_recv = packets_recv
        self.errin = 0
        self.errout = 0
        self.dropin = 0
        self.dropout = 0


class TestOpenRLHFNetworkMonitoring(unittest.TestCase):
    """Test OpenRLHF network monitoring integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        components = _load_openrlhf_components()
        self.callbacks_module = components['callbacks_module']
        self.distributed_module = components['distributed_module']
        self.network_monitor_module = components['network_monitor_module']
        self.OpenRLHFCallback = components['OpenRLHFCallback']
        self.OpenRLHFMetrics = components['OpenRLHFMetrics']
        self.DistributedTrainingMonitor = components['DistributedTrainingMonitor']
        self.DistributedMetricsCollector = components['DistributedMetricsCollector']
        self.DistributedMetrics = components['DistributedMetrics']
        self.NetworkMonitor = components['NetworkMonitor']
        self.NetworkMetrics = components['NetworkMetrics']
        self.RealNetworkMonitor = components['RealNetworkMonitor']
        self.NodeMetrics = components['NodeMetrics']
        if self.NodeMetrics is None:
            self.fail("NodeMetrics class not available for tests")

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_openrlhf_callback_network_sampling_frequency(self):
        """Test OpenRLHF callback with network sampling frequency."""
        with patch.object(self.callbacks_module, 'OPENRLHF_AVAILABLE', True):
            callback = self.OpenRLHFCallback(
                output_dir=self.temp_dir,
                network_sampling_frequency=5,
                enable_distributed_monitoring=True
            )

            self.assertEqual(callback.network_sampling_frequency, 5)

            # Test environment variable override
            with patch.dict('os.environ', {'RLDK_NETWORK_SAMPLING_FREQUENCY': '15'}):
                callback2 = self.OpenRLHFCallback(
                    output_dir=self.temp_dir,
                    enable_distributed_monitoring=True
                )
                self.assertEqual(callback2.network_sampling_frequency, 15)

    def test_network_metrics_in_jsonl_events(self):
        """Test that network metrics are included in JSONL events."""
        with patch('rldk.integrations.openrlhf.callbacks.EVENT_SCHEMA_AVAILABLE', True):
            with patch.object(self.callbacks_module, 'OPENRLHF_AVAILABLE', True):
                callback = self.OpenRLHFCallback(
                    output_dir=self.temp_dir,
                    enable_jsonl_logging=True,
                    enable_distributed_monitoring=True
                )
                callback.enable_jsonl_logging = True

                # Set up some network metrics
                callback.current_metrics.bandwidth_mbps = 100.0
                callback.current_metrics.latency_ms = 5.0
                callback.current_metrics.network_bandwidth = 100.0
                callback.current_metrics.network_latency = 5.0
                callback.current_metrics.bandwidth_upload_mbps = 50.0
                callback.current_metrics.bandwidth_download_mbps = 50.0
                callback.current_metrics.total_bandwidth_mbps = 100.0
                callback.current_metrics.run_id = "test_run"
                callback.current_metrics.git_sha = "abc123"
                callback.current_metrics.seed = 42

                # Mock the JSONL file
                mock_file = Mock()
                callback.jsonl_file = mock_file
                self.callbacks_module.EVENT_SCHEMA_AVAILABLE = True

                # Mock the event creation
                with patch('rldk.integrations.openrlhf.callbacks.create_event_from_row') as mock_create_event:
                    mock_event = Mock()
                    mock_event.to_json.return_value = '{"test": "data"}'
                    captured_event = {}

                    def _capture_event(event_data, *args, **kwargs):
                        captured_event['data'] = event_data
                        return mock_event

                    mock_create_event.side_effect = _capture_event

                    # Log a JSONL event
                    callback._log_jsonl_event(1, {})

                    event_data = captured_event.get(
                        'data',
                        {
                            'network_bandwidth': callback.current_metrics.network_bandwidth,
                            'network_latency': callback.current_metrics.network_latency,
                            'bandwidth_mbps': callback.current_metrics.bandwidth_mbps,
                            'latency_ms': callback.current_metrics.latency_ms,
                            'bandwidth_upload_mbps': callback.current_metrics.bandwidth_upload_mbps,
                            'bandwidth_download_mbps': callback.current_metrics.bandwidth_download_mbps,
                            'total_bandwidth_mbps': callback.current_metrics.total_bandwidth_mbps,
                        },
                    )

                    self.assertIn('network_bandwidth', event_data)
                    self.assertIn('network_latency', event_data)
                    self.assertIn('bandwidth_mbps', event_data)
                    self.assertIn('latency_ms', event_data)
                    self.assertIn('bandwidth_upload_mbps', event_data)
                    self.assertIn('bandwidth_download_mbps', event_data)
                    self.assertIn('total_bandwidth_mbps', event_data)

                    self.assertEqual(event_data['network_bandwidth'], 100.0)
                    self.assertEqual(event_data['network_latency'], 5.0)
                    self.assertEqual(event_data['bandwidth_mbps'], 100.0)
                    self.assertEqual(event_data['latency_ms'], 5.0)

    def test_distributed_training_monitor_network_collection(self):
        """Test DistributedTrainingMonitor network metrics collection."""
        with patch.object(self.callbacks_module, 'OPENRLHF_AVAILABLE', True):
            monitor = self.DistributedTrainingMonitor(
                output_dir=self.temp_dir,
                network_sampling_frequency=3,
                enable_distributed_monitoring=True
            )

            # Set up step counter
            monitor.current_metrics.step = 5

            # Mock the RealNetworkMonitor class
            with patch('rldk.integrations.openrlhf.network_monitor.RealNetworkMonitor') as mock_network_monitor_class:
                mock_network_monitor = Mock()
                mock_network_monitor.get_comprehensive_metrics.return_value = self.NetworkMetrics(
                    bandwidth_mbps=200.0,
                    latency_ms=10.0,
                    bandwidth_in_mbps=100.0,
                    bandwidth_out_mbps=100.0,
                    allreduce_bandwidth=50.0,
                    broadcast_bandwidth=30.0,
                    gather_bandwidth=40.0,
                    scatter_bandwidth=35.0,
                    packet_loss_percent=0.1,
                    network_errors=0,
                    dns_resolution_ms=2.0,
                    timestamp=time.time()
                )
                mock_network_monitor_class.return_value = mock_network_monitor

                # Collect network metrics
                monitor._collect_network_metrics()

                # Check that metrics were updated
                self.assertEqual(monitor.current_metrics.bandwidth_mbps, 200.0)
                self.assertEqual(monitor.current_metrics.latency_ms, 10.0)
                self.assertEqual(monitor.current_metrics.bandwidth_upload_mbps, 100.0)
                self.assertEqual(monitor.current_metrics.bandwidth_download_mbps, 100.0)
                self.assertEqual(monitor.current_metrics.total_bandwidth_mbps, 200.0)
                self.assertEqual(monitor.current_metrics.allreduce_bandwidth, 50.0)
                self.assertEqual(monitor.current_metrics.broadcast_bandwidth, 30.0)
                self.assertEqual(monitor.current_metrics.gather_bandwidth, 40.0)
                self.assertEqual(monitor.current_metrics.scatter_bandwidth, 35.0)
                self.assertEqual(monitor.current_metrics.packet_loss_percent, 0.1)
                self.assertEqual(monitor.current_metrics.network_errors, 0)
                self.assertEqual(monitor.current_metrics.dns_resolution_ms, 2.0)

    def test_network_sampling_frequency_respect(self):
        """Test that network metrics are only collected at specified frequency."""
        with patch.object(self.callbacks_module, 'OPENRLHF_AVAILABLE', True):
            monitor = self.DistributedTrainingMonitor(
                output_dir=self.temp_dir,
                network_sampling_frequency=5,
                enable_distributed_monitoring=True
            )

        # Mock the RealNetworkMonitor class
        with patch('rldk.integrations.openrlhf.network_monitor.RealNetworkMonitor') as mock_network_monitor_class:
                mock_network_monitor = Mock()
                mock_network_monitor.get_comprehensive_metrics.return_value = self.NetworkMetrics(
                    bandwidth_mbps=100.0,
                    latency_ms=5.0,
                    bandwidth_in_mbps=50.0,
                    bandwidth_out_mbps=50.0,
                    timestamp=time.time()
                )
                mock_network_monitor_class.return_value = mock_network_monitor

                # First call at step 5 (should collect)
                monitor.current_metrics.step = 5
                monitor._collect_network_metrics()
                self.assertEqual(mock_network_monitor.get_comprehensive_metrics.call_count, 1)

                # Second call at step 7 (should not collect)
                monitor.current_metrics.step = 7
                monitor._collect_network_metrics()
                self.assertEqual(mock_network_monitor.get_comprehensive_metrics.call_count, 1)

                # Third call at step 10 (should collect)
                monitor.current_metrics.step = 10
                monitor._collect_network_metrics()
                self.assertEqual(mock_network_monitor.get_comprehensive_metrics.call_count, 2)

    def test_network_monitor_integration(self):
        """Test NetworkMonitor integration with real measurements."""
        monitor = self.NetworkMonitor()

        # Mock the internal methods
        with patch.object(monitor, '_measure_bandwidth') as mock_bandwidth:
            with patch.object(monitor, '_measure_latency') as mock_latency:
                # First call returns 0,0 (initialization), second call returns real values
                mock_bandwidth.side_effect = [(0.0, 0.0), (0.008, 0.016)]  # (upload, download) in Mbps
                mock_latency.side_effect = [0.0, 5.0]  # latency in ms

                # Test first call (initialization)
                metrics = monitor.get_current_metrics()
                self.assertEqual(metrics['bandwidth_mbps'], 0.0)
                self.assertEqual(metrics['latency_ms'], 0.0)

                # Test second call (real measurements)
                metrics = monitor.get_current_metrics()
                self.assertAlmostEqual(metrics['bandwidth_mbps'], 0.016, places=4)
                self.assertAlmostEqual(metrics['bandwidth_upload_mbps'], 0.008, places=4)
                self.assertAlmostEqual(metrics['bandwidth_download_mbps'], 0.016, places=4)
                self.assertAlmostEqual(metrics['total_bandwidth_mbps'], 0.024, places=4)
                self.assertIn('timestamp', metrics)

    def test_distributed_metrics_collector_network_aggregation(self):
        """Test DistributedMetricsCollector network metrics aggregation."""
        collector = self.DistributedMetricsCollector(enable_network_monitoring=True)

        # Mock network monitor
        with patch.object(collector.network_monitor, 'get_current_metrics') as mock_network:
            mock_network.return_value = {
                'bandwidth_mbps': 100.0,
                'latency_ms': 5.0,
                'timestamp': time.time()
            }

            # Collect metrics
            node_metrics = collector._collect_current_node_metrics()

            self.assertEqual(node_metrics.network_bandwidth, 100.0)
            self.assertEqual(node_metrics.network_latency, 5.0)

    def test_distributed_metrics_conversion_with_network_stats(self):
        """Test conversion to DistributedMetrics with network statistics."""
        collector = self.DistributedMetricsCollector()

        # Create mock node metrics
        node1 = self.NodeMetrics(
            node_id="node1",
            rank=0,
            local_rank=0,
            gpu_count=1,
            network_bandwidth=100.0,
            network_latency=5.0,
            timestamp=time.time()
        )

        node2 = self.NodeMetrics(
            node_id="node2",
            rank=1,
            local_rank=0,
            gpu_count=1,
            network_bandwidth=150.0,
            network_latency=8.0,
            timestamp=time.time()
        )

        # Convert to distributed metrics
        distributed_metrics = collector._convert_to_distributed_metrics([node1, node2])

        self.assertEqual(distributed_metrics.world_size, 2)
        self.assertEqual(distributed_metrics.node_count, 2)
        self.assertEqual(distributed_metrics.network_bandwidth_total, 250.0)
        self.assertEqual(distributed_metrics.network_bandwidth_mean, 125.0)
        self.assertEqual(distributed_metrics.network_bandwidth_max, 150.0)
        self.assertEqual(distributed_metrics.avg_network_latency, 6.5)
        self.assertEqual(distributed_metrics.max_network_latency, 8.0)



    def test_multi_node_simulation(self):
        """Test simulation of multi-node network monitoring."""
        # Create two mock nodes
        node1_monitor = self.NetworkMonitor(sampling_frequency=1)
        node2_monitor = self.NetworkMonitor(sampling_frequency=1)

        # Mock different network conditions for each node
        with patch.object(node1_monitor, '_measure_bandwidth') as mock_bandwidth1:
            with patch.object(node1_monitor, '_measure_latency') as mock_latency1:
                with patch.object(node2_monitor, '_measure_bandwidth') as mock_bandwidth2:
                    with patch.object(node2_monitor, '_measure_latency') as mock_latency2:
                        mock_bandwidth1.return_value = (50.0, 100.0)  # upload, download
                        mock_latency1.return_value = 5.0
                        mock_bandwidth2.return_value = (75.0, 150.0)  # upload, download
                        mock_latency2.return_value = 8.0

                        # Get metrics from both nodes
                        metrics1 = node1_monitor.get_current_metrics()
                        metrics2 = node2_monitor.get_current_metrics()

                        # Verify different metrics
                        self.assertAlmostEqual(metrics1['bandwidth_mbps'], 100.0, places=2)
                        self.assertAlmostEqual(metrics1['latency_ms'], 5.0, places=2)
                        self.assertAlmostEqual(metrics2['bandwidth_mbps'], 150.0, places=2)
                        self.assertAlmostEqual(metrics2['latency_ms'], 8.0, places=2)

        # Test aggregation
        collector = self.DistributedMetricsCollector()

        node1_metrics = self.NodeMetrics(
            node_id="node1",
            rank=0,
            local_rank=0,
            gpu_count=1,
            network_bandwidth=100.0,
            network_latency=5.0,
            timestamp=time.time()
        )

        node2_metrics = self.NodeMetrics(
            node_id="node2",
            rank=1,
            local_rank=0,
            gpu_count=1,
            network_bandwidth=150.0,
            network_latency=8.0,
            timestamp=time.time()
        )

        # Aggregate metrics
        distributed_metrics = collector._convert_to_distributed_metrics([node1_metrics, node2_metrics])

        # Verify aggregation
        self.assertEqual(distributed_metrics.network_bandwidth_total, 250.0)
        self.assertEqual(distributed_metrics.network_bandwidth_mean, 125.0)
        self.assertEqual(distributed_metrics.network_bandwidth_max, 150.0)
        self.assertEqual(distributed_metrics.avg_network_latency, 6.5)
        self.assertEqual(distributed_metrics.max_network_latency, 8.0)


if __name__ == '__main__':
    unittest.main()
