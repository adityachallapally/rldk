"""Tests for network monitoring functionality."""

import json
import shutil

# Import the modules to test
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rldk.integrations.openrlhf.distributed import NetworkMonitor
from rldk.integrations.openrlhf.network_monitor import (
    DistributedNetworkMonitor,
    NetworkBandwidthMonitor,
    NetworkDiagnostics,
    NetworkInterfaceMonitor,
    NetworkLatencyMonitor,
    NetworkMetrics,
    RealNetworkMonitor,
)


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


class TestNetworkMetrics(unittest.TestCase):
    """Test NetworkMetrics dataclass."""

    def test_network_metrics_creation(self):
        """Test creating NetworkMetrics instance."""
        metrics = NetworkMetrics(
            bandwidth_mbps=100.0,
            latency_ms=5.0,
            packet_loss_percent=0.1
        )

        self.assertEqual(metrics.bandwidth_mbps, 100.0)
        self.assertEqual(metrics.latency_ms, 5.0)
        self.assertEqual(metrics.packet_loss_percent, 0.1)

    def test_network_metrics_to_dict(self):
        """Test converting NetworkMetrics to dictionary."""
        metrics = NetworkMetrics(
            bandwidth_mbps=100.0,
            latency_ms=5.0,
            timestamp=1234567890.0
        )

        metrics_dict = metrics.to_dict()

        self.assertIsInstance(metrics_dict, dict)
        self.assertEqual(metrics_dict['bandwidth_mbps'], 100.0)
        self.assertEqual(metrics_dict['latency_ms'], 5.0)
        self.assertEqual(metrics_dict['timestamp'], 1234567890.0)


class TestNetworkMonitor(unittest.TestCase):
    """Test NetworkMonitor with mocked dependencies."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch('psutil.net_io_counters')
    @patch('time.perf_counter')
    def test_measure_bandwidth(self, mock_perf_counter, mock_net_io):
        """Test bandwidth measurement with mocked psutil."""
        # Mock time.perf_counter to return increasing values
        mock_perf_counter.side_effect = [1000.0, 1001.0]  # 1 second difference

        # Mock psutil.net_io_counters to return different values
        mock_net_io.side_effect = [
            MockNetIOCounters(bytes_sent=1000, bytes_recv=2000),
            MockNetIOCounters(bytes_sent=2000, bytes_recv=4000)  # 1000 bytes sent, 2000 bytes received
        ]

        monitor = NetworkMonitor()

        # First call should return zeros (no previous measurement)
        upload, download = monitor._measure_bandwidth()
        self.assertEqual(upload, 0.0)
        self.assertEqual(download, 0.0)

        # Second call should calculate real bandwidth
        upload, download = monitor._measure_bandwidth()

        # Expected: (1000 bytes * 8 bits) / (1 second * 1,000,000) = 0.008 Mbps upload
        # Expected: (2000 bytes * 8 bits) / (1 second * 1,000,000) = 0.016 Mbps download
        self.assertAlmostEqual(upload, 0.008, places=4)
        self.assertAlmostEqual(download, 0.016, places=4)

    @patch('subprocess.run')
    def test_icmp_ping_success(self, mock_run):
        """Test successful ICMP ping."""
        # Mock successful ping response
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "64 bytes from 8.8.8.8: icmp_seq=1 time=5.23 ms"

        monitor = NetworkMonitor()
        latency = monitor._icmp_ping("8.8.8.8")

        self.assertEqual(latency, 5.23)

    @patch('subprocess.run')
    def test_icmp_ping_failure(self, mock_run):
        """Test failed ICMP ping."""
        # Mock failed ping response
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = "ping: unknown host 8.8.8.8"

        monitor = NetworkMonitor()
        latency = monitor._icmp_ping("8.8.8.8")

        self.assertIsNone(latency)

    @patch('socket.socket')
    def test_tcp_handshake_success(self, mock_socket):
        """Test successful TCP handshake."""
        # Mock successful socket connection
        mock_sock = Mock()
        mock_sock.connect_ex.return_value = 0
        mock_socket.return_value = mock_sock

        monitor = NetworkMonitor()

        with patch('time.perf_counter') as mock_time:
            mock_time.side_effect = [1000.0, 1000.005]  # 5ms difference
            latency = monitor._tcp_handshake("8.8.8.8")

        self.assertAlmostEqual(latency, 5.0, places=2)

    @patch('socket.socket')
    def test_tcp_handshake_failure(self, mock_socket):
        """Test failed TCP handshake."""
        # Mock failed socket connection
        mock_sock = Mock()
        mock_sock.connect_ex.return_value = 1
        mock_socket.return_value = mock_sock

        monitor = NetworkMonitor()
        latency = monitor._tcp_handshake("8.8.8.8")

        self.assertIsNone(latency)

    @patch('psutil.net_io_counters')
    @patch('time.perf_counter')
    @patch('subprocess.run')
    def test_get_current_metrics(self, mock_run, mock_perf_counter, mock_net_io):
        """Test getting current metrics."""
        # Mock successful ping
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "64 bytes from 8.8.8.8: icmp_seq=1 time=10.5 ms"

        # Mock time and network I/O
        mock_perf_counter.side_effect = [1000.0, 1001.0, 1002.0]
        mock_net_io.side_effect = [
            MockNetIOCounters(bytes_sent=1000, bytes_recv=2000),
            MockNetIOCounters(bytes_sent=2000, bytes_recv=4000)
        ]

        monitor = NetworkMonitor()

        # First call to initialize
        metrics = monitor.get_current_metrics()
        self.assertEqual(metrics['bandwidth_mbps'], 0.0)
        # Latency will be measured on first call due to mock
        self.assertAlmostEqual(metrics['latency_ms'], 10.5, places=2)

        # Second call to get real measurements
        metrics = monitor.get_current_metrics()

        self.assertAlmostEqual(metrics['bandwidth_mbps'], 0.016, places=4)  # Download bandwidth
        self.assertAlmostEqual(metrics['latency_ms'], 10.5, places=2)
        self.assertAlmostEqual(metrics['bandwidth_upload_mbps'], 0.008, places=4)
        self.assertAlmostEqual(metrics['bandwidth_download_mbps'], 0.016, places=4)
        self.assertAlmostEqual(metrics['total_bandwidth_mbps'], 0.024, places=4)

    def test_error_handling(self):
        """Test error handling in network monitor."""
        monitor = NetworkMonitor()

        # Test with psutil error
        with patch('psutil.net_io_counters', side_effect=Exception("psutil error")):
            upload, download = monitor._measure_bandwidth()
            self.assertEqual(upload, 0.0)
            self.assertEqual(download, 0.0)
            self.assertIsNotNone(monitor.get_error_status()['bandwidth'])

        # Test with socket error
        with patch('socket.socket', side_effect=Exception("socket error")):
            latency = monitor._measure_latency()
            self.assertEqual(latency, float('inf'))
            self.assertIsNotNone(monitor.get_error_status()['latency'])

    def test_reset_counters(self):
        """Test resetting network counters."""
        monitor = NetworkMonitor()

        # Set some initial state
        monitor.last_net_io = MockNetIOCounters()
        monitor.last_net_time = 1000.0

        # Reset counters
        monitor.reset_counters()

        self.assertIsNone(monitor.last_net_io)
        self.assertIsNone(monitor.last_net_time)


class TestNetworkDiagnostics(unittest.TestCase):
    """Test NetworkDiagnostics class."""

    @patch('subprocess.run')
    def test_ping_host_advanced_success(self, mock_run):
        """Test successful advanced ping."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = """
PING 8.8.8.8 (8.8.8.8) 56(84) bytes of data.
64 bytes from 8.8.8.8: icmp_seq=1 time=5.23 ms
64 bytes from 8.8.8.8: icmp_seq=2 time=6.12 ms
64 bytes from 8.8.8.8: icmp_seq=3 time=4.89 ms
64 bytes from 8.8.8.8: icmp_seq=4 time=5.67 ms
        """

        diagnostics = NetworkDiagnostics()
        result = diagnostics._ping_host_advanced("8.8.8.8")

        self.assertTrue(result['success'])
        self.assertAlmostEqual(result['latency'], 5.48, places=2)  # Average of the 4 measurements
        self.assertEqual(result['packet_loss'], 0.0)

    @patch('subprocess.run')
    def test_ping_host_advanced_failure(self, mock_run):
        """Test failed advanced ping."""
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "ping: unknown host 8.8.8.8"

        diagnostics = NetworkDiagnostics()
        result = diagnostics._ping_host_advanced("8.8.8.8")

        self.assertFalse(result.get('success', False))
        self.assertEqual(result['latency'], float('inf'))
        self.assertIn('error', result)


class TestNetworkInterfaceMonitor(unittest.TestCase):
    """Test NetworkInterfaceMonitor class."""

    @patch('psutil.net_io_counters')
    @patch('time.time')
    def test_get_interface_stats(self, mock_time, mock_net_io):
        """Test getting interface statistics."""
        mock_time.side_effect = [1000.0, 1001.0]  # 1 second difference

        mock_net_io.side_effect = [
            {'eth0': MockNetIOCounters(bytes_sent=1000, bytes_recv=2000)},
            {'eth0': MockNetIOCounters(bytes_sent=2000, bytes_recv=4000)}
        ]

        monitor = NetworkInterfaceMonitor(interface_name='eth0')

        # First call should return empty stats
        stats = monitor.get_interface_stats()
        self.assertEqual(stats['bytes_in_per_sec'], 0.0)
        self.assertEqual(stats['bytes_out_per_sec'], 0.0)

        # Second call should calculate real rates
        stats = monitor.get_interface_stats()

        self.assertAlmostEqual(stats['bytes_in_per_sec'], 2000.0, places=2)
        self.assertAlmostEqual(stats['bytes_out_per_sec'], 1000.0, places=2)
        self.assertAlmostEqual(stats['bytes_in_mbps'], 0.016, places=4)  # 2000 * 8 / 1000000
        self.assertAlmostEqual(stats['bytes_out_mbps'], 0.008, places=4)  # 1000 * 8 / 1000000


class TestNetworkLatencyMonitor(unittest.TestCase):
    """Test NetworkLatencyMonitor class."""

    @patch('subprocess.run')
    def test_measure_latency(self, mock_run):
        """Test latency measurement."""
        # Mock successful ping responses
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "64 bytes from 8.8.8.8: icmp_seq=1 time=5.23 ms"

        monitor = NetworkLatencyMonitor()
        results = monitor.measure_latency()

        # Should have results for all target hosts
        self.assertGreater(len(results), 0)
        for host, latency in results.items():
            if latency != float('inf'):
                self.assertIsInstance(latency, float)
                self.assertGreaterEqual(latency, 0.0)

    def test_get_average_latency(self):
        """Test getting average latency."""
        monitor = NetworkLatencyMonitor()

        # Add some test data
        monitor.latency_history['8.8.8.8'] = [5.0, 6.0, 4.0]
        monitor.latency_history['1.1.1.1'] = [3.0, 4.0, 5.0]

        avg_latency = monitor.get_average_latency()

        # Expected: (5+6+4+3+4+5) / 6 = 4.5
        self.assertAlmostEqual(avg_latency, 4.5, places=2)


class TestNetworkBandwidthMonitor(unittest.TestCase):
    """Test NetworkBandwidthMonitor class."""

    @patch('subprocess.run')
    def test_measure_bandwidth_speedtest(self, mock_run):
        """Test bandwidth measurement with speedtest."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '{"download": 50000000, "upload": 25000000, "ping": 10.5}'

        monitor = NetworkBandwidthMonitor()
        bandwidth = monitor._measure_with_speedtest()

        # Expected: 50000000 / 1000000 = 50 Mbps
        self.assertEqual(bandwidth, 50.0)

    def test_get_bandwidth_stats(self):
        """Test getting bandwidth statistics."""
        monitor = NetworkBandwidthMonitor()

        # Add some test data
        monitor.bandwidth_history = [10.0, 20.0, 30.0, 40.0, 50.0]

        stats = monitor.get_bandwidth_stats()

        self.assertEqual(stats['mean'], 30.0)
        self.assertEqual(stats['min'], 10.0)
        self.assertEqual(stats['max'], 50.0)
        self.assertEqual(stats['median'], 30.0)


class TestDistributedNetworkMonitor(unittest.TestCase):
    """Test DistributedNetworkMonitor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch('psutil.net_io_counters')
    @patch('time.time')
    def test_measure_distributed_metrics(self, mock_time, mock_net_io):
        """Test measuring distributed metrics."""
        mock_time.return_value = 1000.0

        mock_net_io.side_effect = [
            {'eth0': MockNetIOCounters(bytes_sent=1000, bytes_recv=2000)},
            {'eth0': MockNetIOCounters(bytes_sent=2000, bytes_recv=4000)}
        ]

        monitor = DistributedNetworkMonitor(world_size=2, rank=0)

        # Mock latency measurement
        with patch.object(monitor.latency_monitor, 'measure_latency') as mock_latency:
            mock_latency.return_value = {'8.8.8.8': 5.0, '1.1.1.1': 6.0}

            metrics = monitor.measure_distributed_metrics()

        self.assertIsInstance(metrics, NetworkMetrics)
        self.assertEqual(metrics.world_size, 2)  # Should match monitor's world_size
        self.assertEqual(metrics.rank, 0)  # Should match monitor's rank

    def test_get_performance_summary(self):
        """Test getting performance summary."""
        monitor = DistributedNetworkMonitor(world_size=2, rank=0)

        # Add some test metrics
        test_metrics = NetworkMetrics(
            bandwidth_mbps=100.0,
            latency_ms=5.0,
            packet_loss_percent=0.1,
            timestamp=1000.0
        )
        monitor.performance_history = [test_metrics]

        summary = monitor.get_performance_summary()

        self.assertEqual(summary['total_measurements'], 1)
        self.assertEqual(summary['avg_bandwidth'], 100.0)
        self.assertEqual(summary['avg_latency'], 5.0)
        self.assertEqual(summary['avg_packet_loss'], 0.1)


class TestRealNetworkMonitor(unittest.TestCase):
    """Test RealNetworkMonitor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_get_current_metrics(self):
        """Test getting current metrics."""
        monitor = RealNetworkMonitor()

        # Mock the underlying monitors and time to bypass measurement interval
        with patch.object(monitor.bandwidth_monitor, 'measure_bandwidth') as mock_bandwidth:
            with patch.object(monitor.latency_monitor, 'get_average_latency') as mock_latency:
                with patch('time.time') as mock_time:
                    mock_bandwidth.return_value = 100.0
                    mock_latency.return_value = 5.0
                    mock_time.return_value = 1000.0  # Fixed time to bypass interval

                    # Mock the distributed monitor to None to use fallback
                    monitor.distributed_monitor = None

                    metrics = monitor.get_current_metrics()

        self.assertEqual(metrics['bandwidth'], 100.0)
        self.assertEqual(metrics['latency'], 5.0)

    def test_get_network_health_report(self):
        """Test getting network health report."""
        monitor = RealNetworkMonitor()

        # Mock the diagnostics
        with patch.object(monitor.network_diagnostics, 'run_comprehensive_diagnostics') as mock_diagnostics:
            mock_diagnostics.return_value = {
                'ping_tests': {'8.8.8.8': {'success': True, 'latency': 5.0}},
                'dns_tests': {'google.com': {'success': True, 'resolution_time_ms': 10.0}},
                'bandwidth_tests': {'speedtest': {'success': True, 'download_mbps': 100.0}}
            }

            report = monitor.get_network_health_report()

        self.assertIn('overall_health', report)
        self.assertIn('issues', report)
        self.assertIn('recommendations', report)


class TestNetworkMonitorSamplingFrequency(unittest.TestCase):
    """Test NetworkMonitor sampling frequency functionality."""

    def test_sampling_frequency_constructor(self):
        """Test sampling frequency parameter in constructor."""
        monitor = NetworkMonitor(sampling_frequency=5)
        self.assertEqual(monitor.sampling_frequency, 5)

    def test_sampling_frequency_env_var(self):
        """Test sampling frequency from environment variable."""
        with patch.dict('os.environ', {'RLDK_NETWORK_SAMPLING_FREQUENCY': '15'}):
            monitor = NetworkMonitor()
            self.assertEqual(monitor.sampling_frequency, 15)

    def test_sampling_frequency_default(self):
        """Test default sampling frequency when no env var is set."""
        with patch.dict('os.environ', {}, clear=True):
            monitor = NetworkMonitor()
            self.assertEqual(monitor.sampling_frequency, 10)

    def test_get_current_metrics_with_timestamp(self):
        """Test that get_current_metrics includes timestamp."""
        monitor = NetworkMonitor()

        with patch.object(monitor, '_measure_bandwidth') as mock_bandwidth:
            with patch.object(monitor, '_measure_latency') as mock_latency:
                mock_bandwidth.return_value = (10.0, 20.0)
                mock_latency.return_value = 5.0

                metrics = monitor.get_current_metrics()

        self.assertIn('timestamp', metrics)
        self.assertIsInstance(metrics['timestamp'], float)
        self.assertGreater(metrics['timestamp'], 0)


if __name__ == '__main__':
    unittest.main()
