"""Comprehensive thread safety tests for network monitoring components."""

import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

# Add the src directory to the path to import the network monitor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from rldk.integrations.openrlhf.network_monitor import (
    DistributedNetworkMonitor,
    NetworkBandwidthMonitor,
    NetworkDiagnostics,
    NetworkInterfaceMonitor,
    NetworkLatencyMonitor,
    RealNetworkMonitor,
)


class TestNetworkThreadSafety:
    """Test thread safety of network monitoring components."""

    def test_network_diagnostics_thread_safety(self):
        """Test NetworkDiagnostics thread safety with concurrent access."""
        diagnostics = NetworkDiagnostics(sampling_frequency=1)

        def run_diagnostics():
            """Run diagnostics in a thread."""
            try:
                result = diagnostics.run_comprehensive_diagnostics()
                return result is not None
            except Exception as e:
                print(f"Error in diagnostics thread: {e}")
                return False

        # Test with 10 concurrent threads
        num_threads = 10
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(run_diagnostics) for _ in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]

        # All threads should complete without exceptions
        assert all(results), "Some diagnostics threads failed"

        # Check that we have a single lock instance
        assert hasattr(diagnostics, '_lock')
        assert isinstance(diagnostics._lock, threading.RLock)

        # Verify invocation count is correct
        assert diagnostics._invocation_count == num_threads

    def test_network_interface_monitor_thread_safety(self):
        """Test NetworkInterfaceMonitor thread safety with concurrent access."""
        monitor = NetworkInterfaceMonitor(sampling_frequency=1)

        def get_stats():
            """Get interface stats in a thread."""
            try:
                stats = monitor.get_interface_stats()
                return stats is not None and isinstance(stats, dict)
            except Exception as e:
                print(f"Error in interface monitor thread: {e}")
                return False

        # Test with 10 concurrent threads
        num_threads = 10
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(get_stats) for _ in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]

        # All threads should complete without exceptions
        assert all(results), "Some interface monitor threads failed"

        # Check that we have a single lock instance
        assert hasattr(monitor, '_lock')
        assert isinstance(monitor._lock, threading.RLock)

        # Verify invocation count is correct
        assert monitor._invocation_count == num_threads

    def test_network_latency_monitor_thread_safety(self):
        """Test NetworkLatencyMonitor thread safety with concurrent access."""
        monitor = NetworkLatencyMonitor(sampling_frequency=1)

        def measure_latency():
            """Measure latency in a thread."""
            try:
                results = monitor.measure_latency()
                return results is not None and isinstance(results, dict)
            except Exception as e:
                print(f"Error in latency monitor thread: {e}")
                return False

        # Test with 10 concurrent threads
        num_threads = 10
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(measure_latency) for _ in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]

        # All threads should complete without exceptions
        assert all(results), "Some latency monitor threads failed"

        # Check that we have a single lock instance
        assert hasattr(monitor, '_lock')
        assert isinstance(monitor._lock, threading.RLock)

        # Verify invocation count is correct
        assert monitor._invocation_count == num_threads

        # Check that latency history uses deque with maxlen
        assert isinstance(monitor.latency_history, dict)
        for host, history in monitor.latency_history.items():
            assert isinstance(history, deque)
            assert history.maxlen == 100

    def test_network_bandwidth_monitor_thread_safety(self):
        """Test NetworkBandwidthMonitor thread safety with concurrent access."""
        monitor = NetworkBandwidthMonitor(sampling_frequency=1)

        def measure_bandwidth():
            """Measure bandwidth in a thread."""
            try:
                bandwidth = monitor.measure_bandwidth()
                return isinstance(bandwidth, (int, float))
            except Exception as e:
                print(f"Error in bandwidth monitor thread: {e}")
                return False

        # Test with 10 concurrent threads
        num_threads = 10
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(measure_bandwidth) for _ in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]

        # All threads should complete without exceptions
        assert all(results), "Some bandwidth monitor threads failed"

        # Check that we have a single lock instance
        assert hasattr(monitor, '_lock')
        assert isinstance(monitor._lock, threading.RLock)

        # Verify invocation count is correct
        assert monitor._invocation_count == num_threads

        # Check that bandwidth history uses deque with maxlen
        assert isinstance(monitor.bandwidth_history, deque)
        assert monitor.bandwidth_history.maxlen == 100

    def test_distributed_network_monitor_thread_safety(self):
        """Test DistributedNetworkMonitor thread safety with concurrent access."""
        monitor = DistributedNetworkMonitor(sampling_frequency=1)

        def measure_metrics():
            """Measure distributed metrics in a thread."""
            try:
                metrics = monitor.measure_distributed_metrics()
                return metrics is not None
            except Exception as e:
                print(f"Error in distributed monitor thread: {e}")
                return False

        # Test with 10 concurrent threads
        num_threads = 10
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(measure_metrics) for _ in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]

        # All threads should complete without exceptions
        assert all(results), "Some distributed monitor threads failed"

        # Check that we have distributed lock
        assert hasattr(monitor, '_distributed_lock')
        assert isinstance(monitor._distributed_lock, threading.RLock)

        # Verify invocation count is correct
        assert monitor._invocation_count == num_threads

        # Check that all distributed lists use deque with maxlen
        assert isinstance(monitor.allreduce_times, deque)
        assert monitor.allreduce_times.maxlen == 100
        assert isinstance(monitor.broadcast_times, deque)
        assert monitor.broadcast_times.maxlen == 100
        assert isinstance(monitor.gather_times, deque)
        assert monitor.gather_times.maxlen == 100
        assert isinstance(monitor.scatter_times, deque)
        assert monitor.scatter_times.maxlen == 100
        assert isinstance(monitor.performance_history, deque)
        assert monitor.performance_history.maxlen == 1000

    def test_real_network_monitor_thread_safety(self):
        """Test RealNetworkMonitor thread safety with concurrent access."""
        monitor = RealNetworkMonitor(sampling_frequency=1)

        def get_metrics():
            """Get current metrics in a thread."""
            try:
                metrics = monitor.get_current_metrics()
                return metrics is not None and isinstance(metrics, dict)
            except Exception as e:
                print(f"Error in real network monitor thread: {e}")
                return False

        # Test with 10 concurrent threads
        num_threads = 10
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(get_metrics) for _ in range(num_threads)]
            results = [future.result() for future in as_completed(futures)]

        # All threads should complete without exceptions
        assert all(results), "Some real network monitor threads failed"

        # Check that we have history lock
        assert hasattr(monitor, '_history_lock')
        assert isinstance(monitor._history_lock, threading.RLock)

        # Verify invocation count is correct
        assert monitor._invocation_count == num_threads

        # Check that history uses deque with maxlen
        assert isinstance(monitor.bandwidth_history, deque)
        assert monitor.bandwidth_history.maxlen == 100
        assert isinstance(monitor.latency_history, deque)
        assert monitor.latency_history.maxlen == 100

    def test_memory_bounds_enforcement(self):
        """Test that memory bounds are enforced with deque maxlen."""
        # Test latency monitor history bounds
        latency_monitor = NetworkLatencyMonitor(sampling_frequency=1)

        # Fill history beyond maxlen
        for i in range(150):  # More than maxlen=100
            with latency_monitor._lock:
                for host in latency_monitor.target_hosts:
                    latency_monitor.latency_history[host].append(i)

        # Check that history is bounded
        for host in latency_monitor.target_hosts:
            assert len(latency_monitor.latency_history[host]) == 100
            assert latency_monitor.latency_history[host].maxlen == 100

        # Test bandwidth monitor history bounds
        bandwidth_monitor = NetworkBandwidthMonitor(sampling_frequency=1)

        # Fill history beyond maxlen
        for i in range(150):  # More than maxlen=100
            with bandwidth_monitor._lock:
                bandwidth_monitor.bandwidth_history.append(i)

        # Check that history is bounded
        assert len(bandwidth_monitor.bandwidth_history) == 100
        assert bandwidth_monitor.bandwidth_history.maxlen == 100

        # Test distributed monitor history bounds
        distributed_monitor = DistributedNetworkMonitor(sampling_frequency=1)

        # Fill all distributed histories beyond maxlen
        for i in range(150):  # More than maxlen=100
            with distributed_monitor._distributed_lock:
                distributed_monitor.allreduce_times.append(i)
                distributed_monitor.broadcast_times.append(i)
                distributed_monitor.gather_times.append(i)
                distributed_monitor.scatter_times.append(i)

        # Check that all histories are bounded
        assert len(distributed_monitor.allreduce_times) == 100
        assert distributed_monitor.allreduce_times.maxlen == 100
        assert len(distributed_monitor.broadcast_times) == 100
        assert distributed_monitor.broadcast_times.maxlen == 100
        assert len(distributed_monitor.gather_times) == 100
        assert distributed_monitor.gather_times.maxlen == 100
        assert len(distributed_monitor.scatter_times) == 100
        assert distributed_monitor.scatter_times.maxlen == 100

        # Test performance history bounds (maxlen=1000)
        for i in range(1100):  # More than maxlen=1000
            with distributed_monitor._distributed_lock:
                distributed_monitor.performance_history.append(i)

        assert len(distributed_monitor.performance_history) == 1000
        assert distributed_monitor.performance_history.maxlen == 1000

    def test_sampling_frequency_control(self):
        """Test that sampling frequency parameter works correctly."""
        # Test with sampling frequency of 3
        sampling_freq = 3
        monitor = NetworkInterfaceMonitor(sampling_frequency=sampling_freq)

        # Call get_interface_stats multiple times
        results = []
        for i in range(10):
            result = monitor.get_interface_stats()
            results.append(result)

        # Check that invocation count is correct
        assert monitor._invocation_count == 10

        # With sampling_frequency=3, only every 3rd call should do real work
        # The others should return empty stats
        empty_stats = {
            'bytes_in_per_sec': 0.0,
            'bytes_out_per_sec': 0.0,
            'packets_in_per_sec': 0.0,
            'packets_out_per_sec': 0.0,
            'bytes_in_mbps': 0.0,
            'bytes_out_mbps': 0.0,
            'total_bytes_recv': 0.0,
            'total_bytes_sent': 0.0,
            'total_packets_recv': 0.0,
            'total_packets_sent': 0.0,
            'dropin': 0.0,
            'dropout': 0.0,
            'errin': 0.0,
            'errout': 0.0,
        }

        # Most calls should return empty stats due to sampling
        empty_count = sum(1 for result in results if result == empty_stats)
        assert empty_count >= 6  # At least 6 out of 10 should be empty due to sampling

    def test_lock_uniqueness(self):
        """Test that each monitor instance has its own unique lock."""
        # Create multiple instances
        monitor1 = NetworkInterfaceMonitor()
        monitor2 = NetworkInterfaceMonitor()
        monitor3 = NetworkLatencyMonitor()

        # Each should have its own lock
        assert monitor1._lock is not monitor2._lock
        assert monitor1._lock is not monitor3._lock
        assert monitor2._lock is not monitor3._lock

        # All should be RLock instances
        assert isinstance(monitor1._lock, threading.RLock)
        assert isinstance(monitor2._lock, threading.RLock)
        assert isinstance(monitor3._lock, threading.RLock)

    def test_concurrent_read_write_operations(self):
        """Test concurrent read and write operations on shared state."""
        monitor = NetworkLatencyMonitor(sampling_frequency=1)

        def writer():
            """Write to latency history."""
            for i in range(50):
                with monitor._lock:
                    for host in monitor.target_hosts:
                        monitor.latency_history[host].append(i)
                time.sleep(0.001)  # Small delay

        def reader():
            """Read from latency history."""
            for i in range(50):
                with monitor._lock:
                    for host in monitor.target_hosts:
                        history = list(monitor.latency_history[host])
                        # Should not raise exception
                        assert isinstance(history, list)
                time.sleep(0.001)  # Small delay

        # Run readers and writers concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            # 5 writers
            for _ in range(5):
                futures.append(executor.submit(writer))
            # 5 readers
            for _ in range(5):
                futures.append(executor.submit(reader))

            # Wait for all to complete
            for future in as_completed(futures):
                future.result()  # This will raise any exceptions

        # Verify no data corruption occurred
        for host in monitor.target_hosts:
            history = list(monitor.latency_history[host])
            # History should contain only integers
            assert all(isinstance(x, (int, float)) for x in history)
            # History should be bounded
            assert len(history) <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
