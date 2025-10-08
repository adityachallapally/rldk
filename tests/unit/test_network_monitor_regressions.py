"""Targeted regression tests extracted from manual network diagnostics."""

from itertools import count
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rldk.integrations.openrlhf import network_monitor


@pytest.fixture
def stub_psutil(monkeypatch):
    """Provide deterministic psutil responses for network monitor tests."""

    data_sequence = [
        {
            'bytes_recv': 1_000,
            'bytes_sent': 500,
            'packets_recv': 10,
            'packets_sent': 5,
            'dropin': 0,
            'dropout': 0,
            'errin': 0,
            'errout': 0,
        },
        {
            'bytes_recv': 1_600,
            'bytes_sent': 800,
            'packets_recv': 16,
            'packets_sent': 9,
            'dropin': 0,
            'dropout': 0,
            'errin': 0,
            'errout': 0,
        },
        {
            'bytes_recv': 2_200,
            'bytes_sent': 1_100,
            'packets_recv': 22,
            'packets_sent': 13,
            'dropin': 0,
            'dropout': 0,
            'errin': 0,
            'errout': 0,
        },
    ]
    call_index = {'value': 0}

    def fake_net_if_stats():
        return {'eth0': SimpleNamespace(isup=True)}

    def fake_net_io_counters(pernic=True):
        idx = call_index['value']
        if idx >= len(data_sequence):
            idx = len(data_sequence) - 1
        call_index['value'] += 1
        return {'eth0': SimpleNamespace(**data_sequence[idx])}

    monkeypatch.setattr(network_monitor.psutil, 'net_if_stats', fake_net_if_stats)
    monkeypatch.setattr(network_monitor.psutil, 'net_io_counters', fake_net_io_counters)
    monkeypatch.setattr(network_monitor.psutil, 'net_connections', lambda: [])

    yield


@pytest.fixture
def fake_time(monkeypatch):
    """Provide a deterministic clock so rate calculations are stable."""

    clock = count()
    monkeypatch.setattr(network_monitor.time, 'time', lambda: next(clock))
    yield


def test_interface_monitor_sampling_frequency(stub_psutil, fake_time):
    """NetworkInterfaceMonitor should respect sampling frequency and produce rates."""

    monitor = network_monitor.NetworkInterfaceMonitor(sampling_frequency=3)
    empty_stats = monitor._empty_stats()

    results = [monitor.get_interface_stats() for _ in range(6)]
    empty_count = sum(1 for result in results if result == empty_stats)

    assert empty_count == 5  # First five calls skip due to sampling/baseline setup

    final_result = results[-1]
    assert final_result != empty_stats
    assert final_result['bytes_in_per_sec'] > 0
    assert final_result['bytes_out_per_sec'] > 0
    assert final_result['packets_in_per_sec'] > 0
    assert final_result['packets_out_per_sec'] > 0
    assert monitor._invocation_count == 6


def test_latency_monitor_history_bounds():
    """Latency history deques should enforce configured bounds."""

    monitor = network_monitor.NetworkLatencyMonitor()

    for i in range(150):
        with monitor._lock:
            for history in monitor.latency_history.values():
                history.append(float(i))

    for history in monitor.latency_history.values():
        assert len(history) == monitor.max_history_size
        assert history.maxlen == monitor.max_history_size


def test_distributed_monitor_history_bounds(stub_psutil):
    """Distributed monitor histories must remain bounded."""

    monitor = network_monitor.DistributedNetworkMonitor()

    with monitor._distributed_lock:
        for i in range(150):
            monitor.allreduce_times.append(i)
            monitor.broadcast_times.append(i)
            monitor.gather_times.append(i)
            monitor.scatter_times.append(i)

        for i in range(1_100):
            monitor.performance_history.append(i)

    assert len(monitor.allreduce_times) == 100
    assert monitor.allreduce_times.maxlen == 100
    assert len(monitor.broadcast_times) == 100
    assert monitor.broadcast_times.maxlen == 100
    assert len(monitor.gather_times) == 100
    assert monitor.gather_times.maxlen == 100
    assert len(monitor.scatter_times) == 100
    assert monitor.scatter_times.maxlen == 100
    assert len(monitor.performance_history) == 1_000
    assert monitor.performance_history.maxlen == 1_000


def test_real_monitor_history_bounds(stub_psutil):
    """RealNetworkMonitor should bound historical measurements."""

    monitor = network_monitor.RealNetworkMonitor(enable_distributed_monitoring=False)

    with monitor._history_lock:
        for i in range(150):
            monitor.bandwidth_history.append(float(i))
            monitor.latency_history.append(float(i))

    assert len(monitor.bandwidth_history) == 100
    assert monitor.bandwidth_history.maxlen == 100
    assert len(monitor.latency_history) == 100
    assert monitor.latency_history.maxlen == 100


def test_distributed_monitor_sampling_invokes_submonitors(stub_psutil, fake_time, monkeypatch):
    """Distributed monitor should sample sub-monitors when allowed by frequency."""

    monitor = network_monitor.DistributedNetworkMonitor(sampling_frequency=3)

    interface_stats = {
        'bytes_in_per_sec': 0.5,
        'bytes_out_per_sec': 0.75,
        'packets_in_per_sec': 1.0,
        'packets_out_per_sec': 1.5,
        'bytes_in_mbps': 1.0,
        'bytes_out_mbps': 1.5,
        'total_bytes_recv': 0.0,
        'total_bytes_sent': 0.0,
        'total_packets_recv': 0.0,
        'total_packets_sent': 0.0,
        'dropin': 0.0,
        'dropout': 0.0,
        'errin': 0.0,
        'errout': 0.0,
    }

    monitor.interface_monitor.get_interface_stats = MagicMock(return_value=interface_stats)
    monitor.latency_monitor.measure_latency = MagicMock(return_value={'example': 1.2})
    monitor.latency_monitor.get_average_latency = MagicMock(return_value=12.5)
    monitor.bandwidth_monitor.measure_bandwidth = MagicMock(return_value=42.0)

    monkeypatch.setattr(network_monitor.psutil, 'net_connections', lambda: [])

    for _ in range(10):
        monitor.measure_distributed_metrics()

    assert monitor._invocation_count == 10
    assert monitor.interface_monitor.get_interface_stats.call_count == 3
    assert monitor.latency_monitor.measure_latency.call_count == 3
    assert monitor.latency_monitor.get_average_latency.call_count == 3
    assert monitor.bandwidth_monitor.measure_bandwidth.call_count == 3
    assert len(monitor.performance_history) == 3
    assert monitor.performance_history.maxlen == 1_000
    assert monitor.performance_history[-1].bandwidth_mbps == 42.0
    assert monitor.performance_history[-1].latency_ms == 12.5
