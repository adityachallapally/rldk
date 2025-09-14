#!/usr/bin/env python3
"""Test script for OpenRLHF network monitoring implementation."""

import os
import sys
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_network_monitoring():
    """Test the network monitoring implementation."""
    print("Testing OpenRLHF Network Monitoring Implementation")
    print("=" * 50)

    try:
        from rldk.integrations.openrlhf.network_monitor import (
            DistributedNetworkMonitor,
            NetworkBandwidthMonitor,
            NetworkInterfaceMonitor,
            NetworkLatencyMonitor,
            NetworkMetrics,
            RealNetworkMonitor,
        )
        print("✓ Successfully imported network monitoring modules")

        # Test basic network monitoring
        print("\n1. Testing RealNetworkMonitor...")
        monitor = RealNetworkMonitor(enable_distributed_monitoring=False)

        # Get current metrics
        metrics = monitor.get_current_metrics()
        print(f"   Current bandwidth: {metrics['bandwidth']:.2f} Mbps")
        print(f"   Current latency: {metrics['latency']:.2f} ms")

        # Get network stats
        stats = monitor.get_network_stats()
        print(f"   Average bandwidth: {stats['avg_bandwidth']:.2f} Mbps")
        print(f"   Average latency: {stats['avg_latency']:.2f} ms")

        # Test comprehensive metrics
        comprehensive_metrics = monitor.get_comprehensive_metrics()
        print(f"   Comprehensive bandwidth: {comprehensive_metrics.bandwidth_mbps:.2f} Mbps")
        print(f"   Comprehensive latency: {comprehensive_metrics.latency_ms:.2f} ms")
        print(f"   Packet loss: {comprehensive_metrics.packet_loss_percent:.2f}%")
        print(f"   Network errors: {comprehensive_metrics.network_errors}")

        # Test interface monitoring
        print("\n2. Testing NetworkInterfaceMonitor...")
        interface_monitor = NetworkInterfaceMonitor()
        interface_stats = interface_monitor.get_interface_stats()
        print(f"   Interface: {interface_monitor.interface_name}")
        print(f"   Bytes in: {interface_stats['bytes_in_per_sec']:.2f} B/s")
        print(f"   Bytes out: {interface_stats['bytes_out_per_sec']:.2f} B/s")
        print(f"   Packets in: {interface_stats['packets_in_per_sec']:.2f} pkt/s")
        print(f"   Packets out: {interface_stats['packets_out_per_sec']:.2f} pkt/s")

        # Test latency monitoring
        print("\n3. Testing NetworkLatencyMonitor...")
        latency_monitor = NetworkLatencyMonitor()
        latency_results = latency_monitor.measure_latency()
        print(f"   Latency results: {latency_results}")
        avg_latency = latency_monitor.get_average_latency()
        print(f"   Average latency: {avg_latency:.2f} ms")

        # Test bandwidth monitoring
        print("\n4. Testing NetworkBandwidthMonitor...")
        bandwidth_monitor = NetworkBandwidthMonitor()
        bandwidth = bandwidth_monitor.measure_bandwidth()
        print(f"   Measured bandwidth: {bandwidth:.2f} Mbps")

        # Test distributed network monitoring (if PyTorch distributed is available)
        print("\n5. Testing DistributedNetworkMonitor...")
        try:
            import torch
            if torch.cuda.is_available():
                print("   CUDA is available, testing distributed metrics...")
                dist_monitor = DistributedNetworkMonitor(world_size=1, rank=0)
                dist_metrics = dist_monitor.measure_distributed_metrics()
                print(f"   Distributed bandwidth: {dist_metrics.bandwidth_mbps:.2f} Mbps")
                print(f"   Distributed latency: {dist_metrics.latency_ms:.2f} ms")
                print(f"   Allreduce bandwidth: {dist_metrics.allreduce_bandwidth:.2f} Mbps")
                print(f"   Broadcast bandwidth: {dist_metrics.broadcast_bandwidth:.2f} Mbps")
            else:
                print("   CUDA not available, skipping distributed metrics")
        except Exception as e:
            print(f"   Distributed monitoring not available: {e}")

        # Test the updated callbacks
        print("\n6. Testing updated callbacks...")
        from rldk.integrations.openrlhf.callbacks import DistributedTrainingMonitor

        callback = DistributedTrainingMonitor()
        print("   ✓ DistributedTrainingMonitor initialized successfully")

        # Test network metrics collection
        callback._collect_network_metrics()
        print(f"   Network bandwidth: {callback.current_metrics.network_bandwidth:.6f} GB/s")
        print(f"   Network latency: {callback.current_metrics.network_latency:.2f} ms")

        print("\n✓ All network monitoring tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_network_tools():
    """Test if network monitoring tools are available."""
    print("\nTesting Network Monitoring Tools")
    print("=" * 30)

    tools = {
        'speedtest-cli': 'speedtest-cli --version',
        'iperf3': 'iperf3 --version',
        'ping': 'ping -c 1 8.8.8.8',
    }

    for tool, command in tools.items():
        try:
            import subprocess
            result = subprocess.run(command.split(), capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✓ {tool} is available")
            else:
                print(f"✗ {tool} is not available")
        except Exception:
            print(f"✗ {tool} is not available")

if __name__ == "__main__":
    print("OpenRLHF Network Monitoring Test Suite")
    print("=" * 40)

    # Test network tools availability
    test_network_tools()

    # Test network monitoring implementation
    success = test_network_monitoring()

    if success:
        print("\n🎉 All tests passed! Network monitoring is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
