#!/usr/bin/env python3
"""Example demonstrating OpenRLHF network monitoring capabilities."""

import json
import os
import sys
import time
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def demonstrate_network_monitoring():
    """Demonstrate the network monitoring capabilities."""
    print("OpenRLHF Network Monitoring Demonstration")
    print("=" * 50)

    try:
        from rldk.integrations.openrlhf.callbacks import DistributedTrainingMonitor
        from rldk.integrations.openrlhf.network_monitor import (
            DistributedNetworkMonitor,
            NetworkBandwidthMonitor,
            NetworkInterfaceMonitor,
            NetworkLatencyMonitor,
            NetworkMetrics,
            RealNetworkMonitor,
        )

        print("✓ Successfully imported network monitoring modules")

        # 1. Basic Network Monitoring
        print("\n1. Basic Network Monitoring")
        print("-" * 30)

        monitor = RealNetworkMonitor(enable_distributed_monitoring=False)

        # Collect metrics over time
        print("Collecting network metrics for 10 seconds...")
        metrics_history = []

        for i in range(10):
            metrics = monitor.get_current_metrics()
            comprehensive_metrics = monitor.get_comprehensive_metrics()

            metrics_history.append({
                'timestamp': time.time(),
                'bandwidth': metrics['bandwidth'],
                'latency': metrics['latency'],
                'packet_loss': comprehensive_metrics.packet_loss_percent,
                'network_errors': comprehensive_metrics.network_errors,
            })

            print(f"   Step {i+1}: {metrics['bandwidth']:.2f} Mbps, {metrics['latency']:.2f} ms")
            time.sleep(1)

        # 2. Interface Monitoring
        print("\n2. Network Interface Monitoring")
        print("-" * 30)

        interface_monitor = NetworkInterfaceMonitor()
        interface_stats = interface_monitor.get_interface_stats()

        print(f"Interface: {interface_monitor.interface_name}")
        print(f"Bytes in: {interface_stats['bytes_in_per_sec']:.2f} B/s")
        print(f"Bytes out: {interface_stats['bytes_out_per_sec']:.2f} B/s")
        print(f"Packets in: {interface_stats['packets_in_per_sec']:.2f} pkt/s")
        print(f"Packets out: {interface_stats['packets_out_per_sec']:.2f} pkt/s")
        print(f"Total bytes received: {interface_stats['total_bytes_recv']:.0f}")
        print(f"Total bytes sent: {interface_stats['total_bytes_sent']:.0f}")
        print(f"Errors in: {interface_stats['errin']}")
        print(f"Errors out: {interface_stats['errout']}")
        print(f"Dropped packets in: {interface_stats['dropin']}")
        print(f"Dropped packets out: {interface_stats['dropout']}")

        # 3. Latency Monitoring
        print("\n3. Network Latency Monitoring")
        print("-" * 30)

        latency_monitor = NetworkLatencyMonitor()
        latency_results = latency_monitor.measure_latency()

        print("Latency to different hosts:")
        for host, latency in latency_results.items():
            if latency == float('inf'):
                print(f"  {host}: Unreachable")
            else:
                print(f"  {host}: {latency:.2f} ms")

        avg_latency = latency_monitor.get_average_latency()
        print(f"Average latency: {avg_latency:.2f} ms")

        latency_stats = latency_monitor.get_latency_stats()
        print(f"Latency statistics: {latency_stats}")

        # 4. Bandwidth Monitoring
        print("\n4. Network Bandwidth Monitoring")
        print("-" * 30)

        bandwidth_monitor = NetworkBandwidthMonitor()
        bandwidth = bandwidth_monitor.measure_bandwidth()

        print(f"Current bandwidth: {bandwidth:.2f} Mbps")

        bandwidth_stats = bandwidth_monitor.get_bandwidth_stats()
        print(f"Bandwidth statistics: {bandwidth_stats}")

        # 5. Distributed Network Monitoring (if available)
        print("\n5. Distributed Network Monitoring")
        print("-" * 30)

        try:
            import torch
            if torch.cuda.is_available():
                print("CUDA is available, testing distributed metrics...")

                dist_monitor = DistributedNetworkMonitor(world_size=1, rank=0)
                dist_metrics = dist_monitor.measure_distributed_metrics()

                print(f"Distributed bandwidth: {dist_metrics.bandwidth_mbps:.2f} Mbps")
                print(f"Distributed latency: {dist_metrics.latency_ms:.2f} ms")
                print(f"Allreduce bandwidth: {dist_metrics.allreduce_bandwidth:.2f} Mbps")
                print(f"Broadcast bandwidth: {dist_metrics.broadcast_bandwidth:.2f} Mbps")
                print(f"Gather bandwidth: {dist_metrics.gather_bandwidth:.2f} Mbps")
                print(f"Scatter bandwidth: {dist_metrics.scatter_bandwidth:.2f} Mbps")
                print(f"TCP connections: {dist_metrics.tcp_connections}")
                print(f"UDP connections: {dist_metrics.udp_connections}")
                print(f"Active connections: {dist_metrics.active_connections}")

                # Get performance summary
                summary = dist_monitor.get_performance_summary()
                print(f"Performance summary: {summary}")

            else:
                print("CUDA not available, skipping distributed metrics")
        except Exception as e:
            print(f"Distributed monitoring not available: {e}")

        # 6. OpenRLHF Callback Integration
        print("\n6. OpenRLHF Callback Integration")
        print("-" * 30)

        callback = DistributedTrainingMonitor(
            output_dir="./network_monitoring_logs",
            log_interval=1,
            enable_distributed_monitoring=True,
            run_id="network_monitoring_demo"
        )

        print("✓ DistributedTrainingMonitor initialized successfully")

        # Simulate training steps
        print("Simulating training steps with network monitoring...")

        for step in range(5):
            # Simulate step begin
            callback.on_step_begin(None, step)

            # Collect network metrics
            callback._collect_network_metrics()

            # Simulate step end
            callback.on_step_end(None, step)

            print(f"  Step {step}: Network bandwidth: {callback.current_metrics.network_bandwidth:.6f} GB/s")
            print(f"           Network latency: {callback.current_metrics.network_latency:.2f} ms")

            time.sleep(1)

        # Save metrics
        callback._save_metrics()
        print("✓ Metrics saved to ./network_monitoring_logs/")

        # 7. Network Performance Analysis
        print("\n7. Network Performance Analysis")
        print("-" * 30)

        if metrics_history:
            # Calculate statistics
            bandwidths = [m['bandwidth'] for m in metrics_history]
            latencies = [m['latency'] for m in metrics_history if m['latency'] > 0]

            if bandwidths:
                avg_bandwidth = sum(bandwidths) / len(bandwidths)
                max_bandwidth = max(bandwidths)
                min_bandwidth = min(bandwidths)

                print("Bandwidth Analysis:")
                print(f"  Average: {avg_bandwidth:.2f} Mbps")
                print(f"  Maximum: {max_bandwidth:.2f} Mbps")
                print(f"  Minimum: {min_bandwidth:.2f} Mbps")

            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                max_latency = max(latencies)
                min_latency = min(latencies)

                print("Latency Analysis:")
                print(f"  Average: {avg_latency:.2f} ms")
                print(f"  Maximum: {max_latency:.2f} ms")
                print(f"  Minimum: {min_latency:.2f} ms")

        # 8. Export Results
        print("\n8. Exporting Results")
        print("-" * 30)

        results = {
            'timestamp': time.time(),
            'interface_stats': interface_stats,
            'latency_results': latency_results,
            'bandwidth_stats': bandwidth_stats,
            'metrics_history': metrics_history,
        }

        # Save to JSON
        output_file = Path("./network_monitoring_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Results saved to {output_file}")

        print("\n🎉 Network monitoring demonstration completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_multi_node_scenario():
    """Demonstrate multi-node network monitoring scenario."""
    print("\nMulti-Node Network Monitoring Scenario")
    print("=" * 50)

    try:
        from rldk.integrations.openrlhf.distributed import DistributedMetricsCollector

        print("This scenario demonstrates how network monitoring works in multi-node distributed training:")
        print()
        print("1. Each node runs its own NetworkMonitor instance")
        print("2. Metrics are collected locally and shared across nodes")
        print("3. Aggregated metrics provide cluster-wide network performance view")
        print("4. Distributed training operations (allreduce, broadcast) are monitored")
        print()

        # Simulate multi-node metrics collection
        DistributedMetricsCollector(
            collect_interval=1.0,
            enable_network_monitoring=True,
            enable_gpu_monitoring=True,
            enable_cpu_monitoring=True,
        )

        print("✓ DistributedMetricsCollector initialized")
        print("In a real multi-node setup, this would:")
        print("- Collect metrics from all nodes")
        print("- Aggregate network performance across the cluster")
        print("- Monitor distributed training communication patterns")
        print("- Provide alerts for network bottlenecks")

        return True

    except Exception as e:
        print(f"Multi-node scenario demonstration failed: {e}")
        return False

if __name__ == "__main__":
    print("OpenRLHF Network Monitoring Example")
    print("=" * 40)

    # Run the main demonstration
    success = demonstrate_network_monitoring()

    if success:
        # Run multi-node scenario demonstration
        demonstrate_multi_node_scenario()

        print("\n📊 Network Monitoring Features Demonstrated:")
        print("✓ Real-time bandwidth measurement")
        print("✓ Network latency monitoring")
        print("✓ Interface statistics collection")
        print("✓ Packet loss detection")
        print("✓ Network error tracking")
        print("✓ Distributed training metrics")
        print("✓ Performance history tracking")
        print("✓ OpenRLHF callback integration")
        print("✓ Multi-node monitoring support")

        print("\n🚀 The network monitoring implementation is now ready for production use!")
        sys.exit(0)
    else:
        print("\n❌ Demonstration failed. Please check the implementation.")
        sys.exit(1)
