#!/usr/bin/env python3
"""Demo script for OpenRLHF network monitoring functionality."""

import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Import the network monitoring components
from src.rldk.integrations.openrlhf.network_monitor import (
    NetworkMonitor, NetworkDiagnostics, RealNetworkMonitor, NetworkMetrics
)
from src.rldk.integrations.openrlhf.dashboard import OpenRLHFDashboard


def run_network_diagnostics():
    """Run comprehensive network diagnostics."""
    print("🔍 Running comprehensive network diagnostics...")
    
    diagnostics = NetworkDiagnostics()
    results = diagnostics.run_comprehensive_diagnostics()
    
    print("\n📊 Network Diagnostics Results:")
    print("=" * 50)
    
    # Ping tests
    print("\n🌐 Ping Tests:")
    for host, result in results['ping_tests'].items():
        if result.get('success', False):
            print(f"  ✅ {host}: {result['latency']:.2f}ms")
        else:
            print(f"  ❌ {host}: {result.get('error', 'Failed')}")
    
    # DNS tests
    print("\n🔍 DNS Resolution:")
    for host, result in results['dns_tests'].items():
        if result.get('success', False):
            print(f"  ✅ {host}: {result['resolution_time_ms']:.2f}ms")
        else:
            print(f"  ❌ {host}: {result.get('error', 'Failed')}")
    
    # Bandwidth tests
    print("\n📡 Bandwidth Tests:")
    for test_name, result in results['bandwidth_tests'].items():
        if result.get('success', False):
            if 'download_mbps' in result:
                print(f"  ✅ {test_name}: {result['download_mbps']:.2f} Mbps download")
            elif 'bandwidth_mbps' in result:
                print(f"  ✅ {test_name}: {result['bandwidth_mbps']:.2f} Mbps")
        else:
            print(f"  ❌ {test_name}: {result.get('error', 'Failed')}")
    
    return results


def run_real_time_monitoring(duration: int = 60, interval: float = 5.0):
    """Run real-time network monitoring."""
    print(f"\n📈 Running real-time network monitoring for {duration} seconds...")
    print(f"📊 Sampling interval: {interval} seconds")
    
    monitor = NetworkMonitor(sampling_frequency=1, enable_icmp=True)
    
    start_time = time.time()
    measurements = []
    
    try:
        while time.time() - start_time < duration:
            # Get current metrics
            metrics = monitor.get_current_metrics()
            current_time = time.time()
            
            # Add timestamp
            metrics['timestamp'] = current_time
            
            # Display current metrics
            print(f"\n⏰ {time.strftime('%H:%M:%S', time.localtime(current_time))}")
            print(f"  📡 Bandwidth: {metrics['bandwidth_mbps']:.2f} Mbps (↓{metrics['bandwidth_download_mbps']:.2f} ↑{metrics['bandwidth_upload_mbps']:.2f})")
            print(f"  ⏱️  Latency: {metrics['latency_ms']:.2f} ms")
            print(f"  📊 Total: {metrics['total_bandwidth_mbps']:.2f} Mbps")
            
            # Check for errors
            errors = monitor.get_error_status()
            if errors['bandwidth']:
                print(f"  ⚠️  Bandwidth error: {errors['bandwidth']}")
            if errors['latency']:
                print(f"  ⚠️  Latency error: {errors['latency']}")
            
            measurements.append(metrics)
            
            # Wait for next measurement
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    
    # Calculate statistics
    if measurements:
        print("\n📊 Monitoring Statistics:")
        print("=" * 30)
        
        bandwidths = [m['bandwidth_mbps'] for m in measurements if m['bandwidth_mbps'] > 0]
        latencies = [m['latency_ms'] for m in measurements if m['latency_ms'] > 0]
        
        if bandwidths:
            print(f"📡 Bandwidth (Mbps):")
            print(f"  Average: {sum(bandwidths) / len(bandwidths):.2f}")
            print(f"  Min: {min(bandwidths):.2f}")
            print(f"  Max: {max(bandwidths):.2f}")
        
        if latencies:
            print(f"⏱️  Latency (ms):")
            print(f"  Average: {sum(latencies) / len(latencies):.2f}")
            print(f"  Min: {min(latencies):.2f}")
            print(f"  Max: {max(latencies):.2f}")
    
    return measurements


def run_distributed_simulation():
    """Simulate distributed training network monitoring."""
    print("\n🖥️  Simulating distributed training network monitoring...")
    
    # Create a comprehensive network monitor
    monitor = RealNetworkMonitor(
        enable_distributed_monitoring=True,
        enable_distributed_measurements=False  # Safer for demo
    )
    
    # Get comprehensive metrics
    metrics = monitor.get_comprehensive_metrics()
    
    print("\n📊 Comprehensive Network Metrics:")
    print("=" * 40)
    print(f"📡 Bandwidth: {metrics.bandwidth_mbps:.2f} Mbps")
    print(f"⏱️  Latency: {metrics.latency_ms:.2f} ms")
    print(f"📦 Packet Loss: {metrics.packet_loss_percent:.2f}%")
    print(f"❌ Network Errors: {metrics.network_errors}")
    print(f"🔍 DNS Resolution: {metrics.dns_resolution_ms:.2f} ms")
    
    # Distributed training metrics
    print(f"\n🔄 Distributed Training Metrics:")
    print(f"  AllReduce Bandwidth: {metrics.allreduce_bandwidth:.2f} Mbps")
    print(f"  Broadcast Bandwidth: {metrics.broadcast_bandwidth:.2f} Mbps")
    print(f"  Gather Bandwidth: {metrics.gather_bandwidth:.2f} Mbps")
    print(f"  Scatter Bandwidth: {metrics.scatter_bandwidth:.2f} Mbps")
    
    return metrics


def run_dashboard_demo(output_dir: str = "./network_demo"):
    """Run dashboard demo."""
    print(f"\n🌐 Starting dashboard demo...")
    print(f"📁 Output directory: {output_dir}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize dashboard
    dashboard = OpenRLHFDashboard(output_dir=output_path, port=5001)
    
    # Initialize network monitoring
    dashboard.initialize_network_monitoring()
    
    # Generate some sample data
    print("📊 Generating sample network data...")
    
    monitor = NetworkMonitor()
    
    for i in range(10):
        # Get current metrics
        metrics = monitor.get_current_metrics()
        metrics['timestamp'] = time.time()
        metrics['step'] = i
        
        # Add to dashboard
        dashboard.add_metrics(metrics)
        
        # Check thresholds and generate alerts
        dashboard.check_network_thresholds(metrics)
        
        print(f"  Step {i}: Bandwidth={metrics['bandwidth_mbps']:.2f} Mbps, Latency={metrics['latency_ms']:.2f} ms")
        
        time.sleep(1)
    
    # Get dashboard URL
    dashboard_url = dashboard.get_dashboard_url()
    print(f"\n🌐 Dashboard available at: {dashboard_url}")
    print("📊 Open your browser to view real-time network metrics")
    
    return dashboard


def save_metrics_to_jsonl(metrics: list, filename: str):
    """Save metrics to JSONL file."""
    with open(filename, 'w') as f:
        for metric in metrics:
            f.write(json.dumps(metric) + '\n')
    print(f"💾 Metrics saved to {filename}")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="OpenRLHF Network Monitoring Demo")
    parser.add_argument("--mode", choices=["diagnostics", "monitoring", "distributed", "dashboard", "all"], 
                       default="all", help="Demo mode to run")
    parser.add_argument("--duration", type=int, default=60, 
                       help="Duration for real-time monitoring (seconds)")
    parser.add_argument("--interval", type=float, default=5.0, 
                       help="Sampling interval for monitoring (seconds)")
    parser.add_argument("--output", type=str, default="./network_demo", 
                       help="Output directory for dashboard demo")
    
    args = parser.parse_args()
    
    print("🚀 OpenRLHF Network Monitoring Demo")
    print("=" * 50)
    
    if args.mode in ["diagnostics", "all"]:
        run_network_diagnostics()
    
    if args.mode in ["monitoring", "all"]:
        measurements = run_real_time_monitoring(args.duration, args.interval)
        save_metrics_to_jsonl(measurements, f"{args.output}/monitoring_metrics.jsonl")
    
    if args.mode in ["distributed", "all"]:
        run_distributed_simulation()
    
    if args.mode in ["dashboard", "all"]:
        dashboard = run_dashboard_demo(args.output)
        
        if args.mode == "dashboard":
            # Keep dashboard running
            print("\n🔄 Dashboard is running. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n⏹️  Dashboard stopped")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main()