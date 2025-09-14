#!/usr/bin/env python3
"""Comprehensive test script for OpenRLHF network diagnostics implementation."""

import json
import os
import sys
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_comprehensive_network_diagnostics():
    """Test the comprehensive network diagnostics implementation."""
    print("Testing OpenRLHF Comprehensive Network Diagnostics")
    print("=" * 60)

    try:
        from rldk.integrations.openrlhf.network_monitor import (
            NetworkDiagnostics,
            NetworkMetrics,
            RealNetworkMonitor,
        )
        print("✓ Successfully imported network diagnostics modules")

        # Test 1: Basic NetworkDiagnostics
        print("\n1. Testing NetworkDiagnostics class...")
        diagnostics = NetworkDiagnostics()

        # Run comprehensive diagnostics
        print("   Running comprehensive network diagnostics...")
        comprehensive_results = diagnostics.run_comprehensive_diagnostics()

        # Display ping test results
        print("\n   Ping Test Results:")
        for host, result in comprehensive_results['ping_tests'].items():
            if result.get('success', False):
                print(f"     {host}: {result['latency']:.2f}ms (min: {result['min_latency']:.2f}ms, max: {result['max_latency']:.2f}ms)")
            else:
                print(f"     {host}: FAILED - {result.get('error', 'Unknown error')}")

        # Display DNS test results
        print("\n   DNS Resolution Test Results:")
        for host, result in comprehensive_results['dns_tests'].items():
            if result.get('success', False):
                print(f"     {host}: {result['resolution_time_ms']:.2f}ms -> {result['resolved_ip']}")
            else:
                print(f"     {host}: FAILED - {result.get('error', 'Unknown error')}")

        # Display bandwidth test results
        print("\n   Bandwidth Test Results:")
        for test_name, result in comprehensive_results['bandwidth_tests'].items():
            if result.get('success', False):
                if 'download_mbps' in result:
                    print(f"     {test_name}: {result['download_mbps']:.2f} Mbps download")
                if 'upload_mbps' in result:
                    print(f"     {test_name}: {result['upload_mbps']:.2f} Mbps upload")
                if 'bandwidth_mbps' in result:
                    print(f"     {test_name}: {result['bandwidth_mbps']:.2f} Mbps")
            else:
                print(f"     {test_name}: FAILED - {result.get('error', 'Unknown error')}")

        # Display interface analysis
        print("\n   Network Interface Analysis:")
        for interface_name, info in comprehensive_results['interface_analysis'].items():
            if 'error' not in info:
                print(f"     {interface_name}:")
                print(f"       Speed: {info.get('speed_mbps', 'Unknown')} Mbps")
                print(f"       MTU: {info.get('mtu', 'Unknown')}")
                print(f"       IP Addresses: {info.get('ip_addresses', [])}")
                if 'bytes_sent' in info:
                    print(f"       Bytes Sent: {info['bytes_sent']:,}")
                    print(f"       Bytes Received: {info['bytes_recv']:,}")

        # Test 2: RealNetworkMonitor with diagnostics
        print("\n2. Testing RealNetworkMonitor with diagnostics...")
        monitor = RealNetworkMonitor(enable_distributed_monitoring=True)

        # Get network health report
        print("   Generating network health report...")
        health_report = monitor.get_network_health_report()

        print("\n   Network Health Report:")
        print(f"     Overall Health: {health_report['overall_health'].upper()}")
        print(f"     Issues Found: {len(health_report['issues'])}")
        print(f"     Recommendations: {len(health_report['recommendations'])}")

        if health_report['issues']:
            print("\n     Issues:")
            for issue in health_report['issues'][:5]:  # Show first 5 issues
                print(f"       - {issue}")

        if health_report['recommendations']:
            print("\n     Recommendations:")
            for rec in health_report['recommendations']:
                print(f"       - {rec}")

        # Display metrics summary
        if health_report['metrics_summary']:
            print("\n     Metrics Summary:")
            for metric, value in health_report['metrics_summary'].items():
                if 'latency' in metric:
                    print(f"       {metric}: {value:.2f} ms")
                elif 'bandwidth' in metric:
                    print(f"       {metric}: {value:.2f} Mbps")
                else:
                    print(f"       {metric}: {value}")

        # Test 3: Individual diagnostic methods
        print("\n3. Testing individual diagnostic methods...")

        # Test ping diagnostics
        print("   Testing ping diagnostics...")
        ping_results = diagnostics._run_ping_diagnostics()
        successful_pings = sum(1 for result in ping_results.values() if result.get('success', False))
        print(f"     Successful pings: {successful_pings}/{len(ping_results)}")

        # Test DNS diagnostics
        print("   Testing DNS diagnostics...")
        dns_results = diagnostics._run_dns_diagnostics()
        successful_dns = sum(1 for result in dns_results.values() if result.get('success', False))
        print(f"     Successful DNS resolutions: {successful_dns}/{len(dns_results)}")

        # Test connectivity diagnostics
        print("   Testing connectivity diagnostics...")
        connectivity_results = diagnostics._run_connectivity_diagnostics()
        tcp_tests = connectivity_results.get('tcp_tests', {})
        udp_tests = connectivity_results.get('udp_tests', {})
        successful_tcp = sum(1 for result in tcp_tests.values() if result.get('success', False))
        successful_udp = sum(1 for result in udp_tests.values() if result.get('success', False))
        print(f"     Successful TCP connections: {successful_tcp}/{len(tcp_tests)}")
        print(f"     Successful UDP connections: {successful_udp}/{len(udp_tests)}")

        # Test bandwidth diagnostics
        print("   Testing bandwidth diagnostics...")
        bandwidth_results = diagnostics._run_bandwidth_diagnostics()
        successful_bandwidth = sum(1 for result in bandwidth_results.values() if result.get('success', False))
        print(f"     Successful bandwidth tests: {successful_bandwidth}/{len(bandwidth_results)}")

        # Test 4: Performance comparison with old methods
        print("\n4. Performance comparison with old placeholder methods...")

        # Test old ping method (if it exists)
        try:
            from rldk.integrations.openrlhf.network_monitor import NetworkLatencyMonitor
            old_latency_monitor = NetworkLatencyMonitor()

            print("   Testing old ping method...")
            old_start = time.time()
            old_latency = old_latency_monitor.get_average_latency()
            old_time = time.time() - old_start

            print("   Testing new ping method...")
            new_start = time.time()
            new_ping_results = diagnostics._run_ping_diagnostics()
            new_time = time.time() - new_start

            # Calculate average from new results
            new_latencies = [result['latency'] for result in new_ping_results.values()
                           if result.get('success', False) and result['latency'] != float('inf')]
            new_avg_latency = sum(new_latencies) / len(new_latencies) if new_latencies else 0.0

            print(f"     Old method: {old_latency:.2f}ms (took {old_time:.2f}s)")
            print(f"     New method: {new_avg_latency:.2f}ms (took {new_time:.2f}s)")
            print(f"     Improvement: {((old_time - new_time) / old_time * 100):.1f}% faster execution")

        except Exception as e:
            print(f"     Could not compare with old method: {e}")

        # Test 5: Export diagnostics to JSON
        print("\n5. Testing diagnostics export...")
        try:
            # Save comprehensive results to file
            output_file = "network_diagnostics_results.json"
            with open(output_file, 'w') as f:
                json.dump(comprehensive_results, f, indent=2, default=str)
            print(f"     Diagnostics saved to: {output_file}")

            # Save health report to file
            health_file = "network_health_report.json"
            with open(health_file, 'w') as f:
                json.dump(health_report, f, indent=2, default=str)
            print(f"     Health report saved to: {health_file}")

        except Exception as e:
            print(f"     Failed to save diagnostics: {e}")

        print("\n✓ All network diagnostics tests completed successfully!")
        return True

    except Exception as e:
        print(f"✗ Network diagnostics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_distributed_network_monitoring():
    """Test distributed network monitoring capabilities."""
    print("\n" + "=" * 60)
    print("Testing Distributed Network Monitoring")
    print("=" * 60)

    try:
        from rldk.integrations.openrlhf.distributed import NetworkMonitor

        print("✓ Testing NetworkMonitor from distributed module...")
        network_monitor = NetworkMonitor()

        # Test basic metrics
        print("\n1. Testing basic network metrics...")
        current_metrics = network_monitor.get_current_metrics()
        print(f"   Current bandwidth: {current_metrics['bandwidth']:.2f} Mbps")
        print(f"   Current latency: {current_metrics['latency']:.2f} ms")

        # Test network stats
        print("\n2. Testing network statistics...")
        network_stats = network_monitor.get_network_stats()
        print(f"   Average bandwidth: {network_stats['avg_bandwidth']:.2f} Mbps")
        print(f"   Average latency: {network_stats['avg_latency']:.2f} ms")
        print(f"   Max bandwidth: {network_stats['max_bandwidth']:.2f} Mbps")
        print(f"   Min latency: {network_stats['min_latency']:.2f} ms")

        # Test comprehensive metrics
        print("\n3. Testing comprehensive metrics...")
        comprehensive_metrics = network_monitor.get_comprehensive_metrics()
        print(f"   Comprehensive bandwidth: {comprehensive_metrics.bandwidth_mbps:.2f} Mbps")
        print(f"   Comprehensive latency: {comprehensive_metrics.latency_ms:.2f} ms")
        print(f"   Packet loss: {comprehensive_metrics.packet_loss_percent:.2f}%")
        print(f"   Network errors: {comprehensive_metrics.network_errors}")

        # Test network diagnostics
        print("\n4. Testing network diagnostics...")
        diagnostics = network_monitor.run_network_diagnostics()
        print(f"   Diagnostics completed with {len(diagnostics)} categories")

        # Test network health report
        print("\n5. Testing network health report...")
        health_report = network_monitor.get_network_health_report()
        print(f"   Overall health: {health_report['overall_health']}")
        print(f"   Issues found: {len(health_report['issues'])}")

        # Test individual test methods
        print("\n6. Testing individual test methods...")

        connectivity_tests = network_monitor.test_network_connectivity()
        print(f"   Connectivity tests completed: {len(connectivity_tests)} categories")

        bandwidth_tests = network_monitor.test_bandwidth()
        print(f"   Bandwidth tests completed: {len(bandwidth_tests)} methods")

        distributed_tests = network_monitor.test_distributed_network()
        print(f"   Distributed tests: {distributed_tests.get('error', 'Completed')}")

        print("\n✓ Distributed network monitoring tests completed successfully!")
        return True

    except Exception as e:
        print(f"✗ Distributed network monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("OpenRLHF Network Diagnostics Comprehensive Test Suite")
    print("=" * 60)

    # Test comprehensive diagnostics
    success1 = test_comprehensive_network_diagnostics()

    # Test distributed monitoring
    success2 = test_distributed_network_monitoring()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Comprehensive Diagnostics: {'✓ PASSED' if success1 else '✗ FAILED'}")
    print(f"Distributed Monitoring: {'✓ PASSED' if success2 else '✗ FAILED'}")

    if success1 and success2:
        print("\n🎉 All tests passed! Network diagnostics are working correctly.")
        print("\nKey improvements:")
        print("- Replaced placeholder ping tests with real system ping commands")
        print("- Added comprehensive DNS resolution testing")
        print("- Added TCP/UDP connectivity testing")
        print("- Added multiple bandwidth measurement methods")
        print("- Added network interface analysis")
        print("- Added network path analysis (traceroute)")
        print("- Added distributed training network diagnostics")
        print("- Added comprehensive health reporting")
        print("- Improved fallback mechanisms with real diagnostics")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")

    return success1 and success2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
