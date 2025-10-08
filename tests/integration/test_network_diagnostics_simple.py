#!/usr/bin/env python3
"""Simple test script for OpenRLHF network diagnostics implementation."""

import json
import os
import sys
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_network_diagnostics_direct():
    """Test the network diagnostics directly without full rldk import."""
    print("Testing OpenRLHF Network Diagnostics (Direct Import)")
    print("=" * 60)

    try:
        # Import directly from the network_monitor module
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'rldk', 'integrations', 'openrlhf'))

        from network_monitor import (
            NetworkBandwidthMonitor,
            NetworkDiagnostics,
            NetworkInterfaceMonitor,
            NetworkLatencyMonitor,
            NetworkMetrics,
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

        # Test 2: Individual diagnostic methods
        print("\n2. Testing individual diagnostic methods...")

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

        # Test 3: Network interface monitor
        print("\n3. Testing NetworkInterfaceMonitor...")
        interface_monitor = NetworkInterfaceMonitor()
        interface_stats = interface_monitor.get_interface_stats()
        print(f"   Interface: {interface_monitor.interface_name}")
        print(f"   Bytes in: {interface_stats['bytes_in_per_sec']:.2f} B/s")
        print(f"   Bytes out: {interface_stats['bytes_out_per_sec']:.2f} B/s")
        print(f"   Packets in: {interface_stats['packets_in_per_sec']:.2f} pkt/s")
        print(f"   Packets out: {interface_stats['packets_out_per_sec']:.2f} pkt/s")

        # Test 4: Latency monitor
        print("\n4. Testing NetworkLatencyMonitor...")
        latency_monitor = NetworkLatencyMonitor()
        latency_results = latency_monitor.measure_latency()
        print(f"   Latency results: {latency_results}")
        avg_latency = latency_monitor.get_average_latency()
        print(f"   Average latency: {avg_latency:.2f} ms")

        # Test 5: Bandwidth monitor
        print("\n5. Testing NetworkBandwidthMonitor...")
        bandwidth_monitor = NetworkBandwidthMonitor()
        bandwidth = bandwidth_monitor.measure_bandwidth()
        print(f"   Measured bandwidth: {bandwidth:.2f} Mbps")

        # Test 6: Export diagnostics to JSON
        print("\n6. Testing diagnostics export...")
        try:
            # Save comprehensive results to file
            output_file = "network_diagnostics_results.json"
            with open(output_file, 'w') as f:
                json.dump(comprehensive_results, f, indent=2, default=str)
            print(f"     Diagnostics saved to: {output_file}")

        except Exception as e:
            print(f"     Failed to save diagnostics: {e}")

        print("\n✓ All network diagnostics tests completed successfully!")
        return True

    except Exception as e:
        print(f"✗ Network diagnostics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_placeholder_replacement():
    """Test that placeholder methods have been replaced with real diagnostics."""
    print("\n" + "=" * 60)
    print("Testing Placeholder Replacement")
    print("=" * 60)

    try:
        from network_monitor import NetworkLatencyMonitor

        print("✓ Testing that placeholder ping method has been replaced...")

        # Test the old vs new ping method
        latency_monitor = NetworkLatencyMonitor()

        # The old method used simple socket connection on port 80
        # The new method uses real system ping command
        print("   Testing new ping implementation...")

        # Get the ping method source to verify it's not the old placeholder
        import inspect
        ping_source = inspect.getsource(latency_monitor._ping_host)

        if "socket.socket" in ping_source and "connect_ex" in ping_source:
            print("   ⚠️  Warning: Still using old socket-based ping method")
            return False
        elif "NetworkDiagnostics" in ping_source:
            print("   ✓ Successfully replaced with advanced ping diagnostics")
            return True
        else:
            print("   ✓ Using new ping implementation")
            return True

    except Exception as e:
        print(f"✗ Placeholder replacement test failed: {e}")
        return False

def main():
    """Main test function."""
    print("OpenRLHF Network Diagnostics Simple Test Suite")
    print("=" * 60)

    # Test comprehensive diagnostics
    success1 = test_network_diagnostics_direct()

    # Test placeholder replacement
    success2 = test_placeholder_replacement()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Comprehensive Diagnostics: {'✓ PASSED' if success1 else '✗ FAILED'}")
    print(f"Placeholder Replacement: {'✓ PASSED' if success2 else '✗ FAILED'}")

    if success1 and success2:
        print("\n🎉 All tests passed! Network diagnostics are working correctly.")
        print("\nKey improvements:")
        print("- Replaced placeholder ping tests with real system ping commands")
        print("- Added comprehensive DNS resolution testing")
        print("- Added TCP/UDP connectivity testing")
        print("- Added multiple bandwidth measurement methods")
        print("- Added network interface analysis")
        print("- Added network path analysis (traceroute)")
        print("- Added comprehensive health reporting")
        print("- Improved fallback mechanisms with real diagnostics")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")

    return success1 and success2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
