"""Comprehensive network monitoring for OpenRLHF distributed training."""

import json
import platform
import socket
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import psutil

try:
    import torch
    import torch.distributed as dist
    DIST_AVAILABLE = True
except ImportError:
    DIST_AVAILABLE = False


if not isinstance(threading.RLock, type):  # pragma: no cover - environment dependent
    threading.RLock = type(threading.RLock())


@dataclass
class NetworkMetrics:
    """Comprehensive network metrics for distributed training."""
    # Basic metrics
    bandwidth_mbps: float = 0.0
    latency_ms: float = 0.0
    packet_loss_percent: float = 0.0

    # Distributed training info
    world_size: int = 1
    rank: int = 0

    # Advanced metrics
    bandwidth_in_mbps: float = 0.0
    bandwidth_out_mbps: float = 0.0
    packets_in_per_sec: float = 0.0
    packets_out_per_sec: float = 0.0
    bytes_in_per_sec: float = 0.0
    bytes_out_per_sec: float = 0.0

    # Connection metrics
    tcp_connections: int = 0
    udp_connections: int = 0
    active_connections: int = 0

    # Error metrics
    network_errors: int = 0
    dropped_packets: int = 0

    # Distributed training specific
    allreduce_bandwidth: float = 0.0
    broadcast_bandwidth: float = 0.0
    gather_bandwidth: float = 0.0
    scatter_bandwidth: float = 0.0

    # Network diagnostics
    dns_resolution_ms: float = 0.0
    tcp_connectivity_ms: float = 0.0
    udp_connectivity_ms: float = 0.0
    network_path_hops: int = 0
    network_path_latency: float = 0.0

    # Timestamp
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }


class NetworkDiagnostics:
    """Comprehensive network diagnostics for distributed training environments."""

    def __init__(self, sampling_frequency: int = 1):
        """Initialize network diagnostics.

        Args:
            sampling_frequency: Only run diagnostics every N invocations (default: 1)
        """
        self.test_hosts = [
            '8.8.8.8',      # Google DNS
            '1.1.1.1',      # Cloudflare DNS
            '208.67.222.222', # OpenDNS
            'google.com',   # Google
            'github.com',   # GitHub
            'pytorch.org',  # PyTorch
        ]

        self.dns_servers = [
            '8.8.8.8',
            '1.1.1.1',
            '208.67.222.222',
        ]

        self.port_tests = [80, 443, 22, 53]  # HTTP, HTTPS, SSH, DNS

        # Thread safety and sampling control
        self._lock = threading.RLock()
        self.sampling_frequency = sampling_frequency
        self._invocation_count = 0

    def run_comprehensive_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive network diagnostics."""
        with self._lock:
            self._invocation_count += 1

            # Only run diagnostics if sampling frequency allows
            if self._invocation_count % self.sampling_frequency != 0:
                return {
                    'timestamp': time.time(),
                    'skipped': True,
                    'invocation_count': self._invocation_count,
                    'sampling_frequency': self.sampling_frequency,
                }

        diagnostics = {
            'timestamp': time.time(),
            'ping_tests': {},
            'dns_tests': {},
            'connectivity_tests': {},
            'bandwidth_tests': {},
            'interface_analysis': {},
            'path_analysis': {},
            'distributed_tests': {},
        }

        # Run ping tests
        print("Running ping diagnostics...")
        diagnostics['ping_tests'] = self._run_ping_diagnostics()

        # Run DNS tests
        print("Running DNS diagnostics...")
        diagnostics['dns_tests'] = self._run_dns_diagnostics()

        # Run connectivity tests
        print("Running connectivity diagnostics...")
        diagnostics['connectivity_tests'] = self._run_connectivity_diagnostics()

        # Run bandwidth tests
        print("Running bandwidth diagnostics...")
        diagnostics['bandwidth_tests'] = self._run_bandwidth_diagnostics()

        # Run interface analysis
        print("Running interface diagnostics...")
        diagnostics['interface_analysis'] = self._run_interface_diagnostics()

        # Run path analysis
        print("Running network path diagnostics...")
        diagnostics['path_analysis'] = self._run_path_diagnostics()

        # Run distributed tests if available
        if DIST_AVAILABLE:
            print("Running distributed network diagnostics...")
            diagnostics['distributed_tests'] = self._run_distributed_diagnostics()

        return diagnostics

    def _run_ping_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive ping diagnostics."""
        results = {}

        for host in self.test_hosts:
            try:
                ping_result = self._ping_host_advanced(host)
                results[host] = ping_result
            except Exception as e:
                results[host] = {'error': str(e), 'latency': float('inf')}

        return results

    def _ping_host_advanced(self, host: str) -> Dict[str, Any]:
        """Advanced ping test using system ping command."""
        try:
            # Determine ping command based on platform
            if platform.system().lower() == 'windows':
                cmd = ['ping', '-n', '4', host]
            else:
                cmd = ['ping', '-c', '4', host]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                # Parse ping output
                output_lines = result.stdout.split('\n')
                latencies = []

                for line in output_lines:
                    if 'time=' in line or 'time<' in line:
                        # Extract latency value
                        try:
                            if 'time=' in line:
                                time_part = line.split('time=')[1].split()[0]
                                latency = float(time_part)
                                latencies.append(latency)
                            elif 'time<' in line:
                                latencies.append(0.1)  # Very fast response
                        except (ValueError, IndexError):
                            continue

                if latencies:
                    return {
                        'latency': np.mean(latencies),
                        'min_latency': np.min(latencies),
                        'max_latency': np.max(latencies),
                        'std_latency': np.std(latencies),
                        'packet_loss': 0.0,
                        'success': True
                    }
                else:
                    return {'error': 'Could not parse ping output', 'latency': float('inf')}
            else:
                return {'error': f'Ping failed: {result.stderr}', 'latency': float('inf')}

        except subprocess.TimeoutExpired:
            return {'error': 'Ping timeout', 'latency': float('inf')}
        except Exception as e:
            return {'error': str(e), 'latency': float('inf')}

    def _run_dns_diagnostics(self) -> Dict[str, Any]:
        """Run DNS resolution diagnostics."""
        results = {}

        for host in self.test_hosts:
            if not self._is_ip_address(host):
                try:
                    start_time = time.time()
                    resolved_ip = socket.gethostbyname(host)
                    end_time = time.time()

                    results[host] = {
                        'resolved_ip': resolved_ip,
                        'resolution_time_ms': (end_time - start_time) * 1000,
                        'success': True
                    }
                except Exception as e:
                    results[host] = {
                        'error': str(e),
                        'resolution_time_ms': float('inf'),
                        'success': False
                    }

        return results

    def _run_connectivity_diagnostics(self) -> Dict[str, Any]:
        """Run TCP/UDP connectivity diagnostics."""
        results = {
            'tcp_tests': {},
            'udp_tests': {},
            'socket_tests': {}
        }

        # TCP connectivity tests
        for host in self.test_hosts[:3]:  # Test first 3 hosts
            for port in self.port_tests:
                try:
                    start_time = time.time()
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5.0)
                    result = sock.connect_ex((host, port))
                    end_time = time.time()
                    sock.close()

                    key = f"{host}:{port}"
                    results['tcp_tests'][key] = {
                        'connected': result == 0,
                        'connect_time_ms': (end_time - start_time) * 1000,
                        'success': result == 0
                    }
                except Exception as e:
                    key = f"{host}:{port}"
                    results['tcp_tests'][key] = {
                        'error': str(e),
                        'connected': False,
                        'success': False
                    }

        # UDP connectivity tests (DNS)
        for dns_server in self.dns_servers:
            try:
                start_time = time.time()
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(5.0)
                sock.sendto(b'\x00', (dns_server, 53))
                sock.close()
                end_time = time.time()

                results['udp_tests'][dns_server] = {
                    'connect_time_ms': (end_time - start_time) * 1000,
                    'success': True
                }
            except Exception as e:
                results['udp_tests'][dns_server] = {
                    'error': str(e),
                    'success': False
                }

        return results

    def _run_bandwidth_diagnostics(self) -> Dict[str, Any]:
        """Run bandwidth measurement diagnostics."""
        results = {}

        # Test with speedtest-cli
        try:
            speedtest_result = self._test_speedtest_cli()
            results['speedtest'] = speedtest_result
        except Exception as e:
            results['speedtest'] = {'error': str(e)}

        # Test with iperf
        try:
            iperf_result = self._test_iperf()
            results['iperf'] = iperf_result
        except Exception as e:
            results['iperf'] = {'error': str(e)}

        # Test with curl (download speed)
        try:
            curl_result = self._test_curl_download()
            results['curl'] = curl_result
        except Exception as e:
            results['curl'] = {'error': str(e)}

        return results

    def _test_speedtest_cli(self) -> Dict[str, Any]:
        """Test bandwidth using speedtest-cli."""
        try:
            result = subprocess.run(
                ['speedtest-cli', '--simple', '--json'],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                return {
                    'download_mbps': data.get('download', 0.0) / 1_000_000,
                    'upload_mbps': data.get('upload', 0.0) / 1_000_000,
                    'ping_ms': data.get('ping', 0.0),
                    'success': True
                }
            else:
                return {'error': f'Speedtest failed: {result.stderr}', 'success': False}

        except subprocess.TimeoutExpired:
            return {'error': 'Speedtest timeout', 'success': False}
        except Exception as e:
            return {'error': str(e), 'success': False}

    def _test_iperf(self) -> Dict[str, Any]:
        """Test bandwidth using iperf3."""
        try:
            result = subprocess.run(
                ['iperf3', '-c', 'speedtest.tele2.net', '-p', '5202', '-J', '-t', '10'],
                capture_output=True,
                text=True,
                timeout=20
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                return {
                    'bandwidth_mbps': data.get('end', {}).get('streams', [{}])[0].get('receiver', {}).get('bits_per_second', 0) / 1_000_000,
                    'success': True
                }
            else:
                return {'error': f'Iperf failed: {result.stderr}', 'success': False}

        except subprocess.TimeoutExpired:
            return {'error': 'Iperf timeout', 'success': False}
        except Exception as e:
            return {'error': str(e), 'success': False}

    def _test_curl_download(self) -> Dict[str, Any]:
        """Test download speed using curl."""
        try:
            # Download a small file to test speed
            result = subprocess.run(
                ['curl', '-o', '/dev/null', '-s', '-w', '%{speed_download}', 'https://speed.hetzner.de/100MB.bin'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                speed_bps = float(result.stdout)
                return {
                    'download_mbps': speed_bps / 1_000_000,
                    'success': True
                }
            else:
                return {'error': f'Curl failed: {result.stderr}', 'success': False}

        except subprocess.TimeoutExpired:
            return {'error': 'Curl timeout', 'success': False}
        except Exception as e:
            return {'error': str(e), 'success': False}

    def _run_interface_diagnostics(self) -> Dict[str, Any]:
        """Run network interface diagnostics."""
        results = {}

        try:
            interfaces = psutil.net_if_stats()
            addresses = psutil.net_if_addrs()

            for interface_name, stats in interfaces.items():
                if stats.isup:
                    interface_info = {
                        'is_up': stats.isup,
                        'speed_mbps': stats.speed if stats.speed > 0 else None,
                        'mtu': stats.mtu,
                        'duplex': stats.duplex,
                    }

                    # Get IP addresses
                    if interface_name in addresses:
                        ip_addresses = []
                        for addr in addresses[interface_name]:
                            if addr.family == socket.AF_INET:
                                ip_addresses.append(addr.address)
                        interface_info['ip_addresses'] = ip_addresses

                    # Get interface statistics
                    try:
                        interface_stats = psutil.net_io_counters(pernic=True).get(interface_name)
                        if interface_stats:
                            interface_info.update({
                                'bytes_sent': interface_stats.bytes_sent,
                                'bytes_recv': interface_stats.bytes_recv,
                                'packets_sent': interface_stats.packets_sent,
                                'packets_recv': interface_stats.packets_recv,
                                'errin': interface_stats.errin,
                                'errout': interface_stats.errout,
                                'dropin': interface_stats.dropin,
                                'dropout': interface_stats.dropout,
                            })
                    except Exception:
                        pass

                    results[interface_name] = interface_info

        except Exception as e:
            results['error'] = str(e)

        return results

    def _run_path_diagnostics(self) -> Dict[str, Any]:
        """Run network path diagnostics (traceroute)."""
        results = {}

        for host in self.test_hosts[:3]:  # Test first 3 hosts
            try:
                path_result = self._traceroute_host(host)
                results[host] = path_result
            except Exception as e:
                results[host] = {'error': str(e)}

        return results

    def _traceroute_host(self, host: str) -> Dict[str, Any]:
        """Perform traceroute to a host."""
        try:
            if platform.system().lower() == 'windows':
                cmd = ['tracert', host]
            else:
                cmd = ['traceroute', '-n', '-w', '1', host]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                # Parse traceroute output
                lines = result.stdout.split('\n')
                hops = []

                for line in lines:
                    if '*' not in line and 'ms' in line:
                        try:
                            # Extract hop information
                            parts = line.split()
                            if len(parts) >= 3:
                                hop_num = int(parts[0])
                                hop_ip = parts[1]
                                hop_time = float(parts[2].replace('ms', ''))
                                hops.append({
                                    'hop': hop_num,
                                    'ip': hop_ip,
                                    'latency_ms': hop_time
                                })
                        except (ValueError, IndexError):
                            continue

                return {
                    'hops': hops,
                    'total_hops': len(hops),
                    'final_latency_ms': hops[-1]['latency_ms'] if hops else 0.0,
                    'success': True
                }
            else:
                return {'error': f'Traceroute failed: {result.stderr}', 'success': False}

        except subprocess.TimeoutExpired:
            return {'error': 'Traceroute timeout', 'success': False}
        except Exception as e:
            return {'error': str(e), 'success': False}

    def _run_distributed_diagnostics(self) -> Dict[str, Any]:
        """Run distributed training network diagnostics."""
        results = {}

        if not DIST_AVAILABLE or not dist.is_initialized():
            return {'error': 'PyTorch distributed not available or initialized'}

        try:
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            results['world_size'] = world_size
            results['rank'] = rank

            # Test basic distributed operations
            if torch.cuda.is_available():
                test_tensor = torch.randn(100, 100, device='cuda')
            else:
                test_tensor = torch.randn(100, 100)

            # Test allreduce
            start_time = time.time()
            dist.all_reduce(test_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            allreduce_time = time.time() - start_time

            # Prevent division by zero and ensure minimum measurement time
            if allreduce_time <= 0.001:  # Less than 1ms, likely measurement error
                results['allreduce_test'] = {
                    'time_ms': allreduce_time * 1000,
                    'tensor_size_mb': test_tensor.numel() * test_tensor.element_size() / (1024 * 1024),
                    'bandwidth_mbps': 0.0
                }
            else:
                results['allreduce_test'] = {
                    'time_ms': allreduce_time * 1000,
                    'tensor_size_mb': test_tensor.numel() * test_tensor.element_size() / (1024 * 1024),
                    'bandwidth_mbps': (test_tensor.numel() * test_tensor.element_size() * 8) / (allreduce_time * 1_000_000)
                }

            # Test broadcast
            start_time = time.time()
            dist.broadcast(test_tensor, src=0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            broadcast_time = time.time() - start_time

            # Prevent division by zero and ensure minimum measurement time
            if broadcast_time <= 0.001:  # Less than 1ms, likely measurement error
                results['broadcast_test'] = {
                    'time_ms': broadcast_time * 1000,
                    'tensor_size_mb': test_tensor.numel() * test_tensor.element_size() / (1024 * 1024),
                    'bandwidth_mbps': 0.0
                }
            else:
                results['broadcast_test'] = {
                    'time_ms': broadcast_time * 1000,
                    'tensor_size_mb': test_tensor.numel() * test_tensor.element_size() / (1024 * 1024),
                    'bandwidth_mbps': (test_tensor.numel() * test_tensor.element_size() * 8) / (broadcast_time * 1_000_000)
                }

        except Exception as e:
            results['error'] = str(e)

        return results

    def _is_ip_address(self, host: str) -> bool:
        """Check if a string is an IP address."""
        try:
            socket.inet_aton(host)
            return True
        except OSError:
            return False


class NetworkInterfaceMonitor:
    """Monitor network interface statistics."""

    def __init__(self, interface_name: Optional[str] = None, sampling_frequency: int = 1):
        """Initialize network interface monitor.

        Args:
            interface_name: Specific interface to monitor (e.g., 'eth0', 'en0')
                          If None, will auto-detect the primary interface
            sampling_frequency: Only collect metrics every N invocations (default: 1)
        """
        self.interface_name = interface_name or self._detect_primary_interface()
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._last_stats = None
        self._last_time = None
        self.sampling_frequency = sampling_frequency
        self._invocation_count = 0

    def _detect_primary_interface(self) -> str:
        """Detect the primary network interface."""
        try:
            # Get all network interfaces
            interfaces = psutil.net_if_stats()

            # Find the first active interface that's not loopback
            for name, stats in interfaces.items():
                if (stats.isup and
                    not name.startswith('lo') and
                    not name.startswith('docker') and
                    not name.startswith('veth')):
                    return name

            # Fallback to first available interface
            for name in interfaces.keys():
                if not name.startswith('lo'):
                    return name

        except Exception:
            pass

        return 'eth0'  # Default fallback

    def get_interface_stats(self) -> Dict[str, float]:
        """Get current network interface statistics."""
        with self._lock:  # Thread-safe access to shared state
            self._invocation_count += 1

            # Only collect metrics if sampling frequency allows
            if self._invocation_count % self.sampling_frequency != 0:
                return self._empty_stats()

            try:
                # Get current network I/O counters
                net_io = psutil.net_io_counters(pernic=True)

                if self.interface_name not in net_io:
                    # Try to find an available interface
                    available_interfaces = list(net_io.keys())
                    if available_interfaces:
                        self.interface_name = available_interfaces[0]
                    else:
                        return self._empty_stats()

                current_stats = net_io[self.interface_name]
                current_time = time.time()

                if self._last_stats is None:
                    # First measurement
                    self._last_stats = current_stats
                    self._last_time = current_time
                    return self._empty_stats()

                # Calculate rates
                time_diff = current_time - self._last_time
                if time_diff <= 0:
                    return self._empty_stats()

                bytes_in_rate = (current_stats.bytes_recv - self._last_stats.bytes_recv) / time_diff
                bytes_out_rate = (current_stats.bytes_sent - self._last_stats.bytes_sent) / time_diff
                packets_in_rate = (current_stats.packets_recv - self._last_stats.packets_recv) / time_diff
                packets_out_rate = (current_stats.packets_sent - self._last_stats.packets_sent) / time_diff

                # Update last stats atomically
                self._last_stats = current_stats
                self._last_time = current_time

                return {
                    'bytes_in_per_sec': bytes_in_rate,
                    'bytes_out_per_sec': bytes_out_rate,
                    'packets_in_per_sec': packets_in_rate,
                    'packets_out_per_sec': packets_out_rate,
                    'bytes_in_mbps': (bytes_in_rate * 8) / 1_000_000,  # Convert to Mbps
                    'bytes_out_mbps': (bytes_out_rate * 8) / 1_000_000,  # Convert to Mbps
                    'total_bytes_recv': current_stats.bytes_recv,
                    'total_bytes_sent': current_stats.bytes_sent,
                    'total_packets_recv': current_stats.packets_recv,
                    'total_packets_sent': current_stats.packets_sent,
                    'dropin': current_stats.dropin,
                    'dropout': current_stats.dropout,
                    'errin': current_stats.errin,
                    'errout': current_stats.errout,
                }

            except Exception as e:
                print(f"Error getting interface stats: {e}")
                return self._empty_stats()

    def _empty_stats(self) -> Dict[str, float]:
        """Return empty statistics."""
        return {
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


class NetworkLatencyMonitor:
    """Monitor network latency using various methods."""

    def __init__(self, target_hosts: Optional[List[str]] = None, sampling_frequency: int = 1):
        """Initialize latency monitor.

        Args:
            target_hosts: List of hosts to ping for latency measurement
            sampling_frequency: Only collect metrics every N invocations (default: 1)
        """
        self.target_hosts = target_hosts or [
            '8.8.8.8',  # Google DNS
            '1.1.1.1',  # Cloudflare DNS
            '208.67.222.222',  # OpenDNS
        ]
        self._lock = threading.RLock()  # Thread-safe access to history
        # Use deque for thread-safe operations with maxlen
        self.latency_history: Dict[str, deque] = {
            host: deque(maxlen=100) for host in self.target_hosts
        }
        self.max_history_size = 100
        self.sampling_frequency = sampling_frequency
        self._invocation_count = 0

    def measure_latency(self) -> Dict[str, float]:
        """Measure latency to multiple hosts."""
        with self._lock:
            self._invocation_count += 1

            # Only collect metrics if sampling frequency allows
            if self._invocation_count % self.sampling_frequency != 0:
                # Return cached results or empty results
                return dict.fromkeys(self.target_hosts, 0.0)

            # Keep lock during expensive operations to prevent race conditions
            results = {}

            with ThreadPoolExecutor(max_workers=len(self.target_hosts)) as executor:
                future_to_host = {
                    executor.submit(self._ping_host, host): host
                    for host in self.target_hosts
                }

                for future in as_completed(future_to_host, timeout=5.0):
                    host = future_to_host[future]
                    try:
                        latency = future.result()
                        results[host] = latency

                        # Thread-safe update of history
                        self.latency_history[host].append(latency)

                    except Exception as e:
                        print(f"Error measuring latency to {host}: {e}")
                        results[host] = float('inf')

            return results

    def _ping_host(self, host: str) -> float:
        """Ping a specific host and return latency in milliseconds."""
        try:
            # Use advanced ping diagnostics
            diagnostics = NetworkDiagnostics()
            ping_result = diagnostics._ping_host_advanced(host)

            if ping_result.get('success', False):
                return ping_result['latency']
            else:
                return float('inf')

        except Exception:
            return float('inf')

    def get_average_latency(self) -> float:
        """Get average latency across all hosts."""
        with self._lock:
            all_latencies = []
            for host_latencies in self.latency_history.values():
                if host_latencies:
                    # Filter out infinite latencies
                    valid_latencies = [lat for lat in host_latencies if lat != float('inf')]
                    if valid_latencies:
                        all_latencies.extend(valid_latencies)

            if all_latencies:
                return np.mean(all_latencies)
            else:
                return 0.0

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        with self._lock:
            all_latencies = []
            for host_latencies in self.latency_history.values():
                valid_latencies = [lat for lat in host_latencies if lat != float('inf')]
                all_latencies.extend(valid_latencies)

            if not all_latencies:
                return {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'median': 0.0,
                }

            return {
                'mean': np.mean(all_latencies),
                'std': np.std(all_latencies),
                'min': np.min(all_latencies),
                'max': np.max(all_latencies),
                'median': np.median(all_latencies),
            }


class NetworkBandwidthMonitor:
    """Monitor network bandwidth using various methods."""

    def __init__(self, sampling_frequency: int = 1):
        """Initialize bandwidth monitor.

        Args:
            sampling_frequency: Only collect metrics every N invocations (default: 1)
        """
        self._lock = threading.RLock()  # Thread-safe access to shared state
        self.bandwidth_history: deque = deque(maxlen=100)
        self.max_history_size = 100
        self._last_measurement = 0.0
        self.measurement_interval = 10.0  # seconds
        self.sampling_frequency = sampling_frequency
        self._invocation_count = 0

    def measure_bandwidth(self) -> float:
        """Measure network bandwidth in Mbps."""
        with self._lock:
            self._invocation_count += 1
            current_time = time.time()

            # Only measure if enough time has passed and sampling frequency allows
            if (current_time - self._last_measurement < self.measurement_interval or
                self._invocation_count % self.sampling_frequency != 0):
                return self.bandwidth_history[-1] if self.bandwidth_history else 0.0

            try:
                # Method 1: Use speedtest-cli if available
                bandwidth = self._measure_with_speedtest()

                # Method 2: Use iperf if available
                if bandwidth == 0.0:
                    bandwidth = self._measure_with_iperf()

                # Method 3: Estimate from interface statistics
                if bandwidth == 0.0:
                    bandwidth = self._estimate_from_interface()

                # Thread-safe update of history
                self.bandwidth_history.append(bandwidth)
                self._last_measurement = current_time
                return bandwidth

            except Exception as e:
                print(f"Error measuring bandwidth: {e}")
                return 0.0

    def _measure_with_speedtest(self) -> float:
        """Measure bandwidth using speedtest-cli."""
        try:
            result = subprocess.run(
                ['speedtest-cli', '--simple', '--json'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                return data.get('download', 0.0) / 1_000_000  # Convert to Mbps
            else:
                return 0.0

        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            return 0.0

    def _measure_with_iperf(self) -> float:
        """Measure bandwidth using iperf."""
        try:
            # Try to connect to a public iperf server
            result = subprocess.run(
                ['iperf3', '-c', 'speedtest.tele2.net', '-p', '5202', '-J', '-t', '5'],
                capture_output=True,
                text=True,
                timeout=15
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                return data.get('end', {}).get('streams', [{}])[0].get('receiver', {}).get('bits_per_second', 0) / 1_000_000
            else:
                return 0.0

        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            return 0.0

    def _estimate_from_interface(self) -> float:
        """Estimate bandwidth from interface statistics."""
        try:
            # Get interface statistics
            interface_monitor = NetworkInterfaceMonitor()
            stats = interface_monitor.get_interface_stats()

            # Estimate bandwidth as the sum of in/out rates
            total_bandwidth = stats['bytes_in_mbps'] + stats['bytes_out_mbps']

            # This is a rough estimate - actual bandwidth depends on many factors
            return total_bandwidth

        except Exception:
            return 0.0

    def get_bandwidth_stats(self) -> Dict[str, float]:
        """Get bandwidth statistics."""
        with self._lock:
            if not self.bandwidth_history:
                return {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'median': 0.0,
                }

            return {
                'mean': np.mean(self.bandwidth_history),
                'std': np.std(self.bandwidth_history),
                'min': np.min(self.bandwidth_history),
                'max': np.max(self.bandwidth_history),
                'median': np.median(self.bandwidth_history),
            }


class DistributedNetworkMonitor:
    """Monitor network performance specifically for distributed training."""

    def __init__(self, world_size: int = 1, rank: int = 0, enable_distributed_measurements: bool = False, sampling_frequency: int = 1):
        """Initialize distributed network monitor.

        Args:
            world_size: Total number of processes in distributed training
            rank: Rank of current process
            enable_distributed_measurements: Whether to perform active distributed measurements
                                          (can interfere with actual training if True)
                                          Defaults to False for safety.
            sampling_frequency: Only collect metrics every N invocations (default: 1)
        """
        self.world_size = world_size
        self.rank = rank
        self.enable_distributed_measurements = enable_distributed_measurements
        self.sampling_frequency = sampling_frequency
        self._invocation_count = 0

        # Initialize monitors without sampling frequency (parent controls sampling)
        self.interface_monitor = NetworkInterfaceMonitor(sampling_frequency=1)
        self.latency_monitor = NetworkLatencyMonitor(sampling_frequency=1)
        self.bandwidth_monitor = NetworkBandwidthMonitor(sampling_frequency=1)

        # Thread safety for distributed metrics
        self._distributed_lock = threading.RLock()

        # Distributed training metrics (thread-safe with deque)
        self.allreduce_times: deque = deque(maxlen=100)
        self.broadcast_times: deque = deque(maxlen=100)
        self.gather_times: deque = deque(maxlen=100)
        self.scatter_times: deque = deque(maxlen=100)

        # Performance history (thread-safe with deque)
        self.performance_history: deque = deque(maxlen=1000)
        self.max_history_size = 1000

    def measure_distributed_metrics(self) -> NetworkMetrics:
        """Measure comprehensive network metrics for distributed training."""
        with self._distributed_lock:
            self._invocation_count += 1

            # Only collect metrics if sampling frequency allows
            if self._invocation_count % self.sampling_frequency != 0:
                # Return cached metrics or empty metrics
                return NetworkMetrics(
                    timestamp=time.time(),
                    world_size=self.world_size,
                    rank=self.rank
                )

            # Keep lock during expensive operations to prevent race conditions
            metrics = NetworkMetrics(
                timestamp=time.time(),
                world_size=self.world_size,
                rank=self.rank
            )

            # Basic network metrics
            interface_stats = self.interface_monitor.get_interface_stats()
            self.latency_monitor.measure_latency()
            bandwidth = self.bandwidth_monitor.measure_bandwidth()

            # Update metrics
            metrics.bandwidth_mbps = bandwidth
            metrics.latency_ms = self.latency_monitor.get_average_latency()
            metrics.bandwidth_in_mbps = interface_stats['bytes_in_mbps']
            metrics.bandwidth_out_mbps = interface_stats['bytes_out_mbps']
            metrics.packets_in_per_sec = interface_stats['packets_in_per_sec']
            metrics.packets_out_per_sec = interface_stats['packets_out_per_sec']
            metrics.bytes_in_per_sec = interface_stats['bytes_in_per_sec']
            metrics.bytes_out_per_sec = interface_stats['bytes_out_per_sec']

            # Connection metrics
            try:
                connections = psutil.net_connections()
                metrics.tcp_connections = len([c for c in connections if c.status == 'ESTABLISHED' and c.type == socket.SOCK_STREAM])
                metrics.udp_connections = len([c for c in connections if c.type == socket.SOCK_DGRAM])
                metrics.active_connections = len([c for c in connections if c.status == 'ESTABLISHED'])
            except Exception:
                pass

            # Error metrics
            metrics.network_errors = int(interface_stats['errin'] + interface_stats['errout'])
            metrics.dropped_packets = int(interface_stats['dropin'] + interface_stats['dropout'])

            # Calculate packet loss percentage
            total_packets = interface_stats['total_packets_recv'] + interface_stats['total_packets_sent']
            if total_packets > 0:
                metrics.packet_loss_percent = (metrics.dropped_packets / total_packets) * 100

            # Distributed training specific metrics
            if DIST_AVAILABLE and dist.is_initialized() and self.enable_distributed_measurements:
                metrics.allreduce_bandwidth = self._measure_allreduce_bandwidth()
                metrics.broadcast_bandwidth = self._measure_broadcast_bandwidth()
                metrics.gather_bandwidth = self._measure_gather_bandwidth()
                metrics.scatter_bandwidth = self._measure_scatter_bandwidth()
            else:
                # Set to 0 if distributed measurements are disabled or not available
                metrics.allreduce_bandwidth = 0.0
                metrics.broadcast_bandwidth = 0.0
                metrics.gather_bandwidth = 0.0
                metrics.scatter_bandwidth = 0.0

            # Thread-safe update of history
            self.performance_history.append(metrics)

            return metrics

    def _measure_allreduce_bandwidth(self) -> float:
        """Measure bandwidth during allreduce operations."""
        if not DIST_AVAILABLE or not dist.is_initialized():
            return 0.0

        try:
            import torch

            # Create a test tensor
            test_tensor = torch.randn(1000, 1000, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            # Measure allreduce time
            start_time = time.time()
            dist.all_reduce(test_tensor)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

            allreduce_time = end_time - start_time

            # Prevent division by zero and ensure minimum measurement time
            if allreduce_time <= 0.001:  # Less than 1ms, likely measurement error
                return 0.0

            with self._distributed_lock:
                self.allreduce_times.append(allreduce_time)

            # Calculate bandwidth (tensor size / time)
            tensor_size_bytes = test_tensor.numel() * test_tensor.element_size()
            bandwidth_mbps = (tensor_size_bytes * 8) / (allreduce_time * 1_000_000)

            return bandwidth_mbps

        except Exception as e:
            print(f"Error measuring allreduce bandwidth: {e}")
            return 0.0

    def _measure_broadcast_bandwidth(self) -> float:
        """Measure bandwidth during broadcast operations."""
        if not DIST_AVAILABLE or not dist.is_initialized():
            return 0.0

        try:
            import torch

            test_tensor = torch.randn(1000, 1000, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            start_time = time.time()
            dist.broadcast(test_tensor, src=0)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

            broadcast_time = end_time - start_time

            # Prevent division by zero and ensure minimum measurement time
            if broadcast_time <= 0.001:  # Less than 1ms, likely measurement error
                return 0.0

            with self._distributed_lock:
                self.broadcast_times.append(broadcast_time)

            tensor_size_bytes = test_tensor.numel() * test_tensor.element_size()
            bandwidth_mbps = (tensor_size_bytes * 8) / (broadcast_time * 1_000_000)

            return bandwidth_mbps

        except Exception as e:
            print(f"Error measuring broadcast bandwidth: {e}")
            return 0.0

    def _measure_gather_bandwidth(self) -> float:
        """Measure bandwidth during gather operations."""
        if not DIST_AVAILABLE or not dist.is_initialized():
            return 0.0

        try:
            import torch

            test_tensor = torch.randn(100, 100, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            gathered_tensors = [torch.zeros_like(test_tensor) for _ in range(self.world_size)]

            start_time = time.time()
            dist.all_gather(gathered_tensors, test_tensor)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

            gather_time = end_time - start_time

            # Prevent division by zero and ensure minimum measurement time
            if gather_time <= 0.001:  # Less than 1ms, likely measurement error
                return 0.0

            with self._distributed_lock:
                self.gather_times.append(gather_time)

            total_size_bytes = sum(t.numel() * t.element_size() for t in gathered_tensors)
            bandwidth_mbps = (total_size_bytes * 8) / (gather_time * 1_000_000)

            return bandwidth_mbps

        except Exception as e:
            print(f"Error measuring gather bandwidth: {e}")
            return 0.0

    def _measure_scatter_bandwidth(self) -> float:
        """Measure bandwidth during scatter operations."""
        if not DIST_AVAILABLE or not dist.is_initialized():
            return 0.0

        try:
            import torch

            test_tensor = torch.randn(100, 100, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            start_time = time.time()

            # Handle scatter differently for source vs non-source ranks
            if self.rank == 0:
                # Source rank: create scatter_list and call scatter
                scattered_tensors = [torch.zeros_like(test_tensor) for _ in range(self.world_size)]
                dist.scatter(test_tensor, scattered_tensors, src=0)
            else:
                # Non-source ranks: call scatter without scatter_list
                dist.scatter(test_tensor, src=0)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

            scatter_time = end_time - start_time

            # Prevent division by zero and ensure minimum measurement time
            if scatter_time <= 0.001:  # Less than 1ms, likely measurement error
                return 0.0

            with self._distributed_lock:
                self.scatter_times.append(scatter_time)

            # Calculate bandwidth based on rank
            if self.rank == 0:
                # Source rank: total size of all scattered tensors
                scattered_tensors = [torch.zeros_like(test_tensor) for _ in range(self.world_size)]
                total_size_bytes = sum(t.numel() * t.element_size() for t in scattered_tensors)
            else:
                # Non-source ranks: size of received tensor
                total_size_bytes = test_tensor.numel() * test_tensor.element_size()

            bandwidth_mbps = (total_size_bytes * 8) / (scatter_time * 1_000_000)

            return bandwidth_mbps

        except Exception as e:
            print(f"Error measuring scatter bandwidth: {e}")
            return 0.0

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of network performance."""
        with self._distributed_lock:
            if not self.performance_history:
                return {
                    'total_measurements': 0,
                    'avg_bandwidth': 0.0,
                    'avg_latency': 0.0,
                    'avg_packet_loss': 0.0,
                    'distributed_metrics': {},
                }

            recent_metrics = list(self.performance_history)[-100:]  # Last 100 measurements

            summary = {
                'total_measurements': len(self.performance_history),
                'avg_bandwidth': np.mean([m.bandwidth_mbps for m in recent_metrics]),
                'avg_latency': np.mean([m.latency_ms for m in recent_metrics]),
                'avg_packet_loss': np.mean([m.packet_loss_percent for m in recent_metrics]),
                'max_bandwidth': np.max([m.bandwidth_mbps for m in recent_metrics]),
                'min_latency': np.min([m.latency_ms for m in recent_metrics if m.latency_ms > 0]),
                'distributed_metrics': {
                    'avg_allreduce_bandwidth': np.mean([m.allreduce_bandwidth for m in recent_metrics]),
                    'avg_broadcast_bandwidth': np.mean([m.broadcast_bandwidth for m in recent_metrics]),
                    'avg_gather_bandwidth': np.mean([m.gather_bandwidth for m in recent_metrics]),
                    'avg_scatter_bandwidth': np.mean([m.scatter_bandwidth for m in recent_metrics]),
                }
            }

            return summary

    def get_metrics_dataframe(self) -> 'pd.DataFrame':
        """Get all metrics as a DataFrame."""
        try:
            import pandas as pd

            with self._distributed_lock:
                if not self.performance_history:
                    return pd.DataFrame()

                metrics_data = [m.to_dict() for m in self.performance_history]
                return pd.DataFrame(metrics_data)

        except ImportError:
            print("pandas not available for DataFrame export")
            return None


class RealNetworkMonitor:
    """Enhanced network monitor with real measurements for OpenRLHF."""

    def __init__(self, enable_distributed_monitoring: bool = True, enable_distributed_measurements: bool = False, sampling_frequency: int = 1):
        """Initialize real network monitor.

        Args:
            enable_distributed_monitoring: Whether to enable distributed training monitoring
            enable_distributed_measurements: Whether to perform active distributed measurements
                                          (can interfere with actual training if True)
                                          Defaults to False for safety.
            sampling_frequency: Only collect metrics every N invocations (default: 1)
        """
        self.enable_distributed_monitoring = enable_distributed_monitoring
        self.enable_distributed_measurements = enable_distributed_measurements
        self.sampling_frequency = sampling_frequency
        self._invocation_count = 0

        # Initialize monitors without sampling frequency (parent controls sampling)
        self.interface_monitor = NetworkInterfaceMonitor(sampling_frequency=1)
        self.latency_monitor = NetworkLatencyMonitor(sampling_frequency=1)
        self.bandwidth_monitor = NetworkBandwidthMonitor(sampling_frequency=1)

        # Initialize comprehensive diagnostics
        self.network_diagnostics = NetworkDiagnostics(sampling_frequency=1)

        # Distributed monitor
        self.distributed_monitor = None
        if self.enable_distributed_monitoring and DIST_AVAILABLE:
            try:
                world_size = dist.get_world_size() if dist.is_initialized() else 1
                rank = dist.get_rank() if dist.is_initialized() else 0
                self.distributed_monitor = DistributedNetworkMonitor(
                    world_size, rank,
                    enable_distributed_measurements=self.enable_distributed_measurements,
                    sampling_frequency=sampling_frequency
                )
            except Exception as e:
                print(f"Failed to initialize distributed monitor: {e}")

        # Thread safety for performance history
        self._history_lock = threading.RLock()

        # Performance history (thread-safe with deque)
        self.bandwidth_history: deque = deque(maxlen=100)
        self.latency_history: deque = deque(maxlen=100)
        self._last_measurement = 0.0
        self.measurement_interval = 5.0  # seconds

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current network metrics."""
        with self._history_lock:
            self._invocation_count += 1
            current_time = time.time()

            # Only measure if enough time has passed and sampling frequency allows
            if (current_time - self._last_measurement < self.measurement_interval or
                self._invocation_count % self.sampling_frequency != 0):
                return {
                    'bandwidth': self.bandwidth_history[-1] if self.bandwidth_history else 0.0,
                    'latency': self.latency_history[-1] if self.latency_history else 0.0,
                }

            # Keep lock during expensive operations to prevent race conditions
            # Get comprehensive metrics
            if self.distributed_monitor:
                metrics = self.distributed_monitor.measure_distributed_metrics()
                bandwidth = metrics.bandwidth_mbps
                latency = metrics.latency_ms
            else:
                # Fallback to basic measurements
                bandwidth = self.bandwidth_monitor.measure_bandwidth()
                latency = self.latency_monitor.get_average_latency()

            # Thread-safe update of history
            self.bandwidth_history.append(bandwidth)
            self.latency_history.append(latency)
            self._last_measurement = current_time

            return {
                'bandwidth': bandwidth,
                'latency': latency,
            }

    def get_network_stats(self) -> Dict[str, float]:
        """Get network performance statistics."""
        with self._history_lock:
            if not self.bandwidth_history or not self.latency_history:
                return {
                    'avg_bandwidth': 0.0,
                    'avg_latency': 0.0,
                    'max_bandwidth': 0.0,
                    'max_latency': 0.0,
                    'min_bandwidth': 0.0,
                    'min_latency': 0.0,
                }

            return {
                'avg_bandwidth': np.mean(self.bandwidth_history),
                'avg_latency': np.mean(self.latency_history),
                'max_bandwidth': np.max(self.bandwidth_history),
                'max_latency': np.max(self.latency_history),
                'min_bandwidth': np.min(self.bandwidth_history),
                'min_latency': np.min(self.latency_history),
            }

    def get_comprehensive_metrics(self) -> NetworkMetrics:
        """Get comprehensive network metrics."""
        if self.distributed_monitor:
            return self.distributed_monitor.measure_distributed_metrics()
        else:
            # Create basic metrics
            basic_metrics = self.get_current_metrics()
            return NetworkMetrics(
                bandwidth_mbps=basic_metrics['bandwidth'],
                latency_ms=basic_metrics['latency'],
                timestamp=time.time()
            )

    def run_network_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive network diagnostics."""
        return self.network_diagnostics.run_comprehensive_diagnostics()

    def get_network_health_report(self) -> Dict[str, Any]:
        """Get a comprehensive network health report."""
        diagnostics = self.run_network_diagnostics()

        # Analyze diagnostics and create health report
        health_report = {
            'timestamp': time.time(),
            'overall_health': 'unknown',
            'issues': [],
            'recommendations': [],
            'metrics_summary': {},
        }

        # Analyze ping tests
        ping_issues = []
        avg_ping_latency = 0.0
        ping_count = 0

        for host, result in diagnostics['ping_tests'].items():
            if result.get('success', False):
                latency = result.get('latency', 0.0)
                avg_ping_latency += latency
                ping_count += 1

                if latency > 100:  # High latency threshold
                    ping_issues.append(f"High latency to {host}: {latency:.2f}ms")
                elif latency == float('inf'):
                    ping_issues.append(f"Unreachable: {host}")
            else:
                ping_issues.append(f"Ping failed to {host}: {result.get('error', 'Unknown error')}")

        if ping_count > 0:
            avg_ping_latency /= ping_count
            health_report['metrics_summary']['avg_ping_latency_ms'] = avg_ping_latency

        # Analyze DNS tests
        dns_issues = []
        avg_dns_resolution = 0.0
        dns_count = 0

        for host, result in diagnostics['dns_tests'].items():
            if result.get('success', False):
                resolution_time = result.get('resolution_time_ms', 0.0)
                avg_dns_resolution += resolution_time
                dns_count += 1

                if resolution_time > 1000:  # Slow DNS resolution
                    dns_issues.append(f"Slow DNS resolution for {host}: {resolution_time:.2f}ms")
            else:
                dns_issues.append(f"DNS resolution failed for {host}: {result.get('error', 'Unknown error')}")

        if dns_count > 0:
            avg_dns_resolution /= dns_count
            health_report['metrics_summary']['avg_dns_resolution_ms'] = avg_dns_resolution

        # Analyze bandwidth tests
        bandwidth_issues = []
        best_bandwidth = 0.0

        for test_name, result in diagnostics['bandwidth_tests'].items():
            if result.get('success', False):
                if 'download_mbps' in result:
                    bandwidth = result['download_mbps']
                    best_bandwidth = max(best_bandwidth, bandwidth)

                    if bandwidth < 10:  # Low bandwidth threshold
                        bandwidth_issues.append(f"Low bandwidth in {test_name}: {bandwidth:.2f} Mbps")
                elif 'bandwidth_mbps' in result:
                    bandwidth = result['bandwidth_mbps']
                    best_bandwidth = max(best_bandwidth, bandwidth)

                    if bandwidth < 10:  # Low bandwidth threshold
                        bandwidth_issues.append(f"Low bandwidth in {test_name}: {bandwidth:.2f} Mbps")
            else:
                bandwidth_issues.append(f"Bandwidth test failed {test_name}: {result.get('error', 'Unknown error')}")

        health_report['metrics_summary']['best_bandwidth_mbps'] = best_bandwidth

        # Determine overall health
        all_issues = ping_issues + dns_issues + bandwidth_issues

        if len(all_issues) == 0:
            health_report['overall_health'] = 'excellent'
        elif len(all_issues) <= 2:
            health_report['overall_health'] = 'good'
        elif len(all_issues) <= 5:
            health_report['overall_health'] = 'fair'
        else:
            health_report['overall_health'] = 'poor'

        health_report['issues'] = all_issues

        # Generate recommendations
        if ping_issues:
            health_report['recommendations'].append("Check network connectivity and firewall settings")
        if dns_issues:
            health_report['recommendations'].append("Consider using different DNS servers")
        if bandwidth_issues:
            health_report['recommendations'].append("Check network bandwidth and consider upgrading connection")
        if len(all_issues) > 5:
            health_report['recommendations'].append("Consider running network diagnostics during off-peak hours")

        return health_report
