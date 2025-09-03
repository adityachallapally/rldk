"""Comprehensive network monitoring for OpenRLHF distributed training."""

import time
import socket
import threading
import subprocess
import psutil
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

try:
    import torch.distributed as dist
    DIST_AVAILABLE = True
except ImportError:
    DIST_AVAILABLE = False


@dataclass
class NetworkMetrics:
    """Comprehensive network metrics for distributed training."""
    # Basic metrics
    bandwidth_mbps: float = 0.0
    latency_ms: float = 0.0
    packet_loss_percent: float = 0.0
    
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
    
    # Timestamp
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }


class NetworkInterfaceMonitor:
    """Monitor network interface statistics."""
    
    def __init__(self, interface_name: Optional[str] = None):
        """Initialize network interface monitor.
        
        Args:
            interface_name: Specific interface to monitor (e.g., 'eth0', 'en0')
                          If None, will auto-detect the primary interface
        """
        self.interface_name = interface_name or self._detect_primary_interface()
        self.last_stats = None
        self.last_time = None
        
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
            
            if self.last_stats is None:
                # First measurement
                self.last_stats = current_stats
                self.last_time = current_time
                return self._empty_stats()
            
            # Calculate rates
            time_diff = current_time - self.last_time
            if time_diff <= 0:
                return self._empty_stats()
            
            bytes_in_rate = (current_stats.bytes_recv - self.last_stats.bytes_recv) / time_diff
            bytes_out_rate = (current_stats.bytes_sent - self.last_stats.bytes_sent) / time_diff
            packets_in_rate = (current_stats.packets_recv - self.last_stats.packets_recv) / time_diff
            packets_out_rate = (current_stats.packets_sent - self.last_stats.packets_sent) / time_diff
            
            # Update last stats
            self.last_stats = current_stats
            self.last_time = current_time
            
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
    
    def __init__(self, target_hosts: Optional[List[str]] = None):
        """Initialize latency monitor.
        
        Args:
            target_hosts: List of hosts to ping for latency measurement
        """
        self.target_hosts = target_hosts or [
            '8.8.8.8',  # Google DNS
            '1.1.1.1',  # Cloudflare DNS
            '208.67.222.222',  # OpenDNS
        ]
        self.latency_history: Dict[str, List[float]] = {host: [] for host in self.target_hosts}
        self.max_history_size = 100
    
    def measure_latency(self) -> Dict[str, float]:
        """Measure latency to multiple hosts."""
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
                    
                    # Update history
                    self.latency_history[host].append(latency)
                    if len(self.latency_history[host]) > self.max_history_size:
                        self.latency_history[host].pop(0)
                        
                except Exception as e:
                    print(f"Error measuring latency to {host}: {e}")
                    results[host] = float('inf')
        
        return results
    
    def _ping_host(self, host: str) -> float:
        """Ping a specific host and return latency in milliseconds."""
        try:
            # Use socket for basic connectivity test
            start_time = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            result = sock.connect_ex((host, 80))
            sock.close()
            end_time = time.time()
            
            if result == 0:
                return (end_time - start_time) * 1000  # Convert to milliseconds
            else:
                return float('inf')
                
        except Exception:
            return float('inf')
    
    def get_average_latency(self) -> float:
        """Get average latency across all hosts."""
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
    
    def __init__(self):
        """Initialize bandwidth monitor."""
        self.bandwidth_history: List[float] = []
        self.max_history_size = 100
        self.last_measurement = 0.0
        self.measurement_interval = 10.0  # seconds
    
    def measure_bandwidth(self) -> float:
        """Measure network bandwidth in Mbps."""
        current_time = time.time()
        
        # Only measure if enough time has passed
        if current_time - self.last_measurement < self.measurement_interval:
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
            
            # Update history
            self.bandwidth_history.append(bandwidth)
            if len(self.bandwidth_history) > self.max_history_size:
                self.bandwidth_history.pop(0)
            
            self.last_measurement = current_time
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
    
    def __init__(self, world_size: int = 1, rank: int = 0, enable_distributed_measurements: bool = True):
        """Initialize distributed network monitor.
        
        Args:
            world_size: Total number of processes in distributed training
            rank: Rank of current process
            enable_distributed_measurements: Whether to perform active distributed measurements
                                          (can interfere with actual training if True)
        """
        self.world_size = world_size
        self.rank = rank
        self.enable_distributed_measurements = enable_distributed_measurements
        
        # Initialize monitors
        self.interface_monitor = NetworkInterfaceMonitor()
        self.latency_monitor = NetworkLatencyMonitor()
        self.bandwidth_monitor = NetworkBandwidthMonitor()
        
        # Distributed training metrics
        self.allreduce_times: List[float] = []
        self.broadcast_times: List[float] = []
        self.gather_times: List[float] = []
        self.scatter_times: List[float] = []
        
        # Performance history
        self.performance_history: List[NetworkMetrics] = []
        self.max_history_size = 1000
    
    def measure_distributed_metrics(self) -> NetworkMetrics:
        """Measure comprehensive network metrics for distributed training."""
        metrics = NetworkMetrics(timestamp=time.time())
        
        # Basic network metrics
        interface_stats = self.interface_monitor.get_interface_stats()
        latency_results = self.latency_monitor.measure_latency()
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
        
        # Update history
        self.performance_history.append(metrics)
        if len(self.performance_history) > self.max_history_size:
            self.performance_history.pop(0)
        
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
            scattered_tensors = [torch.zeros_like(test_tensor) for _ in range(self.world_size)]
            
            start_time = time.time()
            dist.scatter(test_tensor, scattered_tensors, src=0)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            scatter_time = end_time - start_time
            
            # Prevent division by zero and ensure minimum measurement time
            if scatter_time <= 0.001:  # Less than 1ms, likely measurement error
                return 0.0
            
            self.scatter_times.append(scatter_time)
            
            total_size_bytes = sum(t.numel() * t.element_size() for t in scattered_tensors)
            bandwidth_mbps = (total_size_bytes * 8) / (scatter_time * 1_000_000)
            
            return bandwidth_mbps
            
        except Exception as e:
            print(f"Error measuring scatter bandwidth: {e}")
            return 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of network performance."""
        if not self.performance_history:
            return {
                'total_measurements': 0,
                'avg_bandwidth': 0.0,
                'avg_latency': 0.0,
                'avg_packet_loss': 0.0,
                'distributed_metrics': {},
            }
        
        recent_metrics = self.performance_history[-100:]  # Last 100 measurements
        
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
            
            if not self.performance_history:
                return pd.DataFrame()
            
            metrics_data = [m.to_dict() for m in self.performance_history]
            return pd.DataFrame(metrics_data)
            
        except ImportError:
            print("pandas not available for DataFrame export")
            return None


class RealNetworkMonitor:
    """Enhanced network monitor with real measurements for OpenRLHF."""
    
    def __init__(self, enable_distributed_monitoring: bool = True, enable_distributed_measurements: bool = False):
        """Initialize real network monitor.
        
        Args:
            enable_distributed_monitoring: Whether to enable distributed training monitoring
            enable_distributed_measurements: Whether to perform active distributed measurements
                                          (can interfere with actual training if True)
        """
        self.enable_distributed_monitoring = enable_distributed_monitoring
        self.enable_distributed_measurements = enable_distributed_measurements
        
        # Initialize monitors
        self.interface_monitor = NetworkInterfaceMonitor()
        self.latency_monitor = NetworkLatencyMonitor()
        self.bandwidth_monitor = NetworkBandwidthMonitor()
        
        # Distributed monitor
        self.distributed_monitor = None
        if self.enable_distributed_monitoring and DIST_AVAILABLE:
            try:
                world_size = dist.get_world_size() if dist.is_initialized() else 1
                rank = dist.get_rank() if dist.is_initialized() else 0
                self.distributed_monitor = DistributedNetworkMonitor(
                    world_size, rank, 
                    enable_distributed_measurements=self.enable_distributed_measurements
                )
            except Exception as e:
                print(f"Failed to initialize distributed monitor: {e}")
        
        # Performance history
        self.bandwidth_history = []
        self.latency_history = []
        self.last_measurement = 0.0
        self.measurement_interval = 5.0  # seconds
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current network metrics."""
        current_time = time.time()
        
        # Only measure if enough time has passed
        if current_time - self.last_measurement < self.measurement_interval:
            return {
                'bandwidth': self.bandwidth_history[-1] if self.bandwidth_history else 0.0,
                'latency': self.latency_history[-1] if self.latency_history else 0.0,
            }
        
        # Get comprehensive metrics
        if self.distributed_monitor:
            metrics = self.distributed_monitor.measure_distributed_metrics()
            bandwidth = metrics.bandwidth_mbps
            latency = metrics.latency_ms
        else:
            # Fallback to basic measurements
            bandwidth = self.bandwidth_monitor.measure_bandwidth()
            latency = self.latency_monitor.get_average_latency()
        
        # Update history
        self.bandwidth_history.append(bandwidth)
        self.latency_history.append(latency)
        
        # Keep only last 100 measurements
        if len(self.bandwidth_history) > 100:
            self.bandwidth_history.pop(0)
        if len(self.latency_history) > 100:
            self.latency_history.pop(0)
        
        self.last_measurement = current_time
        
        return {
            'bandwidth': bandwidth,
            'latency': latency,
        }
    
    def get_network_stats(self) -> Dict[str, float]:
        """Get network performance statistics."""
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