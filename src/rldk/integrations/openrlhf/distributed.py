"""Distributed training monitoring for OpenRLHF."""

import os
import platform
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import torch

try:
    import torch.distributed as dist
    DIST_AVAILABLE = True
except ImportError:
    DIST_AVAILABLE = False


@dataclass
class NodeMetrics:
    """Metrics for a single node in distributed training."""
    node_id: str
    rank: int
    local_rank: int
    gpu_count: int
    gpu_memory_used: List[float] = field(default_factory=list)
    gpu_memory_allocated: List[float] = field(default_factory=list)
    gpu_utilization: List[float] = field(default_factory=list)
    cpu_utilization: float = 0.0
    cpu_memory_used: float = 0.0
    network_bandwidth: float = 0.0
    network_latency: float = 0.0
    timestamp: float = 0.0


@dataclass
class DistributedMetrics:
    """Aggregated metrics for distributed training."""
    world_size: int
    node_count: int
    total_gpu_memory_used: float = 0.0
    total_gpu_memory_allocated: float = 0.0
    avg_gpu_utilization: float = 0.0
    avg_cpu_utilization: float = 0.0
    total_cpu_memory_used: float = 0.0
    network_bandwidth_total: float = 0.0
    network_bandwidth_mean: float = 0.0
    network_bandwidth_max: float = 0.0
    avg_network_latency: float = 0.0
    max_network_latency: float = 0.0
    allreduce_time: float = 0.0
    broadcast_time: float = 0.0
    gather_time: float = 0.0
    scatter_time: float = 0.0
    timestamp: float = 0.0


class DistributedMetricsCollector:
    """Collects metrics from all nodes in distributed training."""

    def __init__(
        self,
        collect_interval: float = 1.0,
        enable_network_monitoring: bool = True,
        enable_gpu_monitoring: bool = True,
        enable_cpu_monitoring: bool = True,
    ):
        """Initialize distributed metrics collector.

        Args:
            collect_interval: Interval between metric collections (seconds)
            enable_network_monitoring: Whether to monitor network performance
            enable_gpu_monitoring: Whether to monitor GPU metrics
            enable_cpu_monitoring: Whether to monitor CPU metrics
        """
        self.collect_interval = collect_interval
        self.enable_network_monitoring = enable_network_monitoring
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.enable_cpu_monitoring = enable_cpu_monitoring

        self.metrics_history: List[DistributedMetrics] = []
        self.node_metrics: Dict[str, NodeMetrics] = {}

        self.collecting = False
        self.collector_thread = None

        # Network monitoring
        self.network_monitor = None
        if self.enable_network_monitoring:
            self.network_monitor = NetworkMonitor()

        # GPU monitoring
        self.gpu_monitor = None
        if self.enable_gpu_monitoring:
            self.gpu_monitor = GPUMemoryMonitor()

    def start_collection(self):
        """Start collecting distributed metrics."""
        if self.collecting:
            return

        self.collecting = True
        self.collector_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self.collector_thread.start()

    def stop_collection(self):
        """Stop collecting distributed metrics."""
        self.collecting = False
        if self.collector_thread:
            self.collector_thread.join(timeout=5.0)

    def _collection_loop(self):
        """Main collection loop."""
        while self.collecting:
            try:
                # Collect metrics from current node
                current_node_metrics = self._collect_current_node_metrics()

                # If distributed training is active, collect from other nodes
                if DIST_AVAILABLE and dist.is_initialized():
                    distributed_metrics = self._collect_distributed_metrics()
                    self.metrics_history.append(distributed_metrics)
                else:
                    # Single node training
                    single_node_metrics = self._convert_to_distributed_metrics([current_node_metrics])
                    self.metrics_history.append(single_node_metrics)

                time.sleep(self.collect_interval)

            except Exception as e:
                print(f"Error in distributed metrics collection: {e}")
                time.sleep(self.collect_interval)

    def _collect_current_node_metrics(self) -> NodeMetrics:
        """Collect metrics from the current node."""
        node_id = socket.gethostname()
        rank = dist.get_rank() if DIST_AVAILABLE and dist.is_initialized() else 0
        local_rank = dist.get_local_rank() if DIST_AVAILABLE and dist.is_initialized() else 0

        # Safe GPU count calculation
        gpu_count = 0
        if torch.cuda.is_available():
            try:
                gpu_count = torch.cuda.device_count()
            except Exception:
                gpu_count = 0

        metrics = NodeMetrics(
            node_id=node_id,
            rank=rank,
            local_rank=local_rank,
            gpu_count=gpu_count,
            timestamp=time.time()
        )

        # Collect GPU metrics
        if self.enable_gpu_monitoring and torch.cuda.is_available():
            for device_id in range(metrics.gpu_count):
                try:
                    memory_used = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
                    memory_allocated = torch.cuda.max_memory_allocated(device_id) / 1024**3  # GB
                    utilization = 0.0  # Would need actual GPU utilization monitoring

                    metrics.gpu_memory_used.append(memory_used)
                    metrics.gpu_memory_allocated.append(memory_allocated)
                    metrics.gpu_utilization.append(utilization)

                except Exception as e:
                    print(f"Failed to collect GPU metrics for device {device_id}: {e}")
                    metrics.gpu_memory_used.append(0.0)
                    metrics.gpu_memory_allocated.append(0.0)
                    metrics.gpu_utilization.append(0.0)

        # Collect CPU metrics
        if self.enable_cpu_monitoring:
            metrics.cpu_utilization = psutil.cpu_percent()
            metrics.cpu_memory_used = psutil.virtual_memory().used / 1024**3  # GB

        # Collect network metrics
        if self.enable_network_monitoring and self.network_monitor:
            network_metrics = self.network_monitor.get_current_metrics()
            metrics.network_bandwidth = network_metrics.get('bandwidth_mbps', 0.0) # Changed to bandwidth_mbps
            metrics.network_latency = network_metrics.get('latency_ms', 0.0) # Changed to latency_ms

        return metrics

    def _collect_distributed_metrics(self) -> DistributedMetrics:
        """Collect metrics from all nodes in distributed training."""
        if not DIST_AVAILABLE or not dist.is_initialized():
            return DistributedMetrics(world_size=1, node_count=1, timestamp=time.time())

        world_size = dist.get_world_size()
        dist.get_rank()

        # Collect current node metrics
        current_metrics = self._collect_current_node_metrics()

        # Gather metrics from all nodes
        all_metrics = [None] * world_size
        dist.all_gather_object(all_metrics, current_metrics)

        # Convert to distributed metrics
        distributed_metrics = self._convert_to_distributed_metrics(all_metrics)

        return distributed_metrics

    def _convert_to_distributed_metrics(self, node_metrics_list: List[NodeMetrics]) -> DistributedMetrics:
        """Convert node metrics to distributed metrics."""
        if not node_metrics_list:
            return DistributedMetrics(world_size=1, node_count=1, timestamp=time.time())

        world_size = len(node_metrics_list)
        node_count = len({m.node_id for m in node_metrics_list})

        # Aggregate metrics
        total_gpu_memory_used = sum(sum(m.gpu_memory_used) for m in node_metrics_list)
        total_gpu_memory_allocated = sum(sum(m.gpu_memory_allocated) for m in node_metrics_list)

        all_gpu_utilizations = [util for m in node_metrics_list for util in m.gpu_utilization]
        avg_gpu_utilization = np.mean(all_gpu_utilizations) if all_gpu_utilizations else 0.0

        cpu_utilizations = [m.cpu_utilization for m in node_metrics_list]
        avg_cpu_utilization = np.mean(cpu_utilizations) if cpu_utilizations else 0.0

        total_cpu_memory_used = sum(m.cpu_memory_used for m in node_metrics_list)

        network_bandwidths = [m.network_bandwidth for m in node_metrics_list]
        network_bandwidth_total = sum(network_bandwidths)
        network_bandwidth_mean = np.mean(network_bandwidths) if network_bandwidths else 0.0
        network_bandwidth_max = np.max(network_bandwidths) if network_bandwidths else 0.0

        network_latencies = [m.network_latency for m in node_metrics_list]
        avg_network_latency = np.mean(network_latencies) if network_latencies else 0.0
        max_network_latency = np.max(network_latencies) if network_latencies else 0.0

        return DistributedMetrics(
            world_size=world_size,
            node_count=node_count,
            total_gpu_memory_used=total_gpu_memory_used,
            total_gpu_memory_allocated=total_gpu_memory_allocated,
            avg_gpu_utilization=avg_gpu_utilization,
            avg_cpu_utilization=avg_cpu_utilization,
            total_cpu_memory_used=total_cpu_memory_used,
            network_bandwidth_total=network_bandwidth_total,
            network_bandwidth_mean=network_bandwidth_mean,
            network_bandwidth_max=network_bandwidth_max,
            avg_network_latency=avg_network_latency,
            max_network_latency=max_network_latency,
            timestamp=time.time()
        )

    def get_latest_metrics(self) -> Optional[DistributedMetrics]:
        """Get the latest collected distributed metrics."""
        return self.metrics_history[-1] if self.metrics_history else None

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Get all collected metrics as a DataFrame."""
        if not self.metrics_history:
            return pd.DataFrame()

        metrics_data = []
        for metrics in self.metrics_history:
            metrics_dict = {
                'world_size': metrics.world_size,
                'node_count': metrics.node_count,
                'total_gpu_memory_used': metrics.total_gpu_memory_used,
                'total_gpu_memory_allocated': metrics.total_gpu_memory_allocated,
                'avg_gpu_utilization': metrics.avg_gpu_utilization,
                'avg_cpu_utilization': metrics.avg_cpu_utilization,
                'total_cpu_memory_used': metrics.total_cpu_memory_used,
                'network_bandwidth_total': metrics.network_bandwidth_total,
                'avg_network_latency': metrics.avg_network_latency,
                'allreduce_time': metrics.allreduce_time,
                'broadcast_time': metrics.broadcast_time,
                'gather_time': metrics.gather_time,
                'scatter_time': metrics.scatter_time,
                'timestamp': metrics.timestamp,
            }
            metrics_data.append(metrics_dict)

        return pd.DataFrame(metrics_data)


class MultiNodeMonitor:
    """Monitor for multi-node distributed training."""

    def __init__(self, master_node: str = "localhost", master_port: int = 12355):
        """Initialize multi-node monitor.

        Args:
            master_node: Hostname of the master node
            master_port: Port for communication
        """
        self.master_node = master_node
        self.master_port = master_port
        self.monitoring_active = False
        self.monitor_thread = None

        # Node information
        self.node_info = self._get_node_info()

        # Metrics storage
        self.node_metrics: Dict[str, Dict[str, Any]] = {}
        self.aggregated_metrics: List[Dict[str, Any]] = []

    def _get_node_info(self) -> Dict[str, Any]:
        """Get information about the current node."""
        # Safe GPU count calculation
        gpu_count = 0
        if torch.cuda.is_available():
            try:
                gpu_count = torch.cuda.device_count()
            except Exception:
                gpu_count = 0

        return {
            'hostname': socket.gethostname(),
            'ip_address': socket.gethostbyname(socket.gethostname()),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / 1024**3,  # GB
            'gpu_count': gpu_count,
        }

    def start_monitoring(self):
        """Start multi-node monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop multi-node monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect local metrics
                local_metrics = self._collect_local_metrics()
                self.node_metrics[self.node_info['hostname']] = local_metrics

                # If this is the master node, aggregate metrics
                if self._is_master_node():
                    self._aggregate_metrics()

                time.sleep(1.0)  # Monitor every second

            except Exception as e:
                print(f"Error in multi-node monitoring: {e}")
                time.sleep(5.0)

    def _collect_local_metrics(self) -> Dict[str, Any]:
        """Collect metrics from the local node."""
        metrics = {
            'timestamp': time.time(),
            'cpu_utilization': psutil.cpu_percent(),
            'memory_used': psutil.virtual_memory().used / 1024**3,  # GB
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
        }

        # GPU metrics
        if torch.cuda.is_available():
            gpu_metrics = []
            for device_id in range(torch.cuda.device_count()):
                try:
                    gpu_metric = {
                        'device_id': device_id,
                        'memory_used': torch.cuda.memory_allocated(device_id) / 1024**3,
                        'memory_allocated': torch.cuda.max_memory_allocated(device_id) / 1024**3,
                        'memory_reserved': torch.cuda.memory_reserved(device_id) / 1024**3,
                    }
                    gpu_metrics.append(gpu_metric)
                except Exception as e:
                    print(f"Failed to collect GPU metrics for device {device_id}: {e}")

            metrics['gpu_metrics'] = gpu_metrics

        return metrics

    def _is_master_node(self) -> bool:
        """Check if this is the master node."""
        if not DIST_AVAILABLE or not dist.is_initialized():
            return True  # Single node training

        return dist.get_rank() == 0

    def _aggregate_metrics(self):
        """Aggregate metrics from all nodes."""
        if not DIST_AVAILABLE or not dist.is_initialized():
            # Single node - just use local metrics
            aggregated = {
                'timestamp': time.time(),
                'node_count': 1,
                'total_cpu_utilization': self.node_metrics.get(self.node_info['hostname'], {}).get('cpu_utilization', 0),
                'total_memory_used': self.node_metrics.get(self.node_info['hostname'], {}).get('memory_used', 0),
                'node_metrics': self.node_metrics.copy(),
            }
            self.aggregated_metrics.append(aggregated)
            return

        # Gather metrics from all nodes
        world_size = dist.get_world_size()
        all_node_metrics = [None] * world_size
        dist.all_gather_object(all_node_metrics, self.node_metrics)

        # Aggregate
        total_cpu_utilization = 0
        total_memory_used = 0
        all_node_data = {}

        for node_metrics in all_node_metrics:
            if node_metrics:
                for hostname, metrics in node_metrics.items():
                    total_cpu_utilization += metrics.get('cpu_utilization', 0)
                    total_memory_used += metrics.get('memory_used', 0)
                    all_node_data[hostname] = metrics

        aggregated = {
            'timestamp': time.time(),
            'node_count': len(all_node_data),
            'total_cpu_utilization': total_cpu_utilization,
            'total_memory_used': total_memory_used,
            'node_metrics': all_node_data,
        }

        self.aggregated_metrics.append(aggregated)

    def get_aggregated_metrics(self) -> List[Dict[str, Any]]:
        """Get aggregated metrics from all nodes."""
        return self.aggregated_metrics.copy()

    def get_node_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics from individual nodes."""
        return self.node_metrics.copy()


class GPUMemoryMonitor:
    """Monitor GPU memory usage across all devices."""

    def __init__(self):
        """Initialize GPU memory monitor."""
        # Safe GPU count calculation
        self.gpu_count = 0
        if torch.cuda.is_available():
            try:
                self.gpu_count = torch.cuda.device_count()
            except Exception:
                self.gpu_count = 0

        self.memory_history: Dict[int, List[float]] = {i: [] for i in range(self.gpu_count)}
        self.allocation_history: Dict[int, List[float]] = {i: [] for i in range(self.gpu_count)}

    def get_current_memory_usage(self) -> Dict[int, Dict[str, float]]:
        """Get current memory usage for all GPUs."""
        if not torch.cuda.is_available():
            return {}

        memory_usage = {}
        for device_id in range(self.gpu_count):
            try:
                memory_used = torch.cuda.memory_allocated(device_id) / 1024**3  # GB
                memory_allocated = torch.cuda.max_memory_allocated(device_id) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(device_id) / 1024**3  # GB

                memory_usage[device_id] = {
                    'used': memory_used,
                    'allocated': memory_allocated,
                    'reserved': memory_reserved,
                    'free': memory_reserved - memory_used,
                }

                # Update history
                self.memory_history[device_id].append(memory_used)
                self.allocation_history[device_id].append(memory_allocated)

                # Keep only last 1000 entries
                if len(self.memory_history[device_id]) > 1000:
                    self.memory_history[device_id].pop(0)
                if len(self.allocation_history[device_id]) > 1000:
                    self.allocation_history[device_id].pop(0)

            except Exception as e:
                print(f"Failed to get memory usage for GPU {device_id}: {e}")
                memory_usage[device_id] = {
                    'used': 0.0,
                    'allocated': 0.0,
                    'reserved': 0.0,
                    'free': 0.0,
                }

        return memory_usage

    def get_memory_trends(self) -> Dict[int, Dict[str, float]]:
        """Get memory usage trends for all GPUs."""
        trends = {}

        for device_id in range(self.gpu_count):
            if len(self.memory_history[device_id]) < 2:
                trends[device_id] = {'used_trend': 0.0, 'allocated_trend': 0.0}
                continue

            # Calculate trends using linear regression
            used_trend = np.polyfit(
                range(len(self.memory_history[device_id])),
                self.memory_history[device_id],
                1
            )[0]

            allocated_trend = np.polyfit(
                range(len(self.allocation_history[device_id])),
                self.allocation_history[device_id],
                1
            )[0]

            trends[device_id] = {
                'used_trend': used_trend,
                'allocated_trend': allocated_trend,
            }

        return trends

    def get_memory_stats(self) -> Dict[int, Dict[str, float]]:
        """Get memory statistics for all GPUs."""
        stats = {}

        for device_id in range(self.gpu_count):
            if not self.memory_history[device_id]:
                stats[device_id] = {
                    'used_mean': 0.0,
                    'used_std': 0.0,
                    'used_max': 0.0,
                    'used_min': 0.0,
                    'allocated_mean': 0.0,
                    'allocated_std': 0.0,
                    'allocated_max': 0.0,
                    'allocated_min': 0.0,
                }
                continue

            used_values = self.memory_history[device_id]
            allocated_values = self.allocation_history[device_id]

            stats[device_id] = {
                'used_mean': np.mean(used_values),
                'used_std': np.std(used_values),
                'used_max': np.max(used_values),
                'used_min': np.min(used_values),
                'allocated_mean': np.mean(allocated_values),
                'allocated_std': np.std(allocated_values),
                'allocated_max': np.max(allocated_values),
                'allocated_min': np.min(allocated_values),
            }

        return stats


# Import the new network monitoring implementation
from .network_monitor import NetworkMetrics, RealNetworkMonitor


class NetworkMonitor:
    """Monitor network performance for distributed training with real measurements."""

    def __init__(self, sampling_frequency: int = None, enable_icmp: bool = True):
        """Initialize network monitor.

        Args:
            sampling_frequency: How often to sample metrics (every N steps).
                               If None, will use RLDK_NETWORK_SAMPLING_FREQUENCY env var or default to 10.
            enable_icmp: Whether to use ICMP ping (requires privileges)
        """
        if sampling_frequency is None:
            sampling_frequency = int(os.environ.get('RLDK_NETWORK_SAMPLING_FREQUENCY', '10'))

        self.sampling_frequency = sampling_frequency
        self.enable_icmp = enable_icmp

        # Network I/O counters for bandwidth measurement
        self.last_net_io = None
        self.last_net_time = None

        # Latency measurement targets
        self.latency_targets = [
            '8.8.8.8',      # Google DNS
            '1.1.1.1',      # Cloudflare DNS
            '208.67.222.222', # OpenDNS
        ]

        # TCP fallback ports for latency measurement
        self.tcp_ports = [80, 443, 22]  # HTTP, HTTPS, SSH

        # Thread safety
        self._lock = threading.Lock()

        # Error tracking
        self.last_errors = {
            'bandwidth': None,
            'latency': None
        }

        # Initialize real monitor for comprehensive metrics
        self.real_monitor = RealNetworkMonitor(
            enable_distributed_monitoring=True,
            enable_distributed_measurements=False  # Safer default - doesn't interfere with training
        )

    def _measure_bandwidth(self) -> Tuple[float, float]:
        """Measure real bandwidth using psutil.net_io_counters.

        Returns:
            Tuple of (upload_mbps, download_mbps)
        """
        try:
            with self._lock:
                current_net_io = psutil.net_io_counters()
                current_time = time.perf_counter()

                if self.last_net_io is None:
                    # First measurement
                    self.last_net_io = current_net_io
                    self.last_net_time = current_time
                    self.last_errors['bandwidth'] = None
                    return 0.0, 0.0

                # Calculate time delta
                time_delta = current_time - self.last_net_time
                if time_delta <= 0:
                    return 0.0, 0.0

                # Calculate byte deltas
                bytes_sent_delta = current_net_io.bytes_sent - self.last_net_io.bytes_sent
                bytes_recv_delta = current_net_io.bytes_recv - self.last_net_io.bytes_recv

                # Calculate bandwidth in Mbps (bytes * 8 bits / 1,000,000)
                upload_mbps = (bytes_sent_delta * 8) / (time_delta * 1_000_000)
                download_mbps = (bytes_recv_delta * 8) / (time_delta * 1_000_000)

                # Update last values
                self.last_net_io = current_net_io
                self.last_net_time = current_time

                self.last_errors['bandwidth'] = None
                return upload_mbps, download_mbps

        except psutil.Error as e:
            self.last_errors['bandwidth'] = f"psutil error: {e}"
            return 0.0, 0.0
        except OSError as e:
            self.last_errors['bandwidth'] = f"OS error: {e}"
            return 0.0, 0.0
        except Exception as e:
            self.last_errors['bandwidth'] = f"Unexpected error: {e}"
            return 0.0, 0.0

    def _measure_latency(self) -> float:
        """Measure real latency using ICMP ping or TCP handshake.

        Returns:
            Average latency in milliseconds
        """
        latencies = []

        for target in self.latency_targets:
            try:
                if self.enable_icmp:
                    # Try ICMP ping first
                    latency = self._icmp_ping(target)
                    if latency is not None and latency != float('inf'):
                        latencies.append(latency)
                        continue

                # Fallback to TCP handshake
                latency = self._tcp_handshake(target)
                if latency is not None and latency != float('inf'):
                    latencies.append(latency)

            except Exception as e:
                # Log error but continue with other targets
                print(f"Error measuring latency to {target}: {e}")
                continue

        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            self.last_errors['latency'] = None
            return avg_latency
        else:
            self.last_errors['latency'] = "All latency measurements failed"
            return float('inf')

    def _icmp_ping(self, target: str) -> Optional[float]:
        """Perform ICMP ping to measure latency.

        Args:
            target: Target host to ping

        Returns:
            Latency in milliseconds or None if failed
        """
        try:
            # Use system ping command
            if platform.system().lower() == 'windows':
                cmd = ['ping', '-n', '1', '-w', '1000', target]
            else:
                cmd = ['ping', '-c', '1', '-W', '1', target]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=2.0
            )

            if result.returncode == 0:
                # Parse ping output for latency
                output = result.stdout
                if 'time=' in output:
                    # Extract time value
                    time_part = output.split('time=')[1].split()[0]
                    latency = float(time_part)
                    return latency
                elif 'time<' in output:
                    # Very fast response
                    return 0.1

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
            pass
        except Exception as e:
            print(f"ICMP ping error for {target}: {e}")

        return None

    def _tcp_handshake(self, target: str) -> Optional[float]:
        """Perform TCP handshake to measure latency.

        Args:
            target: Target host to connect to

        Returns:
            Latency in milliseconds or None if failed
        """
        for port in self.tcp_ports:
            try:
                start_time = time.perf_counter()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                result = sock.connect_ex((target, port))
                end_time = time.perf_counter()
                sock.close()

                if result == 0:
                    latency = (end_time - start_time) * 1000  # Convert to milliseconds
                    return latency

            except (socket.timeout, OSError):
                continue
            except Exception as e:
                print(f"TCP handshake error for {target}:{port}: {e}")
                continue

        return None

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current network metrics with real measurements.

        Returns:
            Dictionary with bandwidth_mbps and latency_ms
        """
        upload_mbps, download_mbps = self._measure_bandwidth()
        latency_ms = self._measure_latency()

        return {
            'bandwidth_mbps': download_mbps,  # Use download as primary bandwidth
            'bandwidth_upload_mbps': upload_mbps,
            'bandwidth_download_mbps': download_mbps,
            'latency_ms': latency_ms if latency_ms != float('inf') else 0.0,
            'total_bandwidth_mbps': upload_mbps + download_mbps,
            'timestamp': time.time(),
        }

    def get_network_stats(self) -> Dict[str, float]:
        """Get network performance statistics."""
        return self.real_monitor.get_network_stats()

    def get_comprehensive_metrics(self) -> NetworkMetrics:
        """Get comprehensive network metrics."""
        return self.real_monitor.get_comprehensive_metrics()

    def run_network_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive network diagnostics."""
        return self.real_monitor.run_network_diagnostics()

    def get_network_health_report(self) -> Dict[str, Any]:
        """Get comprehensive network health report."""
        return self.real_monitor.get_network_health_report()

    def get_error_status(self) -> Dict[str, Optional[str]]:
        """Get status of any measurement errors."""
        return self.last_errors.copy()

    def reset_counters(self):
        """Reset network counters for fresh measurement."""
        with self._lock:
            self.last_net_io = None
            self.last_net_time = None

    def test_network_connectivity(self) -> Dict[str, Any]:
        """Test network connectivity to common hosts."""
        diagnostics = self.real_monitor.network_diagnostics
        return {
            'ping_tests': diagnostics._run_ping_diagnostics(),
            'dns_tests': diagnostics._run_dns_diagnostics(),
            'connectivity_tests': diagnostics._run_connectivity_diagnostics(),
        }

    def test_bandwidth(self) -> Dict[str, Any]:
        """Test network bandwidth using multiple methods."""
        diagnostics = self.real_monitor.network_diagnostics
        return diagnostics._run_bandwidth_diagnostics()

    def test_distributed_network(self) -> Dict[str, Any]:
        """Test distributed training network performance."""
        if DIST_AVAILABLE and dist.is_initialized():
            diagnostics = self.real_monitor.network_diagnostics
            return diagnostics._run_distributed_diagnostics()
        else:
            return {'error': 'PyTorch distributed not available or initialized'}
