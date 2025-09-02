"""Distributed training monitoring for OpenRLHF."""

import time
import threading
import queue
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import socket
import subprocess
import psutil

import torch
import numpy as np
import pandas as pd

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
    avg_network_latency: float = 0.0
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
        
        metrics = NodeMetrics(
            node_id=node_id,
            rank=rank,
            local_rank=local_rank,
            gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
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
            metrics.network_bandwidth = network_metrics.get('bandwidth', 0.0)
            metrics.network_latency = network_metrics.get('latency', 0.0)
        
        return metrics
    
    def _collect_distributed_metrics(self) -> DistributedMetrics:
        """Collect metrics from all nodes in distributed training."""
        if not DIST_AVAILABLE or not dist.is_initialized():
            return DistributedMetrics(world_size=1, node_count=1, timestamp=time.time())
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
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
        node_count = len(set(m.node_id for m in node_metrics_list))
        
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
        
        network_latencies = [m.network_latency for m in node_metrics_list]
        avg_network_latency = np.mean(network_latencies) if network_latencies else 0.0
        
        return DistributedMetrics(
            world_size=world_size,
            node_count=node_count,
            total_gpu_memory_used=total_gpu_memory_used,
            total_gpu_memory_allocated=total_gpu_memory_allocated,
            avg_gpu_utilization=avg_gpu_utilization,
            avg_cpu_utilization=avg_cpu_utilization,
            total_cpu_memory_used=total_cpu_memory_used,
            network_bandwidth_total=network_bandwidth_total,
            avg_network_latency=avg_network_latency,
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
        return {
            'hostname': socket.gethostname(),
            'ip_address': socket.gethostbyname(socket.gethostname()),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / 1024**3,  # GB
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
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
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
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


class NetworkMonitor:
    """Monitor network performance for distributed training."""
    
    def __init__(self):
        """Initialize network monitor."""
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
        
        # Measure bandwidth (simplified - would need actual network testing)
        bandwidth = self._measure_bandwidth()
        latency = self._measure_latency()
        
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
    
    def _measure_bandwidth(self) -> float:
        """Measure network bandwidth (simplified implementation)."""
        # This is a placeholder - actual implementation would need
        # to perform network tests or use system monitoring tools
        try:
            # Use a simple approach - measure time to resolve a hostname
            start_time = time.time()
            socket.gethostbyname('google.com')
            end_time = time.time()
            
            # Convert to approximate bandwidth (very rough estimate)
            response_time = end_time - start_time
            if response_time > 0:
                # This is not a real bandwidth measurement
                # In practice, you'd use tools like iperf or netperf
                return 1.0 / response_time  # Placeholder
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _measure_latency(self) -> float:
        """Measure network latency (simplified implementation)."""
        try:
            # Simple ping-like measurement
            start_time = time.time()
            socket.gethostbyname('google.com')
            end_time = time.time()
            
            return (end_time - start_time) * 1000  # Convert to milliseconds
        except Exception:
            return 0.0
    
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