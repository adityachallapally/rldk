# Network Monitoring for OpenRLHF Distributed Training

This document describes the comprehensive network monitoring functionality implemented for OpenRLHF distributed training, including real bandwidth and latency measurements, diagnostics, and dashboard integration.

## Overview

The network monitoring system provides accurate, real-time measurements of network performance during distributed training, replacing the previous DNS-based placeholder methods with actual bandwidth and latency measurements.

## Key Features

- **Real Bandwidth Measurement**: Uses `psutil.net_io_counters` to measure actual bytes sent/received
- **Accurate Latency Measurement**: ICMP ping with TCP handshake fallback
- **Comprehensive Diagnostics**: Ping, DNS, connectivity, and bandwidth tests
- **Distributed Training Support**: AllReduce, Broadcast, Gather, and Scatter bandwidth measurements
- **Real-time Dashboard**: Live monitoring with alerts and thresholds
- **Thread-safe**: Safe for concurrent access during training
- **Error Handling**: Graceful fallbacks and detailed error reporting
- **Sampling Frequency Control**: Configurable network metrics collection frequency
- **Environment Variable Support**: `RLDK_NETWORK_SAMPLING_FREQUENCY` for easy configuration
- **JSONL Event Logging**: Network metrics included in structured event logs
- **Per-node Aggregation**: Mean and max statistics across distributed nodes

## Architecture

### Core Components

1. **NetworkMonitor** (`distributed.py`): Main monitoring class with real measurements
2. **RealNetworkMonitor** (`network_monitor.py`): Comprehensive monitoring with diagnostics
3. **NetworkDiagnostics** (`network_monitor.py`): Comprehensive network testing
4. **OpenRLHFDashboard** (`dashboard.py`): Real-time visualization and alerts

### Measurement Methods

#### Bandwidth Measurement

```python
def _measure_bandwidth(self) -> Tuple[float, float]:
    """Measure real bandwidth using psutil.net_io_counters."""
    current_net_io = psutil.net_io_counters()
    current_time = time.perf_counter()
    
    # Calculate byte deltas
    bytes_sent_delta = current_net_io.bytes_sent - self.last_net_io.bytes_sent
    bytes_recv_delta = current_net_io.bytes_recv - self.last_net_io.bytes_recv
    
    # Calculate bandwidth in Mbps
    upload_mbps = (bytes_sent_delta * 8) / (time_delta * 1_000_000)
    download_mbps = (bytes_recv_delta * 8) / (time_delta * 1_000_000)
    
    return upload_mbps, download_mbps
```

#### Latency Measurement

```python
def _measure_latency(self) -> float:
    """Measure real latency using ICMP ping or TCP handshake."""
    for target in self.latency_targets:
        if self.enable_icmp:
            latency = self._icmp_ping(target)
            if latency is not None:
                return latency
        
        # Fallback to TCP handshake
        latency = self._tcp_handshake(target)
        if latency is not None:
            return latency
    
    return float('inf')
```

## Metrics Structure

### NetworkMetrics Dataclass

```python
@dataclass
class NetworkMetrics:
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
    
    # Network diagnostics
    dns_resolution_ms: float = 0.0
    tcp_connectivity_ms: float = 0.0
    udp_connectivity_ms: float = 0.0
    network_path_hops: int = 0
    network_path_latency: float = 0.0
    
    # Timestamp
    timestamp: float = 0.0
```

### OpenRLHFMetrics Integration

The `OpenRLHFMetrics` class has been updated to include network metrics:

```python
# Network metrics (for multi-node)
bandwidth_mbps: float = 0.0
latency_ms: float = 0.0
bandwidth_upload_mbps: float = 0.0
bandwidth_download_mbps: float = 0.0
total_bandwidth_mbps: float = 0.0
allreduce_time: float = 0.0
allreduce_bandwidth: float = 0.0
broadcast_bandwidth: float = 0.0
gather_bandwidth: float = 0.0
scatter_bandwidth: float = 0.0
packet_loss_percent: float = 0.0
network_errors: int = 0
dns_resolution_ms: float = 0.0
```

## Usage Examples

### Basic Network Monitoring

```python
from src.rldk.integrations.openrlhf.distributed import NetworkMonitor

# Initialize monitor
monitor = NetworkMonitor(sampling_frequency=10, enable_icmp=True)

# Get current metrics
metrics = monitor.get_current_metrics()
print(f"Bandwidth: {metrics['bandwidth_mbps']:.2f} Mbps")
print(f"Latency: {metrics['latency_ms']:.2f} ms")
```

### Comprehensive Diagnostics

```python
from src.rldk.integrations.openrlhf.network_monitor import NetworkDiagnostics

# Run comprehensive diagnostics
diagnostics = NetworkDiagnostics()
results = diagnostics.run_comprehensive_diagnostics()

# Check ping results
for host, result in results['ping_tests'].items():
    if result['success']:
        print(f"{host}: {result['latency']:.2f}ms")
```

### Real-time Monitoring

```python
from src.rldk.integrations.openrlhf.network_monitor import RealNetworkMonitor

# Initialize comprehensive monitor
monitor = RealNetworkMonitor(
    enable_distributed_monitoring=True,
    enable_distributed_measurements=False  # Safer default
)

# Get comprehensive metrics
metrics = monitor.get_comprehensive_metrics()
print(f"Bandwidth: {metrics.bandwidth_mbps:.2f} Mbps")
print(f"Latency: {metrics.latency_ms:.2f} ms")
print(f"Packet Loss: {metrics.packet_loss_percent:.2f}%")
```

### Dashboard Integration

```python
from src.rldk.integrations.openrlhf.dashboard import OpenRLHFDashboard

# Initialize dashboard
dashboard = OpenRLHFDashboard(output_dir="./logs", port=5000)

# Initialize network monitoring
dashboard.initialize_network_monitoring()

# Add metrics and check thresholds
metrics = monitor.get_current_metrics()
dashboard.add_metrics(metrics)
dashboard.check_network_thresholds(metrics)

# Start dashboard
dashboard.start()
```

## Configuration Options

### NetworkMonitor Configuration

```python
monitor = NetworkMonitor(
    sampling_frequency=10,  # Sample every 10 steps
    enable_icmp=True,        # Use ICMP ping (requires privileges)
)
```

### RealNetworkMonitor Configuration

```python
monitor = RealNetworkMonitor(
    enable_distributed_monitoring=True,
    enable_distributed_measurements=False,  # Safer - doesn't interfere with training
)
```

### Dashboard Configuration

```python
dashboard = OpenRLHFDashboard(
    output_dir="./logs",
    port=5000,
    host="localhost",
    enable_auto_refresh=True,
    refresh_interval=1.0,
)
```

## Thresholds and Alerts

### Default Thresholds

```python
thresholds = {
    'latency_ms': 100.0,        # High latency threshold
    'bandwidth_mbps': 10.0,     # Low bandwidth threshold
    'packet_loss_percent': 5.0, # High packet loss threshold
}
```

### Alert Types

- **High Latency**: When latency exceeds 100ms
- **Low Bandwidth**: When bandwidth drops below 10 Mbps
- **High Packet Loss**: When packet loss exceeds 5%

### Custom Alerts

```python
dashboard.add_alert(
    alert_type='network',
    message='Custom network issue detected',
    severity='warning',
    threshold=50.0,
    current_value=75.0
)
```

## Troubleshooting

### Common Issues

#### 1. ICMP Ping Fails

**Symptoms**: Latency measurements return `float('inf')` or very high values

**Causes**:
- Insufficient privileges (ICMP requires root/admin)
- Firewall blocking ICMP packets
- Network configuration issues

**Solutions**:
- Run with elevated privileges
- Configure firewall to allow ICMP
- Use TCP fallback (automatic)
- Set `enable_icmp=False`

#### 2. Bandwidth Measurements Return Zero

**Symptoms**: All bandwidth measurements are 0.0 Mbps

**Causes**:
- `psutil` not available or not working
- Network interface not found
- Insufficient privileges

**Solutions**:
- Install `psutil`: `pip install psutil`
- Check network interface names
- Run with appropriate privileges
- Check error logs for specific issues

#### 3. High CPU Usage

**Symptoms**: Monitoring causes high CPU usage

**Causes**:
- Too frequent sampling
- Inefficient measurement methods
- Large number of targets

**Solutions**:
- Increase `sampling_frequency`
- Reduce number of latency targets
- Use less frequent comprehensive diagnostics

#### 4. Dashboard Not Updating

**Symptoms**: Dashboard shows stale data

**Causes**:
- File permissions issues
- Dashboard not properly initialized
- Network monitor not running

**Solutions**:
- Check file permissions on output directory
- Ensure dashboard is properly initialized
- Verify network monitor is running
- Check for error messages in logs

### Error Handling

The system provides comprehensive error handling:

```python
# Check for errors
errors = monitor.get_error_status()
if errors['bandwidth']:
    print(f"Bandwidth error: {errors['bandwidth']}")
if errors['latency']:
    print(f"Latency error: {errors['latency']}")
```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor will now log detailed information
monitor = NetworkMonitor()
```

## Performance Considerations

### Sampling Frequency

- **High frequency** (every step): Accurate but high overhead
- **Medium frequency** (every 10 steps): Good balance
- **Low frequency** (every 100 steps): Low overhead, less accurate

### Memory Usage

- Metrics history is limited to 1000 entries by default
- Network diagnostics results are not cached
- Dashboard data is kept in memory for real-time updates

### CPU Usage

- Bandwidth measurement: ~0.1ms per measurement
- Latency measurement: ~1-10ms per target (depending on network)
- Comprehensive diagnostics: ~1-5 seconds (run sparingly)

## Best Practices

### For Production Use

1. **Use appropriate sampling frequency**: Every 10-50 steps for most use cases
2. **Enable ICMP when possible**: More accurate than TCP fallback
3. **Monitor error status**: Check for measurement failures
4. **Set up alerts**: Configure appropriate thresholds
5. **Use dashboard sparingly**: Can impact performance if overused

### For Development

1. **Test with demo script**: Use `examples/openrlhf_network_demo.py`
2. **Run diagnostics first**: Use `NetworkDiagnostics` to verify network health
3. **Start with basic monitoring**: Use `NetworkMonitor` before comprehensive monitoring
4. **Check logs**: Monitor for errors and warnings

### For Multi-node Setups

1. **Consistent node identification**: Ensure node IDs are consistent across metrics
2. **Aggregate metrics**: Use `DistributedMetricsCollector` for multi-node aggregation
3. **Monitor all nodes**: Collect metrics from all nodes in the cluster
4. **Set appropriate thresholds**: Consider network topology when setting thresholds

## Testing

### Unit Tests

Run the comprehensive test suite:

```bash
python -m pytest tests/test_network_monitor.py -v
```

### Integration Tests

Test with the demo script:

```bash
python examples/openrlhf_network_demo.py --mode all
```

### Manual Testing

1. **Basic functionality**:
   ```bash
   python examples/openrlhf_network_demo.py --mode monitoring --duration 30
   ```

2. **Dashboard testing**:
   ```bash
   python examples/openrlhf_network_demo.py --mode dashboard
   ```

3. **Diagnostics testing**:
   ```bash
   python examples/openrlhf_network_demo.py --mode diagnostics
   ```

## API Reference

### NetworkMonitor Methods

- `get_current_metrics()`: Get current bandwidth and latency
- `get_error_status()`: Get status of any measurement errors
- `reset_counters()`: Reset network counters for fresh measurement
- `_measure_bandwidth()`: Internal bandwidth measurement
- `_measure_latency()`: Internal latency measurement

### RealNetworkMonitor Methods

- `get_comprehensive_metrics()`: Get full NetworkMetrics object
- `run_network_diagnostics()`: Run comprehensive diagnostics
- `get_network_health_report()`: Get health report with recommendations

### Dashboard Methods

- `update_data()`: Reload metrics from disk
- `add_metrics()`: Add new metrics and trigger refresh
- `add_alert()`: Record alerts for UI display
- `check_network_thresholds()`: Check metrics against thresholds

## Migration Guide

### From Old DNS-based Monitoring

1. **Update imports**:
   ```python
   # Old
   from .network_monitor import NetworkMonitor
   
   # New
   from .distributed import NetworkMonitor
   ```

2. **Update metric names**:
   ```python
   # Old
   metrics.network_bandwidth
   metrics.network_latency
   
   # New
   metrics.bandwidth_mbps
   metrics.latency_ms
   ```

3. **Update dashboard integration**:
   ```python
   # Old
   dashboard.add_metrics(metrics)
   
   # New
   dashboard.add_metrics(metrics.to_dict())
   dashboard.check_network_thresholds(metrics.to_dict())
   ```

## Latest Features (v0.1.0)

### Sampling Frequency Control

The network monitoring system now supports configurable sampling frequency to reduce overhead during training:

```python
# Constructor parameter
monitor = NetworkMonitor(sampling_frequency=10)  # Sample every 10 steps

# Environment variable
export RLDK_NETWORK_SAMPLING_FREQUENCY=5  # Sample every 5 steps

# OpenRLHF callback
callback = OpenRLHFCallback(
    network_sampling_frequency=15,  # Sample every 15 steps
    enable_distributed_monitoring=True
)
```

### Enhanced Metrics Format

The `get_current_metrics()` method now returns a comprehensive set of network metrics:

```python
{
    'bandwidth_mbps': 100.0,           # Primary download bandwidth
    'bandwidth_upload_mbps': 50.0,     # Upload bandwidth
    'bandwidth_download_mbps': 100.0,  # Download bandwidth
    'latency_ms': 5.0,                 # Average latency
    'total_bandwidth_mbps': 150.0,     # Total bandwidth
    'timestamp': 1234567890.0          # Measurement timestamp
}
```

### Distributed Metrics Aggregation

The `DistributedMetricsCollector` now provides per-node statistics:

```python
@dataclass
class DistributedMetrics:
    network_bandwidth_total: float = 0.0    # Sum across all nodes
    network_bandwidth_mean: float = 0.0     # Average across all nodes
    network_bandwidth_max: float = 0.0      # Maximum across all nodes
    avg_network_latency: float = 0.0        # Average latency
    max_network_latency: float = 0.0        # Maximum latency
```

### JSONL Event Logging

Network metrics are now automatically included in JSONL event logs:

```json
{
  "step": 100,
  "timestamp": 1234567890.0,
  "network_bandwidth": 100.0,
  "network_latency": 5.0,
  "bandwidth_mbps": 100.0,
  "latency_ms": 5.0,
  "bandwidth_upload_mbps": 50.0,
  "bandwidth_download_mbps": 100.0,
  "total_bandwidth_mbps": 150.0,
  "allreduce_bandwidth": 25.0,
  "broadcast_bandwidth": 30.0,
  "gather_bandwidth": 20.0,
  "scatter_bandwidth": 15.0,
  "packet_loss_percent": 0.1,
  "network_errors": 0,
  "dns_resolution_ms": 2.0
}
```

### Dependencies

The network monitoring system requires the following packages:

```bash
pip install psutil iperf3
```

- **psutil**: For network I/O counters and system metrics
- **iperf3**: For bandwidth testing (optional, falls back to other methods)

### Error Handling

The system includes comprehensive error handling:

```python
# Individual error handling for different failure modes
try:
    # psutil operations
    current_net_io = psutil.net_io_counters()
except psutil.Error as e:
    self.last_errors['bandwidth'] = f"psutil error: {e}"
    return 0.0, 0.0
except OSError as e:
    self.last_errors['bandwidth'] = f"OS error: {e}"
    return 0.0, 0.0
```

## Future Enhancements

- **GPU Direct RDMA monitoring**: Monitor GPU-to-GPU communication
- **Network topology mapping**: Automatic discovery of network topology
- **Predictive analytics**: Predict network issues before they occur
- **Integration with monitoring systems**: Prometheus, Grafana, etc.
- **Custom measurement protocols**: Support for custom network protocols