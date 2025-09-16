# Monitor Actions Examples

This directory contains examples demonstrating the new gating actions and robust outputs in RLDK monitoring.

## Files

### `minimal_streaming_loop.py`
A simple training loop that emits metrics to JSONL format. This simulates a real training process that can be monitored and controlled by RLDK.

**Usage:**
```bash
python examples/minimal_streaming_loop.py
```

The script will:
- Print its PID for monitoring
- Emit metrics (kl, reward, grad_norm) to `artifacts/run.jsonl`
- Run until stopped manually (Ctrl+C) or by monitoring rules

### `rules.yaml`
Example monitoring rules demonstrating all action types:

- **stop_on_high_kl**: Stops the process when KL divergence exceeds 0.35 for 5 consecutive steps
- **sentinel_on_low_reward**: Creates a sentinel file when reward drops below -0.5
- **shell_on_high_grad_norm**: Executes a shell command when gradient norm exceeds 3.0
- **http_on_extreme_values**: Makes HTTP requests when extreme values are detected

## Quick Start

1. **Start the training loop:**
   ```bash
   python examples/minimal_streaming_loop.py &
   ```
   Note the PID that's printed.

2. **Monitor with auto-stop:**
   ```bash
   rldk monitor --stream artifacts/run.jsonl --rules examples/rules.yaml --pid <PID>
   ```

3. **Monitor with full outputs:**
   ```bash
   rldk monitor \
     --stream artifacts/run.jsonl \
     --rules examples/rules.yaml \
     --pid <PID> \
     --alerts artifacts/alerts.jsonl \
     --summary artifacts/summary.txt
   ```

## Expected Behavior

- The training loop will run and emit metrics
- When KL divergence exceeds 0.35 for 5 consecutive steps, the monitoring will:
  - Print a warning message
  - Send SIGTERM to the training process
  - If the process doesn't stop within 5 seconds, send SIGKILL
- Other rules will trigger based on their conditions
- All alerts will be logged to `artifacts/alerts.jsonl`
- A human-readable summary will be written to `artifacts/summary.txt`

## Testing

Run the comprehensive test suite:
```bash
python test_monitor_actions.py
```

This will test:
- Auto-stop by PID
- Sentinel file creation
- Failed action logging
- Rolling windows
- Human summary generation

## Integration with Real Training

To integrate with your own training code:

1. **Use EventWriter:**
   ```python
   from rldk.emit import EventWriter
   
   writer = EventWriter("artifacts/run.jsonl")
   writer.log(step=step, name="kl", value=kl_value)
   writer.log(step=step, name="reward", value=reward_value)
   ```

2. **Or write JSONL directly:**
   ```python
   import json
   with open("artifacts/run.jsonl", "a") as f:
       f.write(json.dumps({
           "time": "2025-01-16T12:00:00Z",
           "step": step,
           "name": "kl",
           "value": kl_value
       }) + "\n")
   ```

3. **Run monitoring:**
   ```bash
   rldk monitor --stream artifacts/run.jsonl --rules your_rules.yaml --pid <your_pid>
   ```

The monitoring system is framework-agnostic and works with any training code that can emit JSONL metrics.