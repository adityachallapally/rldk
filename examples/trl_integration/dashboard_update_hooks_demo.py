#!/usr/bin/env python3
"""
Demo of TRL Dashboard Update Hooks

This example demonstrates how to use the newly implemented dashboard update hooks:
- update_data(): Refresh dashboard data from files
- add_metrics(): Add new metrics to the dashboard in real-time
- add_alert(): Add new alerts to the dashboard in real-time

The dashboard can now be updated programmatically during training.
"""

import time
import threading
from pathlib import Path
from typing import Dict, Any

from rldk.utils.math_utils import safe_divide

# Import TRL components
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import PPOConfig, PPOTrainer
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    print("TRL not available. Install with: pip install trl")

# Import RLDK components
from rldk.integrations.trl.callbacks import RLDKCallback, RLDKMetrics
from rldk.integrations.trl.dashboard import RLDKDashboard


def create_sample_metrics(step: int) -> RLDKMetrics:
    """Create sample metrics for demonstration."""
    return RLDKMetrics(
        step=step,
        epoch=safe_divide(step, 100.0, 0.0),
        learning_rate=0.0001,
        loss=0.5 + (step % 10) * 0.1,
        grad_norm=1.0 + (step % 5) * 0.2,
        reward_mean=0.7 + (step % 20) * 0.05,
        reward_std=0.1,
        kl_mean=0.02 + (step % 15) * 0.01,
        kl_std=0.005,
        entropy_mean=0.8,
        clip_frac=0.1,
        value_loss=0.3,
        policy_loss=0.4,
        gpu_memory_used=8.5,
        gpu_memory_allocated=9.0,
        cpu_memory_used=4.2,
        step_time=0.1,
        wall_time=step * 0.1,
        tokens_in=512,
        tokens_out=128,
        training_stability_score=0.9,
        convergence_indicator=0.1,
        phase="train",
        run_id="demo_run",
        git_sha="abc123",
        seed=42
    )


def create_sample_alert(step: int, alert_type: str = "high_loss") -> Dict[str, Any]:
    """Create sample alert for demonstration."""
    return {
        "type": alert_type,
        "message": f"Training loss is high at step {step}",
        "step": step,
        "timestamp": time.time(),
        "severity": "warning"
    }


def demo_manual_updates():
    """Demonstrate manual dashboard updates."""
    print("🚀 Starting Manual Dashboard Updates Demo")
    
    # Initialize dashboard
    output_dir = Path("./demo_dashboard")
    output_dir.mkdir(exist_ok=True)
    
    dashboard = RLDKDashboard(
        output_dir=output_dir,
        port=8501,
        auto_refresh=True,
        refresh_interval=3,
        run_id="demo_run"
    )
    
    # Start dashboard in background
    dashboard.start_dashboard(blocking=False)
    
    print("📊 Dashboard started. You can view it at http://localhost:8501")
    print("🔄 Auto-refresh is enabled with 3-second intervals")
    
    # Simulate training steps with manual updates
    for step in range(1, 21):
        print(f"\n📈 Step {step}: Adding metrics and alerts...")
        
        # Add metrics
        metrics = create_sample_metrics(step)
        dashboard.add_metrics(metrics)
        
        # Add alerts occasionally
        if step % 5 == 0:
            alert = create_sample_alert(step, "checkpoint_alert")
            dashboard.add_alert(alert)
            print(f"⚠️  Added alert at step {step}")
        
        # Simulate training time
        time.sleep(2)
    
    print("\n✅ Manual updates demo completed!")
    print("📊 Dashboard will continue to auto-refresh. Press Ctrl+C to stop.")


def demo_callback_integration():
    """Demonstrate dashboard integration with TRL callback."""
    if not TRL_AVAILABLE:
        print("❌ TRL not available. Skipping callback integration demo.")
        return
    
    print("\n🚀 Starting Callback Integration Demo")
    
    # Initialize dashboard
    output_dir = Path("./demo_callback_dashboard")
    output_dir.mkdir(exist_ok=True)
    
    dashboard = RLDKDashboard(
        output_dir=output_dir,
        port=8502,
        auto_refresh=True,
        refresh_interval=2,
        run_id="callback_demo"
    )
    
    # Initialize callback
    callback = RLDKCallback(
        output_dir=output_dir,
        log_interval=2,
        run_id="callback_demo",
        enable_resource_monitoring=True
    )
    
    # Connect dashboard to callback
    dashboard.connect_callback(callback)
    
    # Start dashboard
    dashboard.start_dashboard(blocking=False)
    
    print("📊 Dashboard started. You can view it at http://localhost:8502")
    print("🔗 Dashboard is connected to callback for real-time updates")
    
    # Simulate callback events
    for step in range(1, 11):
        print(f"\n📈 Simulating callback step {step}...")
        
        # Create metrics
        metrics = create_sample_metrics(step)
        callback.current_metrics = metrics
        
        # Trigger callback methods (which will also update dashboard)
        callback._log_detailed_metrics()
        
        # Add alerts occasionally
        if step % 3 == 0:
            callback._add_alert("performance_alert", f"Performance issue at step {step}")
        
        time.sleep(3)
    
    print("\n✅ Callback integration demo completed!")


def demo_programmatic_refresh():
    """Demonstrate programmatic data refresh."""
    print("\n🚀 Starting Programmatic Refresh Demo")
    
    # Initialize dashboard
    output_dir = Path("./demo_refresh_dashboard")
    output_dir.mkdir(exist_ok=True)
    
    dashboard = RLDKDashboard(
        output_dir=output_dir,
        port=8503,
        auto_refresh=False,  # Disable auto-refresh for manual control
        run_id="refresh_demo"
    )
    
    # Start dashboard
    dashboard.start_dashboard(blocking=False)
    
    print("📊 Dashboard started. You can view it at http://localhost:8503")
    print("🔄 Auto-refresh is disabled. Using manual refresh.")
    
    # Simulate data updates with manual refresh
    for step in range(1, 16):
        print(f"\n📈 Step {step}: Adding data and refreshing...")
        
        # Add metrics
        metrics = create_sample_metrics(step)
        dashboard.add_metrics(metrics)
        
        # Add alerts occasionally
        if step % 4 == 0:
            alert = create_sample_alert(step, "refresh_alert")
            dashboard.add_alert(alert)
        
        # Manually refresh dashboard data
        dashboard.update_data()
        print(f"🔄 Dashboard refreshed with {len(dashboard.metrics_data)} metrics, {len(dashboard.alerts_data)} alerts")
        
        time.sleep(2)
    
    print("\n✅ Programmatic refresh demo completed!")


def main():
    """Run all dashboard update hooks demos."""
    print("🎯 TRL Dashboard Update Hooks Demo")
    print("=" * 50)
    
    try:
        # Demo 1: Manual updates
        demo_manual_updates()
        
        # Wait between demos
        time.sleep(5)
        
        # Demo 2: Callback integration
        demo_callback_integration()
        
        # Wait between demos
        time.sleep(5)
        
        # Demo 3: Programmatic refresh
        demo_programmatic_refresh()
        
        print("\n🎉 All demos completed!")
        print("\n📋 Summary of implemented features:")
        print("✅ update_data() - Refresh dashboard data from files")
        print("✅ add_metrics() - Add new metrics to dashboard in real-time")
        print("✅ add_alert() - Add new alerts to dashboard in real-time")
        print("✅ connect_callback() - Connect dashboard to TRL callback")
        print("✅ enable_auto_refresh() - Enable automatic data refresh")
        print("✅ Manual programmatic control - Update dashboard from code")
        
        # Keep running for a while to allow viewing
        print("\n⏰ Keeping demos running for 30 seconds...")
        time.sleep(30)
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    finally:
        print("\n🧹 Cleaning up...")
        # Note: In a real application, you'd want to properly stop the dashboards


if __name__ == "__main__":
    main()