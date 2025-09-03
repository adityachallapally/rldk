#!/usr/bin/env python3
"""
Ultimate Post-Training Demo - Showcase of RLDK's Advanced Features

This demo demonstrates the complete RLDK toolkit for post-training:
- Universal monitoring of any framework
- Intelligent anomaly detection
- One-click training debugging
- Health scoring and recommendations
- Real-time alerts and dashboards
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Import RLDK ultimate features
from rldk import (
    UniversalMonitor, start_monitoring,
    AnomalyDetector, detect_anomalies, detect_training_anomalies,
    TrainingDebugger, debug_training, quick_debug
)

def create_demo_training_logs():
    """Create realistic training logs for demonstration"""
    print("📝 Creating demo training logs...")
    
    # Create demo directory
    demo_dir = Path("demo_training_run")
    demo_dir.mkdir(exist_ok=True)
    
    # Generate realistic training data with some anomalies
    steps = list(range(1000))
    base_loss = 2.0
    base_reward = 0.5
    
    # Create normal training progression
    losses = [base_loss * np.exp(-step/500) + np.random.normal(0, 0.1) for step in steps]
    rewards = [base_reward + step/1000 + np.random.normal(0, 0.05) for step in steps]
    kl_divs = [0.01 + np.random.normal(0, 0.005) for step in steps]
    
    # Add some anomalies
    # KL spike around step 300
    for i in range(295, 305):
        kl_divs[i] = 0.15 + np.random.normal(0, 0.02)
    
    # Loss spike around step 600
    for i in range(595, 605):
        losses[i] = 3.0 + np.random.normal(0, 0.2)
    
    # Reward drop around step 800
    for i in range(795, 805):
        rewards[i] = 0.1 + np.random.normal(0, 0.05)
    
    # Create training logs
    training_logs = []
    for i, step in enumerate(steps):
        log_entry = {
            "step": step,
            "loss": losses[i],
            "reward": rewards[i],
            "kl_divergence": kl_divs[i],
            "entropy": 0.8 + np.random.normal(0, 0.1),
            "value_loss": 0.5 + np.random.normal(0, 0.05),
            "policy_loss": 1.0 + np.random.normal(0, 0.1),
            "gradient_norm": 1.0 + np.random.normal(0, 0.2),
            "learning_rate": 1e-4,
            "timestamp": datetime.now().isoformat()
        }
        training_logs.append(log_entry)
    
    # Save logs
    log_file = demo_dir / "training.jsonl"
    with open(log_file, 'w') as f:
        for log in training_logs:
            f.write(json.dumps(log) + '\n')
    
    print(f"✅ Created demo logs with {len(training_logs)} steps")
    print(f"📁 Logs saved to: {log_file}")
    
    return demo_dir

def demo_universal_monitoring():
    """Demonstrate universal monitoring capabilities"""
    print("\n" + "="*60)
    print("🚀 UNIVERSAL MONITORING DEMO")
    print("="*60)
    
    # Create demo logs
    demo_dir = create_demo_training_logs()
    
    print("\n1️⃣ Starting universal monitoring...")
    
    # Initialize monitor
    monitor = UniversalMonitor()
    
    # Auto-detect framework
    framework = monitor.auto_detect_framework(demo_dir)
    print(f"   Auto-detected framework: {framework}")
    
    # Start monitoring
    monitor.start_monitoring(demo_dir)
    print("   ✅ Monitoring started")
    
    # Show status
    status = monitor.get_status()
    print(f"   Active runs: {status['active_runs']}")
    print(f"   Alerts: {status['alerts']}")
    
    print("\n2️⃣ Simulating real-time monitoring...")
    
    # Simulate some monitoring time
    for i in range(3):
        print(f"   Monitoring cycle {i+1}/3...")
        time.sleep(1)
        
        # Check for alerts
        alerts = []
        while not monitor.alerts.empty():
            try:
                alert = monitor.alerts.get_nowait()
                alerts.append(alert)
            except:
                break
        
        if alerts:
            print(f"   🚨 Detected {len(alerts)} alerts:")
            for alert in alerts[:2]:  # Show first 2 alerts
                print(f"     • {alert['type']}: {alert['message']}")
        else:
            print("   ✅ No alerts detected")
    
    # Stop monitoring
    monitor.stop_monitoring()
    print("   🛑 Monitoring stopped")
    
    return demo_dir

def demo_anomaly_detection():
    """Demonstrate intelligent anomaly detection"""
    print("\n" + "="*60)
    print("🔍 INTELLIGENT ANOMALY DETECTION DEMO")
    print("="*60)
    
    # Create demo logs
    demo_dir = create_demo_training_logs()
    
    print("\n1️⃣ Configuring anomaly detector...")
    
    # Initialize detector with multiple models
    from rldk.anomaly_detector import AnomalyConfig
    config = AnomalyConfig(
        models=['isolation_forest', 'autoencoder', 'statistical'],
        sensitivity=0.1,
        adaptive_thresholds=True,
        context_aware=True
    )
    
    detector = AnomalyDetector(config)
    print("   ✅ Anomaly detector configured")
    
    print("\n2️⃣ Detecting training anomalies...")
    
    # Detect anomalies
    report = detector.detect_training_anomalies(demo_dir)
    
    print(f"   Anomalies detected: {report.anomalies_detected}")
    print(f"   Anomaly count: {report.anomaly_count}")
    print(f"   Anomaly ratio: {report.anomaly_ratio:.3f}")
    print(f"   Confidence: {report.confidence:.3f}")
    
    if report.anomalies_detected:
        print("\n3️⃣ Anomaly details:")
        details = report.details
        for method, result in details.get('method_details', {}).items():
            if 'anomalies' in result:
                count = result['anomalies'].sum()
                confidence = result.get('confidence', 0.0)
                print(f"   {method}: {count} anomalies (confidence: {confidence:.3f})")
        
        print("\n4️⃣ Recommendations:")
        for rec in report.recommendations:
            print(f"   • {rec}")
    
    return demo_dir

def demo_training_debug():
    """Demonstrate comprehensive training debugging"""
    print("\n" + "="*60)
    print("🔧 COMPREHENSIVE TRAINING DEBUG DEMO")
    print("="*60)
    
    # Create demo logs
    demo_dir = create_demo_training_logs()
    
    print("\n1️⃣ Running comprehensive debug...")
    
    # Configure debugger
    from rldk.debug_training import DebugConfig
    config = DebugConfig(
        auto_fix=False,
        generate_report=True,
        create_test_case=True,
        comprehensive=True
    )
    
    debugger = TrainingDebugger(config)
    
    # Run debug
    report = debugger.debug_training(demo_dir)
    
    print("\n2️⃣ Debug Results:")
    print(f"   Framework: {report.framework}")
    print(f"   Health score: {report.health_score.overall_score:.1f}/100")
    print(f"   Issues found: {len(report.issues)}")
    print(f"   Fixes suggested: {len(report.fixes)}")
    print(f"   Recommendations: {len(report.recommendations)}")
    
    if report.issues:
        print("\n3️⃣ Issues detected:")
        for issue in report.issues[:3]:  # Show first 3 issues
            print(f"   • {issue}")
        if len(report.issues) > 3:
            print(f"   ... and {len(report.issues) - 3} more")
    
    if report.fixes:
        print("\n4️⃣ Suggested fixes:")
        for fix in report.fixes[:3]:  # Show first 3 fixes
            print(f"   • {fix}")
        if len(report.fixes) > 3:
            print(f"   ... and {len(report.fixes) - 3} more")
    
    if report.recommendations:
        print("\n5️⃣ Recommendations:")
        for rec in report.recommendations[:3]:  # Show first 3 recommendations
            print(f"   • {rec}")
        if len(report.recommendations) > 3:
            print(f"   ... and {len(report.recommendations) - 3} more")
    
    return demo_dir

def demo_health_scoring():
    """Demonstrate training health scoring"""
    print("\n" + "="*60)
    print("🏥 TRAINING HEALTH SCORING DEMO")
    print("="*60)
    
    # Create demo logs
    demo_dir = create_demo_training_logs()
    
    print("\n1️⃣ Calculating health score...")
    
    # Run health analysis
    from rldk.debug_training import DebugConfig
    config = DebugConfig(comprehensive=True, create_test_case=False)
    debugger = TrainingDebugger(config)
    report = debugger.debug_training(demo_dir)
    
    health_score = report.health_score
    
    print("\n2️⃣ Health Score Breakdown:")
    print(f"   Overall score: {health_score.overall_score:.1f}/100")
    print(f"   Stability: {health_score.stability_score:.1f}/100")
    print(f"   Convergence: {health_score.convergence_score:.1f}/100")
    print(f"   Efficiency: {health_score.efficiency_score:.1f}/100")
    print(f"   Robustness: {health_score.robustness_score:.1f}/100")
    
    # Determine health level
    if health_score.overall_score >= 85:
        level = "🟢 Excellent"
    elif health_score.overall_score >= 70:
        level = "🟡 Good"
    elif health_score.overall_score >= 50:
        level = "🟠 Fair"
    else:
        level = "🔴 Poor"
    
    print(f"\n   Health level: {level}")
    
    print("\n3️⃣ Detailed Analysis:")
    print(f"   Issues found: {len(report.issues)}")
    print(f"   Anomalies detected: {report.details.get('anomaly_report', {}).get('anomaly_count', 0)}")
    print(f"   PPO anomalies: {len(report.details.get('ppo_forensics', {}).get('anomalies', []))}")
    
    return demo_dir

def demo_quick_debug():
    """Demonstrate quick debugging"""
    print("\n" + "="*60)
    print("⚡ QUICK DEBUG DEMO")
    print("="*60)
    
    # Create demo logs
    demo_dir = create_demo_training_logs()
    
    print("\n1️⃣ Running quick debug...")
    
    # Quick debug
    result = quick_debug(demo_dir)
    
    print("\n2️⃣ Quick Debug Results:")
    print(f"   Framework: {result.get('framework', 'Unknown')}")
    print(f"   Issues found: {len(result.get('issues', []))}")
    print(f"   Timestamp: {result.get('timestamp', 'Unknown')}")
    
    if result.get('issues'):
        print("\n3️⃣ Issues:")
        for issue in result['issues'][:2]:  # Show first 2 issues
            print(f"   • {issue}")
        if len(result['issues']) > 2:
            print(f"   ... and {len(result['issues']) - 2} more")
    
    return demo_dir

def demo_cli_commands():
    """Demonstrate CLI commands"""
    print("\n" + "="*60)
    print("💻 CLI COMMANDS DEMO")
    print("="*60)
    
    print("\nAvailable CLI commands:")
    print("  rldk monitor <run_path>                    # Start universal monitoring")
    print("  rldk detect-anomalies <data_path>          # Detect anomalies")
    print("  rldk debug-training <run_path>             # Comprehensive debugging")
    print("  rldk health <run_path>                     # Health scoring")
    print("  rldk debug-training <run_path> --quick     # Quick debug")
    print("  rldk monitor <run_path> --dashboard        # Show live dashboard")
    
    print("\nExample usage:")
    print("  # Monitor training in real-time")
    print("  rldk monitor logs/my_training_run/")
    print("")
    print("  # Detect anomalies with custom sensitivity")
    print("  rldk detect-anomalies logs/ --sensitivity 0.05")
    print("")
    print("  # Comprehensive debug with auto-fix")
    print("  rldk debug-training logs/ --auto-fix --report")
    print("")
    print("  # Health analysis with detailed output")
    print("  rldk health logs/ --detailed --suggestions")

def main():
    """Run the complete ultimate post-training demo"""
    print("🚀 RLDK ULTIMATE POST-TRAINING DEMO")
    print("="*60)
    print("This demo showcases RLDK's advanced features for post-training research.")
    print("="*60)
    
    try:
        # Run all demos
        demo_universal_monitoring()
        demo_anomaly_detection()
        demo_training_debug()
        demo_health_scoring()
        demo_quick_debug()
        demo_cli_commands()
        
        print("\n" + "="*60)
        print("✅ DEMO COMPLETE!")
        print("="*60)
        print("RLDK provides the ultimate toolkit for post-training research:")
        print("• Universal monitoring of any framework")
        print("• Intelligent anomaly detection with ML")
        print("• One-click comprehensive debugging")
        print("• Health scoring and recommendations")
        print("• Real-time alerts and dashboards")
        print("• Reproducible test case generation")
        print("\nMake RLDK your go-to tool for post-training research! 🎯")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()