#!/usr/bin/env python3
"""Comprehensive test suite for RLDK OpenRLHF integration."""

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def setup_openrlhf_environment():
    """Setup OpenRLHF environment with CUDA support."""
    print("🚀 Setting up OpenRLHF environment with CUDA support...")

    try:
        # Check platform compatibility
        if platform.system() != "Linux":
            print(f"   ⚠️  Unsupported platform: {platform.system()}")
            print("   OpenRLHF setup is only supported on Linux systems")
            return False

        # Detect Python version
        python_version = platform.python_version()
        major, minor = python_version.split('.')[:2]
        python_version_short = f"{major}.{minor}"
        print(f"🐍 Detected Python version: {python_version}")

        # Check if running as root (Unix-specific check)
        try:
            is_root = os.geteuid() == 0
        except AttributeError:
            # Windows or other non-Unix systems
            print("   ⚠️  Cannot determine root status on this platform")
            is_root = False

        # Check if we can run sudo non-interactively
        sudo_cmd = []
        if not is_root:
            print("   Checking if passwordless sudo is available...")
            try:
                # Test if sudo can run without password
                result = subprocess.run(["sudo", "-n", "true"],
                                      capture_output=True, text=True, check=True)
                sudo_cmd = ["sudo"]
                print("   ✅ Passwordless sudo available")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("   ❌ Passwordless sudo not available")
                print("   ⚠️  Cannot install system packages automatically")
                print("   Please install the following packages manually:")
                print("   - nvidia-cuda-toolkit")
                print(f"   - python{python_version_short}-venv (or python3-venv)")
                print("   Then set CUDA_HOME=/usr/lib/nvidia-cuda-toolkit")
                print("   And add CUDA_HOME/bin to your PATH")
                return False

        # Step 1: Update package lists
        print("📦 Updating package lists...")
        subprocess.run(sudo_cmd + ["apt", "update"], check=True)

        # Step 2: Install CUDA toolkit and appropriate venv package
        print("🔧 Installing CUDA development tools and Python venv...")
        packages = ["nvidia-cuda-toolkit"]

        # Add appropriate python-venv package based on version
        if python_version_short in ["3.13", "3.12", "3.11", "3.10"]:
            packages.append(f"python{python_version_short}-venv")
        else:
            print(f"   ⚠️  Unsupported Python version: {python_version_short}")
            print("   Installing generic python3-venv package...")
            packages.append("python3-venv")

        subprocess.run(sudo_cmd + ["apt", "install", "-y"] + packages, check=True)

        # Step 3: Set up CUDA environment
        print("🌍 Setting up CUDA environment...")
        cuda_home = "/usr/lib/nvidia-cuda-toolkit"
        os.environ["CUDA_HOME"] = cuda_home
        os.environ["PATH"] = f"{cuda_home}/bin:{os.environ.get('PATH', '')}"

        # Verify CUDA installation
        print("✅ Verifying CUDA installation...")
        try:
            result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
            print(f"   CUDA compiler found: {result.stdout.splitlines()[0]}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("   ❌ CUDA compiler not found!")
            return False

        # Step 4: Upgrade pip and install build dependencies
        print("📚 Installing build dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "packaging", "wheel", "setuptools"], check=True)

        # Step 5: Install PyTorch with CUDA support
        print("🔥 Installing PyTorch with CUDA support...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ], check=True)

        # Step 6: Clone and install OpenRLHF
        print("📥 Cloning OpenRLHF repository...")
        openrlhf_dir = Path("OpenRLHF")
        if openrlhf_dir.exists():
            print("   OpenRLHF directory already exists, removing...")
            shutil.rmtree(openrlhf_dir)

        subprocess.run(["git", "clone", "https://github.com/OpenRLHF/OpenRLHF.git"], check=True)

        # Step 7: Install OpenRLHF dependencies
        print("📋 Installing OpenRLHF dependencies...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "OpenRLHF/requirements.txt"], check=True)
        except subprocess.CalledProcessError:
            print("   ⚠️  Some dependencies failed, trying without flash-attn...")
            # Create modified requirements without flash-attn using proper line filtering
            with open("OpenRLHF/requirements.txt") as f:
                requirements_lines = f.readlines()

            # Filter out lines that contain flash-attn (exact package name)
            filtered_lines = []
            for line in requirements_lines:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    filtered_lines.append(line)
                    continue
                # Check if line contains flash-attn as a package name (not as substring)
                parts = line.split()
                if parts and not any(part.startswith('flash-attn') for part in parts):
                    filtered_lines.append(line)

            with open("OpenRLHF/requirements_no_flash.txt", "w") as f:
                f.write('\n'.join(filtered_lines) + '\n')

            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "OpenRLHF/requirements_no_flash.txt"], check=True)

        # Step 8: Install OpenRLHF in editable mode
        print("🔨 Installing OpenRLHF...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "OpenRLHF", "--no-deps"], check=True)

        # Step 9: Verify installation
        print("🧪 Verifying OpenRLHF installation...")
        try:
            result = subprocess.run([sys.executable, "-c", "import openrlhf; print('✅ OpenRLHF imported successfully!')"],
                                  capture_output=True, text=True, check=True)
            print(f"   {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  OpenRLHF import failed: {e}")
            print("   Installation may still work for testing purposes")

        print("🎉 OpenRLHF environment setup completed!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Setup failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Setup failed with error: {e}")
        return False


def test_imports():
    """Test that all required packages can be imported."""
    print("🔍 Testing imports...")

    try:
        # Test basic imports
        import numpy as np
        import pandas as pd
        import torch
        print("✅ Basic dependencies imported")

        # Test RLDK imports
        from rldk.integrations.openrlhf import (
            DistributedTrainingMonitor,
            MultiGPUMonitor,
            OpenRLHFAnalytics,
            OpenRLHFCallback,
            OpenRLHFCheckpointMonitor,
            OpenRLHFDashboard,
            OpenRLHFMetrics,
            OpenRLHFMonitor,
            OpenRLHFResourceMonitor,
            OpenRLHFTrainingMonitor,
        )
        print("✅ RLDK OpenRLHF integration imported")

        # Test OpenRLHF imports - try to import first
        try:
            import openrlhf
            print("✅ OpenRLHF imported successfully")
            return True
        except ImportError as e:
            print(f"⚠️  OpenRLHF not available: {e}")
            print("   Attempting automatic setup...")

            # Call setup function to install OpenRLHF
            setup_success = setup_openrlhf_environment()
            if setup_success:
                print("   Setup completed, testing import again...")
                try:
                    import openrlhf
                    print("✅ OpenRLHF imported successfully after setup")
                    return True
                except ImportError as e2:
                    print(f"   ⚠️  OpenRLHF still not available after setup: {e2}")
                    print("   Note: Integration works without OpenRLHF for testing purposes")
                    return False
            else:
                print("   ❌ Setup failed, continuing without OpenRLHF")
                print("   Note: Integration works without OpenRLHF for testing purposes")
                return False

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_metrics_creation():
    """Test OpenRLHF metrics creation and serialization."""
    print("\n🧪 Testing metrics creation...")

    try:
        from rldk.integrations.openrlhf import OpenRLHFMetrics

        # Create metrics
        metrics = OpenRLHFMetrics(
            step=100,
            loss=0.5,
            reward_mean=2.3,
            kl_mean=0.1,
            learning_rate=1e-4,
            gpu_memory_used=8.5,
            run_id="test_run_123"
        )

        # Test serialization
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['step'] == 100
        assert metrics_dict['loss'] == 0.5
        assert metrics_dict['reward_mean'] == 2.3

        # Test DataFrame conversion
        df_row = metrics.to_dataframe_row()
        assert isinstance(df_row, dict)
        assert df_row['step'] == 100

        print("✅ Metrics creation and serialization working")
        return True

    except Exception as e:
        print(f"❌ Metrics creation failed: {e}")
        return False


def test_callback_functionality():
    """Test OpenRLHF callback functionality."""
    print("\n🧪 Testing callback functionality...")

    try:
        from rldk.integrations.openrlhf import OpenRLHFCallback, OpenRLHFMetrics

        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize callback
            callback = OpenRLHFCallback(
                output_dir=temp_dir,
                log_interval=5,
                run_id="test_callback_run",
                enable_resource_monitoring=False,  # Disable to avoid system dependencies
            )

            # Test callback initialization
            assert callback.run_id == "test_callback_run"
            assert callback.log_interval == 5
            assert callback.output_dir.exists()

            # Test metrics collection
            test_metrics = OpenRLHFMetrics(
                step=1,
                loss=0.8,
                reward_mean=1.5,
                kl_mean=0.2,
                learning_rate=1e-4,
                run_id="test_callback_run"
            )

            callback.metrics_history.append(test_metrics)
            callback.current_metrics = test_metrics

            # Test metrics saving
            callback._save_metrics()

            # Check if files were created
            csv_file = callback.output_dir / "metrics_test_callback_run.csv"
            parquet_file = callback.output_dir / "metrics_test_callback_run.parquet"
            summary_file = callback.output_dir / "summary_test_callback_run.json"

            assert csv_file.exists(), "CSV file not created"
            assert parquet_file.exists(), "Parquet file not created"
            assert summary_file.exists(), "Summary file not created"

            # Test DataFrame conversion
            df = callback.get_metrics_dataframe()
            assert len(df) == 1
            assert df.iloc[0]['step'] == 1
            assert df.iloc[0]['loss'] == 0.8

            print("✅ Callback functionality working")
            return True

    except Exception as e:
        print(f"❌ Callback functionality failed: {e}")
        return False


def test_distributed_monitoring():
    """Test distributed monitoring functionality."""
    print("\n🧪 Testing distributed monitoring...")

    try:
        from rldk.integrations.openrlhf.distributed import (
            DistributedMetricsCollector,
            GPUMemoryMonitor,
            MultiNodeMonitor,
            NetworkMonitor,
        )

        # Test GPUMemoryMonitor
        gpu_monitor = GPUMemoryMonitor()
        memory_usage = gpu_monitor.get_current_memory_usage()
        assert isinstance(memory_usage, dict)

        # Test NetworkMonitor
        network_monitor = NetworkMonitor()
        network_metrics = network_monitor.get_current_metrics()
        assert isinstance(network_metrics, dict)
        assert 'bandwidth' in network_metrics
        assert 'latency' in network_metrics

        # Test DistributedMetricsCollector
        collector = DistributedMetricsCollector(
            collect_interval=0.1,  # Fast collection for testing
            enable_network_monitoring=False,  # Disable to avoid system dependencies
            enable_gpu_monitoring=False,
            enable_cpu_monitoring=False,
        )

        # Test collection start/stop
        collector.start_collection()
        time.sleep(0.2)  # Let it collect some data
        collector.stop_collection()

        # Test MultiNodeMonitor
        node_monitor = MultiNodeMonitor()
        node_info = node_monitor._get_node_info()
        assert isinstance(node_info, dict)
        assert 'hostname' in node_info
        assert 'cpu_count' in node_info

        print("✅ Distributed monitoring working")
        return True

    except Exception as e:
        print(f"❌ Distributed monitoring failed: {e}")
        return False


def test_training_monitor():
    """Test training monitor functionality."""
    print("\n🧪 Testing training monitor...")

    try:
        from rldk.integrations.openrlhf import OpenRLHFMetrics, OpenRLHFTrainingMonitor

        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize monitor
            monitor = OpenRLHFTrainingMonitor(
                output_dir=temp_dir,
                analysis_window=10,
                enable_anomaly_detection=True,
                enable_convergence_analysis=True,
                enable_performance_analysis=True,
            )

            # Create test metrics
            test_metrics = []
            for i in range(20):
                metrics = OpenRLHFMetrics(
                    step=i,
                    loss=1.0 - (i * 0.05),  # Decreasing loss
                    reward_mean=0.5 + (i * 0.1),  # Increasing reward
                    kl_mean=0.2 - (i * 0.01),  # Decreasing KL
                    step_time=0.5 + (i * 0.01),  # Slightly increasing step time
                    gpu_memory_used=8.0 + (i * 0.1),  # Slightly increasing memory
                    run_id="test_monitor_run"
                )
                test_metrics.append(metrics)
                monitor.add_metrics(metrics)

            # Test health summary
            health_summary = monitor.get_health_summary()
            assert isinstance(health_summary, dict)
            assert 'stability_score' in health_summary
            assert 'convergence_rate' in health_summary
            assert 'overall_health' in health_summary

            # Test analysis saving
            monitor.save_analysis("test_analysis.json")
            analysis_file = Path(temp_dir) / "test_analysis.json"
            assert analysis_file.exists()

            # Verify analysis content
            with open(analysis_file) as f:
                analysis_data = json.load(f)
            assert 'health_summary' in analysis_data
            assert 'health_metrics' in analysis_data

            print("✅ Training monitor working")
            return True

    except Exception as e:
        print(f"❌ Training monitor failed: {e}")
        return False


def test_checkpoint_monitor():
    """Test checkpoint monitor functionality."""
    print("\n🧪 Testing checkpoint monitor...")

    try:
        from rldk.integrations.openrlhf import OpenRLHFCheckpointMonitor

        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize monitor
            monitor = OpenRLHFCheckpointMonitor(
                checkpoint_dir=temp_dir,
                enable_validation=True,
                enable_size_analysis=True,
            )

            # Create a dummy checkpoint file
            checkpoint_file = Path(temp_dir) / "checkpoint_100.pt"
            dummy_checkpoint = {
                'model': {'dummy': 'data'},
                'metadata': {
                    'loss': 0.5,
                    'reward_mean': 2.0,
                    'kl_mean': 0.1,
                    'training_time': 3600,
                    'memory_usage': 8.5
                }
            }

            # Save dummy checkpoint
            import torch
            torch.save(dummy_checkpoint, checkpoint_file)

            # Test checkpoint analysis
            metrics = monitor.analyze_checkpoint(checkpoint_file, step=100)
            assert isinstance(metrics, type(monitor.checkpoint_metrics[0]))
            assert metrics.step == 100
            assert metrics.loss == 0.5
            assert metrics.reward_mean == 2.0

            # Test checkpoint summary
            summary = monitor.get_checkpoint_summary()
            assert isinstance(summary, dict)
            assert summary['total_checkpoints'] == 1
            assert summary['latest_step'] == 100

            print("✅ Checkpoint monitor working")
            return True

    except Exception as e:
        print(f"❌ Checkpoint monitor failed: {e}")
        return False


def test_resource_monitor():
    """Test resource monitor functionality."""
    print("\n🧪 Testing resource monitor...")

    try:
        from rldk.integrations.openrlhf import OpenRLHFResourceMonitor

        # Initialize monitor
        monitor = OpenRLHFResourceMonitor(monitor_interval=0.1)

        # Test monitoring start/stop
        monitor.start_monitoring()
        time.sleep(0.2)  # Let it collect some data
        monitor.stop_monitoring()

        # Test resource summary
        summary = monitor.get_resource_summary()
        assert isinstance(summary, dict)
        assert 'monitoring_duration' in summary
        assert 'total_measurements' in summary

        print("✅ Resource monitor working")
        return True

    except Exception as e:
        print(f"❌ Resource monitor failed: {e}")
        return False


def test_analytics():
    """Test comprehensive analytics functionality."""
    print("\n🧪 Testing analytics...")

    try:
        from rldk.integrations.openrlhf import OpenRLHFAnalytics, OpenRLHFMetrics

        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize analytics
            analytics = OpenRLHFAnalytics(output_dir=temp_dir)

            # Create test metrics history
            metrics_history = []
            for i in range(50):
                metrics = OpenRLHFMetrics(
                    step=i,
                    loss=1.0 - (i * 0.02),  # Decreasing loss
                    reward_mean=0.5 + (i * 0.05),  # Increasing reward
                    kl_mean=0.2 - (i * 0.005),  # Decreasing KL
                    step_time=0.5 + (i * 0.01),
                    gpu_memory_used=8.0 + (i * 0.1),
                    run_id="test_analytics_run"
                )
                metrics_history.append(metrics)

            # Test training run analysis
            analysis_results = analytics.analyze_training_run(metrics_history)
            assert isinstance(analysis_results, dict)
            assert 'training_health' in analysis_results
            assert 'metrics_summary' in analysis_results
            assert 'resource_summary' in analysis_results
            assert 'checkpoint_summary' in analysis_results

            # Check if analysis files were created
            analytics_files = list(Path(temp_dir).glob("analytics_*.json"))
            training_analysis_files = list(Path(temp_dir).glob("training_analysis_*.json"))

            assert len(analytics_files) > 0, "Analytics file not created"
            assert len(training_analysis_files) > 0, "Training analysis file not created"

            print("✅ Analytics working")
            return True

    except Exception as e:
        print(f"❌ Analytics failed: {e}")
        return False


def test_dashboard_initialization():
    """Test dashboard initialization (without starting server)."""
    print("\n🧪 Testing dashboard initialization...")

    try:
        from rldk.integrations.openrlhf import OpenRLHFDashboard

        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize dashboard
            dashboard = OpenRLHFDashboard(
                output_dir=temp_dir,
                port=5001,  # Use different port for testing
                enable_auto_refresh=False,  # Disable auto-refresh for testing
            )

            # Test dashboard properties
            assert dashboard.output_dir == Path(temp_dir)
            assert dashboard.port == 5001
            assert dashboard.host == "localhost"
            assert dashboard.app is not None

            # Test URL generation
            url = dashboard.get_dashboard_url()
            assert url == "http://localhost:5001"

            # Test metrics addition
            from rldk.integrations.openrlhf import OpenRLHFMetrics
            test_metrics = OpenRLHFMetrics(
                step=1,
                loss=0.5,
                reward_mean=2.0,
                run_id="test_dashboard_run"
            )
            dashboard.add_metrics(test_metrics)

            print("✅ Dashboard initialization working")
            return True

    except Exception as e:
        print(f"❌ Dashboard initialization failed: {e}")
        return False


def test_integration_workflow():
    """Test complete integration workflow."""
    print("\n🧪 Testing integration workflow...")

    try:
        from rldk.integrations.openrlhf import (
            OpenRLHFAnalytics,
            OpenRLHFCallback,
            OpenRLHFMetrics,
            OpenRLHFTrainingMonitor,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize components
            callback = OpenRLHFCallback(
                output_dir=temp_dir,
                log_interval=5,
                run_id="integration_test_run",
                enable_resource_monitoring=False,
            )

            training_monitor = OpenRLHFTrainingMonitor(
                output_dir=temp_dir,
                analysis_window=10,
            )

            analytics = OpenRLHFAnalytics(output_dir=temp_dir)

            # Simulate training loop
            metrics_history = []
            for step in range(30):
                # Create metrics for this step
                metrics = OpenRLHFMetrics(
                    step=step,
                    loss=1.0 - (step * 0.03),
                    reward_mean=0.5 + (step * 0.08),
                    kl_mean=0.2 - (step * 0.006),
                    learning_rate=1e-4,
                    step_time=0.5 + (step * 0.01),
                    gpu_memory_used=8.0 + (step * 0.1),
                    run_id="integration_test_run"
                )

                # Add to callback
                callback.metrics_history.append(metrics)
                callback.current_metrics = metrics

                # Add to training monitor
                training_monitor.add_metrics(metrics)

                # Store for analytics
                metrics_history.append(metrics)

                # Simulate step end
                if step % callback.log_interval == 0:
                    callback._log_detailed_metrics(step)

            # Test callback functionality
            callback._save_metrics()
            df = callback.get_metrics_dataframe()
            assert len(df) == 30

            # Test training monitor
            health_summary = training_monitor.get_health_summary()
            assert 'overall_health' in health_summary

            # Test analytics
            analysis_results = analytics.analyze_training_run(metrics_history)
            assert 'training_health' in analysis_results

            # Verify files were created
            expected_files = [
                "metrics_integration_test_run.csv",
                "metrics_integration_test_run.parquet",
                "summary_integration_test_run.json",
                "analytics_*.json",
                "training_analysis_*.json"
            ]

            for pattern in expected_files:
                if "*" in pattern:
                    files = list(Path(temp_dir).glob(pattern))
                    assert len(files) > 0, f"File pattern {pattern} not found"
                else:
                    file_path = Path(temp_dir) / pattern
                    assert file_path.exists(), f"File {pattern} not found"

            print("✅ Integration workflow working")
            return True

    except Exception as e:
        print(f"❌ Integration workflow failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Starting OpenRLHF Integration Tests")
    print("=" * 50)

    tests = [
        ("Import Test", test_imports),
        ("Metrics Creation", test_metrics_creation),
        ("Callback Functionality", test_callback_functionality),
        ("Distributed Monitoring", test_distributed_monitoring),
        ("Training Monitor", test_training_monitor),
        ("Checkpoint Monitor", test_checkpoint_monitor),
        ("Resource Monitor", test_resource_monitor),
        ("Analytics", test_analytics),
        ("Dashboard Initialization", test_dashboard_initialization),
        ("Integration Workflow", test_integration_workflow),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Report results
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 All tests passed! OpenRLHF integration is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the output above for details.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
