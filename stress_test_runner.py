#!/usr/bin/env python3
"""
Stress Test Runner for RLDK

This script runs multiple iterations of comprehensive tests to trigger edge cases,
anomalies, and stress conditions to ensure the RLDK repository is robust under
various failure scenarios.
"""

import os
import sys
import json
import time
import random
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import uuid
import gc
import psutil
import signal
import subprocess

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import our test scripts
try:
    from comprehensive_rl_llm_test import ComprehensiveTester, TestConfig
    from rldk_function_validator import RLDKFunctionValidator
    COMPREHENSIVE_TEST_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import test scripts: {e}")
    COMPREHENSIVE_TEST_AVAILABLE = False


class StressTestConfig:
    """Configuration for stress testing."""
    
    def __init__(self):
        # Test parameters
        self.num_iterations = 20  # Run many iterations to catch edge cases
        self.parallel_workers = 4  # Run tests in parallel
        self.memory_pressure_enabled = True
        self.cpu_pressure_enabled = True
        self.anomaly_injection_enabled = True
        
        # Stress parameters
        self.memory_pressure_probability = 0.3  # 30% chance of memory pressure
        self.cpu_pressure_probability = 0.2     # 20% chance of CPU pressure
        self.anomaly_probability = 0.25         # 25% chance of anomaly injection
        
        # Output parameters
        self.output_dir = Path("./stress_test_results")
        self.save_detailed_logs = True
        self.save_crash_dumps = True
        
        # Test types to run
        self.run_comprehensive_tests = True
        self.run_function_validation = True
        self.run_memory_stress_tests = True
        self.run_concurrent_tests = True
        self.run_failure_recovery_tests = True
        
        # Timeout parameters
        self.test_timeout = 300  # 5 minutes per test
        self.iteration_timeout = 1800  # 30 minutes per iteration


class MemoryStressor:
    """Create memory pressure to test system robustness."""
    
    def __init__(self, target_memory_percent: float = 80.0):
        self.target_memory_percent = target_memory_percent
        self.stress_threads = []
        self.running = False
    
    def start_memory_stress(self):
        """Start memory stress threads."""
        if self.running:
            return
        
        self.running = True
        
        # Calculate target memory usage
        total_memory = psutil.virtual_memory().total
        target_memory = total_memory * (self.target_memory_percent / 100.0)
        
        # Start multiple threads to allocate memory
        for i in range(4):  # 4 threads for memory allocation
            thread = threading.Thread(target=self._allocate_memory, args=(target_memory / 4,))
            thread.daemon = True
            thread.start()
            self.stress_threads.append(thread)
    
    def stop_memory_stress(self):
        """Stop memory stress threads."""
        self.running = False
        # Threads will stop when they see running=False
    
    def _allocate_memory(self, target_size: int):
        """Allocate memory in a separate thread."""
        allocated_memory = []
        chunk_size = min(100 * 1024 * 1024, target_size // 10)  # 100MB chunks
        
        while self.running and len(allocated_memory) * chunk_size < target_size:
            try:
                # Allocate memory chunk
                chunk = bytearray(chunk_size)
                allocated_memory.append(chunk)
                time.sleep(0.1)  # Small delay to prevent overwhelming
            except MemoryError:
                break  # Out of memory
        
        # Keep memory allocated while running
        while self.running:
            time.sleep(1)
        
        # Clean up
        allocated_memory.clear()
        gc.collect()


class CPUStressor:
    """Create CPU pressure to test system robustness."""
    
    def __init__(self, target_cpu_percent: float = 80.0):
        self.target_cpu_percent = target_cpu_percent
        self.stress_threads = []
        self.running = False
    
    def start_cpu_stress(self):
        """Start CPU stress threads."""
        if self.running:
            return
        
        self.running = True
        
        # Start multiple threads for CPU stress
        num_threads = multiprocessing.cpu_count()
        for i in range(num_threads):
            thread = threading.Thread(target=self._cpu_stress_worker)
            thread.daemon = True
            thread.start()
            self.stress_threads.append(thread)
    
    def stop_cpu_stress(self):
        """Stop CPU stress threads."""
        self.running = False
    
    def _cpu_stress_worker(self):
        """CPU stress worker thread."""
        while self.running:
            # Do some CPU-intensive work
            start_time = time.time()
            while time.time() - start_time < 0.1:  # 100ms of work
                # Simple CPU-intensive operation
                _ = sum(i * i for i in range(1000))
            
            # Sleep to control CPU usage
            time.sleep(0.1)


class AnomalyInjector:
    """Inject various system-level anomalies."""
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.injected_anomalies = []
    
    def should_inject_anomaly(self) -> bool:
        """Determine if an anomaly should be injected."""
        return random.random() < self.config.anomaly_probability
    
    def inject_system_anomaly(self) -> Dict[str, Any]:
        """Inject a system-level anomaly."""
        anomaly_types = [
            'memory_pressure',
            'cpu_pressure',
            'disk_io_pressure',
            'network_delay',
            'file_system_error',
            'process_interruption'
        ]
        
        anomaly_type = random.choice(anomaly_types)
        anomaly_info = {
            'type': anomaly_type,
            'timestamp': time.time(),
            'description': f"Injected {anomaly_type} anomaly"
        }
        
        try:
            if anomaly_type == 'memory_pressure':
                self._inject_memory_pressure()
            elif anomaly_type == 'cpu_pressure':
                self._inject_cpu_pressure()
            elif anomaly_type == 'disk_io_pressure':
                self._inject_disk_io_pressure()
            elif anomaly_type == 'network_delay':
                self._inject_network_delay()
            elif anomaly_type == 'file_system_error':
                self._inject_file_system_error()
            elif anomaly_type == 'process_interruption':
                self._inject_process_interruption()
            
            anomaly_info['success'] = True
            self.injected_anomalies.append(anomaly_info)
            
        except Exception as e:
            anomaly_info['success'] = False
            anomaly_info['error'] = str(e)
        
        return anomaly_info
    
    def _inject_memory_pressure(self):
        """Inject memory pressure."""
        # Allocate large amounts of memory
        large_arrays = []
        for _ in range(10):
            try:
                arr = bytearray(50 * 1024 * 1024)  # 50MB
                large_arrays.append(arr)
            except MemoryError:
                break
        time.sleep(2)  # Keep memory allocated for 2 seconds
        large_arrays.clear()
        gc.collect()
    
    def _inject_cpu_pressure(self):
        """Inject CPU pressure."""
        # Do CPU-intensive work for a short time
        start_time = time.time()
        while time.time() - start_time < 1.0:  # 1 second of CPU stress
            _ = sum(i * i for i in range(10000))
    
    def _inject_disk_io_pressure(self):
        """Inject disk I/O pressure."""
        # Create temporary files to stress disk I/O
        temp_files = []
        for _ in range(5):
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(b'0' * (10 * 1024 * 1024))  # 10MB
            temp_file.close()
            temp_files.append(temp_file.name)
        
        time.sleep(1)  # Keep files for 1 second
        
        # Clean up
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def _inject_network_delay(self):
        """Inject network delay (simulated)."""
        # Simulate network delay by sleeping
        time.sleep(random.uniform(0.5, 2.0))
    
    def _inject_file_system_error(self):
        """Inject file system error."""
        # Try to write to a non-existent directory
        try:
            with open('/tmp/nonexistent/directory/file.txt', 'w') as f:
                f.write('test')
        except (OSError, IOError):
            pass  # Expected error
    
    def _inject_process_interruption(self):
        """Inject process interruption."""
        # Send SIGTERM to current process (but catch it)
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except:
            pass


class StressTestRunner:
    """Main stress test runner."""
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize stress components
        self.memory_stressor = MemoryStressor()
        self.cpu_stressor = CPUStressor()
        self.anomaly_injector = AnomalyInjector(config)
        
        # Test results
        self.test_results = {
            'start_time': time.time(),
            'config': self.config.__dict__,
            'iterations': [],
            'anomalies_injected': [],
            'system_info': self._get_system_info(),
            'overall_success': True,
            'errors': [],
            'warnings': []
        }
    
    def setup_logging(self):
        """Setup logging for stress testing."""
        log_file = self.config.output_dir / "stress_test.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'cpu_count': multiprocessing.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'platform': sys.platform,
            'python_version': sys.version,
            'pid': os.getpid()
        }
    
    def run_single_iteration(self, iteration: int) -> Dict[str, Any]:
        """Run a single stress test iteration."""
        self.logger.info(f"Starting stress test iteration {iteration}")
        
        iteration_results = {
            'iteration': iteration,
            'start_time': time.time(),
            'tests_run': [],
            'anomalies_injected': [],
            'system_stress': [],
            'success': True,
            'errors': []
        }
        
        try:
            # Inject system anomalies
            if self.config.anomaly_injection_enabled and self.anomaly_injector.should_inject_anomaly():
                anomaly = self.anomaly_injector.inject_system_anomaly()
                iteration_results['anomalies_injected'].append(anomaly)
                self.test_results['anomalies_injected'].append(anomaly)
            
            # Apply memory stress
            if self.config.memory_pressure_enabled and random.random() < self.config.memory_pressure_probability:
                self.logger.info(f"Iteration {iteration}: Applying memory stress")
                self.memory_stressor.start_memory_stress()
                iteration_results['system_stress'].append({
                    'type': 'memory_pressure',
                    'start_time': time.time()
                })
                time.sleep(5)  # Apply stress for 5 seconds
                self.memory_stressor.stop_memory_stress()
            
            # Apply CPU stress
            if self.config.cpu_pressure_enabled and random.random() < self.config.cpu_pressure_probability:
                self.logger.info(f"Iteration {iteration}: Applying CPU stress")
                self.cpu_stressor.start_cpu_stress()
                iteration_results['system_stress'].append({
                    'type': 'cpu_pressure',
                    'start_time': time.time()
                })
                time.sleep(5)  # Apply stress for 5 seconds
                self.cpu_stressor.stop_cpu_stress()
            
            # Run comprehensive tests if available
            if self.config.run_comprehensive_tests and COMPREHENSIVE_TEST_AVAILABLE:
                self.logger.info(f"Iteration {iteration}: Running comprehensive tests")
                test_config = TestConfig(
                    num_iterations=2,  # Reduced for stress testing
                    enable_anomaly_injection=True,
                    anomaly_probability=0.3,  # Higher probability
                    output_dir=self.config.output_dir / f"iteration_{iteration}_comprehensive"
                )
                
                tester = ComprehensiveTester(test_config)
                comprehensive_results = tester.run_comprehensive_test()
                iteration_results['tests_run'].append({
                    'type': 'comprehensive',
                    'success': comprehensive_results['overall_success'],
                    'duration': comprehensive_results['total_duration']
                })
            
            # Run function validation
            if self.config.run_function_validation:
                self.logger.info(f"Iteration {iteration}: Running function validation")
                validator = RLDKFunctionValidator(self.config.output_dir / f"iteration_{iteration}_validation")
                validation_results = validator.run_comprehensive_validation()
                iteration_results['tests_run'].append({
                    'type': 'function_validation',
                    'success': validation_results['overall_success'],
                    'duration': validation_results['total_duration']
                })
            
            # Run memory stress tests
            if self.config.run_memory_stress_tests:
                self.logger.info(f"Iteration {iteration}: Running memory stress tests")
                memory_test_results = self._run_memory_stress_tests()
                iteration_results['tests_run'].append({
                    'type': 'memory_stress',
                    'success': memory_test_results['success'],
                    'duration': memory_test_results['duration']
                })
            
            # Run concurrent tests
            if self.config.run_concurrent_tests:
                self.logger.info(f"Iteration {iteration}: Running concurrent tests")
                concurrent_test_results = self._run_concurrent_tests()
                iteration_results['tests_run'].append({
                    'type': 'concurrent',
                    'success': concurrent_test_results['success'],
                    'duration': concurrent_test_results['duration']
                })
            
            # Run failure recovery tests
            if self.config.run_failure_recovery_tests:
                self.logger.info(f"Iteration {iteration}: Running failure recovery tests")
                recovery_test_results = self._run_failure_recovery_tests()
                iteration_results['tests_run'].append({
                    'type': 'failure_recovery',
                    'success': recovery_test_results['success'],
                    'duration': recovery_test_results['duration']
                })
            
        except Exception as e:
            self.logger.error(f"Error in iteration {iteration}: {e}")
            iteration_results['success'] = False
            iteration_results['errors'].append(str(e))
            self.test_results['errors'].append(f"Iteration {iteration}: {e}")
        
        iteration_results['end_time'] = time.time()
        iteration_results['duration'] = iteration_results['end_time'] - iteration_results['start_time']
        
        self.logger.info(f"Completed stress test iteration {iteration} in {iteration_results['duration']:.2f} seconds")
        
        return iteration_results
    
    def _run_memory_stress_tests(self) -> Dict[str, Any]:
        """Run memory stress tests."""
        start_time = time.time()
        
        try:
            # Test memory allocation and deallocation
            large_objects = []
            for i in range(10):
                # Allocate large objects
                obj = bytearray(10 * 1024 * 1024)  # 10MB
                large_objects.append(obj)
                
                # Force garbage collection periodically
                if i % 3 == 0:
                    gc.collect()
            
            # Test memory pressure handling
            memory_before = psutil.virtual_memory().percent
            
            # Create memory pressure
            pressure_objects = []
            try:
                while psutil.virtual_memory().percent < 90:  # Stop at 90% memory usage
                    obj = bytearray(50 * 1024 * 1024)  # 50MB
                    pressure_objects.append(obj)
            except MemoryError:
                pass  # Expected when out of memory
            
            memory_after = psutil.virtual_memory().percent
            
            # Clean up
            large_objects.clear()
            pressure_objects.clear()
            gc.collect()
            
            return {
                'success': True,
                'duration': time.time() - start_time,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_handled': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'duration': time.time() - start_time,
                'error': str(e)
            }
    
    def _run_concurrent_tests(self) -> Dict[str, Any]:
        """Run concurrent execution tests."""
        start_time = time.time()
        
        try:
            # Test concurrent file operations
            def file_worker(worker_id: int, num_files: int):
                for i in range(num_files):
                    temp_file = self.config.output_dir / f"concurrent_test_{worker_id}_{i}.tmp"
                    with open(temp_file, 'w') as f:
                        f.write(f"Worker {worker_id} file {i}")
                    time.sleep(0.01)  # Small delay
            
            # Start multiple threads
            threads = []
            for i in range(4):
                thread = threading.Thread(target=file_worker, args=(i, 5))
                thread.start()
                threads.append(thread)
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Clean up test files
            for temp_file in self.config.output_dir.glob("concurrent_test_*.tmp"):
                try:
                    temp_file.unlink()
                except:
                    pass
            
            return {
                'success': True,
                'duration': time.time() - start_time,
                'threads_completed': len(threads)
            }
            
        except Exception as e:
            return {
                'success': False,
                'duration': time.time() - start_time,
                'error': str(e)
            }
    
    def _run_failure_recovery_tests(self) -> Dict[str, Any]:
        """Run failure recovery tests."""
        start_time = time.time()
        
        try:
            # Test handling of various failure scenarios
            failures_handled = 0
            total_failures = 5
            
            # Test 1: File not found
            try:
                with open('/nonexistent/file.txt', 'r') as f:
                    f.read()
            except FileNotFoundError:
                failures_handled += 1
            
            # Test 2: Permission denied
            try:
                with open('/root/test.txt', 'w') as f:
                    f.write('test')
            except PermissionError:
                failures_handled += 1
            
            # Test 3: Memory error
            try:
                large_list = []
                while True:
                    large_list.append(bytearray(100 * 1024 * 1024))  # 100MB
            except MemoryError:
                failures_handled += 1
                large_list.clear()
                gc.collect()
            
            # Test 4: Division by zero
            try:
                result = 1 / 0
            except ZeroDivisionError:
                failures_handled += 1
            
            # Test 5: Key error
            try:
                d = {}
                value = d['nonexistent_key']
            except KeyError:
                failures_handled += 1
            
            return {
                'success': failures_handled == total_failures,
                'duration': time.time() - start_time,
                'failures_handled': failures_handled,
                'total_failures': total_failures
            }
            
        except Exception as e:
            return {
                'success': False,
                'duration': time.time() - start_time,
                'error': str(e)
            }
    
    def run_stress_tests(self) -> Dict[str, Any]:
        """Run all stress tests."""
        self.logger.info(f"Starting stress tests with {self.config.num_iterations} iterations")
        
        start_time = time.time()
        
        # Run iterations
        for i in range(self.config.num_iterations):
            self.logger.info(f"Running iteration {i + 1}/{self.config.num_iterations}")
            
            iteration_results = self.run_single_iteration(i)
            self.test_results['iterations'].append(iteration_results)
            
            # Check if iteration failed
            if not iteration_results['success']:
                self.test_results['overall_success'] = False
            
            # Force garbage collection between iterations
            gc.collect()
            
            # Small delay between iterations
            time.sleep(1)
        
        self.test_results['end_time'] = time.time()
        self.test_results['total_duration'] = self.test_results['end_time'] - start_time
        
        # Analyze results
        self._analyze_results()
        
        # Save results
        results_file = self.config.output_dir / "stress_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        self.logger.info(f"Stress tests completed. Results saved to {results_file}")
        
        return self.test_results
    
    def _analyze_results(self):
        """Analyze stress test results."""
        # Count successful iterations
        successful_iterations = sum(1 for iteration in self.test_results['iterations'] if iteration['success'])
        total_iterations = len(self.test_results['iterations'])
        
        # Count test types
        test_type_counts = {}
        for iteration in self.test_results['iterations']:
            for test in iteration['tests_run']:
                test_type = test['type']
                if test_type not in test_type_counts:
                    test_type_counts[test_type] = {'total': 0, 'successful': 0}
                test_type_counts[test_type]['total'] += 1
                if test['success']:
                    test_type_counts[test_type]['successful'] += 1
        
        # Count anomalies
        total_anomalies = len(self.test_results['anomalies_injected'])
        successful_anomalies = sum(1 for anomaly in self.test_results['anomalies_injected'] if anomaly.get('success', False))
        
        self.test_results['analysis'] = {
            'successful_iterations': successful_iterations,
            'total_iterations': total_iterations,
            'success_rate': successful_iterations / total_iterations if total_iterations > 0 else 0,
            'test_type_counts': test_type_counts,
            'total_anomalies': total_anomalies,
            'successful_anomalies': successful_anomalies,
            'anomaly_success_rate': successful_anomalies / total_anomalies if total_anomalies > 0 else 0
        }
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of the stress test results."""
        report = []
        report.append("=" * 80)
        report.append("STRESS TEST SUMMARY REPORT")
        report.append("=" * 80)
        
        # Overall results
        report.append(f"Overall Success: {'✅ PASSED' if self.test_results['overall_success'] else '❌ FAILED'}")
        report.append(f"Total Duration: {self.test_results['total_duration']:.2f} seconds")
        report.append(f"Total Iterations: {self.test_results['analysis']['total_iterations']}")
        report.append(f"Successful Iterations: {self.test_results['analysis']['successful_iterations']}")
        report.append(f"Success Rate: {self.test_results['analysis']['success_rate']:.2%}")
        report.append("")
        
        # Test type analysis
        report.append("TEST TYPE ANALYSIS:")
        report.append("-" * 40)
        for test_type, counts in self.test_results['analysis']['test_type_counts'].items():
            success_rate = counts['successful'] / counts['total'] if counts['total'] > 0 else 0
            report.append(f"{test_type.upper()}: {counts['successful']}/{counts['total']} ({success_rate:.2%})")
        report.append("")
        
        # Anomaly analysis
        report.append("ANOMALY ANALYSIS:")
        report.append("-" * 40)
        report.append(f"Total Anomalies Injected: {self.test_results['analysis']['total_anomalies']}")
        report.append(f"Successful Anomalies: {self.test_results['analysis']['successful_anomalies']}")
        report.append(f"Anomaly Success Rate: {self.test_results['analysis']['anomaly_success_rate']:.2%}")
        report.append("")
        
        # Error summary
        if self.test_results['errors']:
            report.append("ERRORS ENCOUNTERED:")
            report.append("-" * 40)
            for error in self.test_results['errors']:
                report.append(f"❌ {error}")
            report.append("")
        
        # System info
        report.append("SYSTEM INFORMATION:")
        report.append("-" * 40)
        for key, value in self.test_results['system_info'].items():
            report.append(f"{key}: {value}")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main function to run stress tests."""
    print("🔥 Starting RLDK Stress Tests")
    print("=" * 60)
    
    # Create stress test configuration
    config = StressTestConfig()
    
    # Create stress test runner
    runner = StressTestRunner(config)
    
    # Run stress tests
    results = runner.run_stress_tests()
    
    # Generate and print summary
    summary = runner.generate_summary_report()
    print(summary)
    
    # Save summary to file
    summary_file = config.output_dir / "stress_test_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    print(f"\n📁 Detailed results saved to: {config.output_dir}")
    print(f"📄 Summary report saved to: {summary_file}")
    
    return results


if __name__ == "__main__":
    main()