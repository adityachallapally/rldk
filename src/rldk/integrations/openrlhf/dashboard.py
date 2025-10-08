"""Real-time dashboard for OpenRLHF training monitoring."""

import json
import threading
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import plotly.graph_objs as go
import plotly.utils
from flask import Flask, jsonify, render_template

from .callbacks import OpenRLHFMetrics
from .distributed import DistributedMetricsCollector
from .monitors import (
    OpenRLHFResourceMonitor,
    OpenRLHFTrainingMonitor,
)


class OpenRLHFDashboard:
    """Real-time dashboard for OpenRLHF training monitoring."""

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        port: int = 5000,
        host: str = "localhost",
        enable_auto_refresh: bool = True,
        refresh_interval: float = 1.0,
    ):
        """Initialize OpenRLHF dashboard.

        Args:
            output_dir: Directory containing training logs and metrics
            port: Port for the dashboard server
            host: Host for the dashboard server
            enable_auto_refresh: Whether to automatically refresh data
            refresh_interval: Interval between refreshes (seconds)
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./rldk_logs")
        self.port = port
        self.host = host
        self.enable_auto_refresh = enable_auto_refresh
        self.refresh_interval = refresh_interval

        # Initialize Flask app
        self.app = Flask(__name__,
                        template_folder=self._get_template_folder(),
                        static_folder=self._get_static_folder())

        # Dashboard state
        self.metrics_data: List[Dict[str, Any]] = []
        self.health_data: List[Dict[str, Any]] = []
        self.resource_data: List[Dict[str, Any]] = []
        self.distributed_data: List[Dict[str, Any]] = []
        self.network_data: List[Dict[str, Any]] = []  # New network metrics data

        # Monitoring components
        self.training_monitor = OpenRLHFTrainingMonitor(output_dir=self.output_dir)
        self.resource_monitor = OpenRLHFResourceMonitor()
        self.distributed_collector = DistributedMetricsCollector()

        # Dashboard refresh thread
        self.refresh_thread = None
        self.dashboard_active = False

        # Network monitoring
        self.network_monitor = None
        self.network_alerts: List[Dict[str, Any]] = []

        # Setup routes
        self._setup_routes()

    def _get_template_folder(self) -> str:
        """Get template folder path."""
        return str(Path(__file__).parent / "templates")

    def _get_static_folder(self) -> str:
        """Get static folder path."""
        return str(Path(__file__).parent / "static")

    def _setup_routes(self):
        """Setup Flask routes for the dashboard."""

        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return render_template('dashboard.html')

        @self.app.route('/api/metrics')
        def get_metrics():
            """Get training metrics data."""
            return jsonify(self.metrics_data)

        @self.app.route('/api/health')
        def get_health():
            """Get training health data."""
            return jsonify(self.health_data)

        @self.app.route('/api/resources')
        def get_resources():
            """Get resource usage data."""
            return jsonify(self.resource_data)

        @self.app.route('/api/network')
        def get_network():
            """Get network metrics data."""
            return jsonify(self.network_data)

        @self.app.route('/api/network/alerts')
        def get_network_alerts():
            """Get network alerts."""
            return jsonify(self.network_alerts)

        @self.app.route('/api/network/health')
        def get_network_health():
            """Get network health report."""
            if self.network_monitor:
                return jsonify(self.network_monitor.get_network_health_report())
            else:
                return jsonify({'error': 'Network monitor not initialized'})

        @self.app.route('/api/distributed')
        def get_distributed():
            """Get distributed training data."""
            return jsonify(self.distributed_data)

        @self.app.route('/api/summary')
        def get_summary():
            """Get training summary."""
            summary = self._get_training_summary()
            return jsonify(summary)

        @self.app.route('/api/plots/<plot_type>')
        def get_plot(plot_type):
            """Get plot data for specific plot type."""
            plot_data = self._generate_plot(plot_type)
            return jsonify(plot_data)

        @self.app.route('/api/alerts')
        def get_alerts():
            """Get active alerts."""
            alerts = self._get_active_alerts()
            return jsonify(alerts)

        @self.app.route('/api/export/<data_type>')
        def export_data(data_type):
            """Export data in various formats."""
            return self._export_data(data_type)

    def start_dashboard(self):
        """Start the dashboard server."""
        if self.dashboard_active:
            return

        self.dashboard_active = True

        # Start data refresh thread
        if self.enable_auto_refresh:
            self.refresh_thread = threading.Thread(
                target=self._refresh_loop,
                daemon=True
            )
            self.refresh_thread.start()

        # Start resource monitoring
        self.resource_monitor.start_monitoring()

        # Start distributed monitoring
        self.distributed_collector.start_collection()

        print(f"🚀 OpenRLHF Dashboard starting at http://{self.host}:{self.port}")
        print(f"📊 Monitoring training logs in: {self.output_dir}")

        # Start Flask app
        self.app.run(host=self.host, port=self.port, debug=False, use_reloader=False)

    def stop_dashboard(self):
        """Stop the dashboard server."""
        self.dashboard_active = False

        # Stop monitoring
        self.resource_monitor.stop_monitoring()
        self.distributed_collector.stop_collection()

        if self.refresh_thread:
            self.refresh_thread.join(timeout=5.0)

    def _refresh_loop(self):
        """Main refresh loop for dashboard data."""
        while self.dashboard_active:
            try:
                # Load latest metrics
                self._load_metrics_data()
                self._load_health_data()
                self._load_resource_data()
                self._load_distributed_data()

                time.sleep(self.refresh_interval)

            except Exception as e:
                warnings.warn(f"Error in dashboard refresh loop: {e}", stacklevel=2)
                time.sleep(self.refresh_interval)

    def _load_metrics_data(self):
        """Load training metrics data."""
        try:
            # Look for metrics files
            metrics_files = list(self.output_dir.glob("metrics_*.jsonl"))
            if not metrics_files:
                return

            # Load latest metrics file
            latest_file = max(metrics_files, key=lambda f: f.stat().st_mtime)

            metrics_data = []
            with open(latest_file) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        metrics_data.append(data)
                    except json.JSONDecodeError:
                        continue

            # Keep only recent data (last 1000 points)
            self.metrics_data = metrics_data[-1000:]

        except Exception as e:
            warnings.warn(f"Failed to load metrics data: {e}", stacklevel=2)

    def _load_health_data(self):
        """Load training health data."""
        try:
            # Get health summary from training monitor
            health_summary = self.training_monitor.get_health_summary()
            if health_summary:
                health_summary['timestamp'] = time.time()
                self.health_data.append(health_summary)

                # Keep only recent data
                if len(self.health_data) > 1000:
                    self.health_data = self.health_data[-1000:]

        except Exception as e:
            warnings.warn(f"Failed to load health data: {e}", stacklevel=2)

    def _load_resource_data(self):
        """Load resource usage data."""
        try:
            # Get resource summary
            resource_summary = self.resource_monitor.get_resource_summary()
            if resource_summary:
                resource_summary['timestamp'] = time.time()
                self.resource_data.append(resource_summary)

                # Keep only recent data
                if len(self.resource_data) > 1000:
                    self.resource_data = self.resource_data[-1000:]

        except Exception as e:
            warnings.warn(f"Failed to load resource data: {e}", stacklevel=2)

    def _load_distributed_data(self):
        """Load distributed training data."""
        try:
            # Get latest distributed metrics
            latest_metrics = self.distributed_collector.get_latest_metrics()
            if latest_metrics:
                metrics_dict = asdict(latest_metrics)
                self.distributed_data.append(metrics_dict)

                # Keep only recent data
                if len(self.distributed_data) > 1000:
                    self.distributed_data = self.distributed_data[-1000:]

        except Exception as e:
            warnings.warn(f"Failed to load distributed data: {e}", stacklevel=2)

    def _get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        summary = {
            'timestamp': time.time(),
            'total_metrics': len(self.metrics_data),
            'total_health_points': len(self.health_data),
            'total_resource_points': len(self.resource_data),
            'total_distributed_points': len(self.distributed_data),
        }

        # Add latest metrics if available
        if self.metrics_data:
            latest_metrics = self.metrics_data[-1]
            summary.update({
                'latest_step': latest_metrics.get('step', 0),
                'latest_loss': latest_metrics.get('loss', 0.0),
                'latest_reward': latest_metrics.get('reward_mean', 0.0),
                'latest_kl': latest_metrics.get('kl_mean', 0.0),
                'latest_lr': latest_metrics.get('learning_rate', 0.0),
            })

        # Add latest health if available
        if self.health_data:
            latest_health = self.health_data[-1]
            summary.update({
                'stability_score': latest_health.get('stability_score', 0.0),
                'convergence_rate': latest_health.get('convergence_rate', 0.0),
                'overall_health': latest_health.get('overall_health', 0.0),
                'anomaly_score': latest_health.get('anomaly_score', 0.0),
            })

        # Add resource summary if available
        if self.resource_data:
            latest_resources = self.resource_data[-1]
            summary.update({
                'avg_cpu_utilization': latest_resources.get('avg_cpu_utilization', 0.0),
                'max_cpu_utilization': latest_resources.get('max_cpu_utilization', 0.0),
                'avg_gpu_memory_usage': latest_resources.get('avg_gpu_memory_usage', 0.0),
                'max_gpu_memory_usage': latest_resources.get('max_gpu_memory_usage', 0.0),
            })

        return summary

    def _generate_plot(self, plot_type: str) -> Dict[str, Any]:
        """Generate plot data for specific plot type."""
        if not self.metrics_data:
            return {'error': 'No data available'}

        df = pd.DataFrame(self.metrics_data)

        if plot_type == 'loss':
            return self._create_loss_plot(df)
        elif plot_type == 'reward':
            return self._create_reward_plot(df)
        elif plot_type == 'kl':
            return self._create_kl_plot(df)
        elif plot_type == 'resources':
            return self._create_resources_plot()
        elif plot_type == 'health':
            return self._create_health_plot()
        else:
            return {'error': f'Unknown plot type: {plot_type}'}

    def _create_loss_plot(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create loss plot."""
        if 'step' not in df.columns or 'loss' not in df.columns:
            return {'error': 'Missing required columns for loss plot'}

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['step'],
            y=df['loss'],
            mode='lines',
            name='Training Loss',
            line={"color": 'blue', "width": 2}
        ))

        fig.update_layout(
            title='Training Loss Over Time',
            xaxis_title='Step',
            yaxis_title='Loss',
            hovermode='x unified'
        )

        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def _create_reward_plot(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create reward plot."""
        if 'step' not in df.columns or 'reward_mean' not in df.columns:
            return {'error': 'Missing required columns for reward plot'}

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['step'],
            y=df['reward_mean'],
            mode='lines',
            name='Reward Mean',
            line={"color": 'green', "width": 2}
        ))

        # Add reward std if available
        if 'reward_std' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['step'],
                y=df['reward_mean'] + df['reward_std'],
                mode='lines',
                name='Reward +1σ',
                line={"color": 'green', "width": 1, "dash": 'dash'},
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=df['step'],
                y=df['reward_mean'] - df['reward_std'],
                mode='lines',
                name='Reward -1σ',
                line={"color": 'green', "width": 1, "dash": 'dash'},
                fill='tonexty',
                fillcolor='rgba(0,255,0,0.1)',
                showlegend=False
            ))

        fig.update_layout(
            title='Reward Over Time',
            xaxis_title='Step',
            yaxis_title='Reward',
            hovermode='x unified'
        )

        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def _create_kl_plot(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create KL divergence plot."""
        if 'step' not in df.columns or 'kl_mean' not in df.columns:
            return {'error': 'Missing required columns for KL plot'}

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['step'],
            y=df['kl_mean'],
            mode='lines',
            name='KL Divergence',
            line={"color": 'red', "width": 2}
        ))

        fig.update_layout(
            title='KL Divergence Over Time',
            xaxis_title='Step',
            yaxis_title='KL Divergence',
            hovermode='x unified'
        )

        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def _create_resources_plot(self) -> Dict[str, Any]:
        """Create resource usage plot."""
        if not self.resource_data:
            return {'error': 'No resource data available'}

        df = pd.DataFrame(self.resource_data)

        fig = go.Figure()

        # CPU utilization
        if 'avg_cpu_utilization' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['avg_cpu_utilization'],
                mode='lines',
                name='CPU Utilization (%)',
                line={"color": 'blue', "width": 2}
            ))

        # GPU memory usage
        if 'avg_gpu_memory_usage' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['avg_gpu_memory_usage'],
                mode='lines',
                name='GPU Memory Usage (GB)',
                line={"color": 'green', "width": 2},
                yaxis='y2'
            ))

        fig.update_layout(
            title='Resource Usage Over Time',
            xaxis_title='Time',
            yaxis={"title": 'CPU Utilization (%)', "side": 'left'},
            yaxis2={"title": 'GPU Memory Usage (GB)', "side": 'right', "overlaying": 'y'},
            hovermode='x unified'
        )

        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def _create_health_plot(self) -> Dict[str, Any]:
        """Create training health plot."""
        if not self.health_data:
            return {'error': 'No health data available'}

        df = pd.DataFrame(self.health_data)

        fig = go.Figure()

        # Overall health
        if 'overall_health' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['overall_health'],
                mode='lines',
                name='Overall Health',
                line={"color": 'purple', "width": 3}
            ))

        # Stability score
        if 'stability_score' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['stability_score'],
                mode='lines',
                name='Stability Score',
                line={"color": 'blue', "width": 2}
            ))

        # Convergence rate
        if 'convergence_rate' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['convergence_rate'],
                mode='lines',
                name='Convergence Rate',
                line={"color": 'green', "width": 2}
            ))

        fig.update_layout(
            title='Training Health Over Time',
            xaxis_title='Time',
            yaxis_title='Health Score',
            yaxis={"range": [0, 1]},
            hovermode='x unified'
        )

        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts."""
        alerts = []

        # Check for health alerts
        if self.health_data:
            latest_health = self.health_data[-1]

            if latest_health.get('anomaly_score', 0) > 0.5:
                alerts.append({
                    'type': 'anomaly',
                    'severity': 'high',
                    'message': f'High anomaly score detected: {latest_health["anomaly_score"]:.3f}',
                    'timestamp': time.time()
                })

            if latest_health.get('stability_score', 1) < 0.5:
                alerts.append({
                    'type': 'stability',
                    'severity': 'medium',
                    'message': f'Low stability score: {latest_health["stability_score"]:.3f}',
                    'timestamp': time.time()
                })

        # Check for resource alerts
        if self.resource_data:
            latest_resources = self.resource_data[-1]

            if latest_resources.get('max_cpu_utilization', 0) > 90:
                alerts.append({
                    'type': 'resource',
                    'severity': 'high',
                    'message': f'High CPU utilization: {latest_resources["max_cpu_utilization"]:.1f}%',
                    'timestamp': time.time()
                })

            if latest_resources.get('max_gpu_memory_usage', 0) > 20:  # 20GB threshold
                alerts.append({
                    'type': 'resource',
                    'severity': 'medium',
                    'message': f'High GPU memory usage: {latest_resources["max_gpu_memory_usage"]:.1f}GB',
                    'timestamp': time.time()
                })

        return alerts

    def _export_data(self, data_type: str):
        """Export data in various formats."""
        if data_type == 'metrics':
            if not self.metrics_data:
                return jsonify({'error': 'No metrics data available'})

            df = pd.DataFrame(self.metrics_data)
            csv_data = df.to_csv(index=False)

            return csv_data, 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': 'attachment; filename=training_metrics.csv'
            }

        elif data_type == 'health':
            if not self.health_data:
                return jsonify({'error': 'No health data available'})

            df = pd.DataFrame(self.health_data)
            csv_data = df.to_csv(index=False)

            return csv_data, 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': 'attachment; filename=training_health.csv'
            }

        elif data_type == 'resources':
            if not self.resource_data:
                return jsonify({'error': 'No resource data available'})

            df = pd.DataFrame(self.resource_data)
            csv_data = df.to_csv(index=False)

            return csv_data, 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': 'attachment; filename=resource_usage.csv'
            }

        else:
            return jsonify({'error': f'Unknown data type: {data_type}'})

    def update_data(self):
        """Reload metrics from disk or in-memory buffer."""
        try:
            # Load metrics from JSONL files
            metrics_file = self.output_dir / "metrics.jsonl"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    lines = f.readlines()
                    self.metrics_data = [json.loads(line.strip()) for line in lines if line.strip()]

            # Load network metrics
            network_file = self.output_dir / "network_metrics.jsonl"
            if network_file.exists():
                with open(network_file) as f:
                    lines = f.readlines()
                    self.network_data = [json.loads(line.strip()) for line in lines if line.strip()]

            # Load health data
            health_file = self.output_dir / "health.jsonl"
            if health_file.exists():
                with open(health_file) as f:
                    lines = f.readlines()
                    self.health_data = [json.loads(line.strip()) for line in lines if line.strip()]

            # Load resource data
            resource_file = self.output_dir / "resources.jsonl"
            if resource_file.exists():
                with open(resource_file) as f:
                    lines = f.readlines()
                    self.resource_data = [json.loads(line.strip()) for line in lines if line.strip()]

            # Load distributed data
            distributed_file = self.output_dir / "distributed.jsonl"
            if distributed_file.exists():
                with open(distributed_file) as f:
                    lines = f.readlines()
                    self.distributed_data = [json.loads(line.strip()) for line in lines if line.strip()]

        except Exception as e:
            print(f"Error updating dashboard data: {e}")

    def add_metrics(self, metrics: Union[Dict[str, Any], 'OpenRLHFMetrics']):
        """Append new metrics entry and trigger refresh."""
        try:
            # Convert dataclass to dictionary if needed
            if hasattr(metrics, 'to_dict'):
                metrics_dict = metrics.to_dict()
            else:
                metrics_dict = metrics

            # Add timestamp if not present
            if 'timestamp' not in metrics_dict:
                metrics_dict['timestamp'] = time.time()

            # Append to appropriate data list
            if 'bandwidth_mbps' in metrics_dict or 'latency_ms' in metrics_dict:
                self.network_data.append(metrics_dict)
                # Keep only last 1000 entries
                if len(self.network_data) > 1000:
                    self.network_data = self.network_data[-1000:]
            else:
                self.metrics_data.append(metrics_dict)
                # Keep only last 1000 entries
                if len(self.metrics_data) > 1000:
                    self.metrics_data = self.metrics_data[-1000:]

            # Write to JSONL file
            self._write_metrics_to_file(metrics_dict)

            # Trigger refresh if using Streamlit
            try:
                import streamlit as st
                st.experimental_rerun()
            except ImportError:
                pass  # Not using Streamlit

        except Exception as e:
            print(f"Error adding metrics: {e}")

    def add_alert(self, alert_type: str, message: str, severity: str = "warning",
                  threshold: Optional[float] = None, current_value: Optional[float] = None):
        """Record network alerts and surface them in the UI."""
        try:
            alert = {
                'timestamp': time.time(),
                'type': alert_type,
                'message': message,
                'severity': severity,
                'threshold': threshold,
                'current_value': current_value
            }

            if alert_type == 'network':
                self.network_alerts.append(alert)
                # Keep only last 100 alerts
                if len(self.network_alerts) > 100:
                    self.network_alerts = self.network_alerts[-100:]

            # Write alert to file
            alert_file = self.output_dir / "alerts.jsonl"
            with open(alert_file, 'a') as f:
                f.write(json.dumps(alert) + '\n')

        except Exception as e:
            print(f"Error adding alert: {e}")

    def _write_metrics_to_file(self, metrics: Dict[str, Any]):
        """Write metrics to appropriate JSONL file."""
        try:
            # Determine file based on metrics type
            if 'bandwidth_mbps' in metrics or 'latency_ms' in metrics:
                filename = "network_metrics.jsonl"
            elif 'gpu_memory_used' in metrics or 'cpu_utilization' in metrics:
                filename = "resources.jsonl"
            elif 'world_size' in metrics and metrics['world_size'] > 1:
                filename = "distributed.jsonl"
            else:
                filename = "metrics.jsonl"

            filepath = self.output_dir / filename
            with open(filepath, 'a') as f:
                f.write(json.dumps(metrics) + '\n')

        except Exception as e:
            print(f"Error writing metrics to file: {e}")

    def initialize_network_monitoring(self):
        """Initialize network monitoring component."""
        try:
            from .network_monitor import RealNetworkMonitor
            self.network_monitor = RealNetworkMonitor(
                enable_distributed_monitoring=True,
                enable_distributed_measurements=False
            )
            print("Network monitoring initialized successfully")
        except Exception as e:
            print(f"Failed to initialize network monitoring: {e}")

    def check_network_thresholds(self, metrics: Union[Dict[str, Any], 'OpenRLHFMetrics']):
        """Check network metrics against thresholds and generate alerts."""
        if not self.network_monitor:
            return

        # Convert dataclass to dictionary if needed
        if hasattr(metrics, 'to_dict'):
            metrics_dict = metrics.to_dict()
        else:
            metrics_dict = metrics

        # Define thresholds
        thresholds = {
            'latency_ms': 100.0,  # High latency threshold
            'bandwidth_mbps': 10.0,  # Low bandwidth threshold
            'packet_loss_percent': 5.0,  # High packet loss threshold
        }

        # Check latency
        if 'latency_ms' in metrics_dict:
            latency = metrics_dict['latency_ms']
            if latency > thresholds['latency_ms']:
                self.add_alert(
                    'network',
                    f"High latency detected: {latency:.2f}ms",
                    'warning',
                    thresholds['latency_ms'],
                    latency
                )

        # Check bandwidth
        if 'bandwidth_mbps' in metrics_dict:
            bandwidth = metrics_dict['bandwidth_mbps']
            if bandwidth < thresholds['bandwidth_mbps']:
                self.add_alert(
                    'network',
                    f"Low bandwidth detected: {bandwidth:.2f} Mbps",
                    'warning',
                    thresholds['bandwidth_mbps'],
                    bandwidth
                )

        # Check packet loss
        if 'packet_loss_percent' in metrics_dict:
            packet_loss = metrics_dict['packet_loss_percent']
            if packet_loss > thresholds['packet_loss_percent']:
                self.add_alert(
                    'network',
                    f"High packet loss detected: {packet_loss:.2f}%",
                    'error',
                    thresholds['packet_loss_percent'],
                    packet_loss
                )

    def get_dashboard_url(self) -> str:
        """Get the dashboard URL."""
        return f"http://{self.host}:{self.port}"
