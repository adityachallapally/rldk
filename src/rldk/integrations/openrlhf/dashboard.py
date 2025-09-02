"""Real-time dashboard for OpenRLHF training monitoring."""

import time
import threading
import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import asdict
import warnings

import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, send_from_directory
import plotly.graph_objs as go
import plotly.utils

from .callbacks import OpenRLHFMetrics, OpenRLHFCallback
from .monitors import OpenRLHFTrainingMonitor, OpenRLHFCheckpointMonitor, OpenRLHFResourceMonitor
from .distributed import DistributedMetricsCollector, MultiNodeMonitor


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
        
        # Monitoring components
        self.training_monitor = OpenRLHFTrainingMonitor(output_dir=self.output_dir)
        self.resource_monitor = OpenRLHFResourceMonitor()
        self.distributed_collector = DistributedMetricsCollector()
        
        # Dashboard refresh thread
        self.refresh_thread = None
        self.dashboard_active = False
        
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
                warnings.warn(f"Error in dashboard refresh loop: {e}")
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
            with open(latest_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        metrics_data.append(data)
                    except json.JSONDecodeError:
                        continue
            
            # Keep only recent data (last 1000 points)
            self.metrics_data = metrics_data[-1000:]
            
        except Exception as e:
            warnings.warn(f"Failed to load metrics data: {e}")
    
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
            warnings.warn(f"Failed to load health data: {e}")
    
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
            warnings.warn(f"Failed to load resource data: {e}")
    
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
            warnings.warn(f"Failed to load distributed data: {e}")
    
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
            line=dict(color='blue', width=2)
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
            line=dict(color='green', width=2)
        ))
        
        # Add reward std if available
        if 'reward_std' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['step'],
                y=df['reward_mean'] + df['reward_std'],
                mode='lines',
                name='Reward +1σ',
                line=dict(color='green', width=1, dash='dash'),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=df['step'],
                y=df['reward_mean'] - df['reward_std'],
                mode='lines',
                name='Reward -1σ',
                line=dict(color='green', width=1, dash='dash'),
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
            line=dict(color='red', width=2)
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
                line=dict(color='blue', width=2)
            ))
        
        # GPU memory usage
        if 'avg_gpu_memory_usage' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['avg_gpu_memory_usage'],
                mode='lines',
                name='GPU Memory Usage (GB)',
                line=dict(color='green', width=2),
                yaxis='y2'
            ))
        
        fig.update_layout(
            title='Resource Usage Over Time',
            xaxis_title='Time',
            yaxis=dict(title='CPU Utilization (%)', side='left'),
            yaxis2=dict(title='GPU Memory Usage (GB)', side='right', overlaying='y'),
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
                line=dict(color='purple', width=3)
            ))
        
        # Stability score
        if 'stability_score' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['stability_score'],
                mode='lines',
                name='Stability Score',
                line=dict(color='blue', width=2)
            ))
        
        # Convergence rate
        if 'convergence_rate' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['convergence_rate'],
                mode='lines',
                name='Convergence Rate',
                line=dict(color='green', width=2)
            ))
        
        fig.update_layout(
            title='Training Health Over Time',
            xaxis_title='Time',
            yaxis_title='Health Score',
            yaxis=dict(range=[0, 1]),
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
    
    def add_metrics(self, metrics: OpenRLHFMetrics):
        """Add new metrics to the dashboard."""
        self.training_monitor.add_metrics(metrics)
    
    def get_dashboard_url(self) -> str:
        """Get the dashboard URL."""
        return f"http://{self.host}:{self.port}"