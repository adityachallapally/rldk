"""Real-time dashboard for TRL training monitoring."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import threading
import queue

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from .callbacks import RLDKCallback, RLDKMetrics
from .monitors import PPOMonitor, CheckpointMonitor


class RLDKDashboard:
    """Real-time dashboard for RLDK training monitoring."""
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        port: int = 8501,
        auto_refresh: bool = True,
        refresh_interval: int = 5,
        run_id: Optional[str] = None,
    ):
        """Initialize RLDK dashboard.
        
        Args:
            output_dir: Directory containing RLDK logs
            port: Port for Streamlit dashboard
            auto_refresh: Whether to auto-refresh the dashboard
            refresh_interval: Refresh interval in seconds
            run_id: Specific run ID to monitor
        """
        if not STREAMLIT_AVAILABLE:
            raise ImportError(
                "Streamlit and Plotly are required for RLDKDashboard. "
                "Install with: pip install streamlit plotly"
            )
        
        self.output_dir = Path(output_dir) if output_dir else Path("./rldk_logs")
        self.port = port
        self.auto_refresh = auto_refresh
        self.refresh_interval = refresh_interval
        self.run_id = run_id
        
        # Data storage
        self.metrics_data: List[Dict[str, Any]] = []
        self.alerts_data: List[Dict[str, Any]] = []
        self.ppo_data: List[Dict[str, Any]] = []
        self.checkpoint_data: List[Dict[str, Any]] = []
        
        # Dashboard state
        self.is_running = False
        self.dashboard_thread: Optional[threading.Thread] = None
        
        print(f"📊 RLDK Dashboard initialized - Port: {self.port}")
        print(f"📁 Monitoring directory: {self.output_dir}")
    
    def start_dashboard(self, blocking: bool = False):
        """Start the dashboard server."""
        if self.is_running:
            print("Dashboard is already running")
            return
        
        self.is_running = True
        
        if blocking:
            self._run_dashboard()
        else:
            self.dashboard_thread = threading.Thread(target=self._run_dashboard)
            self.dashboard_thread.daemon = True
            self.dashboard_thread.start()
            print(f"🚀 Dashboard started on http://localhost:{self.port}")
    
    def stop_dashboard(self):
        """Stop the dashboard server."""
        self.is_running = False
        if self.dashboard_thread:
            self.dashboard_thread.join(timeout=5)
        print("🛑 Dashboard stopped")
    
    def _run_dashboard(self):
        """Run the Streamlit dashboard."""
        import subprocess
        import sys
        
        # Create a temporary Streamlit app file
        app_file = self.output_dir / "dashboard_app.py"
        self._create_dashboard_app(app_file)
        
        # Run Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(app_file), "--server.port", str(self.port),
            "--server.headless", "true"
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Dashboard error: {e}")
        except KeyboardInterrupt:
            print("Dashboard interrupted")
        finally:
            self.is_running = False
    
    def _create_dashboard_app(self, app_file: Path):
        """Create the Streamlit dashboard app."""
        app_content = f'''
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
from pathlib import Path

# Configuration
OUTPUT_DIR = Path("{self.output_dir}")
RUN_ID = "{self.run_id}" if "{self.run_id}" else None
REFRESH_INTERVAL = {self.refresh_interval}

st.set_page_config(
    page_title="RLDK Training Monitor",
    page_icon="🎯",
    layout="wide"
)

def load_metrics_data():
    """Load metrics data from files."""
    metrics_data = []
    alerts_data = []
    ppo_data = []
    checkpoint_data = []
    
    # Find all metrics files
    if RUN_ID:
        metrics_files = list(OUTPUT_DIR.glob(f"{{RUN_ID}}_metrics.json"))
        alerts_files = list(OUTPUT_DIR.glob(f"{{RUN_ID}}_alerts.json"))
        ppo_files = list(OUTPUT_DIR.glob(f"{{RUN_ID}}_ppo_metrics.csv"))
        checkpoint_files = list(OUTPUT_DIR.glob(f"{{RUN_ID}}_checkpoint_summary.csv"))
    else:
        metrics_files = list(OUTPUT_DIR.glob("*_metrics.json"))
        alerts_files = list(OUTPUT_DIR.glob("*_alerts.json"))
        ppo_files = list(OUTPUT_DIR.glob("*_ppo_metrics.csv"))
        checkpoint_files = list(OUTPUT_DIR.glob("*_checkpoint_summary.csv"))
    
    # Load metrics
    for file in metrics_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    metrics_data.extend(data)
                else:
                    metrics_data.append(data)
        except Exception as e:
            st.error(f"Error loading {{file}}: {{e}}")
    
    # Load alerts
    for file in alerts_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    alerts_data.extend(data)
                else:
                    alerts_data.append(data)
        except Exception as e:
            st.error(f"Error loading {{file}}: {{e}}")
    
    # Load PPO data
    for file in ppo_files:
        try:
            df = pd.read_csv(file)
            ppo_data.extend(df.to_dict('records'))
        except Exception as e:
            st.error(f"Error loading {{file}}: {{e}}")
    
    # Load checkpoint data
    for file in checkpoint_files:
        try:
            df = pd.read_csv(file)
            checkpoint_data.extend(df.to_dict('records'))
        except Exception as e:
            st.error(f"Error loading {{file}}: {{e}}")
    
    return metrics_data, alerts_data, ppo_data, checkpoint_data

def create_metrics_plots(metrics_df):
    """Create metrics visualization plots."""
    if metrics_df.empty:
        return None, None, None
    
    # Training metrics plot
    fig_metrics = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Loss', 'Learning Rate', 'Reward', 'KL Divergence'),
        specs=[[{{"secondary_y": False}}, {{"secondary_y": False}}],
               [{{"secondary_y": False}}, {{"secondary_y": False}}]]
    )
    
    if 'loss' in metrics_df.columns:
        fig_metrics.add_trace(
            go.Scatter(x=metrics_df['step'], y=metrics_df['loss'], 
                      name='Loss', line=dict(color='blue')),
            row=1, col=1
        )
    
    if 'learning_rate' in metrics_df.columns:
        fig_metrics.add_trace(
            go.Scatter(x=metrics_df['step'], y=metrics_df['learning_rate'], 
                      name='Learning Rate', line=dict(color='green')),
            row=1, col=2
        )
    
    if 'reward_mean' in metrics_df.columns:
        fig_metrics.add_trace(
            go.Scatter(x=metrics_df['step'], y=metrics_df['reward_mean'], 
                      name='Reward', line=dict(color='orange')),
            row=2, col=1
        )
    
    if 'kl_mean' in metrics_df.columns:
        fig_metrics.add_trace(
            go.Scatter(x=metrics_df['step'], y=metrics_df['kl_mean'], 
                      name='KL Divergence', line=dict(color='red')),
            row=2, col=2
        )
    
    fig_metrics.update_layout(height=600, showlegend=False, title_text="Training Metrics")
    
    # Resource usage plot
    fig_resources = go.Figure()
    
    if 'gpu_memory_used' in metrics_df.columns:
        fig_resources.add_trace(
            go.Scatter(x=metrics_df['step'], y=metrics_df['gpu_memory_used'], 
                      name='GPU Memory (GB)', line=dict(color='purple'))
        )
    
    if 'cpu_memory_used' in metrics_df.columns:
        fig_resources.add_trace(
            go.Scatter(x=metrics_df['step'], y=metrics_df['cpu_memory_used'], 
                      name='CPU Memory (GB)', line=dict(color='brown'))
        )
    
    fig_resources.update_layout(title="Resource Usage", xaxis_title="Step", yaxis_title="Memory (GB)")
    
    # Training health plot
    fig_health = go.Figure()
    
    if 'training_stability_score' in metrics_df.columns:
        fig_health.add_trace(
            go.Scatter(x=metrics_df['step'], y=metrics_df['training_stability_score'], 
                      name='Training Stability', line=dict(color='green'))
        )
    
    if 'convergence_indicator' in metrics_df.columns:
        fig_health.add_trace(
            go.Scatter(x=metrics_df['step'], y=metrics_df['convergence_indicator'], 
                      name='Convergence Indicator', line=dict(color='blue'))
        )
    
    fig_health.update_layout(title="Training Health", xaxis_title="Step", yaxis_title="Score")
    
    return fig_metrics, fig_resources, fig_health

def create_ppo_plots(ppo_df):
    """Create PPO-specific plots."""
    if ppo_df.empty:
        return None, None
    
    # PPO metrics plot
    fig_ppo = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Reward Distribution', 'Policy KL', 'Entropy', 'Clip Fraction'),
        specs=[[{{"secondary_y": False}}, {{"secondary_y": False}}],
               [{{"secondary_y": False}}, {{"secondary_y": False}}]]
    )
    
    if 'rollout_reward_mean' in ppo_df.columns and 'rollout_reward_std' in ppo_df.columns:
        fig_ppo.add_trace(
            go.Scatter(x=ppo_df.index, y=ppo_df['rollout_reward_mean'], 
                      error_y=dict(type='data', array=ppo_df['rollout_reward_std']),
                      name='Reward', line=dict(color='orange')),
            row=1, col=1
        )
    
    if 'policy_kl_mean' in ppo_df.columns:
        fig_ppo.add_trace(
            go.Scatter(x=ppo_df.index, y=ppo_df['policy_kl_mean'], 
                      name='KL Divergence', line=dict(color='red')),
            row=1, col=2
        )
    
    if 'policy_entropy_mean' in ppo_df.columns:
        fig_ppo.add_trace(
            go.Scatter(x=ppo_df.index, y=ppo_df['policy_entropy_mean'], 
                      name='Entropy', line=dict(color='blue')),
            row=2, col=1
        )
    
    if 'policy_clip_frac' in ppo_df.columns:
        fig_ppo.add_trace(
            go.Scatter(x=ppo_df.index, y=ppo_df['policy_clip_frac'], 
                      name='Clip Fraction', line=dict(color='green')),
            row=2, col=2
        )
    
    fig_ppo.update_layout(height=600, showlegend=False, title_text="PPO Metrics")
    
    # PPO health plot
    fig_ppo_health = go.Figure()
    
    if 'policy_collapse_risk' in ppo_df.columns:
        fig_ppo_health.add_trace(
            go.Scatter(x=ppo_df.index, y=ppo_df['policy_collapse_risk'], 
                      name='Policy Collapse Risk', line=dict(color='red'))
        )
    
    if 'reward_hacking_risk' in ppo_df.columns:
        fig_ppo_health.add_trace(
            go.Scatter(x=ppo_df.index, y=ppo_df['reward_hacking_risk'], 
                      name='Reward Hacking Risk', line=dict(color='orange'))
        )
    
    if 'training_stability' in ppo_df.columns:
        fig_ppo_health.add_trace(
            go.Scatter(x=ppo_df.index, y=ppo_df['training_stability'], 
                      name='Training Stability', line=dict(color='green'))
        )
    
    fig_ppo_health.update_layout(title="PPO Health Indicators", xaxis_title="Step", yaxis_title="Risk Score")
    
    return fig_ppo, fig_ppo_health

# Main dashboard
st.title("🎯 RLDK Training Monitor")

# Auto-refresh
if st.checkbox("Auto-refresh", value=True):
    time.sleep(REFRESH_INTERVAL)
    st.rerun()

# Load data
metrics_data, alerts_data, ppo_data, checkpoint_data = load_metrics_data()

# Display summary
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Steps", len(metrics_data) if metrics_data else 0)

with col2:
    st.metric("Active Alerts", len(alerts_data) if alerts_data else 0)

with col3:
    st.metric("PPO Steps", len(ppo_data) if ppo_data else 0)

with col4:
    st.metric("Checkpoints", len(checkpoint_data) if checkpoint_data else 0)

# Display alerts
if alerts_data:
    st.subheader("🚨 Active Alerts")
    for alert in alerts_data[-10:]:  # Show last 10 alerts
        st.warning(f"**{{alert.get('type', 'Unknown')}}**: {{alert.get('message', 'No message')}}")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Training Metrics", "PPO Analysis", "Checkpoint Health", "Raw Data"])

with tab1:
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        fig_metrics, fig_resources, fig_health = create_metrics_plots(metrics_df)
        
        if fig_metrics:
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        if fig_resources:
            st.plotly_chart(fig_resources, use_container_width=True)
        
        if fig_health:
            st.plotly_chart(fig_health, use_container_width=True)
    else:
        st.info("No training metrics data available")

with tab2:
    if ppo_data:
        ppo_df = pd.DataFrame(ppo_data)
        fig_ppo, fig_ppo_health = create_ppo_plots(ppo_df)
        
        if fig_ppo:
            st.plotly_chart(fig_ppo, use_container_width=True)
        
        if fig_ppo_health:
            st.plotly_chart(fig_ppo_health, use_container_width=True)
    else:
        st.info("No PPO data available")

with tab3:
    if checkpoint_data:
        checkpoint_df = pd.DataFrame(checkpoint_data)
        
        # Checkpoint health plot
        fig_checkpoint = go.Figure()
        
        if 'model_health_score' in checkpoint_df.columns:
            fig_checkpoint.add_trace(
                go.Scatter(x=checkpoint_df['step'], y=checkpoint_df['model_health_score'], 
                          name='Model Health Score', line=dict(color='green'))
            )
        
        if 'gradient_flow_health' in checkpoint_df.columns:
            fig_checkpoint.add_trace(
                go.Scatter(x=checkpoint_df['step'], y=checkpoint_df['gradient_flow_health'], 
                          name='Gradient Flow Health', line=dict(color='blue'))
            )
        
        fig_checkpoint.update_layout(title="Checkpoint Health", xaxis_title="Step", yaxis_title="Health Score")
        st.plotly_chart(fig_checkpoint, use_container_width=True)
        
        # Checkpoint summary table
        st.subheader("Checkpoint Summary")
        st.dataframe(checkpoint_df[['step', 'model_health_score', 'gradient_flow_health', 'total_parameters']])
    else:
        st.info("No checkpoint data available")

with tab4:
    st.subheader("Raw Metrics Data")
    if metrics_data:
        st.dataframe(pd.DataFrame(metrics_data))
    else:
        st.info("No raw data available")
'''
        
        with open(app_file, 'w') as f:
            f.write(app_content)
    
    def update_data(self):
        """Update dashboard data from files."""
        # This would be called by the callback to update data
        # For now, the dashboard reads directly from files
        pass
    
    def add_metrics(self, metrics: RLDKMetrics):
        """Add new metrics to the dashboard."""
        # This could be used for real-time updates
        pass
    
    def add_alert(self, alert: Dict[str, Any]):
        """Add new alert to the dashboard."""
        # This could be used for real-time alerts
        pass