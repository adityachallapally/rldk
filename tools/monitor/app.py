"""
Streamlit-based monitoring dashboard for RLHF training profiler.

This dashboard provides real-time visualization of profiler metrics,
access to profiler artifacts, and training alerts.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Check dependencies before importing
try:
    from utils.dependency_checker import check_streamlit_dependencies
    check_streamlit_dependencies()

    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
    from profiler.report import ProfilerReport

except ImportError as e:
    print(f"❌ Error: {e}")
    print("\n💡 To fix this, install the missing dependencies:")
    print("   pip install streamlit plotly")
    print("   # or with --break-system-packages if needed:")
    print("   pip install streamlit plotly --break-system-packages")
    sys.exit(1)


class ProfilerDashboard:
    """Streamlit dashboard for profiler monitoring."""

    def __init__(self):
        self.runs_dir = Path("runs")
        self.runs_dir.mkdir(exist_ok=True)

    def run(self):
        """Run the Streamlit dashboard."""
        st.set_page_config(
            page_title="RLHF Profiler Dashboard",
            page_icon="📊",
            layout="wide"
        )

        st.title("🚀 RLHF Training Profiler Dashboard")
        st.markdown("Real-time monitoring of training performance and profiler artifacts")

        # Sidebar for run selection
        self._render_sidebar()

        # Main content
        if st.session_state.get("selected_run"):
            self._render_main_content()
        else:
            self._render_welcome()

    def _render_sidebar(self):
        """Render the sidebar with run selection."""
        st.sidebar.title("📁 Run Selection")

        # Get available runs
        runs = self._get_available_runs()

        if not runs:
            st.sidebar.warning("No runs found. Start a training run with profiling enabled.")
            return

        # Run selection
        selected_run = st.sidebar.selectbox(
            "Select Run:",
            options=runs,
            format_func=lambda x: f"{x} ({self._get_run_info(x)})"
        )

        st.session_state["selected_run"] = selected_run

        # Refresh button
        if st.sidebar.button("🔄 Refresh"):
            st.rerun()

        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=False)
        if auto_refresh:
            st.rerun()

    def _render_welcome(self):
        """Render welcome message when no run is selected."""
        st.markdown("""
        ## Welcome to the RLHF Profiler Dashboard! 🎯

        This dashboard helps you monitor and analyze the performance of your RLHF training runs.

        ### Features:
        - 📊 **Real-time Metrics**: Live visualization of training performance
        - 🔍 **Profiler Artifacts**: Access to trace files, operation stats, and timing data
        - ⚠️ **Training Alerts**: Warnings and recommendations for optimization
        - 📈 **Performance Analysis**: Detailed breakdowns of training stages

        ### Getting Started:
        1. Start a training run with profiling enabled: `python train.py --profiler on`
        2. Select the run from the sidebar
        3. Explore the profiler data and visualizations

        ### Profiler Artifacts:
        - `trace.json`: Chrome trace file for detailed performance analysis
        - `op_stats.csv`: Operation-level timing and memory statistics
        - `stage_times.json`: Training stage timing breakdown
        - `memory_stats.json`: Memory usage patterns
        """)

    def _render_main_content(self):
        """Render the main dashboard content."""
        run_id = st.session_state["selected_run"]
        run_dir = self.runs_dir / run_id

        # Run info header
        self._render_run_header(run_dir)

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "⏱️ Timing", "💾 Memory", "🔧 Artifacts"])

        with tab1:
            self._render_overview_tab(run_dir)

        with tab2:
            self._render_timing_tab(run_dir)

        with tab3:
            self._render_memory_tab(run_dir)

        with tab4:
            self._render_artifacts_tab(run_dir)

    def _render_run_header(self, run_dir: Path):
        """Render run information header."""
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Run ID", run_dir.name)

        with col2:
            if run_dir.exists():
                mod_time = time.ctime(run_dir.stat().st_mtime)
                st.metric("Last Modified", mod_time)
            else:
                st.metric("Status", "Not Found")

        with col3:
            artifacts = self._get_profiler_artifacts(run_dir)
            st.metric("Profiler Artifacts", f"{sum(artifacts.values())}/4")

    def _render_overview_tab(self, run_dir: Path):
        """Render the overview tab."""
        st.header("📊 Training Overview")

        # Check for profiler report
        profiler_report_path = run_dir / "profiler_report.json"
        if profiler_report_path.exists():
            with open(profiler_report_path) as f:
                report = json.load(f)

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Profiling Status", "✅ Success" if report["summary"]["profiling_successful"] else "❌ Failed")

            with col2:
                if "stage_summary" in report["summary"]:
                    total_steps = report["summary"]["stage_summary"]["total_steps"]
                    st.metric("Total Steps", total_steps)
                else:
                    st.metric("Total Steps", "N/A")

            with col3:
                if "memory_summary" in report["summary"]:
                    peak_memory = report["summary"]["memory_summary"]["peak_cuda_memory"]
                    st.metric("Peak CUDA Memory", f"{peak_memory:,} bytes")
                else:
                    st.metric("Peak CUDA Memory", "N/A")

            with col4:
                if "operation_summary" in report["summary"]:
                    total_ops = report["summary"]["operation_summary"]["total_operations"]
                    st.metric("Total Operations", total_ops)
                else:
                    st.metric("Total Operations", "N/A")

            # Artifacts status
            st.subheader("📁 Profiler Artifacts")
            artifacts = report["artifacts"]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Trace File", "✅" if artifacts["trace_json"] else "❌")

            with col2:
                st.metric("Operation Stats", "✅" if artifacts["op_stats_csv"] else "❌")

            with col3:
                st.metric("Stage Times", "✅" if artifacts["stage_times_json"] else "❌")

            with col4:
                st.metric("Memory Stats", "✅" if artifacts["memory_stats_json"] else "❌")

        else:
            st.warning("No profiler report found. Profiling may not have been enabled or completed.")

    def _render_timing_tab(self, run_dir: Path):
        """Render the timing analysis tab."""
        st.header("⏱️ Timing Analysis")

        # Stage timing data
        stage_times_path = run_dir / "stage_times.json"
        if stage_times_path.exists():
            with open(stage_times_path) as f:
                stage_data = json.load(f)

            # Stage timing chart
            if "average_times" in stage_data and stage_data["average_times"]:
                st.subheader("Average Stage Times")

                stages = list(stage_data["average_times"].keys())
                times = list(stage_data["average_times"].values())

                fig = px.bar(
                    x=stages,
                    y=times,
                    title="Average Time per Stage",
                    labels={"x": "Stage", "y": "Time (seconds)"}
                )
                st.plotly_chart(fig, use_container_width=True)

            # Detailed stage times
            if "stage_times" in stage_data:
                st.subheader("Detailed Stage Times")

                # Create DataFrame for detailed view
                stage_times_df = pd.DataFrame([
                    {"Stage": stage, "Time": time_val, "Run": i}
                    for stage, times in stage_data["stage_times"].items()
                    for i, time_val in enumerate(times)
                ])

                if not stage_times_df.empty:
                    fig = px.box(
                        stage_times_df,
                        x="Stage",
                        y="Time",
                        title="Stage Time Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("No stage timing data found.")

        # Operation statistics
        op_stats_path = run_dir / "op_stats.csv"
        if op_stats_path.exists():
            st.subheader("Operation Statistics")

            df = pd.read_csv(op_stats_path)

            # Top operations by CPU time
            if not df.empty and "CPU Time (μs)" in df.columns:
                df["CPU Time (μs)"] = pd.to_numeric(df["CPU Time (μs)"], errors='coerce')
                top_ops = df.nlargest(10, "CPU Time (μs)")

                fig = px.bar(
                    top_ops,
                    x="Name",
                    y="CPU Time (μs)",
                    title="Top 10 Operations by CPU Time"
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

    def _render_memory_tab(self, run_dir: Path):
        """Render the memory analysis tab."""
        st.header("💾 Memory Analysis")

        # Memory statistics
        memory_stats_path = run_dir / "memory_stats.json"
        if memory_stats_path.exists():
            with open(memory_stats_path) as f:
                memory_data = json.load(f)

            col1, col2 = st.columns(2)

            with col1:
                peak_cpu = memory_data.get("peak_memory_usage", {}).get("cpu", 0)
                st.metric("Peak CPU Memory", f"{peak_cpu:,} bytes")

            with col2:
                peak_cuda = memory_data.get("peak_memory_usage", {}).get("cuda", 0)
                st.metric("Peak CUDA Memory", f"{peak_cuda:,} bytes")

        # Operation memory usage
        op_stats_path = run_dir / "op_stats.csv"
        if op_stats_path.exists():
            st.subheader("Operation Memory Usage")

            df = pd.read_csv(op_stats_path)

            if not df.empty and "CUDA Memory (bytes)" in df.columns:
                df["CUDA Memory (bytes)"] = pd.to_numeric(df["CUDA Memory (bytes)"], errors='coerce')
                memory_ops = df[df["CUDA Memory (bytes)"] > 0].nlargest(10, "CUDA Memory (bytes)")

                if not memory_ops.empty:
                    fig = px.bar(
                        memory_ops,
                        x="Name",
                        y="CUDA Memory (bytes)",
                        title="Top 10 Operations by CUDA Memory Usage"
                    )
                    fig.update_xaxis(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)

    def _render_artifacts_tab(self, run_dir: Path):
        """Render the artifacts tab."""
        st.header("🔧 Profiler Artifacts")

        artifacts = self._get_profiler_artifacts(run_dir)

        for artifact_name, exists in artifacts.items():
            col1, col2 = st.columns([3, 1])

            with col1:
                status = "✅ Available" if exists else "❌ Missing"
                st.write(f"**{artifact_name}**: {status}")

            with col2:
                if exists:
                    artifact_path = run_dir / artifact_name
                    if st.button(f"Download {artifact_name}", key=f"download_{artifact_name}"):
                        with open(artifact_path, 'rb') as f:
                            st.download_button(
                                label=f"Download {artifact_name}",
                                data=f.read(),
                                file_name=artifact_name,
                                mime="application/octet-stream"
                            )

        # Instructions for using artifacts
        st.subheader("📖 How to Use Profiler Artifacts")

        st.markdown("""
        **Chrome Trace (`trace.json`)**:
        - Open in Chrome by navigating to `chrome://tracing`
        - Load the trace file for detailed performance analysis
        - Shows CPU/GPU activity timeline and call stacks

        **Operation Stats (`op_stats.csv`)**:
        - Import into Excel, pandas, or any CSV viewer
        - Contains detailed timing and memory usage per operation
        - Useful for identifying performance bottlenecks

        **Stage Times (`stage_times.json`)**:
        - JSON format with training stage breakdown
        - Shows average times and individual measurements
        - Useful for understanding training pipeline performance

        **Memory Stats (`memory_stats.json`)**:
        - Peak memory usage information
        - Memory usage patterns during training
        - Helps with memory optimization
        """)

    def _get_available_runs(self) -> List[str]:
        """Get list of available run directories."""
        if not self.runs_dir.exists():
            return []

        runs = []
        for item in self.runs_dir.iterdir():
            if item.is_dir():
                runs.append(item.name)

        return sorted(runs, reverse=True)  # Most recent first

    def _get_run_info(self, run_id: str) -> str:
        """Get brief info about a run."""
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            return "Not found"

        artifacts = self._get_profiler_artifacts(run_dir)
        artifact_count = sum(artifacts.values())
        return f"{artifact_count}/4 artifacts"

    def _get_profiler_artifacts(self, run_dir: Path) -> Dict[str, bool]:
        """Check which profiler artifacts exist in a run directory."""
        artifacts = {
            "trace.json": (run_dir / "trace.json").exists(),
            "op_stats.csv": (run_dir / "op_stats.csv").exists(),
            "stage_times.json": (run_dir / "stage_times.json").exists(),
            "memory_stats.json": (run_dir / "memory_stats.json").exists()
        }
        return artifacts


def main():
    """Main function to run the dashboard."""
    dashboard = ProfilerDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
