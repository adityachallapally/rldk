# RLDK Architecture

This document provides a comprehensive overview of RLDK's architecture, design principles, and implementation details.

## Overview

RLDK (RL Debug Kit) is designed as a modular, extensible toolkit for reinforcement learning experiment management, analysis, and reproducibility. The architecture emphasizes minimal overhead, robust error handling, and seamless integration with existing RL frameworks.

## Core Design Principles

### 1. Modularity
Each component can be used independently:
```python
# Use only tracking
from rldk.tracking import ExperimentTracker

# Use only forensics
from rldk.forensics import ComprehensivePPOForensics

# Use only determinism checking
from rldk.determinism import check
```

### 2. Extensibility
Easy to add new adapters, metrics, and integrations:
```python
# Custom adapter
class MyAdapter(BaseAdapter):
    def can_handle(self, source: str) -> bool: ...
    def ingest(self, source: str) -> pd.DataFrame: ...

# Custom metric
class MyMetric(BaseMetric):
    def evaluate(self, model, tokenizer, data) -> Dict: ...
```

### 3. Performance
Minimal overhead during training:
- Lazy loading of heavy dependencies
- Efficient data structures
- Optional components
- Configurable logging frequency

### 4. Reliability
Robust error handling and graceful degradation:
- Comprehensive exception handling
- Fallback mechanisms
- Validation at multiple levels
- Clear error messages

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        RLDK Core                            │
├─────────────────────────────────────────────────────────────┤
│  CLI Interface  │  Python API  │  Framework Integrations    │
├─────────────────┼──────────────┼─────────────────────────────┤
│                 │              │                             │
│  rldk forensics │  Tracking    │  TRL Callbacks             │
│  rldk evals     │  Forensics   │  OpenRLHF Monitors         │
│  rldk replay    │  Determinism │  WandB Integration          │
│  rldk ingest    │  Evaluation  │                             │
│                 │  Replay      │                             │
├─────────────────┴──────────────┴─────────────────────────────┤
│                     Core Components                          │
├─────────────────────────────────────────────────────────────┤
│  Data Ingestion  │  Analysis Engine  │  Output Generation   │
│                  │                   │                      │
│  • Adapters      │  • Forensics      │  • Writers           │
│  • Validators    │  • Metrics        │  • Formatters        │
│  • Normalizers   │  • Comparators    │  • Visualizers       │
├─────────────────────────────────────────────────────────────┤
│                    Utility Layer                             │
├─────────────────────────────────────────────────────────────┤
│  Seed Management │  Error Handling  │  Progress Tracking    │
│  Math Utils      │  Validation      │  Configuration        │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Tracking System (`src/rldk/tracking/`)

**Purpose**: Complete experiment lifecycle management and reproducibility

**Key Classes**:
- `ExperimentTracker`: Main orchestrator
- `DatasetTracker`: Dataset versioning and checksums
- `ModelTracker`: Model fingerprinting
- `EnvironmentTracker`: System state capture
- `SeedTracker`: Random seed management
- `GitTracker`: Repository state tracking

**Architecture**:
```python
class ExperimentTracker:
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.trackers = {
            'dataset': DatasetTracker(config),
            'model': ModelTracker(config),
            'environment': EnvironmentTracker(config),
            'seed': SeedTracker(config),
            'git': GitTracker(config)
        }
    
    def start_experiment(self):
        """Initialize all enabled trackers."""
        for name, tracker in self.trackers.items():
            if self.config.is_enabled(name):
                tracker.initialize()
```

**Data Flow**:
```
User Code → ExperimentTracker → Individual Trackers → Storage
    ↓              ↓                    ↓               ↓
  track_*()   coordinate()         compute()        save()
```

### 2. Forensics System (`src/rldk/forensics/`)

**Purpose**: Real-time training analysis and anomaly detection

**Key Classes**:
- `ComprehensivePPOForensics`: Main analysis engine
- `GradientNormsAnalyzer`: Gradient instability detection
- `KLScheduleTracker`: KL divergence monitoring
- `AdvantageStatisticsTracker`: Advantage function analysis

**Architecture**:
```python
class ComprehensivePPOForensics:
    def __init__(self, config):
        self.analyzers = [
            GradientNormsAnalyzer(config),
            KLScheduleTracker(config),
            AdvantageStatisticsTracker(config)
        ]
        self.anomaly_detector = AnomalyDetector(config)
    
    def update(self, step: int, **metrics) -> ForensicsResult:
        """Analyze current training step."""
        results = []
        for analyzer in self.analyzers:
            result = analyzer.analyze(step, metrics)
            results.append(result)
        
        return self.anomaly_detector.detect(results)
```

**Anomaly Detection Pipeline**:
```
Metrics → Analyzers → Anomaly Detection → Recommendations
   ↓         ↓             ↓                    ↓
 Raw Data  Analysis    Classification      Actionable
          Results      & Severity         Suggestions
```

### 3. Determinism System (`src/rldk/determinism/`)

**Purpose**: Reproducibility verification and validation

**Key Components**:
- `check()`: Main determinism checking function
- `DeterminismRunner`: Execution environment management
- `MetricComparator`: Cross-run comparison
- `ReportGenerator`: Results analysis

**Architecture**:
```python
def check(cmd: str, compare: List[str], replicas: int = 3) -> DeterminismReport:
    """Check if command produces deterministic results."""
    
    # 1. Set up controlled environment
    env = DeterministicEnvironment()
    
    # 2. Run multiple replicas
    results = []
    for i in range(replicas):
        result = env.run_command(cmd)
        results.append(result)
    
    # 3. Compare results
    comparator = MetricComparator(tolerance=config.tolerance)
    mismatches = comparator.compare_all(results, compare)
    
    # 4. Generate report
    return DeterminismReport(
        passed=len(mismatches) == 0,
        mismatches=mismatches,
        recommendations=generate_fixes(mismatches)
    )
```

### 4. Evaluation System (`src/rldk/evals/`)

**Purpose**: Standardized model evaluation and benchmarking

**Key Components**:
- `EvaluationRunner`: Orchestrates evaluation suites
- `MetricRegistry`: Manages available metrics
- `SuiteManager`: Predefined evaluation configurations
- Individual metrics: `ThroughputMetric`, `BiasMetric`, `ToxicityMetric`

**Architecture**:
```python
class EvaluationRunner:
    def __init__(self):
        self.metrics = MetricRegistry()
        self.suites = SuiteManager()
    
    def run_evaluation(self, data, suite_name: str) -> EvaluationResult:
        """Run evaluation suite on data."""
        suite = self.suites.get_suite(suite_name)
        results = {}
        
        for metric_name in suite.metrics:
            metric = self.metrics.get_metric(metric_name)
            result = metric.evaluate(data, suite.config)
            results[metric_name] = result
        
        return EvaluationResult(
            overall_score=compute_overall_score(results),
            individual_results=results
        )
```

### 5. Data Ingestion System (`src/rldk/ingest/`)

**Purpose**: Unified data loading and normalization

**Key Components**:
- `AdapterRegistry`: Manages data format adapters
- `BaseAdapter`: Abstract adapter interface
- Framework-specific adapters: `TRLAdapter`, `OpenRLHFAdapter`, `WandBAdapter`
- `DataValidator`: Schema validation and quality checks

**Architecture**:
```python
def ingest_runs(source: str, adapter_hint: str = None) -> pd.DataFrame:
    """Ingest training data from various sources."""
    
    # 1. Detect appropriate adapter
    registry = AdapterRegistry()
    adapter = registry.get_adapter(source, hint=adapter_hint)
    
    # 2. Ingest data
    raw_data = adapter.ingest(source)
    
    # 3. Validate and normalize
    validator = DataValidator()
    if not validator.validate(raw_data):
        raise ValidationError("Data validation failed")
    
    normalized_data = adapter.normalize(raw_data)
    
    return normalized_data
```

## Data Models

### 1. Event Schema

Central data structure for normalized training events:

```python
@dataclass
class Event:
    step: int
    timestamp: Optional[float] = None
    seed: Optional[int] = None
    
    # Core metrics
    loss: Optional[float] = None
    reward_mean: Optional[float] = None
    reward_std: Optional[float] = None
    kl_divergence: Optional[float] = None
    entropy: Optional[float] = None
    
    # Gradient metrics
    policy_grad_norm: Optional[float] = None
    value_grad_norm: Optional[float] = None
    
    # PPO-specific
    advantage_mean: Optional[float] = None
    advantage_std: Optional[float] = None
    clip_fraction: Optional[float] = None
    kl_coefficient: Optional[float] = None
    
    # Extensible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 2. Configuration System

Hierarchical configuration with validation:

```python
@dataclass
class RLDKConfig:
    """Global RLDK configuration."""
    
    # Tracking configuration
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    
    # Forensics configuration
    forensics: ForensicsConfig = field(default_factory=ForensicsConfig)
    
    # Evaluation configuration
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Global settings
    log_level: str = "INFO"
    cache_dir: str = "~/.cache/rldk"
    config_dir: str = "~/.rldk"
    
    def validate(self) -> None:
        """Validate configuration."""
        # Validation logic
        pass
```

## Integration Architecture

### 1. Framework Callbacks

**TRL Integration**:
```python
class RLDKCallback(TrainerCallback):
    """TRL trainer callback for RLDK integration."""
    
    def __init__(self, config: RLDKConfig):
        self.tracker = ExperimentTracker(config.tracking)
        self.forensics = ComprehensivePPOForensics(config.forensics)
        self.evaluator = EvaluationRunner(config.evaluation)
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize tracking at training start."""
        self.tracker.start_experiment()
        
    def on_step_end(self, args, state, control, **kwargs):
        """Process metrics after each training step."""
        metrics = extract_metrics(state.log_history[-1])
        
        # Forensics analysis
        forensics_result = self.forensics.update(state.global_step, **metrics)
        
        # Handle anomalies
        if forensics_result.has_anomalies:
            self.handle_anomalies(forensics_result.anomalies)
```

**OpenRLHF Integration**:
```python
class OpenRLHFMonitor:
    """Monitor for OpenRLHF distributed training."""
    
    def __init__(self, config: RLDKConfig):
        self.network_monitor = NetworkMonitor(config)
        self.performance_analyzer = PerformanceAnalyzer(config)
        self.distributed_tracker = DistributedTracker(config)
    
    def monitor_step(self, step: int, metrics: Dict):
        """Monitor distributed training step."""
        # Network monitoring
        network_metrics = self.network_monitor.get_metrics()
        
        # Performance analysis
        perf_metrics = self.performance_analyzer.analyze(metrics)
        
        # Distributed coordination
        self.distributed_tracker.sync_metrics(step, metrics)
```

### 2. CLI Architecture

Command-line interface built with Typer:

```python
# src/rldk/cli.py
import typer

app = typer.Typer()

# Sub-applications
forensics_app = typer.Typer()
evals_app = typer.Typer()

app.add_typer(forensics_app, name="forensics")
app.add_typer(evals_app, name="evals")

@forensics_app.command("log-scan")
def forensics_log_scan(
    run_path: str,
    kl_threshold: float = 0.1,
    output: Optional[str] = None
):
    """Scan training logs for anomalies."""
    # Implementation
    pass

@evals_app.command("evaluate")
def evals_evaluate(
    data_file: str,
    suite: str = "quick",
    output: Optional[str] = None
):
    """Run evaluation suite on data."""
    # Implementation
    pass
```

## Error Handling Strategy

### 1. Exception Hierarchy

```python
class RLDKError(Exception):
    """Base exception for RLDK."""
    pass

class ConfigurationError(RLDKError):
    """Configuration-related errors."""
    pass

class DataIngestionError(RLDKError):
    """Data ingestion and validation errors."""
    pass

class ForensicsError(RLDKError):
    """Forensics analysis errors."""
    pass

class DeterminismError(RLDKError):
    """Determinism checking errors."""
    pass
```

### 2. Graceful Degradation

```python
def safe_track_model(model, name: str) -> bool:
    """Safely track model with fallback."""
    try:
        tracker.track_model(model, name)
        return True
    except Exception as e:
        logger.warning(f"Model tracking failed: {e}")
        # Continue without model tracking
        return False

def robust_forensics_update(**metrics) -> Optional[ForensicsResult]:
    """Robust forensics update with error handling."""
    try:
        return forensics.update(**metrics)
    except Exception as e:
        logger.error(f"Forensics analysis failed: {e}")
        # Return minimal result
        return ForensicsResult(has_anomalies=False, anomalies=[])
```

## Performance Considerations

### 1. Lazy Loading

```python
class LazyImport:
    """Lazy import for heavy dependencies."""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self._module = None
    
    def __getattr__(self, name: str):
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
        return getattr(self._module, name)

# Usage
torch = LazyImport("torch")
transformers = LazyImport("transformers")
```

### 2. Efficient Data Structures

```python
class CircularBuffer:
    """Memory-efficient circular buffer for metrics."""
    
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self.data = [None] * maxsize
        self.index = 0
        self.size = 0
    
    def append(self, item):
        self.data[self.index] = item
        self.index = (self.index + 1) % self.maxsize
        self.size = min(self.size + 1, self.maxsize)
```

### 3. Caching Strategy

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=128)
def compute_dataset_checksum(data_hash: str) -> str:
    """Cached dataset checksum computation."""
    # Expensive computation
    return checksum

def get_data_hash(data) -> str:
    """Generate hash for data caching."""
    return hashlib.sha256(str(data).encode()).hexdigest()
```

## Testing Architecture

### 1. Test Structure

```
tests/
├── unit/                 # Unit tests
│   ├── test_tracking.py
│   ├── test_forensics.py
│   └── test_utils.py
├── integration/          # Integration tests
│   ├── test_cli.py
│   ├── test_trl_integration.py
│   └── test_end_to_end.py
├── e2e/                  # End-to-end tests
│   └── test_workflows.py
├── fixtures/             # Test data
└── conftest.py          # Shared fixtures
```

### 2. Test Utilities

```python
# tests/utils.py
class MockTrainer:
    """Mock trainer for testing integrations."""
    
    def __init__(self, config):
        self.config = config
        self.callbacks = []
    
    def add_callback(self, callback):
        self.callbacks.append(callback)
    
    def simulate_training(self, steps: int):
        """Simulate training for testing."""
        for step in range(steps):
            metrics = generate_mock_metrics(step)
            for callback in self.callbacks:
                callback.on_step_end(step, metrics)

def generate_mock_metrics(step: int) -> Dict:
    """Generate realistic mock metrics."""
    return {
        "loss": 1.0 - step * 0.01,
        "reward_mean": step * 0.1,
        "kl_divergence": 0.1 + random.normal(0, 0.01)
    }
```

## Deployment and Distribution

### 1. Package Structure

```
rldk/
├── pyproject.toml        # Package configuration
├── src/rldk/            # Source code
├── tests/               # Test suite
├── docs/                # Documentation
├── examples/            # Example scripts
└── scripts/             # Utility scripts
```

### 2. Optional Dependencies

```toml
# pyproject.toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
    "black>=23.0"
]
openrlhf = [
    "openrlhf>=0.1.0",
    "torch-distributed>=1.0"
]
parquet = [
    "pyarrow>=10.0",
    "fastparquet>=0.8"
]
```

### 3. Entry Points

```toml
[project.scripts]
rldk = "rldk.cli:app"

[project.entry-points."rldk.adapters"]
trl = "rldk.adapters.trl:TRLAdapter"
openrlhf = "rldk.adapters.openrlhf:OpenRLHFAdapter"
wandb = "rldk.adapters.wandb:WandBAdapter"
```

## Future Architecture Considerations

### 1. Plugin System

```python
class PluginManager:
    """Manage RLDK plugins."""
    
    def __init__(self):
        self.plugins = {}
    
    def register_plugin(self, name: str, plugin_class):
        """Register a new plugin."""
        self.plugins[name] = plugin_class
    
    def load_plugin(self, name: str, config):
        """Load and initialize plugin."""
        if name in self.plugins:
            return self.plugins[name](config)
        raise ValueError(f"Unknown plugin: {name}")
```

### 2. Distributed Architecture

```python
class DistributedRLDK:
    """Distributed RLDK for multi-node training."""
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.coordinator = DistributedCoordinator(rank, world_size)
    
    def sync_metrics(self, metrics: Dict):
        """Synchronize metrics across nodes."""
        all_metrics = self.coordinator.all_gather(metrics)
        return aggregate_metrics(all_metrics)
```

### 3. Cloud Integration

```python
class CloudStorage:
    """Cloud storage backend for RLDK."""
    
    def __init__(self, provider: str, config: Dict):
        self.provider = provider
        self.client = self._create_client(provider, config)
    
    def save_experiment(self, experiment_data: Dict, path: str):
        """Save experiment to cloud storage."""
        self.client.upload(experiment_data, path)
    
    def load_experiment(self, path: str) -> Dict:
        """Load experiment from cloud storage."""
        return self.client.download(path)
```

This architecture provides a solid foundation for RLDK's current functionality while remaining flexible enough to accommodate future enhancements and integrations.
