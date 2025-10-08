# 🔗 RLDK Integration Guide

## **Overview**
This guide shows how to integrate RLDK with popular RL frameworks and custom training loops. RLDK works seamlessly with any RL training setup, providing debugging capabilities regardless of your framework choice.

## **Integration with TRL (Transformers RL)**

### **Setup**
```bash
pip install trl rldk
```

### **Basic Integration**
```python
from trl import PPOConfig, PPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
import rldk

# Initialize model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Configure PPO training
config = PPOConfig(
    learning_rate=1e-5,
    batch_size=4,
    mini_batch_size=1,
    gradient_accumulation_steps=4,
    optimize_cuda_cache=True,
    seed=42
)

# Initialize trainer (TRL v0.22.2+ API)
trainer = PPOTrainer(
    args=config,
    model=model,
    ref_model=ref_model,
    reward_model=reward_model,
    value_model=value_model,
    processing_class=tokenizer,
    train_dataset=dataset,
)

# Add RLDK monitoring
rldk_monitor = rldk.TrainingMonitor(
    trainer=trainer,
    metrics=["reward", "kl_divergence", "policy_loss"],
    output_dir="trl_training_analysis"
)

# Training loop with RLDK
for step in range(100):
    # Run PPO step
    stats = trainer.step()
    
    # RLDK automatically monitors
    rldk_monitor.record_step(step, stats)
    
    # Check for issues every 10 steps
    if step % 10 == 0:
        issues = rldk_monitor.check_health()
        if issues:
            print(f"🚨 Issues detected at step {step}: {issues}")
```

### **Advanced TRL Integration**
```python
from rldk import DeterminismChecker, RewardHealthAnalyzer

class RLDTrainer(PPOTrainer):
    """TRL trainer with integrated RLDK debugging."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize RLDK components
        self.determinism_checker = DeterminismChecker()
        self.reward_analyzer = RewardHealthAnalyzer()
        self.divergence_detector = DivergenceDetector()
        
        # Set up monitoring
        self.setup_rldk_monitoring()
    
    def setup_rldk_monitoring(self):
        """Set up comprehensive RLDK monitoring."""
        self.metrics_history = []
        self.health_checks = []
        
    def step(self, *args, **kwargs):
        """Enhanced step with RLDK monitoring."""
        # Record pre-step state
        pre_metrics = self.get_current_metrics()
        
        # Run original step
        result = super().step(*args, **kwargs)
        
        # Record post-step state
        post_metrics = self.get_current_metrics()
        
        # RLDK analysis
        self.analyze_step(pre_metrics, post_metrics, result)
        
        return result
    
    def analyze_step(self, pre_metrics, post_metrics, result):
        """Run RLDK analysis on this training step."""
        # Record metrics
        step_data = {
            "step": self.state.global_step,
            "reward": result.get("reward", 0),
            "kl_divergence": result.get("kl_div", 0),
            "policy_loss": result.get("policy_loss", 0),
            "value_loss": result.get("value_loss", 0),
            "entropy": result.get("entropy", 0),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "wall_time": time.time()
        }
        
        self.metrics_history.append(step_data)
        
        # Check for issues
        if len(self.metrics_history) > 10:
            self.check_training_health()
    
    def check_training_health(self):
        """Run comprehensive health checks."""
        recent_metrics = self.metrics_history[-10:]
        
        # Check determinism
        if len(self.metrics_history) > 50:
            determinism_report = self.determinism_checker.check(
                self.metrics_history[-50:]
            )
            if not determinism_report.passed:
                print(f"🚨 Determinism issues: {determinism_report.issues}")
        
        # Check reward health
        reward_health = self.reward_analyzer.analyze(recent_metrics)
        if not reward_health.passed:
            print(f"🚨 Reward health issues: {reward_health.issues}")
        
        # Check for divergence
        divergence = self.divergence_detector.check(recent_metrics)
        if divergence.detected:
            print(f"🚨 Divergence detected: {divergence.details}")
    
    def save_rldk_report(self, output_dir):
        """Save comprehensive RLDK analysis report."""
        import json
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save metrics
        with open(output_path / "training_metrics.jsonl", "w") as f:
            for metrics in self.metrics_history:
                f.write(json.dumps(metrics) + "\n")
        
        # Generate RLDK reports
        self.generate_rldk_reports(output_path)
        
        print(f"✅ RLDK reports saved to: {output_dir}")
    
    def generate_rldk_reports(self, output_path):
        """Generate all RLDK analysis reports."""
        # Reward health report
        reward_health = self.reward_analyzer.analyze(self.metrics_history)
        self.reward_analyzer.save_report(reward_health, output_path / "reward_health")
        
        # Determinism report
        if len(self.metrics_history) > 50:
            determinism_report = self.determinism_checker.check(self.metrics_history)
            self.determinism_checker.save_report(determinism_report, output_path / "determinism")
        
        # Divergence report
        divergence_report = self.divergence_detector.check(self.metrics_history)
        self.divergence_detector.save_report(divergence_report, output_path / "divergence")

# Usage
trainer = RLDTrainer(config=config, model=model, tokenizer=tokenizer)

# Train with automatic RLDK monitoring
trainer.train()

# Save comprehensive RLDK report
trainer.save_rldk_report("trl_rldk_analysis")
```

### **TRL + RLDK CLI Integration**
```bash
# After training, analyze with RLDK CLI
rldk diff --a trl_run_1 --b trl_run_2 --signals reward,kl_divergence --output-dir trl_analysis

rldk check-determinism --cmd "python train_trl.py --seed 42" --compare reward --output-dir trl_determinism

rldk reward-health --run trl_training_metrics.jsonl --output-dir trl_reward_health
```

---

## **Integration with OpenRLHF**

### **Setup**
```bash
pip install openrlhf rldk
```

### **Basic Integration**
```python
from openrlhf import OpenRLHF
import rldk

# Initialize OpenRLHF
rlhf = OpenRLHF(
    model_name="gpt2",
    reward_model_name="gpt2",
    dataset_name="helpful-harmless",
    seed=42
)

# Add RLDK monitoring
rldk_monitor = rldk.TrainingMonitor(
    trainer=rlhf,
    metrics=["reward", "kl_divergence", "policy_loss"],
    output_dir="openrlhf_analysis"
)

# Training with RLDK
for epoch in range(10):
    # Run training epoch
    stats = rlhf.train_epoch()
    
    # RLDK monitoring
    rldk_monitor.record_epoch(epoch, stats)
    
    # Health check
    if epoch % 2 == 0:
        health = rldk_monitor.check_health()
        if health.issues:
            print(f"🚨 Health issues at epoch {epoch}: {health.issues}")
```

### **Advanced OpenRLHF Integration**
```python
class OpenRLHFWithRLDK(OpenRLHF):
    """OpenRLHF with integrated RLDK debugging."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize RLDK components
        self.setup_rldk()
    
    def setup_rldk(self):
        """Set up RLDK monitoring."""
        from rldk import (
            DeterminismChecker, 
            RewardHealthAnalyzer, 
            DivergenceDetector,
            DataLineageTracker
        )
        
        self.determinism_checker = DeterminismChecker()
        self.reward_analyzer = RewardHealthAnalyzer()
        self.divergence_detector = DivergenceDetector()
        self.data_lineage = DataLineageTracker()
        
        # Metrics storage
        self.training_metrics = []
        self.validation_metrics = []
        
    def train_epoch(self, *args, **kwargs):
        """Enhanced training with RLDK monitoring."""
        # Record pre-epoch state
        pre_metrics = self.get_epoch_metrics()
        
        # Run original training
        result = super().train_epoch(*args, **kwargs)
        
        # Record post-epoch state
        post_metrics = self.get_epoch_metrics()
        
        # RLDK analysis
        self.analyze_epoch(pre_metrics, post_metrics, result)
        
        return result
    
    def analyze_epoch(self, pre_metrics, post_metrics, result):
        """Run RLDK analysis on this epoch."""
        epoch_data = {
            "epoch": self.current_epoch,
            "reward_mean": result.get("reward_mean", 0),
            "reward_std": result.get("reward_std", 0),
            "kl_divergence": result.get("kl_divergence", 0),
            "policy_loss": result.get("policy_loss", 0),
            "value_loss": result.get("value_loss", 0),
            "entropy": result.get("entropy", 0),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "wall_time": time.time()
        }
        
        self.training_metrics.append(epoch_data)
        
        # Run health checks
        if len(self.training_metrics) > 5:
            self.run_health_checks()
    
    def run_health_checks(self):
        """Run comprehensive health checks."""
        # Check determinism
        if len(self.training_metrics) > 10:
            det_report = self.determinism_checker.check(self.training_metrics)
            if not det_report.passed:
                print(f"🚨 Determinism issues: {det_report.issues}")
        
        # Check reward health
        reward_health = self.reward_analyzer.analyze(self.training_metrics)
        if not reward_health.passed:
            print(f"🚨 Reward health issues: {reward_health.issues}")
        
        # Check for divergence
        divergence = self.divergence_detector.check(self.training_metrics)
        if divergence.detected:
            print(f"🚨 Divergence detected: {divergence.details}")
    
    def save_rldk_report(self, output_dir):
        """Save comprehensive RLDK analysis."""
        import json
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save metrics
        with open(output_path / "training_metrics.jsonl", "w") as f:
            for metrics in self.training_metrics:
                f.write(json.dumps(metrics) + "\n")
        
        # Generate reports
        self.generate_reports(output_path)
        
        print(f"✅ RLDK reports saved to: {output_dir}")
    
    def generate_reports(self, output_path):
        """Generate all RLDK reports."""
        # Reward health
        reward_health = self.reward_analyzer.analyze(self.training_metrics)
        self.reward_analyzer.save_report(reward_health, output_path / "reward_health")
        
        # Determinism
        if len(self.training_metrics) > 10:
            det_report = self.determinism_checker.check(self.training_metrics)
            self.determinism_checker.save_report(det_report, output_path / "determinism")
        
        # Divergence
        div_report = self.divergence_detector.check(self.training_metrics)
        self.divergence_detector.save_report(div_report, output_path / "divergence")

# Usage
rlhf = OpenRLHFWithRLDK(
    model_name="gpt2",
    reward_model_name="gpt2",
    dataset_name="helpful-harmless",
    seed=42
)

# Train with RLDK monitoring
for epoch in range(10):
    rlhf.train_epoch()

# Save RLDK analysis
rlhf.save_rldk_report("openrlhf_rldk_analysis")
```

---

## **Integration with Custom Training Loops**

### **Basic Custom Loop Integration**
```python
import torch
import torch.nn as nn
from torch.optim import AdamW
import rldk

class CustomRLTrainer:
    """Custom RL trainer with RLDK integration."""
    
    def __init__(self, model, reward_model, config):
        self.model = model
        self.reward_model = reward_model
        self.config = config
        
        # Initialize RLDK
        self.setup_rldk()
        
        # Training components
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        self.metrics_history = []
        
    def setup_rldk(self):
        """Set up RLDK monitoring."""
        from rldk import (
            DeterminismChecker,
            RewardHealthAnalyzer,
            DivergenceDetector
        )
        
        self.determinism_checker = DeterminismChecker()
        self.reward_analyzer = RewardHealthAnalyzer()
        self.divergence_detector = DivergenceDetector()
        
    def training_step(self, batch):
        """Single training step with RLDK monitoring."""
        # Record pre-step metrics
        pre_metrics = self.get_step_metrics()
        
        # Forward pass
        outputs = self.model(batch)
        rewards = self.reward_model(batch, outputs)
        
        # Compute loss
        loss = self.compute_loss(outputs, rewards)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Record post-step metrics
        post_metrics = self.get_step_metrics()
        
        # RLDK analysis
        self.analyze_step(pre_metrics, post_metrics, loss.item(), rewards)
        
        return {"loss": loss.item(), "rewards": rewards.mean().item()}
    
    def analyze_step(self, pre_metrics, post_metrics, loss, rewards):
        """Run RLDK analysis on this step."""
        step_data = {
            "step": len(self.metrics_history),
            "loss": loss,
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "wall_time": time.time()
        }
        
        self.metrics_history.append(step_data)
        
        # Health checks every 10 steps
        if len(self.metrics_history) % 10 == 0:
            self.run_health_checks()
    
    def run_health_checks(self):
        """Run RLDK health checks."""
        recent_metrics = self.metrics_history[-10:]
        
        # Check reward health
        reward_health = self.reward_analyzer.analyze(recent_metrics)
        if not reward_health.passed:
            print(f"🚨 Reward health issues: {reward_health.issues}")
        
        # Check for divergence
        if len(self.metrics_history) > 20:
            divergence = self.divergence_detector.check(recent_metrics)
            if divergence.detected:
                print(f"🚨 Divergence detected: {divergence.details}")
    
    def train(self, dataloader, num_epochs):
        """Main training loop."""
        for epoch in range(num_epochs):
            for batch in dataloader:
                result = self.training_step(batch)
                
                if len(self.metrics_history) % 100 == 0:
                    print(f"Step {len(self.metrics_history)}, Loss: {result['loss']:.4f}")
    
    def save_rldk_report(self, output_dir):
        """Save RLDK analysis report."""
        import json
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save metrics
        with open(output_path / "training_metrics.jsonl", "w") as f:
            for metrics in self.metrics_history:
                f.write(json.dumps(metrics) + "\n")
        
        # Generate reports
        self.generate_reports(output_path)
        
        print(f"✅ RLDK reports saved to: {output_dir}")
    
    def generate_reports(self, output_path):
        """Generate all RLDK reports."""
        # Reward health
        reward_health = self.reward_analyzer.analyze(self.metrics_history)
        self.reward_analyzer.save_report(reward_health, output_path / "reward_health")
        
        # Divergence
        if len(self.metrics_history) > 20:
            div_report = self.divergence_detector.check(self.metrics_history)
            self.divergence_detector.save_report(div_report, output_path / "divergence")

# Usage
trainer = CustomRLTrainer(model, reward_model, config)
trainer.train(dataloader, num_epochs=10)
trainer.save_rldk_report("custom_training_analysis")
```

---

## **Integration with Weights & Biases (wandb)**

### **Setup**
```bash
pip install wandb rldk
```

### **Basic Integration**
```python
import wandb
import rldk

# Initialize wandb
wandb.init(project="rlhf-training", name="experiment-1")

# Initialize RLDK with wandb integration
rldk_monitor = rldk.TrainingMonitor(
    wandb_run=wandb.run,
    metrics=["reward", "kl_divergence", "policy_loss"],
    output_dir="wandb_rldk_analysis"
)

# Training loop
for step in range(1000):
    # Your training code here
    result = training_step()
    
    # Log to wandb
    wandb.log({
        "reward": result["reward"],
        "kl_divergence": result["kl_divergence"],
        "policy_loss": result["policy_loss"]
    })
    
    # RLDK monitoring
    rldk_monitor.record_step(step, result)
    
    # Health checks
    if step % 100 == 0:
        health = rldk_monitor.check_health()
        if health.issues:
            wandb.alert(
                title="Training Issues Detected",
                text=f"RLDK detected issues: {health.issues}",
                level=wandb.AlertLevel.WARNING
            )
```

### **Advanced wandb Integration**
```python
class WandBRLDKIntegration:
    """Advanced wandb + RLDK integration."""
    
    def __init__(self, project_name, run_name):
        self.wandb_run = wandb.init(project=project_name, name=run_name)
        self.setup_rldk()
        
    def setup_rldk(self):
        """Set up RLDK with wandb integration."""
        from rldk import (
            TrainingMonitor,
            DeterminismChecker,
            RewardHealthAnalyzer,
            DivergenceDetector
        )
        
        self.monitor = TrainingMonitor(
            wandb_run=self.wandb_run,
            metrics=["reward", "kl_divergence", "policy_loss"],
            output_dir="wandb_rldk_analysis"
        )
        
        self.determinism_checker = DeterminismChecker()
        self.reward_analyzer = RewardHealthAnalyzer()
        self.divergence_detector = DivergenceDetector()
        
        # Metrics storage
        self.metrics_history = []
        
    def log_metrics(self, step, metrics):
        """Log metrics to both wandb and RLDK."""
        # Log to wandb
        wandb.log(metrics, step=step)
        
        # Store for RLDK
        step_data = {"step": step, **metrics, "wall_time": time.time()}
        self.metrics_history.append(step_data)
        
        # RLDK monitoring
        self.monitor.record_step(step, metrics)
        
        # Health checks
        if step % 100 == 0:
            self.run_health_checks(step)
    
    def run_health_checks(self, step):
        """Run RLDK health checks and log to wandb."""
        recent_metrics = self.metrics_history[-10:]
        
        # Check reward health
        reward_health = self.reward_analyzer.analyze(recent_metrics)
        if not reward_health.passed:
            wandb.alert(
                title="Reward Health Issues",
                text=f"RLDK detected reward issues: {reward_health.issues}",
                level=wandb.AlertLevel.WARNING
            )
        
        # Check for divergence
        if len(self.metrics_history) > 20:
            divergence = self.divergence_detector.check(recent_metrics)
            if divergence.detected:
                wandb.alert(
                    title="Training Divergence",
                    text=f"RLDK detected divergence: {divergence.details}",
                    level=wandb.AlertLevel.ERROR
                )
        
        # Log health metrics
        wandb.log({
            "rldk/reward_health_score": reward_health.score if hasattr(reward_health, 'score') else 0,
            "rldk/divergence_detected": divergence.detected if 'divergence' in locals() else False
        }, step=step)
    
    def finish(self):
        """Finish the run and save RLDK reports."""
        # Save RLDK analysis
        self.monitor.save_reports("wandb_rldk_analysis")
        
        # Upload RLDK reports to wandb
        import os
        for file_path in Path("wandb_rldk_analysis").rglob("*.md"):
            artifact = wandb.Artifact(
                name=f"rldk-analysis-{self.wandb_run.id}",
                type="rldk-analysis"
            )
            artifact.add_file(str(file_path))
            wandb.log_artifact(artifact)
        
        # Finish wandb run
        self.wandb_run.finish()

# Usage
wandb_rldk = WandBRLDKIntegration("rlhf-project", "experiment-1")

# Training loop
for step in range(1000):
    result = training_step()
    
    # Log with RLDK monitoring
    wandb_rldk.log_metrics(step, {
        "reward": result["reward"],
        "kl_divergence": result["kl_divergence"],
        "policy_loss": result["policy_loss"]
    })

# Finish and save reports
wandb_rldk.finish()
```

---

## **Integration with TensorBoard**

### **Setup**
```bash
pip install tensorboard rldk
```

### **Basic Integration**
```python
from torch.utils.tensorboard import SummaryWriter
import rldk

# Initialize TensorBoard
writer = SummaryWriter("runs/rlhf_experiment")

# Initialize RLDK with TensorBoard integration
rldk_monitor = rldk.TrainingMonitor(
    tensorboard_writer=writer,
    metrics=["reward", "kl_divergence", "policy_loss"],
    output_dir="tensorboard_rldk_analysis"
)

# Training loop
for step in range(1000):
    # Your training code here
    result = training_step()
    
    # Log to TensorBoard
    writer.add_scalar("Loss/Policy", result["policy_loss"], step)
    writer.add_scalar("Loss/Value", result["value_loss"], step)
    writer.add_scalar("Metrics/Reward", result["reward"], step)
    writer.add_scalar("Metrics/KL_Divergence", result["kl_divergence"], step)
    
    # RLDK monitoring
    rldk_monitor.record_step(step, result)
    
    # Health checks
    if step % 100 == 0:
        health = rldk_monitor.check_health()
        if health.issues:
            # Log issues to TensorBoard
            writer.add_text("RLDK/Issues", str(health.issues), step)

# Close TensorBoard
writer.close()
```

---

## **CI/CD Integration**

### **GitHub Actions Example**
```yaml
name: RLDK Training Validation

on: [push, pull_request]

jobs:
  rldk-validation:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install rldk torch transformers
    
    - name: Run training with RLDK
      run: |
        python train_model.py --steps 100 --output-dir training_output
    
    - name: Run RLDK analysis
      run: |
        rldk reward-health --run training_output --output-dir rldk_analysis
        rldk check-determinism --cmd "python train_model.py --steps 10" --compare reward_mean
    
    - name: Check RLDK results
      run: |
        # Check if any critical issues were found
        if grep -q "FAILED" rldk_analysis/*.md; then
          echo "🚨 RLDK detected training issues!"
          exit 1
        fi
        echo "✅ RLDK validation passed"
    
    - name: Upload RLDK reports
      uses: actions/upload-artifact@v3
      with:
        name: rldk-analysis
        path: rldk_analysis/
```

### **GitLab CI Example**
```yaml
stages:
  - train
  - analyze
  - validate

train:
  stage: train
  script:
    - pip install rldk torch transformers
    - python train_model.py --steps 100 --output-dir training_output
  artifacts:
    paths:
      - training_output/

analyze:
  stage: analyze
  script:
    - pip install rldk
    - rldk reward-health --run training_output --output-dir rldk_analysis
    - rldk check-determinism --cmd "python train_model.py --steps 10" --compare reward_mean
  artifacts:
    paths:
      - rldk_analysis/
  dependencies:
    - train

validate:
  stage: validate
  script:
    - |
      if grep -q "FAILED" rldk_analysis/*.md; then
        echo "🚨 RLDK detected training issues!"
        exit 1
      fi
      echo "✅ RLDK validation passed"
  dependencies:
    - analyze
```

---

## **Best Practices**

### **1. Early Integration**
- Integrate RLDK from the start of your project
- Don't wait until you have issues to add monitoring

### **2. Comprehensive Monitoring**
- Monitor all key metrics (reward, loss, KL divergence)
- Set appropriate thresholds for your use case
- Run health checks regularly

### **3. Automated Alerts**
- Set up alerts for critical issues
- Integrate with your existing monitoring systems
- Use CI/CD to catch issues before deployment

### **4. Regular Analysis**
- Run full RLDK analysis after each training run
- Keep historical reports for trend analysis
- Use RLDK reports for team discussions

### **5. Custom Metrics**
- Add custom metrics specific to your task
- Extend RLDK analyzers for domain-specific issues
- Share custom analyzers with the community

---

## **Getting Help**

### **Documentation**
- **RLDK Docs**: `docs/` directory
- **Examples**: `reference/` directory
- **Case Studies**: `reference/CASE_STUDIES.md`

### **Community**
- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and share solutions
- **Contributions**: Submit pull requests and improvements

### **Support**
- **Installation**: `reference/INSTALLATION.md`
- **Integration**: This guide
- **Troubleshooting**: Check common issues in documentation

---

**RLDK integrates seamlessly with any RL framework, providing consistent debugging capabilities regardless of your setup. Start integrating today and make your RL training runs bulletproof!** 🚀