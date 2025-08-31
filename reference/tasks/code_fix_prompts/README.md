# 💻 Code Fix Prompts Task

## **Task Overview**
Train a language model to generate code fixes for buggy Python code using reinforcement learning. This task demonstrates RLDK's ability to catch bugs in code generation RL training.

## **RL Setup**
- **Objective**: Maximize code correctness and readability scores
- **Method**: PPO with automated code evaluation
- **Base Model**: 7B parameter model (GPU recommended)
- **Training Steps**: 500 steps with intentional code generation bugs

## **Intentional Code Generation Bugs for RLDK to Catch**

### **1. Memory Leak in Training Loop**
- **Bug**: GPU memory grows unbounded due to missing cleanup
- **RLDK Detection**: 2.1x memory growth rate over time
- **Expected Output**: Compute profiling report showing memory leak

### **2. Reward Model Calibration Issues**
- **Bug**: Reward model overconfident, poor calibration
- **RLDK Detection**: 0.65 calibration score (threshold 0.7)
- **Expected Output**: Calibration plots and poor calibration warning

### **3. Training Instability (Gradient Explosion)**
- **Bug**: Gradients explode after step 300 due to learning rate
- **RLDK Detection**: Gradient norm spikes and training divergence
- **Expected Output**: Drift card showing gradient explosion

## **Training Script**
```bash
python train_code_fix.py \
  --model 7b \
  --dataset code_bugs \
  --learning_rate 1e-5 \
  --max_grad_norm 1.0 \
  --seed 42
```

## **Expected RLDK Outputs**
1. **Compute Profile**: `compute_profile.md` - Shows memory leak
2. **Calibration Report**: `calibration_report.md` - Poor reward calibration
3. **Gradient Drift Card**: `gradient_drift_card.md` - Catches explosion
4. **Training Metrics**: `training_metrics.jsonl` - Raw training data

## **Files in This Directory**
- `train_code_fix.py` - Training script with code generation bugs
- `code_evaluator.py` - Code evaluation with calibration issues
- `dataset.py` - Dataset of buggy Python code
- `model.py` - 7B parameter model wrapper
- `config.yaml` - Configuration with problematic hyperparameters
- `expected_outputs/` - Expected RLDK analysis results

## **How to Run**
```bash
# Install dependencies
pip install transformers torch datasets black pylint

# Run training (will have code generation bugs)
python train_code_fix.py

# Run RLDK analysis
rldk diff --a run_1 --b run_2 --signals gradient_norm,code_score
rldk reward-health --run run_1 --output-dir code_analysis
rldk evals --run run_1 --suite code --output-dir code_eval
```

## **What RLDK Will Catch**
- ✅ **Memory leak**: 2.1x memory growth rate
- ✅ **Poor calibration**: 0.65 calibration score
- ✅ **Gradient explosion**: Spikes after step 300
- ✅ **Training instability**: Divergence in code quality metrics

## **Success Criteria**
- RLDK detects all 3 code generation bugs
- Generates actionable debugging reports
- Demonstrates RLDK's value for code generation RL