# 🛡️ Refusal Safety Task

## **Task Overview**
Train a language model to refuse harmful requests while maintaining helpfulness on safe requests. This task demonstrates RLDK's ability to catch safety-critical bugs in RL training.

## **RL Setup**
- **Objective**: Maximize helpfulness on safe requests while refusing harmful ones
- **Method**: PPO with preference data (safe vs. harmful)
- **Base Model**: 1B parameter model (CPU with RAM)
- **Training Steps**: 200 steps with intentional safety bugs

## **Intentional Safety Bugs for RLDK to Catch**

### **1. Data Leakage (Train/Val Contamination)**
- **Bug**: Validation examples accidentally included in training
- **RLDK Detection**: 0.92 correlation between train/val refusal rates
- **Expected Output**: Data lineage report showing contamination

### **2. Reward Hacking (Shortcut Learning)**
- **Bug**: Model learns to refuse ALL requests instead of just harmful ones
- **RLDK Detection**: Shortcut signal detection with 0.78 confidence
- **Expected Output**: Reward health report identifying shortcut behavior

### **3. Safety Degradation Over Time**
- **Bug**: Safety performance degrades after step 150
- **RLDK Detection**: Divergence in safety metrics at step 150
- **Expected Output**: Drift card showing safety regression

## **Training Script**
```bash
python train_refusal_safety.py \
  --model 1b \
  --dataset safety_preferences \
  --learning_rate 5e-5 \
  --safety_weight 2.0 \
  --seed 42
```

## **Expected RLDK Outputs**
1. **Data Lineage Report**: `data_lineage.md` - Shows train/val contamination
2. **Safety Drift Card**: `safety_drift_card.md` - Catches safety degradation
3. **Reward Health Report**: `reward_health_card.md` - Detects shortcut learning
4. **Training Metrics**: `training_metrics.jsonl` - Raw training data

## **Files in This Directory**
- `train_refusal_safety.py` - Training script with safety bugs
- `dataset.py` - Dataset with intentional contamination
- `safety_evaluator.py` - Safety evaluation with known issues
- `reward.py` - Reward function that encourages shortcuts
- `config.yaml` - Configuration with problematic safety weights
- `expected_outputs/` - Expected RLDK analysis results

## **How to Run**
```bash
# Install dependencies
pip install transformers torch datasets

# Run training (will have safety bugs)
python train_refusal_safety.py

# Run RLDK analysis
rldk diff --a run_1 --b run_2 --signals safety_score,refusal_rate
rldk reward-health --run run_1 --output-dir safety_analysis
rldk evals --run run_1 --suite safety --output-dir safety_eval
```

## **What RLDK Will Catch**
- ✅ **Data contamination**: 0.92 train/val correlation
- ✅ **Safety degradation**: Regression at step 150
- ✅ **Shortcut learning**: 0.78 shortcut signal confidence
- ✅ **Reward hacking**: Model refuses everything

## **Success Criteria**
- RLDK detects all 3 safety-related bugs
- Generates actionable safety reports
- Demonstrates RLDK's value for safety-critical applications