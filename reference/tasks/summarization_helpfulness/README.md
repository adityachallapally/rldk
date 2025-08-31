# 📝 Summarization Helpfulness Task

## **Task Overview**
Train a language model to generate helpful summaries using human feedback through reinforcement learning. This task demonstrates RLDK's ability to catch common RL training issues in text generation.

## **RL Setup**
- **Objective**: Maximize human preference scores for summary helpfulness
- **Method**: PPO with human feedback (RLHF)
- **Base Model**: GPT-2 (125M parameters)
- **Training Steps**: 100 steps with intentional bugs

## **Intentional Bugs for RLDK to Catch**

### **1. KL Divergence Spike (Step 47)**
- **Bug**: Learning rate too high causes policy to diverge from reference
- **RLDK Detection**: First divergence detection at step 47
- **Expected Output**: Drift card showing 2.3x normal KL divergence

### **2. Non-Deterministic Training**
- **Bug**: Missing seed setting in data loader
- **RLDK Detection**: Determinism harness catches 15% variance
- **Expected Output**: Determinism report with fix recommendations

### **3. Reward Saturation**
- **Bug**: Reward scaling too aggressive, losing signal
- **RLDK Detection**: Reward health analysis shows 0.87 saturation
- **Expected Output**: Calibration plots and saturation warnings

## **Training Script**
```bash
python train_summarization.py \
  --model gpt2 \
  --dataset summarization_preferences \
  --learning_rate 1e-4 \
  --kl_penalty 0.2 \
  --seed 42
```

## **Expected RLDK Outputs**
1. **Drift Card**: `drift_card.md` - Shows divergence at step 47
2. **Determinism Report**: `determinism_card.md` - Catches non-determinism
3. **Reward Health**: `reward_health_card.md` - Detects saturation
4. **Training Metrics**: `training_metrics.jsonl` - Raw training data

## **Files in This Directory**
- `train_summarization.py` - Training script with intentional bugs
- `dataset.py` - Dataset loading with non-deterministic augmentation
- `model.py` - GPT-2 model wrapper
- `reward.py` - Reward function with scaling issues
- `config.yaml` - Configuration with problematic hyperparameters
- `expected_outputs/` - Expected RLDK analysis results

## **How to Run**
```bash
# Install dependencies
pip install transformers torch datasets

# Run training (will have bugs)
python train_summarization.py

# Run RLDK analysis
rldk diff --a run_1 --b run_2 --signals kl_divergence,reward_mean
rldk check-determinism --cmd "python train_summarization.py" --compare reward_mean
rldk reward-health --run run_1 --output-dir analysis
```

## **What RLDK Will Catch**
- ✅ **Divergence**: Step 47 KL spike (2.3x normal)
- ✅ **Non-determinism**: 15% metric variance across replicas  
- ✅ **Reward issues**: 0.87 saturation score, poor calibration
- ✅ **Data problems**: Inconsistent data augmentation

## **Success Criteria**
- RLDK detects all 3 intentional bugs
- Generates actionable reports with fix recommendations
- Demonstrates real debugging value in text generation RL