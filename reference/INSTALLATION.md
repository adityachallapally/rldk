# 🚀 RLDK Reference Suite Installation Guide

## **Quick Start (5 minutes)**

### **1. Clone the Repository**
```bash
git clone <your-rldk-repo>
cd rldk/reference
```

### **2. Install Dependencies**
```bash
# Create virtual environment (recommended)
python -m venv rldk_env
source rldk_env/bin/activate  # On Windows: rldk_env\Scripts\activate

# Install RLDK and dependencies
pip install -e ..  # Install RLDK from source
pip install transformers torch datasets matplotlib seaborn scikit-learn
```

### **3. Run the 2-Minute Test**
```bash
python smoke_tests/cpu_2min_test.py
```

**That's it!** You should see RLDK catch real bugs in under 2 minutes.

## **Detailed Installation**

### **System Requirements**

#### **Minimum (CPU-only)**
- **CPU**: 4+ cores (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 8GB
- **Storage**: 5GB free space
- **Python**: 3.8+
- **Time**: 2-5 minutes for basic test

#### **Recommended (GPU)**
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3080, A100, etc.)
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Storage**: 10GB free space
- **CUDA**: 11.8+
- **Time**: 1 hour for full test

### **Dependencies**

#### **Core RLDK**
```bash
pip install rldk
```

#### **ML Libraries**
```bash
pip install torch torchvision torchaudio
pip install transformers datasets
pip install accelerate
```

#### **Data Science**
```bash
pip install numpy pandas scipy
pip install matplotlib seaborn
pip install scikit-learn
```

#### **Optional (for full functionality)**
```bash
pip install wandb  # Weights & Biases integration
pip install tensorboard  # TensorBoard logging
pip install jupyter  # Jupyter notebooks
```

### **GPU Setup (Optional)**

#### **CUDA Installation**
```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### **Verify GPU Access**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name()}")
```

## **Running the Reference Suite**

### **Option 1: Quick Test (2 minutes)**
```bash
# Run the basic CPU test
python smoke_tests/cpu_2min_test.py
```

**What you'll see:**
- Training script runs with intentional bugs
- RLDK catches KL divergence spike at step 47
- RLDK detects non-deterministic training
- RLDK identifies reward saturation
- Reports are generated and saved

### **Option 2: Full Test (1 hour)**
```bash
# Run the comprehensive GPU test
python smoke_tests/gpu_1hr_test.py
```

**What you'll see:**
- All three tasks complete training
- Full RLDK analysis suite runs
- All 10 capabilities demonstrated
- Comprehensive reports generated

### **Option 3: Individual Tasks**
```bash
# Run just the summarization task
python reference/tasks/summarization_helpfulness/train_summarization.py --steps 50

# Run just the safety task
python reference/tasks/refusal_safety/train_refusal_safety.py --steps 100

# Run just the code generation task
python reference/tasks/code_fix_prompts/train_code_fix.py --steps 150
```

## **Troubleshooting**

### **Common Issues**

#### **Import Errors**
```bash
# If you get "No module named 'rldk'"
pip install -e ..  # Install from source
export PYTHONPATH=$PYTHONPATH:$(pwd)/src  # Add to path
```

#### **CUDA Issues**
```bash
# If PyTorch can't find CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### **Memory Issues**
```bash
# Reduce batch size for CPU training
python train_summarization.py --batch_size 8

# Use smaller models
python train_summarization.py --model 125M
```

#### **Permission Issues**
```bash
# If you get permission errors
sudo chown -R $USER:$USER .
chmod +x smoke_tests/*.py
```

### **Getting Help**

#### **Check System Status**
```bash
# Python version
python --version

# PyTorch version and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Available packages
pip list | grep -E "(torch|transformers|rldk)"
```

#### **Debug Mode**
```bash
# Run with verbose output
python smoke_tests/cpu_2min_test.py --verbose

# Check individual components
python -c "from rldk import ingest_runs; print('RLDK import successful')"
```

## **Expected Outputs**

### **After Running CPU Test**
```
smoke_test_outputs/
├── training_metrics.jsonl          # Raw training data
├── smoke_test_outputs_diff/        # Divergence analysis
│   ├── diff_report.md
│   └── drift_card.md
├── smoke_test_outputs_determinism/ # Determinism check
│   └── determinism_card.md
└── smoke_test_outputs_health/      # Reward health
    ├── reward_health_card.md
    └── calibration_plots.png
```

### **After Running GPU Test**
```
comprehensive_test_report.md         # Full test summary
gpu_test_summarization/             # Summarization task outputs
gpu_test_safety/                    # Safety task outputs
gpu_test_code/                      # Code generation outputs
```

## **Next Steps**

### **1. Explore the Reports**
- Read the drift cards to understand bugs
- Check determinism reports for reproducibility issues
- Review reward health analysis for pathologies

### **2. Try with Your Data**
```bash
# Analyze your own training runs
rldk diff --a your_run_1 --b your_run_2 --signals reward_mean,loss

# Check determinism
rldk check-determinism --cmd "python your_training.py" --compare reward_mean

# Analyze reward health
rldk reward-health --run your_training_logs.jsonl
```

### **3. Integrate with Your Workflow**
- Add RLDK to your CI/CD pipeline
- Use RLDK for debugging production training runs
- Share RLDK reports with your team

## **Performance Benchmarks**

| Test Type | Hardware | Time | Memory | Outputs |
|-----------|----------|------|---------|---------|
| CPU 2-min | 4-core CPU, 8GB RAM | 2-3 min | 2GB RAM | Basic reports |
| CPU Full | 8-core CPU, 16GB RAM | 10-15 min | 8GB RAM | All reports |
| GPU Full | RTX 3080, 16GB VRAM | 45-60 min | 12GB VRAM | Complete suite |

## **Support**

### **Documentation**
- **README**: `reference/README.md`
- **Installation**: `reference/INSTALLATION.md` (this file)
- **Task Guides**: `reference/tasks/*/README.md`

### **Community**
- **Issues**: GitHub Issues for bugs
- **Discussions**: GitHub Discussions for questions
- **Contributions**: Pull requests welcome!

---

**Ready to make your RL training runs bulletproof? Start with the 2-minute test and see RLDK catch real bugs in action!** 🚀