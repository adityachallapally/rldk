# 📁 RLDK Reference Suite - Complete File Index

## **Overview**
This document provides a complete listing of all files in the RLDK reference suite, organized by category and purpose. Use this as a quick reference to understand what each file contains and how to use it.

## **📚 Core Documentation**

### **Main Overview Documents**
| File | Purpose | Content |
|------|---------|---------|
| `README.md` | **Main suite overview** | Complete introduction, quick start, and suite structure |
| `SUITE_SUMMARY.md` | **Comprehensive summary** | Complete overview of all components and capabilities |
| `FILE_INDEX.md` | **This file index** | Complete listing of all files and their purposes |

### **Setup and Installation**
| File | Purpose | Content |
|------|---------|---------|
| `INSTALLATION.md` | **Installation guide** | Step-by-step setup, dependencies, troubleshooting |
| `INTEGRATION_GUIDE.md` | **Framework integration** | TRL, OpenRLHF, custom loops, CI/CD examples |

### **Examples and Analysis**
| File | Purpose | Content |
|------|---------|---------|
| `CASE_STUDIES.md` | **Real bug examples** | 5 detailed case studies with actual outputs |
| `PERFORMANCE_BENCHMARKS.md` | **Performance data** | Hardware requirements, timing, scalability data |

---

## **🎯 Task Implementation Files**

### **Summarization Helpfulness Task**
| File | Purpose | Content |
|------|---------|---------|
| `tasks/summarization_helpfulness/README.md` | **Task overview** | Description, setup, expected outputs |
| `tasks/summarization_helpfulness/train_summarization.py` | **Training script** | Complete training with intentional bugs |

**Intentional Bugs:**
1. KL divergence spike at step 47
2. Non-deterministic training
3. Reward saturation

### **Refusal Safety Task**
| File | Purpose | Content |
|------|---------|---------|
| `tasks/refusal_safety/README.md` | **Task overview** | Description, setup, expected outputs |

**Intentional Bugs:**
1. Data leakage (train/val contamination)
2. Safety degradation over time
3. Shortcut learning

### **Code Fix Prompts Task**
| File | Purpose | Content |
|------|---------|---------|
| `tasks/code_fix_prompts/README.md` | **Task overview** | Description, setup, expected outputs |

**Intentional Bugs:**
1. Memory leak in training loop
2. Poor reward calibration
3. Gradient explosion

---

## **🧪 Testing and Validation**

### **Smoke Tests**
| File | Purpose | Content |
|------|---------|---------|
| `smoke_tests/cpu_2min_test.py` | **Quick CPU test** | 2-minute validation of core capabilities |
| `smoke_tests/gpu_1hr_test.py` | **Full GPU test** | 1-hour comprehensive validation |

### **Expected Outputs**
After running the tests, you'll get:
```
smoke_test_outputs/
├── training_metrics.jsonl          # Raw training data
├── smoke_test_outputs_diff/        # Divergence analysis
├── smoke_test_outputs_determinism/ # Determinism check
└── smoke_test_outputs_health/      # Reward health
```

---

## **📊 Data and Manifests**

### **Data Lineage**
| File | Purpose | Content |
|------|---------|---------|
| `datasets/manifests/data_lineage.md` | **Data lineage manifest** | Content-addressed data, contamination examples |

**Demonstrates:**
- Data flow tracking
- Contamination detection
- Hash-based validation

---

## **🚀 Quick Start Guide**

### **1. Start Here**
```bash
# Read the main overview
cat reference/README.md

# Check the complete summary
cat reference/SUITE_SUMMARY.md
```

### **2. Setup and Installation**
```bash
# Follow the installation guide
cat reference/INSTALLATION.md

# Check integration examples
cat reference/INTEGRATION_GUIDE.md
```

### **3. Run the Tests**
```bash
# Quick CPU test (2 minutes)
python reference/smoke_tests/cpu_2min_test.py

# Full GPU test (1 hour)
python reference/smoke_tests/gpu_1hr_test.py
```

### **4. Explore Examples**
```bash
# Check individual task implementations
cat reference/tasks/summarization_helpfulness/README.md
cat reference/tasks/refusal_safety/README.md
cat reference/tasks/code_fix_prompts/README.md
```

---

## **📖 Reading Order**

### **For New Users**
1. `README.md` - Get the big picture
2. `INSTALLATION.md` - Set up your environment
3. `smoke_tests/cpu_2min_test.py` - Run your first test
4. `CASE_STUDIES.md` - See real examples

### **For Advanced Users**
1. `SUITE_SUMMARY.md` - Complete technical overview
2. `INTEGRATION_GUIDE.md` - Framework integration
3. `PERFORMANCE_BENCHMARKS.md` - Performance data
4. Individual task implementations

### **For Framework Integration**
1. `INTEGRATION_GUIDE.md` - Complete integration examples
2. `CASE_STUDIES.md` - Real-world usage patterns
3. `PERFORMANCE_BENCHMARKS.md` - Performance expectations

---

## **🔍 File Details**

### **Markdown Files (.md)**
- **Total**: 9 markdown files
- **Purpose**: Documentation, guides, examples
- **Format**: GitHub-flavored markdown with emojis and tables

### **Python Files (.py)**
- **Total**: 3 Python files
- **Purpose**: Training scripts and smoke tests
- **Dependencies**: PyTorch, transformers, datasets (optional for mock mode)

### **Directory Structure**
```
reference/
├── 9 markdown files (documentation)
├── 3 Python files (implementation)
├── 3 task directories (examples)
├── 1 dataset directory (data lineage)
└── 1 smoke test directory (validation)
```

---

## **🎯 What Each File Teaches You**

### **Core Concepts**
- **`README.md`**: What RLDK is and why it matters
- **`SUITE_SUMMARY.md`**: Complete technical overview
- **`INSTALLATION.md`**: How to get started

### **Practical Skills**
- **`INTEGRATION_GUIDE.md`**: How to use RLDK with your framework
- **`CASE_STUDIES.md`**: Real debugging scenarios and solutions
- **`PERFORMANCE_BENCHMARKS.md`**: What to expect performance-wise

### **Hands-on Experience**
- **`smoke_tests/*.py`**: Run RLDK and see it work
- **`tasks/*/README.md`**: Understand different RL scenarios
- **`tasks/*/train_*.py`**: See training code with intentional bugs

---

## **🚀 Next Steps After Reading**

### **1. Run the Tests**
```bash
# Start with the 2-minute test
python reference/smoke_tests/cpu_2min_test.py
```

### **2. Explore the Examples**
```bash
# Check out the task implementations
ls reference/tasks/*/
cat reference/tasks/*/README.md
```

### **3. Apply to Your Work**
```bash
# Follow the integration guide
cat reference/INTEGRATION_GUIDE.md

# Check case studies for examples
cat reference/CASE_STUDIES.md
```

### **4. Share Your Experience**
- Report issues and improvements
- Share custom analyzers
- Contribute to the community

---

## **📞 Getting Help**

### **Documentation Order**
1. **Start**: `README.md` for overview
2. **Setup**: `INSTALLATION.md` for installation
3. **Examples**: `smoke_tests/*.py` for hands-on
4. **Integration**: `INTEGRATION_GUIDE.md` for your framework
5. **Advanced**: `CASE_STUDIES.md` and `PERFORMANCE_BENCHMARKS.md`

### **Community Support**
- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and share solutions
- **Documentation**: Use this file index to find what you need

---

## **🎉 Success Metrics**

### **After Running the Suite, You Should:**
✅ **Understand** what RLDK does and why it's valuable
✅ **Have seen** RLDK catch real bugs in action
✅ **Know how** to integrate RLDK with your framework
✅ **Be confident** using RLDK for debugging
✅ **Have examples** to share with your team

### **The RLDK Promise**
**RLDK transforms RL debugging from a time-consuming mystery to a quick, systematic process.**

---

**This reference suite proves that reliable RL debugging is not just possible—it's easy, fast, and comprehensive. Start your journey to bulletproof RL training today!** 🚀