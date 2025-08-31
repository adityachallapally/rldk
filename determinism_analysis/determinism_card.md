# Determinism Check Card

## ✅ Determinism Check Passed

The training run appears to be deterministic.

## ⚙️ Environment Settings

### RNG Settings
- **torch_seed:** Set via torch.manual_seed()
- **cuda_seed:** Set via torch.cuda.manual_seed_all()
- **numpy_seed:** Set via np.random.seed()
- **python_hash:** 42

### Deterministic Flags Set
- **PYTHONHASHSEED:** Set to 42
- **OMP_NUM_THREADS:** Set to 1
- **MKL_NUM_THREADS:** Set to 1
- **NUMEXPR_NUM_THREADS:** Set to 1
- **OPENBLAS_NUM_THREADS:** Set to 1
- **VECLIB_MAXIMUM_THREADS:** Set to 1
- **CUDA_LAUNCH_BLOCKING:** Set to 1
- **TORCH_USE_CUDA_DSA:** Set to 1
- **CUBLAS_WORKSPACE_CONFIG:** Set to :4096:8

### PyTorch Deterministic Settings Applied
- **torch.backends.cudnn.deterministic = True**
- **torch.backends.cudnn.benchmark = False**
- **torch.use_deterministic_algorithms(True)**
- **torch.backends.cuda.matmul.allow_tf32 = False**
- **torch.backends.cudnn.allow_tf32 = False**
- **torch.manual_seed(42)**
- **torch.cuda.manual_seed(42)** (if CUDA available)

### Random Seeds Set
- **Python random.seed(42)**
- **NumPy np.random.seed(42)**
- **PyTorch torch.manual_seed(42)**

## 🔧 Recommended Fixes

- Set torch.backends.cudnn.deterministic = True
- Set torch.backends.cudnn.benchmark = False
- Use torch.manual_seed() consistently
- Disable dropout or use deterministic=True
- Use deterministic reduction operations

## 📁 Report Location

Full report saved to: `determinism_analysis/determinism_card.md`
