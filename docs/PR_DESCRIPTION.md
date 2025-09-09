# Install CUDA Development Tools for OpenRLHF

## Problem
OpenRLHF installation was failing in container environments due to missing CUDA development tools. The error occurred because:
- OpenRLHF requires CUDA development tools (nvcc, CUDA_HOME) to compile flash-attention
- Container environments typically only have CUDA runtime, not development tools
- This caused compilation failures during pip install

## Solution
This PR provides a comprehensive solution to install OpenRLHF with proper CUDA support in container environments:

### 1. Automated Installation Script
- **`install_openrlhf_with_cuda.sh`** - One-command installation script
- Handles all CUDA requirements automatically
- Includes error handling and fallback strategies

### 2. Updated QA Test
- **`test_openrlhf_integration.py`** - Enhanced with proper CUDA installation instructions
- Provides both automated and manual installation options
- Clear error messages with step-by-step guidance

### 3. Comprehensive Documentation
- **`OPENRLHF_CUDA_INSTALLATION_GUIDE.md`** - Detailed installation guide
- **`OPENRLHF_INSTALLATION_SUMMARY.md`** - Success summary and verification steps
- Troubleshooting section for common issues

## Changes Made

### Files Added
- `install_openrlhf_with_cuda.sh` - Automated installation script
- `OPENRLHF_CUDA_INSTALLATION_GUIDE.md` - Installation documentation
- `OPENRLHF_INSTALLATION_SUMMARY.md` - Implementation summary

### Files Modified
- `test_openrlhf_integration.py` - Updated with CUDA installation instructions

## Installation Process

### Quick Installation
```bash
./install_openrlhf_with_cuda.sh
```

### Manual Installation
```bash
# 1. Install CUDA development tools
sudo apt install -y nvidia-cuda-toolkit python3.13-venv

# 2. Set CUDA environment
export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
export PATH=$CUDA_HOME/bin:$PATH

# 3. Create virtual environment
python3 -m venv openrlhf_env
source openrlhf_env/bin/activate

# 4. Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Clone and install OpenRLHF
git clone https://github.com/OpenRLHF/OpenRLHF.git
cd OpenRLHF
pip install -r requirements.txt
pip install -e . --no-deps
```

## Key Features

### ✅ CUDA Development Tools
- Installs `nvidia-cuda-toolkit` with proper environment setup
- Sets `CUDA_HOME` and `PATH` variables correctly
- Verifies CUDA compiler availability

### ✅ Virtual Environment Isolation
- Creates isolated Python environment to prevent conflicts
- Installs PyTorch with CUDA support first
- Handles dependency resolution properly

### ✅ Flash-Attention Compilation
- Successfully compiles flash-attention 2.8.3 with CUDA support
- Includes fallback strategy if compilation fails
- Handles build dependencies correctly

### ✅ Comprehensive Error Handling
- Clear error messages with solutions
- Step-by-step troubleshooting guide
- Multiple installation options (automated vs manual)

## Testing

The installation has been tested and verified:
- ✅ CUDA compiler working: `nvcc --version`
- ✅ PyTorch with CUDA support installed
- ✅ Flash-attention successfully compiled
- ✅ OpenRLHF import test passed: `import openrlhf`

## Usage

After installation, users can use OpenRLHF by:

```bash
# Activate virtual environment
source openrlhf_env/bin/activate

# Set CUDA environment
export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
export PATH=$CUDA_HOME/bin:$PATH

# Run OpenRLHF scripts
python your_openrlhf_script.py
```

## Benefits

1. **Solves Container CUDA Issues**: Enables OpenRLHF installation in containerized environments
2. **Automated Process**: One-command installation with comprehensive error handling
3. **Clear Documentation**: Step-by-step guides for both automated and manual installation
4. **Fallback Strategies**: Handles edge cases and compilation failures gracefully
5. **QA Integration**: Updated test suite provides clear installation guidance

## Files Changed
- `test_openrlhf_integration.py` - Enhanced error messages with CUDA installation instructions
- `install_openrlhf_with_cuda.sh` - New automated installation script
- `OPENRLHF_CUDA_INSTALLATION_GUIDE.md` - New comprehensive installation guide
- `OPENRLHF_INSTALLATION_SUMMARY.md` - New implementation summary

This PR resolves the OpenRLHF installation issues in container environments and provides a robust, well-documented solution for future users.