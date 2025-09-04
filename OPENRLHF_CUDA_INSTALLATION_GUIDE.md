# OpenRLHF CUDA Installation Guide

This guide provides step-by-step instructions for installing OpenRLHF with CUDA support in container environments.

## Problem

OpenRLHF requires CUDA development tools (nvcc, CUDA_HOME) to compile flash-attention and other CUDA-dependent components. Many containerized environments only have CUDA runtime, not development tools, which causes installation failures.

## Solution

This guide installs the necessary CUDA development tools and sets up the proper environment for OpenRLHF installation.

## Prerequisites

- Ubuntu/Debian-based container environment
- Root or sudo access for package installation
- Internet connection for downloading packages

## Quick Installation

Run the automated installation script:

```bash
./install_openrlhf_with_cuda.sh
```

## Manual Installation

### Step 1: Install CUDA Development Tools

```bash
# Update package lists
sudo apt update

# Install CUDA toolkit and Python venv
sudo apt install -y nvidia-cuda-toolkit python3.13-venv
```

### Step 2: Set Up CUDA Environment

```bash
# Set CUDA environment variables
export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
export PATH=$CUDA_HOME/bin:$PATH

# Verify CUDA installation
nvcc --version
```

### Step 3: Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv openrlhf_env
source openrlhf_env/bin/activate

# Upgrade pip and install build dependencies
pip install --upgrade pip
pip install packaging wheel setuptools
```

### Step 4: Install PyTorch with CUDA Support

```bash
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 5: Clone and Install OpenRLHF

```bash
# Clone OpenRLHF repository
git clone https://github.com/OpenRLHF/OpenRLHF.git
cd OpenRLHF

# Install dependencies (may need to skip flash-attn)
pip install -r requirements.txt || {
    echo "Some dependencies failed, trying without flash-attn..."
    grep -v "flash-attn" requirements.txt > requirements_no_flash.txt
    pip install -r requirements_no_flash.txt
}

# Install OpenRLHF in editable mode
pip install -e . --no-deps
```

### Step 6: Verify Installation

```bash
# Test OpenRLHF import
python -c "import openrlhf; print('OpenRLHF imported successfully!')"
```

## Environment Setup

To use OpenRLHF after installation, always activate the virtual environment and set CUDA variables:

```bash
# Activate virtual environment
source openrlhf_env/bin/activate

# Set CUDA environment
export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
export PATH=$CUDA_HOME/bin:$PATH

# Now you can use OpenRLHF
python your_openrlhf_script.py
```

## Troubleshooting

### CUDA Compiler Not Found

If `nvcc --version` fails:
```bash
# Check if CUDA toolkit is installed
ls -la /usr/lib/nvidia-cuda-toolkit/bin/nvcc

# If not found, reinstall
sudo apt install --reinstall nvidia-cuda-toolkit
```

### Flash-Attention Compilation Fails

Flash-attention requires specific CUDA versions and can be problematic. The installation script handles this by:
1. Installing dependencies without flash-attn if compilation fails
2. Installing OpenRLHF without dependencies to avoid conflicts

### Import Errors

If OpenRLHF import fails:
```bash
# Check if virtual environment is activated
which python

# Check if OpenRLHF is installed
pip list | grep openrlhf

# Reinstall if necessary
cd OpenRLHF
pip install -e . --no-deps
```

## Testing

Run the integration test to verify everything works:

```bash
python test_openrlhf_integration.py
```

## Notes

- This installation works in container environments that lack CUDA development tools
- Flash-attention may not compile in all environments but OpenRLHF can still function
- The virtual environment isolates OpenRLHF dependencies from system Python
- CUDA environment variables must be set each time you use OpenRLHF

## Files Created

- `openrlhf_env/` - Python virtual environment
- `OpenRLHF/` - OpenRLHF source code
- `install_openrlhf_with_cuda.sh` - Automated installation script
- `requirements_no_flash.txt` - Modified requirements without flash-attn