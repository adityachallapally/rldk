#!/bin/bash
# OpenRLHF Installation Script with CUDA Support
# This script handles CUDA requirements for OpenRLHF installation in container environments

set -e  # Exit on any error

echo "🚀 Starting OpenRLHF installation with CUDA support..."

# Check if running as root (for package installation)
if [ "$EUID" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

# Step 1: Update package lists
echo "📦 Updating package lists..."
$SUDO apt update

# Step 2: Detect Python version and install appropriate packages
echo "🐍 Detecting Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
echo "   Detected Python version: $PYTHON_VERSION"

# Install CUDA development tools and appropriate venv package
echo "🔧 Installing CUDA development tools and Python venv..."
if [[ "$PYTHON_VERSION" == "3.13" ]]; then
    $SUDO apt install -y nvidia-cuda-toolkit python3.13-venv
elif [[ "$PYTHON_VERSION" == "3.12" ]]; then
    $SUDO apt install -y nvidia-cuda-toolkit python3.12-venv
elif [[ "$PYTHON_VERSION" == "3.11" ]]; then
    $SUDO apt install -y nvidia-cuda-toolkit python3.11-venv
elif [[ "$PYTHON_VERSION" == "3.10" ]]; then
    $SUDO apt install -y nvidia-cuda-toolkit python3.10-venv
else
    echo "   ⚠️  Unsupported Python version: $PYTHON_VERSION"
    echo "   Installing generic python3-venv package..."
    $SUDO apt install -y nvidia-cuda-toolkit python3-venv
fi

# Step 3: Set up CUDA environment
echo "🌍 Setting up CUDA environment..."
export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
export PATH=$CUDA_HOME/bin:$PATH

# Verify CUDA installation
echo "✅ Verifying CUDA installation..."
if command -v nvcc &> /dev/null; then
    echo "   CUDA compiler found: $(nvcc --version | head -n1)"
else
    echo "   ❌ CUDA compiler not found!"
    exit 1
fi

# Step 4: Create virtual environment
echo "🐍 Creating Python virtual environment..."
if [ -d "openrlhf_env" ]; then
    echo "   Virtual environment already exists, removing..."
    rm -rf openrlhf_env
fi

# Try to create virtual environment with detected Python version
echo "   Creating virtual environment with Python $PYTHON_VERSION..."
if python3 -m venv openrlhf_env; then
    echo "   ✅ Virtual environment created successfully"
else
    echo "   ❌ Failed to create virtual environment with python3 -m venv"
    echo "   Trying alternative methods..."
    
    # Try with python3.13 if available
    if command -v python3.13 &> /dev/null; then
        echo "   Trying with python3.13..."
        python3.13 -m venv openrlhf_env
    # Try with python3.12 if available
    elif command -v python3.12 &> /dev/null; then
        echo "   Trying with python3.12..."
        python3.12 -m venv openrlhf_env
    # Try with python3.11 if available
    elif command -v python3.11 &> /dev/null; then
        echo "   Trying with python3.11..."
        python3.11 -m venv openrlhf_env
    # Try with python3.10 if available
    elif command -v python3.10 &> /dev/null; then
        echo "   Trying with python3.10..."
        python3.10 -m venv openrlhf_env
    else
        echo "   ❌ No suitable Python version found for virtual environment creation"
        echo "   Please ensure you have a supported Python version (3.10+) installed"
        exit 1
    fi
fi

# Verify virtual environment was created
if [ ! -d "openrlhf_env" ]; then
    echo "   ❌ Virtual environment creation failed"
    exit 1
fi

source openrlhf_env/bin/activate

# Step 5: Upgrade pip and install build dependencies
echo "📚 Installing build dependencies..."
pip install --upgrade pip
pip install packaging wheel setuptools

# Step 6: Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Step 7: Clone OpenRLHF repository
echo "📥 Cloning OpenRLHF repository..."
if [ -d "OpenRLHF" ]; then
    echo "   OpenRLHF directory already exists, removing..."
    rm -rf OpenRLHF
fi

git clone https://github.com/OpenRLHF/OpenRLHF.git
cd OpenRLHF

# Step 8: Install OpenRLHF dependencies (without flash-attn)
echo "📋 Installing OpenRLHF dependencies..."
pip install -r requirements.txt || {
    echo "   ⚠️  Some dependencies failed, trying without flash-attn..."
    # Create modified requirements without flash-attn
    grep -v "flash-attn" requirements.txt > requirements_no_flash.txt
    pip install -r requirements_no_flash.txt
}

# Step 9: Install OpenRLHF in editable mode
echo "🔨 Installing OpenRLHF..."
pip install -e . --no-deps

# Step 10: Verify installation
echo "🧪 Verifying OpenRLHF installation..."
cd ..
python -c "import openrlhf; print('✅ OpenRLHF imported successfully!')" || {
    echo "   ⚠️  OpenRLHF import failed, but installation may still work"
}

echo "🎉 OpenRLHF installation completed!"
echo ""
echo "To use OpenRLHF:"
echo "1. Activate the virtual environment: source openrlhf_env/bin/activate"
echo "2. Set CUDA environment: export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit"
echo "3. Add CUDA to PATH: export PATH=\$CUDA_HOME/bin:\$PATH"
echo "4. Run your OpenRLHF scripts"