# Installation

This guide will help you install RLDK and its dependencies for your specific use case.

## System Requirements

### Python Version
RLDK requires Python 3.8 or higher. We recommend Python 3.10 or 3.11 for the best experience.

### Operating System
RLDK is designed to work on:
- **Linux** (Ubuntu 18.04+, CentOS 7+, RHEL 7+)
- **macOS** (10.14+)
- **Windows** (Windows 10+)

### Hardware Requirements
- **CPU**: Any modern multi-core processor
- **Memory**: Minimum 4GB RAM (8GB+ recommended for large models)
- **Storage**: At least 1GB free space for installation
- **GPU**: Optional, but recommended for training (CUDA 11.0+)

## Installation Options

### Option 1: Core Package (Recommended)

The core package includes all essential features for experiment tracking, forensics analysis, and reproducibility checking.

```bash
pip install rldk
```

### Option 2: Development Package

For developers and contributors, install with development dependencies:

```bash
pip install rldk[dev]
```

This includes:
- Testing frameworks (pytest, pytest-cov)
- Code quality tools (black, isort, ruff, mypy)
- Documentation tools (mkdocs, mkdocstrings)

### Option 3: Optional Dependencies

Install additional features as needed:

```bash
# For Parquet file support (better performance with large datasets)
pip install rldk[parquet]

# For OpenRLHF integration
pip install rldk[openrlhf]

# For all optional dependencies
pip install rldk[dev,parquet,openrlhf]
```

## Verification

After installation, verify that RLDK is working correctly:

```bash
# Check version
rldk version

# Run basic CLI test
rldk --help

# Test core functionality
python -c "import rldk; print('RLDK installed successfully!')"
```

## Virtual Environment (Recommended)

We strongly recommend using a virtual environment to avoid dependency conflicts:

### Using venv
```bash
# Create virtual environment
python -m venv rldk-env

# Activate virtual environment
# On Linux/macOS:
source rldk-env/bin/activate
# On Windows:
rldk-env\Scripts\activate

# Install RLDK
pip install rldk[dev]
```

### Using conda
```bash
# Create conda environment
conda create -n rldk-env python=3.10

# Activate environment
conda activate rldk-env

# Install RLDK
pip install rldk[dev]
```

## Docker Installation

For containerized environments, you can use our Docker image:

```bash
# Pull the official image
docker pull your-registry/rldk:latest

# Run with mounted volume for data
docker run -it -v $(pwd):/workspace your-registry/rldk:latest
```

Or build from source:

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .
RUN pip install -e .[dev]

CMD ["rldk", "--help"]
```

## Troubleshooting

### Common Installation Issues

#### 1. Permission Errors
If you encounter permission errors during installation:

```bash
# Use user installation
pip install --user rldk

# Or use virtual environment (recommended)
python -m venv rldk-env
source rldk-env/bin/activate
pip install rldk
```

#### 2. CUDA/GPU Issues
If you have CUDA-related issues:

```bash
# Check CUDA installation
nvidia-smi

# Install CPU-only PyTorch if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install rldk
```

#### 3. Dependency Conflicts
If you have dependency conflicts:

```bash
# Create fresh environment
python -m venv fresh-rldk-env
source fresh-rldk-env/bin/activate

# Install with specific versions
pip install rldk==0.1.0

# Or install dependencies separately
pip install torch>=2.1.0
pip install transformers>=4.45.0
pip install rldk
```

#### 4. Missing System Dependencies

On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install build-essential python3-dev
pip install rldk
```

On CentOS/RHEL:
```bash
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel
pip install rldk
```

On macOS:
```bash
# Install Xcode command line tools
xcode-select --install

# Install via Homebrew (optional)
brew install python
pip install rldk
```

### Performance Issues

#### Slow Import Times
If RLDK takes too long to import:

```bash
# Check import time
python -c "import time; start=time.time(); import rldk; print(f'Import time: {time.time()-start:.2f}s')"

# Should be < 2 seconds. If not, check for:
# 1. Slow network connections (for model downloads)
# 2. Large number of installed packages
# 3. Antivirus software scanning
```

#### Memory Issues
If you encounter memory issues:

```bash
# Check memory usage
python -c "import psutil, rldk; print(f'Memory: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB')"

# Should be < 200MB for basic operations
```

### Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Look for detailed error messages
2. **Run diagnostics**: `rldk seed --show` to check system state
3. **Create minimal reproduction**: Isolate the issue with a simple example
4. **Search issues**: Check our [GitHub Issues](https://github.com/your-org/rldk/issues)
5. **Ask for help**: Open a new issue with detailed information

## Next Steps

Once RLDK is installed, you can:

1. **[Quick Start](quickstart.md)** - Get up and running in 5 minutes
2. **[Configuration](configuration.md)** - Set up RLDK for your environment
3. **[User Guide](../user-guide/tracking.md)** - Learn about experiment tracking
4. **[Examples](../examples/basic-ppo-cartpole.md)** - Try example notebooks

## Uninstallation

To remove RLDK:

```bash
# Remove package
pip uninstall rldk

# Remove configuration (optional)
rm -rf ~/.rldk

# Remove cached data (optional)
rm -rf ~/.cache/rldk
```