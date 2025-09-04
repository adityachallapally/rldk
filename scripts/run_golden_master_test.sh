#!/bin/bash
# Golden Master Test Runner for RL Debug Kit
#
# This script runs the complete golden master testing workflow:
# 1. Creates a fresh virtual environment
# 2. Installs the package in development mode
# 3. Captures golden master
# 4. Replays and compares with golden master
# 5. Reports results

# Set deterministic environment variables
export PYTHONHASHSEED=0
export TZ=UTC
export LC_ALL=C
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:16:8
export SOURCE_DATE_EPOCH=315532800

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VENV_NAME="rldk_golden_master_test"
PYTHON_VERSION="python3"
PIP_VERSION="pip3"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to cleanup on exit
cleanup() {
    print_status "Cleaning up..."
    if [ -d "$VENV_NAME" ]; then
        rm -rf "$VENV_NAME"
        print_status "Removed virtual environment: $VENV_NAME"
    fi
}

# Set up cleanup on script exit
trap cleanup EXIT

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Python
    if ! command -v $PYTHON_VERSION &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check pip
    if ! command -v $PIP_VERSION &> /dev/null; then
        print_error "pip3 is required but not installed"
        exit 1
    fi
    
    # Check if we're in the right directory
    if [ ! -f "pyproject.toml" ]; then
        print_error "This script must be run from the project root directory"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to create and activate virtual environment
setup_venv() {
    print_status "Creating fresh virtual environment: $VENV_NAME"
    
    # Remove existing venv if it exists
    if [ -d "$VENV_NAME" ]; then
        rm -rf "$VENV_NAME"
    fi
    
    # Create new virtual environment
    $PYTHON_VERSION -m venv "$VENV_NAME"
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    
    # Upgrade pip
    $PIP_VERSION install --upgrade pip
    
    print_success "Virtual environment created and activated"
}

# Function to install package
install_package() {
    print_status "Installing RL Debug Kit in development mode..."
    
    # Install the package
    $PIP_VERSION install -e .
    
    # Install additional dependencies that might be needed
    $PIP_VERSION install jsonschema
    
    print_success "Package installed successfully"
}

# Function to run golden master capture
run_capture() {
    print_status "Running golden master capture..."
    
    # Run the capture script
    python scripts/capture_golden_master.py
    
    if [ $? -eq 0 ]; then
        print_success "Golden master capture completed successfully"
    else
        print_error "Golden master capture failed"
        exit 1
    fi
}

# Function to run golden master replay
run_replay() {
    print_status "Running golden master replay..."
    
    # Run the replay script
    python scripts/replay_golden_master.py --golden-master golden_master_output --output-dir replay_output
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_success "Golden master replay completed successfully - all tests passed!"
    else
        print_error "Golden master replay failed - some tests did not pass"
        # Don't exit here, we want to show the results
    fi
    
    return $exit_code
}

# Function to display results
display_results() {
    print_status "Displaying test results..."
    
    if [ -f "golden_master.zip" ]; then
        print_success "Golden master zip file created: golden_master.zip"
        ls -lh golden_master.zip
    fi
    
    if [ -f "replay_output/replay_summary.json" ]; then
        print_status "Replay summary:"
        cat replay_output/replay_summary.json | python -m json.tool
    fi
    
    if [ -f "replay_output/replay_comparison_report.txt" ]; then
        print_status "Detailed comparison report available in: replay_output/replay_comparison_report.txt"
    fi
}

# Function to create final report
create_final_report() {
    print_status "Creating final test report..."
    
    cat > golden_master_test_report.md << EOF
# RL Debug Kit Golden Master Test Report

## Test Summary
- **Test Date**: $(date)
- **Python Version**: $($PYTHON_VERSION --version)
- **Package Version**: $(python -c "import rldk; print(rldk.__version__)" 2>/dev/null || echo "Unknown")

## Files Generated
- **Golden Master**: golden_master_output/
- **Golden Master Zip**: golden_master.zip
- **Replay Results**: replay_output/
- **Test Report**: golden_master_test_report.md

## Test Results
$(if [ -f "replay_output/replay_summary.json" ]; then
    python -c "
import json
with open('replay_output/replay_summary.json') as f:
    data = json.load(f)
print(f'- Total Commands: {data[\"total_commands\"]}')
print(f'- Passed: {data[\"passed_commands\"]}')
print(f'- Failed: {data[\"failed_commands\"]}')
print(f'- All Passed: {data[\"all_passed\"]}')
"
fi)

## Next Steps
- If all tests passed: ✅ No behavior changes detected
- If some tests failed: Review the differences in replay_output/replay_comparison_report.txt
- To update golden master: Run this script again and commit the new golden_master.zip

EOF

    print_success "Final test report created: golden_master_test_report.md"
}

# Main execution
main() {
    print_status "Starting RL Debug Kit Golden Master Test"
    print_status "=========================================="
    
    # Check prerequisites
    check_prerequisites
    
    # Setup virtual environment
    setup_venv
    
    # Install package
    install_package
    
    # Run capture
    run_capture
    
    # Run replay
    run_replay
    replay_exit_code=$?
    
    # Display results
    display_results
    
    # Create final report
    create_final_report
    
    # Final status
    if [ $replay_exit_code -eq 0 ]; then
        print_success "🎉 All golden master tests passed! No behavior changes detected."
        exit 0
    else
        print_error "❌ Some golden master tests failed. Check replay_output/ for details."
        exit 1
    fi
}

# Run main function
main "$@"