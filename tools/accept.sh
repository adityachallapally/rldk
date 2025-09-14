#!/usr/bin/env bash
set -euo pipefail


echo "🚀 Starting RLDK Refactor Acceptance Tests"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "INFO")
            echo -e "${BLUE}ℹ️  $message${NC}"
            ;;
        "SUCCESS")
            echo -e "${GREEN}✅ $message${NC}"
            ;;
        "WARNING")
            echo -e "${YELLOW}⚠️  $message${NC}"
            ;;
        "ERROR")
            echo -e "${RED}❌ $message${NC}"
            ;;
    esac
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_status "ERROR" "Please run this script from the RLDK root directory"
    exit 1
fi

print_status "INFO" "Setting up fresh environment..."
python -m venv .venv && source .venv/bin/activate
python -V
pip install -U pip
pip install -e .[dev]

print_status "INFO" "Running lint and format checks..."
ruff check .
ruff format --check .

print_status "INFO" "Running tests..."
pytest -q

print_status "INFO" "Running CLI smoke tests..."
set -x
rldk evals list-suites || true
rldk seed --show || true
rldk check-determinism --cmd "python -c 'print(1)'" --compare loss || true
rldk replay --help | head -n 5 || true
rldk forensics --help | head -n 5 || true
rldk ingest --help | head -n 5 || true
rldk card --help | head -n 5 || true
set +x

print_status "INFO" "Running examples smoke test..."
python examples/basic_ppo_cartpole.py || true

print_status "SUCCESS" "🎉 Refactor acceptance checks completed!"
echo "Ready for next refactor step! 🚀"
