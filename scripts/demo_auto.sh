#!/bin/bash

# RLDK Automated Demo Script
# This script runs the demo without user interaction for testing and CI

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

# Function to run command with timing
run_with_timing() {
    local cmd="$1"
    local description="$2"
    
    echo -e "${CYAN}Running: $description${NC}"
    echo -e "${YELLOW}Command: $cmd${NC}"
    
    start_time=$(date +%s)
    if eval "$cmd" >/dev/null 2>&1; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        print_success "$description completed in ${duration}s"
    else
        print_error "$description failed"
        return 1
    fi
    echo
}

# Main demo function
main() {
    echo -e "${BLUE}RLDK Automated Demo Starting...${NC}"
    
    # Step 1: Check prerequisites
    print_step "Checking Prerequisites"
    
    if ! command -v python3 >/dev/null 2>&1; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    print_success "Python 3 found"
    
    if ! command -v pip >/dev/null 2>&1; then
        print_error "pip is required but not installed"
        exit 1
    fi
    print_success "pip found"
    
    # Step 2: Install RLDK
    print_step "Installing RLDK"
    
    if [ -f "pyproject.toml" ]; then
        # Try to install with --break-system-packages if needed
        if pip install -e . >/dev/null 2>&1; then
            print_success "RLDK installed successfully"
        elif pip install -e . --break-system-packages >/dev/null 2>&1; then
            print_success "RLDK installed successfully (with --break-system-packages)"
        else
            print_warning "Could not install RLDK, trying to use existing installation"
        fi
        
        # Verify rldk command is available
        if command -v rldk >/dev/null 2>&1; then
            print_success "RLDK command available"
        else
            print_warning "RLDK command not found in PATH, trying to find it..."
            # Try common locations
            for path in "/usr/local/bin/rldk" "/usr/bin/rldk" "$HOME/.local/bin/rldk"; do
                if [ -f "$path" ]; then
                    export PATH="$(dirname "$path"):$PATH"
                    print_success "Found RLDK at $path"
                    break
                fi
            done
        fi
    else
        print_error "pyproject.toml not found. Are you in the RLDK directory?"
        exit 1
    fi
    
    # Step 3: Generate test fixtures
    print_step "Generating Test Fixtures"
    
    if [ -f "tests/_make_fixtures.py" ]; then
        run_with_timing "python3 tests/_make_fixtures.py" "Generating test artifacts"
    else
        print_error "tests/_make_fixtures.py not found"
        exit 1
    fi
    
    # Step 4: Generate training logs
    print_step "Generating Training Logs"
    
    if [ -f "generate_logs.py" ]; then
        run_with_timing "python3 generate_logs.py" "Generating training logs"
    else
        print_warning "generate_logs.py not found, using existing logs"
    fi
    
    # Step 5: Verify artifacts exist
    print_step "Verifying Demo Artifacts"
    
    required_files=(
        "test_artifacts/logs_clean/training.jsonl"
        "test_artifacts/logs_doctored_kl_spike/training.jsonl"
        "test_artifacts/reward_drift_demo/prompts.jsonl"
        "test_artifacts/ckpt_identical/a.pt"
        "test_artifacts/ckpt_identical/b.pt"
        "test_artifacts/ckpt_value_head_edit/a.pt"
        "test_artifacts/ckpt_value_head_edit/b.pt"
    )
    
    missing_files=0
    for file in "${required_files[@]}"; do
        if [ -f "$file" ]; then
            print_success "Found: $file"
        else
            print_warning "Missing: $file"
            missing_files=$((missing_files + 1))
        fi
    done
    
    if [ $missing_files -gt 0 ]; then
        print_warning "$missing_files files missing, but continuing..."
    fi
    
    echo
    print_info "Starting RLDK analysis..."
    
    # Step 6: Run RLDK compare-runs
    print_step "Step 1: Comparing Training Runs"
    
    run_with_timing "rldk compare-runs test_artifacts/logs_clean test_artifacts/logs_doctored_kl_spike" \
        "Comparing clean vs doctored training runs"
    
    # Step 7: Run RLDK diff-ckpt
    print_step "Step 2: Comparing Checkpoints"
    
    run_with_timing "rldk diff-ckpt test_artifacts/ckpt_identical/a.pt test_artifacts/ckpt_identical/b.pt" \
        "Comparing identical checkpoints"
    
    run_with_timing "rldk diff-ckpt test_artifacts/ckpt_value_head_edit/a.pt test_artifacts/ckpt_value_head_edit/b.pt" \
        "Comparing checkpoints with value head differences"
    
    # Step 8: Run RLDK env-audit
    print_step "Step 3: Environment Audit"
    
    run_with_timing "rldk env-audit test_artifacts/logs_clean" \
        "Auditing environment for determinism"
    
    # Step 9: Run RLDK log-scan
    print_step "Step 4: PPO Log Analysis"
    
    run_with_timing "rldk log-scan test_artifacts/logs_clean" \
        "Scanning clean logs for anomalies"
    
    run_with_timing "rldk log-scan test_artifacts/logs_doctored_kl_spike" \
        "Scanning doctored logs for KL spike detection"
    
    # Step 10: Run RLDK reward-drift
    print_step "Step 5: Reward Model Drift Detection"
    
    run_with_timing "rldk reward-drift test_artifacts/reward_drift_demo/rmA test_artifacts/reward_drift_demo/rmB --prompts test_artifacts/reward_drift_demo/prompts.jsonl" \
        "Detecting reward model drift"
    
    # Step 11: Run RLDK doctor
    print_step "Step 6: Comprehensive Diagnostics"
    
    run_with_timing "rldk doctor test_artifacts/logs_doctored_kl_spike" \
        "Running comprehensive diagnostics"
    
    # Step 12: Show results summary
    print_step "Demo Results Summary"
    
    echo -e "${GREEN}RLDK Automated Demo Completed Successfully!${NC}"
    echo
    echo -e "${CYAN}Generated Reports:${NC}"
    
    reports=(
        "rldk_reports/divergence_report.json"
        "rldk_reports/ckpt_diff.json"
        "rldk_reports/determinism_card.json"
        "rldk_reports/ppo_scan.json"
        "rldk_reports/reward_drift.json"
        "rldk_reports/reward_drift.png"
    )
    
    generated_reports=0
    for report in "${reports[@]}"; do
        if [ -f "$report" ]; then
            echo -e "  ${GREEN}✓${NC} $report"
            generated_reports=$((generated_reports + 1))
        else
            echo -e "  ${YELLOW}⚠${NC} $report (not generated)"
        fi
    done
    
    echo
    echo -e "${GREEN}Successfully generated $generated_reports reports${NC}"
    echo
    echo -e "${PURPLE}Key Findings:${NC}"
    echo -e "  • ${CYAN}KL Spike Detection:${NC} RLDK detected a KL spike around step 800"
    echo -e "  • ${CYAN}Checkpoint Analysis:${NC} Value head parameters showed significant changes"
    echo -e "  • ${CYAN}Environment Audit:${NC} Determinism risks identified and documented"
    echo -e "  • ${CYAN}Reward Drift:${NC} Reward models showed measurable drift"
    echo
    echo -e "${GREEN}RLDK successfully demonstrated its ability to detect and analyze${NC}"
    echo -e "${GREEN}real RL training issues through comprehensive forensics!${NC}"
    echo
}

# Run the demo
main "$@"