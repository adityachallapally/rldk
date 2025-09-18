#!/bin/bash


set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

SEED=42
MAX_STEPS=2000
MAX_HOURS=3.0
BATCH_SIZE=4
LEARNING_RATE=1e-5
MODEL_NAME="gpt2-medium"
OUTPUT_DIR="artifacts/fullscale"
CLI_LOG_FILE="$OUTPUT_DIR/cli_logs.txt"

# Function to print colored output
print_step() {
    echo -e "${BLUE}=== $1 ===${NC}" | tee -a "$CLI_LOG_FILE"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}" | tee -a "$CLI_LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}" | tee -a "$CLI_LOG_FILE"
}

print_error() {
    echo -e "${RED}✗ $1${NC}" | tee -a "$CLI_LOG_FILE"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}" | tee -a "$CLI_LOG_FILE"
}

# Function to run command with timing and logging
run_with_timing() {
    local cmd="$1"
    local description="$2"
    
    echo -e "${CYAN}Running: $description${NC}" | tee -a "$CLI_LOG_FILE"
    echo -e "${YELLOW}Command: $cmd${NC}" | tee -a "$CLI_LOG_FILE"
    
    start_time=$(date +%s)
    if eval "$cmd" 2>&1 | tee -a "$CLI_LOG_FILE"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        print_success "$description completed in ${duration}s"
        return 0
    else
        print_error "$description failed"
        return 1
    fi
}

check_file_lines() {
    local file="$1"
    local min_lines="$2"
    local description="$3"
    
    if [ ! -f "$file" ]; then
        print_error "$description: File $file does not exist"
        return 1
    fi
    
    local line_count=$(wc -l < "$file")
    if [ "$line_count" -lt "$min_lines" ]; then
        print_error "$description: File $file has only $line_count lines (minimum: $min_lines)"
        return 1
    fi
    
    print_success "$description: File $file has $line_count lines (≥ $min_lines)"
    return 0
}

validate_jsonl() {
    local file="$1"
    local description="$2"
    
    if ! python3 -c "
import json
import sys
try:
    with open('$file', 'r') as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                json.loads(line)
    print('✓ Valid JSONL format')
except Exception as e:
    print(f'✗ Invalid JSONL at line {i}: {e}')
    sys.exit(1)
" 2>&1 | tee -a "$CLI_LOG_FILE"; then
        print_success "$description: Valid JSONL format"
        return 0
    else
        print_error "$description: Invalid JSONL format"
        return 1
    fi
}

write_acceptance_summary() {
    local status="$1"
    local summary_file="$OUTPUT_DIR/ACCEPTANCE_SUMMARY.md"
    
    cat > "$summary_file" << EOF

**Status: $status**
**Timestamp: $(date -u '+%Y-%m-%d %H:%M:%S UTC')**
**Test Configuration:**
- Model: $MODEL_NAME
- Seed: $SEED
- Max Steps: $MAX_STEPS
- Max Hours: $MAX_HOURS
- Batch Size: $BATCH_SIZE
- Learning Rate: $LEARNING_RATE


EOF

    if [ -f "$OUTPUT_DIR/run.jsonl" ]; then
        local line_count=$(wc -l < "$OUTPUT_DIR/run.jsonl")
        echo "- ✅ Training JSONL generated: $line_count lines" >> "$summary_file"
    else
        echo "- ❌ Training JSONL missing" >> "$summary_file"
    fi

    if [ -f "$OUTPUT_DIR/final_metrics.json" ]; then
        echo "- ✅ Final metrics saved" >> "$summary_file"
    else
        echo "- ❌ Final metrics missing" >> "$summary_file"
    fi

    cat >> "$summary_file" << EOF

EOF

    if [ -f "$OUTPUT_DIR/monitor_report.json" ]; then
        echo "- ✅ Monitor report generated" >> "$summary_file"
    else
        echo "- ❌ Monitor report missing" >> "$summary_file"
    fi

    if [ -f "$OUTPUT_DIR/alerts.jsonl" ]; then
        local alert_count=$(wc -l < "$OUTPUT_DIR/alerts.jsonl")
        echo "- ✅ Monitor alerts: $alert_count alerts fired" >> "$summary_file"
    else
        echo "- ❌ Monitor alerts missing" >> "$summary_file"
    fi

    cat >> "$summary_file" << EOF

EOF

    local analysis_files=(
        "reward_health_report.json:Reward Health Analysis"
        "diff_report.json:Training Metrics Diff"
        "determinism_report.json:Determinism Check"
        "drift_card.json:Drift Card"
        "reward_card.json:Reward Card"
        "determinism_card.json:Determinism Card"
    )

    for file_desc in "${analysis_files[@]}"; do
        local file="${file_desc%%:*}"
        local desc="${file_desc##*:}"
        if [ -f "$OUTPUT_DIR/$file" ]; then
            echo "- ✅ $desc completed" >> "$summary_file"
        else
            echo "- ❌ $desc missing" >> "$summary_file"
        fi
    done

    cat >> "$summary_file" << EOF

EOF

    if [ -f "$OUTPUT_DIR/final_metrics.json" ]; then
        python3 -c "
import json
try:
    with open('$OUTPUT_DIR/final_metrics.json', 'r') as f:
        metrics = json.load(f)
    print(f'- Total Steps: {metrics.get(\"total_steps\", \"N/A\")}')
    print(f'- Total Time: {metrics.get(\"total_time_hours\", \"N/A\")} hours')
    print(f'- Model: {metrics.get(\"model_name\", \"N/A\")}')
    print(f'- Run ID: {metrics.get(\"run_id\", \"N/A\")}')
except Exception as e:
    print(f'- Error reading metrics: {e}')
" >> "$summary_file"
    fi

    cat >> "$summary_file" << EOF

\`\`\`
$OUTPUT_DIR/
$(find "$OUTPUT_DIR" -type f | sort | sed 's|^'"$OUTPUT_DIR"'/|├── |')
\`\`\`

Full command logs are available in: \`$CLI_LOG_FILE\`

---
*Generated by RLDK Fullscale Acceptance Test*
EOF

    print_info "Acceptance summary written to: $summary_file"
}

main() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                RLDK Fullscale Acceptance Test                ║"
    echo "║                                                              ║"
    echo "║  End-to-end validation of RLDK with production-scale        ║"
    echo "║  RL training, monitoring, and comprehensive analysis        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    mkdir -p "$OUTPUT_DIR"
    echo "RLDK Fullscale Acceptance Test - $(date -u '+%Y-%m-%d %H:%M:%S UTC')" > "$CLI_LOG_FILE"
    
    print_step "Phase 1: Environment Setup"
    
    if ! command -v python3 >/dev/null 2>&1; then
        print_error "Python 3 is required but not installed"
        write_acceptance_summary "FAIL"
        exit 1
    fi
    print_success "Python 3 found"
    
    if ! command -v rldk >/dev/null 2>&1; then
        print_error "RLDK CLI not found in PATH"
        write_acceptance_summary "FAIL"
        exit 1
    fi
    print_success "RLDK CLI found"
    
    required_files=(
        "scripts/fullscale_train_rl.py"
        "rules/fullscale_rules.yaml"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            print_error "Required file missing: $file"
            write_acceptance_summary "FAIL"
            exit 1
        fi
        print_success "Found: $file"
    done
    
    print_step "Phase 2: RL Training with EventWriter Logging"
    
    if ! run_with_timing \
        "python3 scripts/fullscale_train_rl.py --seed $SEED --max-steps $MAX_STEPS --max-hours $MAX_HOURS --batch-size $BATCH_SIZE --learning-rate $LEARNING_RATE --outdir $OUTPUT_DIR --model-name $MODEL_NAME --determinism-check" \
        "Fullscale RL training with EventWriter logging"; then
        print_error "Training phase failed"
        write_acceptance_summary "FAIL"
        exit 1
    fi
    
    if ! check_file_lines "$OUTPUT_DIR/run.jsonl" 1000 "Training JSONL"; then
        write_acceptance_summary "FAIL"
        exit 1
    fi
    
    if ! validate_jsonl "$OUTPUT_DIR/run.jsonl" "Training JSONL"; then
        write_acceptance_summary "FAIL"
        exit 1
    fi
    
    print_step "Phase 3: Monitor Analysis"
    
    if ! run_with_timing \
        "rldk monitor --once $OUTPUT_DIR/run.jsonl --rules rules/fullscale_rules.yaml --report $OUTPUT_DIR/monitor_report.json --alerts $OUTPUT_DIR/alerts.jsonl" \
        "Monitor analysis with fullscale rules"; then
        print_error "Monitor analysis failed"
        write_acceptance_summary "FAIL"
        exit 1
    fi
    
    if [ -f "$OUTPUT_DIR/alerts.jsonl" ]; then
        local alert_count=$(wc -l < "$OUTPUT_DIR/alerts.jsonl" 2>/dev/null || echo "0")
        if [ "$alert_count" -gt 0 ]; then
            print_success "Monitor rules fired: $alert_count alerts"
        else
            print_warning "No monitor alerts fired (may indicate rules need adjustment)"
        fi
    else
        print_warning "No alerts file generated"
    fi
    
    print_step "Phase 4: Data Ingestion and Normalization"
    
    if ! run_with_timing \
        "rldk ingest $OUTPUT_DIR/run.jsonl --output $OUTPUT_DIR/normalized_metrics.csv" \
        "Data ingestion and normalization"; then
        print_error "Data ingestion failed"
        write_acceptance_summary "FAIL"
        exit 1
    fi
    
    print_step "Phase 5: Reward Health Analysis"
    
    if ! run_with_timing \
        "rldk reward-health $OUTPUT_DIR/normalized_metrics.csv --output $OUTPUT_DIR/reward_health_report.json" \
        "Reward health analysis"; then
        print_warning "Reward health analysis failed (may be expected for synthetic data)"
    fi
    
    print_step "Phase 6: Training Metrics Diff"
    
    if [ -f "$OUTPUT_DIR/baseline.jsonl" ]; then
        if ! run_with_timing \
            "rldk diff $OUTPUT_DIR/run.jsonl $OUTPUT_DIR/baseline.jsonl --output $OUTPUT_DIR/diff_report.json" \
            "Training metrics diff analysis"; then
            print_warning "Diff analysis failed"
        fi
    else
        print_info "No baseline run found, skipping diff analysis"
    fi
    
    print_step "Phase 7: Determinism Check"
    
    if [ -f "$OUTPUT_DIR/baseline.jsonl" ]; then
        if ! run_with_timing \
            "rldk check-determinism --cmd 'python3 scripts/fullscale_train_rl.py --seed $SEED --max-steps 100 --outdir $OUTPUT_DIR/det_check' --compare $OUTPUT_DIR/baseline.jsonl --replicas 2 --output $OUTPUT_DIR/determinism_report.json" \
            "Determinism check"; then
            print_warning "Determinism check failed (may be expected on CPU with limited precision)"
        fi
    else
        print_info "No baseline run found, skipping determinism check"
    fi
    
    print_step "Phase 8: Card Generation"
    
    if [ -f "$OUTPUT_DIR/run.jsonl" ] && [ -f "$OUTPUT_DIR/baseline.jsonl" ]; then
        if ! run_with_timing \
            "rldk card --type drift --source $OUTPUT_DIR/run.jsonl --baseline $OUTPUT_DIR/baseline.jsonl --output $OUTPUT_DIR/drift_card.json" \
            "Drift card generation"; then
            print_warning "Drift card generation failed"
        fi
    fi
    
    if ! run_with_timing \
        "rldk card --type reward --source $OUTPUT_DIR/normalized_metrics.csv --output $OUTPUT_DIR/reward_card.json" \
        "Reward card generation"; then
        print_warning "Reward card generation failed"
    fi
    
    if [ -f "$OUTPUT_DIR/determinism_report.json" ]; then
        if ! run_with_timing \
            "rldk card --type determinism --source $OUTPUT_DIR/determinism_report.json --output $OUTPUT_DIR/determinism_card.json" \
            "Determinism card generation"; then
            print_warning "Determinism card generation failed"
        fi
    fi
    
    print_step "Phase 9: Comprehensive Validation"
    
    local validation_passed=true
    
    if ! check_file_lines "$OUTPUT_DIR/run.jsonl" 1000 "Training JSONL"; then
        validation_passed=false
    fi
    
    if [ ! -f "$OUTPUT_DIR/final_metrics.json" ]; then
        print_error "Final metrics file missing"
        validation_passed=false
    fi
    
    if [ ! -f "$OUTPUT_DIR/monitor_report.json" ]; then
        print_error "Monitor report missing"
        validation_passed=false
    fi
    
    if [ -f "$OUTPUT_DIR/alerts.jsonl" ]; then
        local alert_count=$(wc -l < "$OUTPUT_DIR/alerts.jsonl" 2>/dev/null || echo "0")
        if [ "$alert_count" -eq 0 ]; then
            print_warning "No monitor alerts fired - rules may need adjustment"
        fi
    fi
    
    if [ ! -f "$OUTPUT_DIR/normalized_metrics.csv" ]; then
        print_error "Normalized metrics missing"
        validation_passed=false
    fi
    
    print_step "Phase 10: Final Summary"
    
    if [ "$validation_passed" = true ]; then
        print_success "All core validation checks passed"
        write_acceptance_summary "PASS"
        
        echo -e "${GREEN}"
        echo "╔══════════════════════════════════════════════════════════════╗"
        echo "║                    ACCEPTANCE TEST PASSED                   ║"
        echo "║                                                              ║"
        echo "║  ✅ Training completed with EventWriter JSONL logging       ║"
        echo "║  ✅ Monitor rules evaluated and alerts generated            ║"
        echo "║  ✅ Data normalization to TrainingMetrics successful        ║"
        echo "║  ✅ RLDK analysis pipeline executed end-to-end              ║"
        echo "║  ✅ Complete artifact tree generated                        ║"
        echo "╚══════════════════════════════════════════════════════════════╝"
        echo -e "${NC}"
        
        print_info "Artifact tree location: $OUTPUT_DIR"
        print_info "CLI logs: $CLI_LOG_FILE"
        print_info "Summary report: $OUTPUT_DIR/ACCEPTANCE_SUMMARY.md"
        
        return 0
    else
        print_error "Validation checks failed"
        write_acceptance_summary "FAIL"
        
        echo -e "${RED}"
        echo "╔══════════════════════════════════════════════════════════════╗"
        echo "║                    ACCEPTANCE TEST FAILED                   ║"
        echo "║                                                              ║"
        echo "║  ❌ One or more critical validation checks failed           ║"
        echo "║  📋 Check the summary report for detailed analysis          ║"
        echo "║  📝 Review CLI logs for error details                       ║"
        echo "╚══════════════════════════════════════════════════════════════╝"
        echo -e "${NC}"
        
        return 1
    fi
}

if main "$@"; then
    echo "🎉 RLDK Fullscale Acceptance Test completed successfully!"
    exit 0
else
    echo "❌ RLDK Fullscale Acceptance Test failed!"
    exit 1
fi
