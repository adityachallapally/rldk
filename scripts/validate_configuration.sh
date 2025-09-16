#!/bin/bash

# Configuration Validation Script for RLDK
# This script validates configuration usage and identifies hardcoded values

set -e

echo "🔧 RLDK Configuration Validation Script"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
ERRORS=0
WARNINGS=0

# Function to report errors
report_error() {
    echo -e "${RED}❌ ERROR: $1${NC}"
    ((ERRORS++))
}

# Function to report warnings
report_warning() {
    echo -e "${YELLOW}⚠️  WARNING: $1${NC}"
    ((WARNINGS++))
}

# Function to report success
report_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

echo ""
echo "🔍 Step 1: Checking for hardcoded values"
echo "======================================="

# Check for hardcoded numeric values in source code
echo "Checking for hardcoded numeric values..."

# Check for hardcoded thresholds
HARDCODED_THRESHOLDS=$(grep -r "if.*> [0-9]" src/ --include="*.py" | grep -v "config\." | grep -v "test_" | grep -v "import" | wc -l)
if [ $HARDCODED_THRESHOLDS -gt 0 ]; then
    report_error "Hardcoded thresholds found:"
    grep -r "if.*> [0-9]" src/ --include="*.py" | grep -v "config\." | grep -v "test_" | grep -v "import"
    echo "These should use configuration values instead"
else
    report_success "No hardcoded thresholds found"
fi

HARDCODED_THRESHOLDS_LT=$(grep -r "if.*< [0-9]" src/ --include="*.py" | grep -v "config\." | grep -v "test_" | grep -v "import" | wc -l)
if [ $HARDCODED_THRESHOLDS_LT -gt 0 ]; then
    report_error "Hardcoded thresholds found:"
    grep -r "if.*< [0-9]" src/ --include="*.py" | grep -v "config\." | grep -v "test_" | grep -v "import"
    echo "These should use configuration values instead"
fi

# Check for hardcoded sample sizes
HARDCODED_SAMPLES=$(grep -r "len.*> [0-9]" src/ --include="*.py" | grep -v "config\." | grep -v "test_" | wc -l)
if [ $HARDCODED_SAMPLES -gt 0 ]; then
    report_error "Hardcoded sample sizes found:"
    grep -r "len.*> [0-9]" src/ --include="*.py" | grep -v "config\." | grep -v "test_"
    echo "These should use configuration values instead"
fi

# Check for hardcoded decimal values
HARDCODED_DECIMALS=$(grep -r "0\.[0-9]" src/ --include="*.py" | grep -v "config\." | grep -v "test_" | grep -v "import" | wc -l)
if [ $HARDCODED_DECIMALS -gt 0 ]; then
    report_warning "Potential hardcoded decimal values found:"
    grep -r "0\.[0-9]" src/ --include="*.py" | grep -v "config\." | grep -v "test_" | grep -v "import" | head -10
    echo "Review these to ensure they should use configuration"
fi

echo ""
echo "🔍 Step 2: Checking configuration imports"
echo "========================================="

# Check for missing config imports
echo "Checking for missing config imports..."
MISSING_CONFIG_IMPORTS=$(find src/rldk -name "*.py" -exec grep -L "from.*config\|import.*config" {} \; 2>/dev/null | wc -l)
if [ $MISSING_CONFIG_IMPORTS -gt 0 ]; then
    report_warning "Files that might need config imports:"
    find src/rldk -name "*.py" -exec grep -L "from.*config\|import.*config" {} \; 2>/dev/null | head -5
    echo "Review these files to see if they need configuration"
fi

# Check for proper config usage patterns
echo "Checking for proper config usage patterns..."
CONFIG_USAGE=$(grep -r "config\." src/ --include="*.py" | wc -l)
if [ $CONFIG_USAGE -gt 0 ]; then
    report_success "Found $CONFIG_USAGE configuration usages"
else
    report_warning "No configuration usage found - this might indicate missing config integration"
fi

echo ""
echo "🔍 Step 3: Checking configuration files"
echo "======================================="

# Check if configuration files exist
CONFIG_FILES=(
    "src/rldk/config/evaluation_config.py"
    "src/rldk/config/forensics_config.py"
    "src/rldk/config/visualization_config.py"
    "src/rldk/config/suite_config.py"
)

for config_file in "${CONFIG_FILES[@]}"; do
    if [ -f "$config_file" ]; then
        report_success "Configuration file exists: $config_file"
    else
        report_error "Missing configuration file: $config_file"
    fi
done

# Check for configuration validation
echo "Checking for configuration validation..."
if [ -f "src/rldk/config/__init__.py" ]; then
    if grep -q "validate_all_configs" src/rldk/config/__init__.py; then
        report_success "Configuration validation function found"
    else
        report_warning "Configuration validation function not found"
    fi
else
    report_error "Configuration __init__.py not found"
fi

echo ""
echo "🔍 Step 4: Checking function signatures"
echo "======================================="

# Check for functions with hardcoded defaults
echo "Checking for functions with hardcoded defaults..."
HARDCODED_DEFAULTS=$(grep -r "def.*= [0-9]" src/ --include="*.py" | grep -v "test_" | wc -l)
if [ $HARDCODED_DEFAULTS -gt 0 ]; then
    report_error "Functions with hardcoded defaults found:"
    grep -r "def.*= [0-9]" src/ --include="*.py" | grep -v "test_"
    echo "These should use configuration instead"
else
    report_success "No functions with hardcoded defaults found"
fi

# Check for functions missing config parameter
echo "Checking for functions missing config parameter..."
FUNCTIONS_WITHOUT_CONFIG=$(grep -r "def.*(" src/ --include="*.py" | grep -v "test_" | grep -v "config=None" | wc -l)
if [ $FUNCTIONS_WITHOUT_CONFIG -gt 0 ]; then
    report_warning "Functions that might need config parameter:"
    grep -r "def.*(" src/ --include="*.py" | grep -v "test_" | grep -v "config=None" | head -5
    echo "Review these functions to see if they need configuration"
fi

echo ""
echo "🔍 Step 5: Checking environment variable usage"
echo "=============================================="

# Check for environment variable usage
echo "Checking for environment variable usage..."
ENV_VAR_USAGE=$(grep -r "os\.environ" src/ --include="*.py" | wc -l)
if [ $ENV_VAR_USAGE -gt 0 ]; then
    report_success "Found $ENV_VAR_USAGE environment variable usages"
    echo "Environment variables found:"
    grep -r "os\.environ" src/ --include="*.py" | head -5
else
    report_warning "No environment variable usage found"
fi

echo ""
echo "🔍 Step 6: Checking configuration presets"
echo "========================================="

# Check for configuration presets
PRESET_DIR="src/rldk/config/presets"
if [ -d "$PRESET_DIR" ]; then
    PRESET_FILES=$(find "$PRESET_DIR" -name "*.py" | wc -l)
    if [ $PRESET_FILES -gt 0 ]; then
        report_success "Found $PRESET_FILES configuration preset files"
    else
        report_warning "No configuration preset files found"
    fi
else
    report_warning "Configuration presets directory not found"
fi

echo ""
echo "🔍 Step 7: Running configuration validation"
echo "==========================================="

# Try to run configuration validation
echo "Attempting to run configuration validation..."
if python3 -c "from src.rldk.config import validate_all_configs; validate_all_configs()" 2>/dev/null; then
    report_success "Configuration validation passed"
else
    report_error "Configuration validation failed"
    echo "Trying to identify the issue..."
    python3 -c "from src.rldk.config import validate_all_configs; validate_all_configs()" 2>&1 | head -10
fi

echo ""
echo "🔍 Step 8: Checking for configuration documentation"
echo "==================================================="

# Check for configuration documentation
CONFIG_DOCS=(
    "docs/configuration.md"
    "docs/reference/configuration.md"
    "src/rldk/config/README.md"
)

for doc_file in "${CONFIG_DOCS[@]}"; do
    if [ -f "$doc_file" ]; then
        report_success "Configuration documentation exists: $doc_file"
    else
        report_warning "Missing configuration documentation: $doc_file"
    fi
done

echo ""
echo "📊 Summary:"
echo "==========="

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}🎉 All configuration checks passed!${NC}"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  $WARNINGS warnings found. Consider addressing these issues.${NC}"
    exit 0
else
    echo -e "${RED}❌ $ERRORS errors and $WARNINGS warnings found.${NC}"
    echo ""
    echo "Please fix the configuration issues:"
    echo "1. Replace hardcoded values with configuration parameters"
    echo "2. Add config parameter to functions that need it"
    echo "3. Update function signatures to use configuration"
    echo "4. Ensure all configuration files exist and are valid"
    exit 1
fi