#!/bin/bash

# File Organization Checker for RLDK
# This script checks for common file organization issues

set -e

echo "🔍 Checking file organization..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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
echo "📁 Checking for files in wrong locations..."

# Check for Python files in root
echo "Checking for Python files in root directory..."
PYTHON_FILES_IN_ROOT=$(find . -maxdepth 1 -name "*.py" -not -name "setup.py" -not -name "conftest.py" | wc -l)
if [ $PYTHON_FILES_IN_ROOT -gt 0 ]; then
    report_error "Python files found in root directory:"
    find . -maxdepth 1 -name "*.py" -not -name "setup.py" -not -name "conftest.py"
    echo "These should be moved to src/rldk/ or examples/"
else
    report_success "No Python files in root directory"
fi

# Check for test files in root
echo "Checking for test files in root directory..."
TEST_FILES_IN_ROOT=$(find . -maxdepth 1 -name "test_*.py" | wc -l)
if [ $TEST_FILES_IN_ROOT -gt 0 ]; then
    report_error "Test files found in root directory:"
    find . -maxdepth 1 -name "test_*.py"
    echo "These should be moved to tests/"
else
    report_success "No test files in root directory"
fi

# Check for summary/documentation files in root
echo "Checking for summary/documentation files in root directory..."
SUMMARY_FILES_IN_ROOT=$(find . -maxdepth 1 -name "*_SUMMARY.md" -o -name "*_IMPLEMENTATION*.md" -o -name "*_FIXES*.md" | wc -l)
if [ $SUMMARY_FILES_IN_ROOT -gt 0 ]; then
    report_error "Summary/documentation files found in root directory:"
    find . -maxdepth 1 -name "*_SUMMARY.md" -o -name "*_IMPLEMENTATION*.md" -o -name "*_FIXES*.md"
    echo "These should be moved to docs/"
else
    report_success "No summary/documentation files in root directory"
fi

# Check for temporary files
echo "Checking for temporary files..."
TEMP_FILES=$(find . -name "temp_*" -o -name "*_temp" -o -name "tmp_*" -o -name "*_tmp" | wc -l)
if [ $TEMP_FILES -gt 0 ]; then
    report_warning "Temporary files found:"
    find . -name "temp_*" -o -name "*_temp" -o -name "tmp_*" -o -name "*_tmp"
    echo "Consider removing these or adding to .gitignore"
else
    report_success "No temporary files found"
fi

echo ""
echo "🔍 Checking for duplicate functionality..."

# Check for potential duplicate files
echo "Checking for potential duplicate files..."
DUPLICATE_UTILS=$(find . -name "*utils*" -type f | wc -l)
if [ $DUPLICATE_UTILS -gt 1 ]; then
    report_warning "Multiple utility files found:"
    find . -name "*utils*" -type f
    echo "Consider consolidating these"
else
    report_success "No duplicate utility files found"
fi

DUPLICATE_HELPERS=$(find . -name "*helper*" -type f | wc -l)
if [ $DUPLICATE_HELPERS -gt 1 ]; then
    report_warning "Multiple helper files found:"
    find . -name "*helper*" -type f
    echo "Consider consolidating these"
else
    report_success "No duplicate helper files found"
fi

DUPLICATE_COMMON=$(find . -name "*common*" -type f | wc -l)
if [ $DUPLICATE_COMMON -gt 1 ]; then
    report_warning "Multiple common files found:"
    find . -name "*common*" -type f
    echo "Consider consolidating these"
else
    report_success "No duplicate common files found"
fi

echo ""
echo "🔍 Checking for hardcoded values..."

# Check for hardcoded values in source code
echo "Checking for hardcoded values in source code..."
HARDCODED_VALUES=$(grep -r "if.*> [0-9]" src/ --include="*.py" | grep -v "config\." | grep -v "test_" | wc -l)
if [ $HARDCODED_VALUES -gt 0 ]; then
    report_error "Hardcoded values found in source code:"
    grep -r "if.*> [0-9]" src/ --include="*.py" | grep -v "config\." | grep -v "test_"
    echo "These should use configuration values instead"
else
    report_success "No hardcoded values found in source code"
fi

HARDCODED_VALUES_LT=$(grep -r "if.*< [0-9]" src/ --include="*.py" | grep -v "config\." | grep -v "test_" | wc -l)
if [ $HARDCODED_VALUES_LT -gt 0 ]; then
    report_error "Hardcoded values found in source code:"
    grep -r "if.*< [0-9]" src/ --include="*.py" | grep -v "config\." | grep -v "test_"
    echo "These should use configuration values instead"
fi

echo ""
echo "🔍 Checking configuration usage..."

# Check for missing config imports
echo "Checking for missing config imports..."
MISSING_CONFIG_IMPORTS=$(grep -L "from.*config" src/rldk/*.py 2>/dev/null | wc -l)
if [ $MISSING_CONFIG_IMPORTS -gt 0 ]; then
    report_warning "Files missing config imports:"
    grep -L "from.*config" src/rldk/*.py 2>/dev/null
    echo "Consider adding config imports if these files use parameters"
fi

echo ""
echo "🔍 Checking test organization..."

# Check for tests in wrong locations
echo "Checking for tests in wrong locations..."
TESTS_IN_WRONG_LOCATION=$(find . -name "test_*.py" -not -path "./tests/*" | wc -l)
if [ $TESTS_IN_WRONG_LOCATION -gt 0 ]; then
    report_error "Test files found outside tests/ directory:"
    find . -name "test_*.py" -not -path "./tests/*"
    echo "These should be moved to tests/"
else
    report_success "All test files are in tests/ directory"
fi

echo ""
echo "🔍 Checking documentation organization..."

# Check for documentation in wrong locations
echo "Checking for documentation in wrong locations..."
DOCS_IN_WRONG_LOCATION=$(find . -name "*.md" -not -path "./docs/*" -not -path "./examples/*" -not -path "./reference/*" -not -name "README.md" -not -name "CONTRIBUTING.md" -not -name "agents.md" -not -name "PROJECT_STRUCTURE.md" -not -name "PR_GUIDELINES.md" | wc -l)
if [ $DOCS_IN_WRONG_LOCATION -gt 0 ]; then
    report_warning "Documentation files found outside docs/ directory:"
    find . -name "*.md" -not -path "./docs/*" -not -path "./examples/*" -not -path "./reference/*" -not -name "README.md" -not -name "CONTRIBUTING.md" -not -name "agents.md" -not -name "PROJECT_STRUCTURE.md" -not -name "PR_GUIDELINES.md"
    echo "Consider moving these to docs/ or appropriate subdirectory"
fi

echo ""
echo "🔍 Checking for circular dependencies..."

# Check for potential circular imports
echo "Checking for potential circular imports..."
CIRCULAR_IMPORTS=$(python3 -c "
import ast
import os

def find_imports(filepath):
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports
    except:
        return []

# Check for potential circular imports
for root, dirs, files in os.walk('src/rldk'):
    for file in files:
        if file.endswith('.py') and file != '__init__.py':
            filepath = os.path.join(root, file)
            imports = find_imports(filepath)
            for imp in imports:
                if imp.startswith('rldk.'):
                    print(f'{filepath}: {imp}')
" 2>/dev/null | wc -l)

if [ $CIRCULAR_IMPORTS -gt 0 ]; then
    report_warning "Potential circular imports found:"
    python3 -c "
import ast
import os

def find_imports(filepath):
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports
    except:
        return []

# Check for potential circular imports
for root, dirs, files in os.walk('src/rldk'):
    for file in files:
        if file.endswith('.py') and file != '__init__.py':
            filepath = os.path.join(root, file)
            imports = find_imports(filepath)
            for imp in imports:
                if imp.startswith('rldk.'):
                    print(f'{filepath}: {imp}')
" 2>/dev/null
    echo "Review these imports for potential circular dependencies"
else
    report_success "No obvious circular imports found"
fi

echo ""
echo "📊 Summary:"
echo "==========="

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}🎉 All checks passed! File organization looks good.${NC}"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  $WARNINGS warnings found. Consider addressing these issues.${NC}"
    exit 0
else
    echo -e "${RED}❌ $ERRORS errors and $WARNINGS warnings found.${NC}"
    echo ""
    echo "Please fix the errors before proceeding:"
    echo "1. Move files to correct directories"
    echo "2. Replace hardcoded values with configuration"
    echo "3. Remove temporary files"
    echo "4. Consolidate duplicate functionality"
    exit 1
fi