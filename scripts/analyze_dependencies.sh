#!/bin/bash

# Dependency Analysis Script for RLDK
# This script analyzes dependencies and identifies potential issues

set -e

echo "🔍 RLDK Dependency Analysis Script"
echo "=================================="

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
echo "🔍 Step 1: Analyzing import patterns"
echo "===================================="

# Find all imports
echo "All imports in source code:"
ALL_IMPORTS=$(grep -r "from.*import\|import.*" src/ --include="*.py" | wc -l)
echo "Total imports found: $ALL_IMPORTS"

# Find local imports
echo "Local imports (rldk.*):"
LOCAL_IMPORTS=$(grep -r "from rldk\|import rldk" src/ --include="*.py" | wc -l)
echo "Local imports found: $LOCAL_IMPORTS"

# Find third-party imports
echo "Third-party imports:"
THIRD_PARTY_IMPORTS=$(grep -r "from.*import\|import.*" src/ --include="*.py" | grep -v "from rldk\|import rldk\|from \.\|import \." | wc -l)
echo "Third-party imports found: $THIRD_PARTY_IMPORTS"

echo ""
echo "🔍 Step 2: Checking for circular dependencies"
echo "============================================"

# Check for circular import patterns
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

# Build dependency graph
dependencies = {}
for root, dirs, files in os.walk('src/rldk'):
    for file in files:
        if file.endswith('.py') and file != '__init__.py':
            filepath = os.path.join(root, file)
            module_name = filepath.replace('src/rldk/', '').replace('.py', '').replace('/', '.')
            imports = find_imports(filepath)
            dependencies[module_name] = [imp for imp in imports if imp.startswith('rldk.')]

# Check for cycles using DFS
def has_cycle(module, visited, rec_stack):
    visited.add(module)
    rec_stack.add(module)
    
    for dep in dependencies.get(module, []):
        dep_name = dep[5:]  # Remove 'rldk.'
        if dep_name not in visited:
            if has_cycle(dep_name, visited, rec_stack):
                return True
        elif dep_name in rec_stack:
            return True
    
    rec_stack.remove(module)
    return False

# Check all modules for cycles
cycles_found = False
for module in dependencies:
    if has_cycle(module, set(), set()):
        print(f'Circular dependency found involving: {module}')
        cycles_found = True

if not cycles_found:
    print('No circular dependencies found')
" 2>/dev/null | wc -l)

if [ $CIRCULAR_IMPORTS -gt 0 ]; then
    report_error "Circular dependencies found"
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

# Build dependency graph
dependencies = {}
for root, dirs, files in os.walk('src/rldk'):
    for file in files:
        if file.endswith('.py') and file != '__init__.py':
            filepath = os.path.join(root, file)
            module_name = filepath.replace('src/rldk/', '').replace('.py', '').replace('/', '.')
            imports = find_imports(filepath)
            dependencies[module_name] = [imp for imp in imports if imp.startswith('rldk.')]

# Check for cycles using DFS
def has_cycle(module, visited, rec_stack):
    visited.add(module)
    rec_stack.add(module)
    
    for dep in dependencies.get(module, []):
        dep_name = dep[5:]  # Remove 'rldk.'
        if dep_name not in visited:
            if has_cycle(dep_name, visited, rec_stack):
                return True
        elif dep_name in rec_stack:
            return True
    
    rec_stack.remove(module)
    return False

# Check all modules for cycles
cycles_found = False
for module in dependencies:
    if has_cycle(module, set(), set()):
        print(f'Circular dependency found involving: {module}')
        cycles_found = True

if not cycles_found:
    print('No circular dependencies found')
" 2>/dev/null
else
    report_success "No circular dependencies found"
fi

echo ""
echo "🔍 Step 3: Checking for unused imports"
echo "======================================="

# Check for unused imports
echo "Checking for unused imports..."
UNUSED_IMPORTS=$(python3 -c "
import ast
import os

def find_unused_imports(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        tree = ast.parse(content)
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        # Check if imports are used
        unused = []
        for imp in imports:
            if imp not in content.replace('import ' + imp, ''):
                unused.append(imp)
        
        return unused
    except:
        return []

# Check for unused imports
unused_found = False
for root, dirs, files in os.walk('src/rldk'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            unused = find_unused_imports(filepath)
            if unused:
                print(f'{filepath}: {unused}')
                unused_found = True

if not unused_found:
    print('No unused imports found')
" 2>/dev/null | wc -l)

if [ $UNUSED_IMPORTS -gt 0 ]; then
    report_warning "Unused imports found"
    python3 -c "
import ast
import os

def find_unused_imports(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        tree = ast.parse(content)
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        # Check if imports are used
        unused = []
        for imp in imports:
            if imp not in content.replace('import ' + imp, ''):
                unused.append(imp)
        
        return unused
    except:
        return []

# Check for unused imports
unused_found = False
for root, dirs, files in os.walk('src/rldk'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            unused = find_unused_imports(filepath)
            if unused:
                print(f'{filepath}: {unused}')
                unused_found = True

if not unused_found:
    print('No unused imports found')
" 2>/dev/null
else
    report_success "No unused imports found"
fi

echo ""
echo "🔍 Step 4: Checking for missing imports"
echo "======================================="

# Check for missing imports
echo "Checking for missing imports..."
MISSING_IMPORTS=$(python3 -c "
import ast
import os

def find_missing_imports(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        tree = ast.parse(content)
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        # Check for common missing imports
        missing = []
        if 'json' in content and 'json' not in imports:
            missing.append('json')
        if 'os' in content and 'os' not in imports:
            missing.append('os')
        if 'sys' in content and 'sys' not in imports:
            missing.append('sys')
        if 'logging' in content and 'logging' not in imports:
            missing.append('logging')
        
        return missing
    except:
        return []

# Check for missing imports
missing_found = False
for root, dirs, files in os.walk('src/rldk'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            missing = find_missing_imports(filepath)
            if missing:
                print(f'{filepath}: {missing}')
                missing_found = True

if not missing_found:
    print('No missing imports found')
" 2>/dev/null | wc -l)

if [ $MISSING_IMPORTS -gt 0 ]; then
    report_error "Missing imports found"
    python3 -c "
import ast
import os

def find_missing_imports(filepath):
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        tree = ast.parse(content)
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        # Check for common missing imports
        missing = []
        if 'json' in content and 'json' not in imports:
            missing.append('json')
        if 'os' in content and 'os' not in imports:
            missing.append('os')
        if 'sys' in content and 'sys' not in imports:
            missing.append('sys')
        if 'logging' in content and 'logging' not in imports:
            missing.append('logging')
        
        return missing
    except:
        return []

# Check for missing imports
missing_found = False
for root, dirs, files in os.walk('src/rldk'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            missing = find_missing_imports(filepath)
            if missing:
                print(f'{filepath}: {missing}')
                missing_found = True

if not missing_found:
    print('No missing imports found')
" 2>/dev/null
else
    report_success "No missing imports found"
fi

echo ""
echo "🔍 Step 5: Analyzing dependency depth"
echo "===================================="

# Analyze dependency depth
echo "Analyzing dependency depth..."
DEPTH_ANALYSIS=$(python3 -c "
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

# Build dependency graph
dependencies = {}
for root, dirs, files in os.walk('src/rldk'):
    for file in files:
        if file.endswith('.py') and file != '__init__.py':
            filepath = os.path.join(root, file)
            module_name = filepath.replace('src/rldk/', '').replace('.py', '').replace('/', '.')
            imports = find_imports(filepath)
            dependencies[module_name] = [imp for imp in imports if imp.startswith('rldk.')]

# Calculate dependency depth
def calculate_depth(module, visited):
    if module in visited:
        return 0
    visited.add(module)
    
    max_depth = 0
    for dep in dependencies.get(module, []):
        dep_name = dep[5:]  # Remove 'rldk.'
        depth = calculate_depth(dep_name, visited.copy())
        max_depth = max(max_depth, depth + 1)
    
    return max_depth

# Calculate depth for all modules
max_depth = 0
for module in dependencies:
    depth = calculate_depth(module, set())
    max_depth = max(max_depth, depth)

print(f'Maximum dependency depth: {max_depth}')
" 2>/dev/null)

echo "$DEPTH_ANALYSIS"

echo ""
echo "🔍 Step 6: Checking import organization"
echo "======================================"

# Check import organization
echo "Checking import organization..."
ORGANIZATION_ISSUES=$(python3 -c "
import ast
import os

def check_import_organization(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        imports = []
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                imports.append((i, line.strip()))
        
        # Check if imports are organized
        stdlib_imports = []
        third_party_imports = []
        local_imports = []
        
        for line_num, imp in imports:
            if 'rldk' in imp:
                local_imports.append((line_num, imp))
            elif any(stdlib in imp for stdlib in ['os', 'sys', 'json', 'math', 'random', 'datetime']):
                stdlib_imports.append((line_num, imp))
            else:
                third_party_imports.append((line_num, imp))
        
        # Check if imports are in correct order
        issues = []
        if stdlib_imports and third_party_imports:
            stdlib_max = max([line_num for line_num, _ in stdlib_imports])
            third_party_min = min([line_num for line_num, _ in third_party_imports])
            if stdlib_max > third_party_min:
                issues.append('Standard library imports after third-party imports')
        
        if third_party_imports and local_imports:
            third_party_max = max([line_num for line_num, _ in third_party_imports])
            local_min = min([line_num for line_num, _ in local_imports])
            if third_party_max > local_min:
                issues.append('Third-party imports after local imports')
        
        return issues
    except:
        return []

# Check all files
organization_issues = []
for root, dirs, files in os.walk('src/rldk'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            issues = check_import_organization(filepath)
            if issues:
                organization_issues.append((filepath, issues))

if organization_issues:
    for filepath, issues in organization_issues:
        print(f'{filepath}: {issues}')
else:
    print('No import organization issues found')
" 2>/dev/null | wc -l)

if [ $ORGANIZATION_ISSUES -gt 0 ]; then
    report_warning "Import organization issues found"
    python3 -c "
import ast
import os

def check_import_organization(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        imports = []
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                imports.append((i, line.strip()))
        
        # Check if imports are organized
        stdlib_imports = []
        third_party_imports = []
        local_imports = []
        
        for line_num, imp in imports:
            if 'rldk' in imp:
                local_imports.append((line_num, imp))
            elif any(stdlib in imp for stdlib in ['os', 'sys', 'json', 'math', 'random', 'datetime']):
                stdlib_imports.append((line_num, imp))
            else:
                third_party_imports.append((line_num, imp))
        
        # Check if imports are in correct order
        issues = []
        if stdlib_imports and third_party_imports:
            stdlib_max = max([line_num for line_num, _ in stdlib_imports])
            third_party_min = min([line_num for line_num, _ in third_party_imports])
            if stdlib_max > third_party_min:
                issues.append('Standard library imports after third-party imports')
        
        if third_party_imports and local_imports:
            third_party_max = max([line_num for line_num, _ in third_party_imports])
            local_min = min([line_num for line_num, _ in local_imports])
            if third_party_max > local_min:
                issues.append('Third-party imports after local imports')
        
        return issues
    except:
        return []

# Check all files
organization_issues = []
for root, dirs, files in os.walk('src/rldk'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            issues = check_import_organization(filepath)
            if issues:
                organization_issues.append((filepath, issues))

if organization_issues:
    for filepath, issues in organization_issues:
        print(f'{filepath}: {issues}')
else:
    print('No import organization issues found')
" 2>/dev/null
else
    report_success "No import organization issues found"
fi

echo ""
echo "📊 Summary:"
echo "==========="

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}🎉 All dependency checks passed!${NC}"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  $WARNINGS warnings found. Consider addressing these issues.${NC}"
    exit 0
else
    echo -e "${RED}❌ $ERRORS errors and $WARNINGS warnings found.${NC}"
    echo ""
    echo "Please fix the dependency issues:"
    echo "1. Resolve circular dependencies"
    echo "2. Remove unused imports"
    echo "3. Add missing imports"
    echo "4. Organize imports properly"
    exit 1
fi