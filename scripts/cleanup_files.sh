#!/bin/bash

# File Cleanup Script for RLDK
# This script helps clean up common file organization issues

set -e

echo "🧹 RLDK File Cleanup Script"
echo "=========================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to ask for confirmation
confirm() {
    read -p "$1 (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

# Function to move file with confirmation
move_file() {
    local source="$1"
    local destination="$2"
    local description="$3"
    
    if [ -f "$source" ]; then
        echo -e "${BLUE}Found: $source${NC}"
        echo -e "${YELLOW}Description: $description${NC}"
        echo -e "${BLUE}Suggested destination: $destination${NC}"
        
        if confirm "Move this file?"; then
            # Create destination directory if it doesn't exist
            mkdir -p "$(dirname "$destination")"
            
            # Move the file
            mv "$source" "$destination"
            echo -e "${GREEN}✅ Moved $source to $destination${NC}"
            return 0
        else
            echo -e "${YELLOW}⏭️  Skipped $source${NC}"
            return 1
        fi
    fi
    return 0
}

# Function to remove file with confirmation
remove_file() {
    local file="$1"
    local description="$2"
    
    if [ -f "$file" ]; then
        echo -e "${BLUE}Found: $file${NC}"
        echo -e "${YELLOW}Description: $description${NC}"
        
        if confirm "Remove this file?"; then
            rm "$file"
            echo -e "${GREEN}✅ Removed $file${NC}"
            return 0
        else
            echo -e "${YELLOW}⏭️  Skipped $file${NC}"
            return 1
        fi
    fi
    return 0
}

echo ""
echo "🔍 Step 1: Cleaning up Python files in root directory"
echo "====================================================="

# Find Python files in root
PYTHON_FILES=$(find . -maxdepth 1 -name "*.py" -not -name "setup.py" -not -name "conftest.py")

if [ -n "$PYTHON_FILES" ]; then
    echo "Found Python files in root directory:"
    for file in $PYTHON_FILES; do
        filename=$(basename "$file")
        
        # Determine appropriate destination based on filename
        if [[ "$filename" =~ ^test_.* ]]; then
            destination="tests/unit/$filename"
            description="Test file - should be in tests/"
        elif [[ "$filename" =~ ^.*_demo.* ]] || [[ "$filename" =~ ^.*_example.* ]]; then
            destination="examples/$filename"
            description="Example/demo file - should be in examples/"
        elif [[ "$filename" =~ ^.*_script.* ]] || [[ "$filename" =~ ^.*_tool.* ]]; then
            destination="scripts/$filename"
            description="Script/tool file - should be in scripts/"
        else
            destination="src/rldk/$filename"
            description="Source file - should be in src/rldk/"
        fi
        
        move_file "$file" "$destination" "$description"
    done
else
    echo -e "${GREEN}✅ No Python files found in root directory${NC}"
fi

echo ""
echo "🔍 Step 2: Cleaning up test files in wrong locations"
echo "====================================================="

# Find test files outside tests directory
TEST_FILES=$(find . -name "test_*.py" -not -path "./tests/*")

if [ -n "$TEST_FILES" ]; then
    echo "Found test files outside tests/ directory:"
    for file in $TEST_FILES; do
        # Determine if it's a unit, integration, or e2e test
        if [[ "$file" =~ .*integration.* ]]; then
            destination="tests/integration/$(basename "$file")"
            description="Integration test - should be in tests/integration/"
        elif [[ "$file" =~ .*e2e.* ]] || [[ "$file" =~ .*end_to_end.* ]]; then
            destination="tests/e2e/$(basename "$file")"
            description="E2E test - should be in tests/e2e/"
        else
            destination="tests/unit/$(basename "$file")"
            description="Unit test - should be in tests/unit/"
        fi
        
        move_file "$file" "$destination" "$description"
    done
else
    echo -e "${GREEN}✅ No test files found outside tests/ directory${NC}"
fi

echo ""
echo "🔍 Step 3: Cleaning up documentation files in wrong locations"
echo "=============================================================="

# Find documentation files outside docs directory
DOC_FILES=$(find . -name "*.md" -not -path "./docs/*" -not -path "./examples/*" -not -path "./reference/*" -not -name "README.md" -not -name "CONTRIBUTING.md" -not -name "agents.md" -not -name "PROJECT_STRUCTURE.md" -not -name "PR_GUIDELINES.md")

if [ -n "$DOC_FILES" ]; then
    echo "Found documentation files outside docs/ directory:"
    for file in $DOC_FILES; do
        filename=$(basename "$file")
        
        # Determine appropriate destination based on filename
        if [[ "$filename" =~ .*SUMMARY.* ]] || [[ "$filename" =~ .*IMPLEMENTATION.* ]] || [[ "$filename" =~ .*FIXES.* ]]; then
            destination="docs/implementation/$filename"
            description="Implementation documentation - should be in docs/implementation/"
        elif [[ "$filename" =~ .*API.* ]] || [[ "$filename" =~ .*REFERENCE.* ]]; then
            destination="docs/reference/$filename"
            description="API documentation - should be in docs/reference/"
        elif [[ "$filename" =~ .*TUTORIAL.* ]] || [[ "$filename" =~ .*GUIDE.* ]]; then
            destination="docs/tutorials/$filename"
            description="Tutorial documentation - should be in docs/tutorials/"
        else
            destination="docs/$filename"
            description="General documentation - should be in docs/"
        fi
        
        move_file "$file" "$destination" "$description"
    done
else
    echo -e "${GREEN}✅ No documentation files found outside docs/ directory${NC}"
fi

echo ""
echo "🔍 Step 4: Cleaning up temporary files"
echo "======================================"

# Find temporary files
TEMP_FILES=$(find . -name "temp_*" -o -name "*_temp" -o -name "tmp_*" -o -name "*_tmp" -o -name "*.tmp" -o -name "*.temp")

if [ -n "$TEMP_FILES" ]; then
    echo "Found temporary files:"
    for file in $TEMP_FILES; do
        remove_file "$file" "Temporary file - can be safely removed"
    done
else
    echo -e "${GREEN}✅ No temporary files found${NC}"
fi

echo ""
echo "🔍 Step 5: Cleaning up duplicate functionality"
echo "==============================================="

# Check for duplicate utility files
UTIL_FILES=$(find . -name "*utils*" -type f)
if [ -n "$UTIL_FILES" ]; then
    echo "Found utility files:"
    for file in $UTIL_FILES; do
        echo -e "${BLUE}$file${NC}"
    done
    
    if confirm "Consolidate utility files into src/rldk/utils/?"; then
        for file in $UTIL_FILES; do
            if [ "$file" != "src/rldk/utils/"* ]; then
                filename=$(basename "$file")
                destination="src/rldk/utils/$filename"
                move_file "$file" "$destination" "Utility file - consolidating into src/rldk/utils/"
            fi
        done
    fi
fi

# Check for duplicate helper files
HELPER_FILES=$(find . -name "*helper*" -type f)
if [ -n "$HELPER_FILES" ]; then
    echo "Found helper files:"
    for file in $HELPER_FILES; do
        echo -e "${BLUE}$file${NC}"
    done
    
    if confirm "Consolidate helper files into src/rldk/utils/?"; then
        for file in $HELPER_FILES; do
            if [ "$file" != "src/rldk/utils/"* ]; then
                filename=$(basename "$file")
                destination="src/rldk/utils/$filename"
                move_file "$file" "$destination" "Helper file - consolidating into src/rldk/utils/"
            fi
        done
    fi
fi

echo ""
echo "🔍 Step 6: Creating missing directories"
echo "======================================="

# Create missing directories if they don't exist
DIRECTORIES=(
    "src/rldk/config/environments"
    "src/rldk/config/presets"
    "tests/unit"
    "tests/integration"
    "tests/e2e"
    "tests/data"
    "tests/fixtures"
    "docs/getting-started"
    "docs/reference"
    "docs/tutorials"
    "docs/architecture"
    "docs/implementation"
    "examples/basic_usage"
    "examples/advanced_features"
    "examples/integrations"
    "examples/demos"
    "examples/notebooks"
)

for dir in "${DIRECTORIES[@]}"; do
    if [ ! -d "$dir" ]; then
        echo -e "${BLUE}Creating directory: $dir${NC}"
        mkdir -p "$dir"
        echo -e "${GREEN}✅ Created $dir${NC}"
    fi
done

echo ""
echo "🔍 Step 7: Updating .gitignore"
echo "==============================="

# Check if .gitignore needs updates
GITIGNORE_UPDATES=(
    "# Temporary files"
    "temp_*"
    "*_temp"
    "tmp_*"
    "*_tmp"
    "*.tmp"
    "*.temp"
    ""
    "# Generated files"
    "*.pyc"
    "__pycache__/"
    "*.pyo"
    "*.pyd"
    ".Python"
    ""
    "# IDE files"
    ".vscode/"
    ".idea/"
    "*.swp"
    "*.swo"
    ""
    "# OS files"
    ".DS_Store"
    "Thumbs.db"
)

if confirm "Update .gitignore with common patterns?"; then
    echo "" >> .gitignore
    echo "# Added by cleanup script" >> .gitignore
    for pattern in "${GITIGNORE_UPDATES[@]}"; do
        echo "$pattern" >> .gitignore
    done
    echo -e "${GREEN}✅ Updated .gitignore${NC}"
fi

echo ""
echo "🎉 Cleanup completed!"
echo "===================="
echo ""
echo "Next steps:"
echo "1. Review the moved files to ensure they're in the right locations"
echo "2. Update any import statements that may have been affected"
echo "3. Run tests to ensure everything still works"
echo "4. Update documentation if needed"
echo ""
echo "To check file organization again, run:"
echo "  ./scripts/check_file_organization.sh"