# Pull Request Guidelines for RLDK

## 🎯 **Overview**

This document provides comprehensive guidelines for creating, reviewing, and managing pull requests in the RLDK project. It addresses common issues like incomplete PRs, lack of broader impact consideration, and inconsistent changes.

## 📋 **Pre-PR Checklist**

### **Before Creating a PR**

- [ ] **File Organization**
  - [ ] All files are in correct directories
  - [ ] No temporary or duplicate files
  - [ ] Proper naming conventions followed
  - [ ] No files created in root directory

- [ ] **Code Quality**
  - [ ] Uses configuration instead of hardcoded values
  - [ ] Follows existing patterns and conventions
  - [ ] Includes proper error handling
  - [ ] Has type hints and docstrings
  - [ ] No circular dependencies introduced

- [ ] **Testing**
  - [ ] Unit tests for new functionality
  - [ ] Integration tests if applicable
  - [ ] All existing tests still pass
  - [ ] Test coverage maintained or improved
  - [ ] Tests are in correct directories

- [ ] **Configuration**
  - [ ] New parameters added to appropriate config files
  - [ ] Configuration validation updated
  - [ ] Environment-specific configs considered
  - [ ] No hardcoded values remain

- [ ] **Documentation**
  - [ ] User documentation updated
  - [ ] API documentation updated
  - [ ] Examples updated if applicable
  - [ ] README updated if needed
  - [ ] Documentation is in correct directories

- [ ] **Dependencies**
  - [ ] No circular dependencies introduced
  - [ ] Import statements updated
  - [ ] Requirements updated if needed
  - [ ] All imports use proper paths

- [ ] **Backward Compatibility**
  - [ ] Existing APIs still work
  - [ ] Configuration changes are backward compatible
  - [ ] Migration path provided if breaking changes

## 🔍 **Impact Analysis**

### **Before Making Changes**

1. **Identify Affected Components**
   ```bash
   # Find all imports of the module you're changing
   grep -r "from.*your_module" src/
   grep -r "import.*your_module" src/
   
   # Find all references to functions/classes
   grep -r "your_function" src/
   grep -r "YourClass" src/
   ```

2. **Check Configuration Dependencies**
   ```bash
   # Find config usage
   grep -r "config\." src/
   grep -r "get_.*_config" src/
   ```

3. **Check Test Dependencies**
   ```bash
   # Find test dependencies
   grep -r "your_module" tests/
   ```

4. **Check Documentation References**
   ```bash
   # Find documentation references
   grep -r "your_feature" docs/
   grep -r "your_function" docs/
   ```

### **Configuration Impact Analysis**

When changing configuration:

1. **Identify All Config Users**
   ```python
   # Search for config usage
   grep -r "config\." src/
   grep -r "get_.*_config" src/
   ```

2. **Check Environment Overrides**
   ```python
   # Check if environment variables are used
   grep -r "os\.environ" src/
   ```

3. **Validate Configuration**
   ```python
   # Run validation
   from src.rldk.config import validate_all_configs
   issues = validate_all_configs()
   if issues:
       print("Configuration issues found:", issues)
   ```

### **API Impact Analysis**

When changing APIs:

1. **Find All Callers**
   ```bash
   # Find function calls
   grep -r "your_function(" src/
   grep -r "YourClass(" src/
   ```

2. **Check Import Patterns**
   ```bash
   # Check import statements
   grep -r "from.*your_module" src/
   grep -r "import.*your_module" src/
   ```

3. **Check Documentation**
   ```bash
   # Check if API is documented
   grep -r "your_function" docs/
   ```

## 📝 **PR Description Template**

### **Standard PR Template**

```markdown
## Summary
Brief description of what this PR does and why it's needed.

## Changes Made
- [ ] Added new feature X
- [ ] Updated configuration Y
- [ ] Fixed bug Z
- [ ] Improved performance of W

## Files Changed
### Source Code
- `src/rldk/new_feature/` - New feature implementation
- `src/rldk/config/evaluation_config.py` - Configuration updates
- `src/rldk/utils/helpers.py` - Utility function updates

### Tests
- `tests/unit/test_new_feature.py` - Unit tests for new feature
- `tests/integration/test_new_feature_integration.py` - Integration tests

### Documentation
- `docs/new_feature.md` - User documentation
- `docs/reference/api.md` - API documentation updates
- `examples/new_feature_example.py` - Usage example

### Configuration
- `src/rldk/config/new_feature_config.py` - New configuration module
- `src/rldk/config/presets/strict.py` - Updated preset

## Configuration Impact
- New parameter: `NEW_PARAMETER` in `EvaluationConfig`
- Default value: `0.5`
- Environment override: `RLDK_NEW_PARAMETER`
- Validation: Must be non-negative

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] E2E tests pass
- [ ] Manual testing completed
- [ ] Backward compatibility verified
- [ ] Performance impact assessed

## Documentation
- [ ] User docs updated
- [ ] API docs updated
- [ ] Examples updated
- [ ] README updated if needed
- [ ] Migration guide provided if needed

## Breaking Changes
- None (or list if any)

## Migration Guide
- None (or provide if breaking changes)

## Performance Impact
- No performance impact (or describe impact)

## Security Considerations
- No security implications (or describe implications)

## Dependencies
- No new dependencies (or list new dependencies)
- No dependency updates (or list updates)

## Checklist
- [ ] Code follows project conventions
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Configuration is properly handled
- [ ] No hardcoded values remain
- [ ] Error handling is included
- [ ] Type hints are present
- [ ] Docstrings are complete
```

### **Bug Fix PR Template**

```markdown
## Summary
Fixes bug in [component] where [description of issue].

## Problem
Describe the problem that was occurring.

## Solution
Describe how the problem was fixed.

## Files Changed
- `src/rldk/component/file.py` - Fixed the bug
- `tests/unit/test_component.py` - Added regression test

## Testing
- [ ] Regression test added
- [ ] All existing tests pass
- [ ] Manual testing completed

## Breaking Changes
- None

## Migration Guide
- None
```

### **Feature Addition PR Template**

```markdown
## Summary
Adds new feature [feature name] to [component].

## Feature Description
Describe what the new feature does and how it works.

## Usage Example
```python
from rldk.new_feature import NewFeature

feature = NewFeature()
result = feature.process(data)
```

## Files Changed
### Source Code
- `src/rldk/new_feature/` - New feature implementation
- `src/rldk/config/new_feature_config.py` - Configuration

### Tests
- `tests/unit/test_new_feature.py` - Unit tests
- `tests/integration/test_new_feature_integration.py` - Integration tests

### Documentation
- `docs/new_feature.md` - User documentation
- `examples/new_feature_example.py` - Usage example

## Configuration
- New parameters: `PARAM1`, `PARAM2`
- Default values: `0.5`, `100`
- Environment overrides: `RLDK_PARAM1`, `RLDK_PARAM2`

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Breaking Changes
- None

## Migration Guide
- None
```

## 🔄 **Review Process**

### **Review Checklist**

#### **For Reviewers**

- [ ] **Code Quality**
  - [ ] Code follows project conventions
  - [ ] No hardcoded values
  - [ ] Proper error handling
  - [ ] Type hints present
  - [ ] Docstrings complete

- [ ] **File Organization**
  - [ ] Files in correct directories
  - [ ] No temporary files
  - [ ] Proper naming conventions
  - [ ] No duplicate functionality

- [ ] **Configuration**
  - [ ] New parameters in appropriate config files
  - [ ] Configuration validation updated
  - [ ] Environment overrides considered
  - [ ] No hardcoded values remain

- [ ] **Testing**
  - [ ] Unit tests for new functionality
  - [ ] Integration tests if applicable
  - [ ] All existing tests pass
  - [ ] Test coverage maintained

- [ ] **Documentation**
  - [ ] User documentation updated
  - [ ] API documentation updated
  - [ ] Examples updated if applicable
  - [ ] README updated if needed

- [ ] **Dependencies**
  - [ ] No circular dependencies
  - [ ] Import statements correct
  - [ ] Requirements updated if needed

- [ ] **Backward Compatibility**
  - [ ] Existing APIs still work
  - [ ] Configuration changes backward compatible
  - [ ] Migration path provided if breaking changes

#### **For Authors**

- [ ] **Pre-submission**
  - [ ] All checklist items completed
  - [ ] Impact analysis performed
  - [ ] All tests pass
  - [ ] Documentation updated

- [ ] **During Review**
  - [ ] Respond to reviewer feedback
  - [ ] Make requested changes
  - [ ] Update tests if needed
  - [ ] Update documentation if needed

### **Review Guidelines**

#### **What to Look For**

1. **File Organization Issues**
   - Files in wrong directories
   - Temporary or duplicate files
   - Inconsistent naming

2. **Configuration Issues**
   - Hardcoded values
   - Missing configuration
   - Inconsistent config usage

3. **Testing Issues**
   - Missing tests
   - Tests in wrong locations
   - Inadequate test coverage

4. **Documentation Issues**
   - Missing documentation
   - Outdated documentation
   - Documentation in wrong locations

5. **Dependency Issues**
   - Circular dependencies
   - Incorrect imports
   - Missing dependencies

#### **Common Issues to Flag**

1. **Hardcoded Values**
   ```python
   # ❌ Flag this
   if len(data) > 10:
       # process data
   
   # ✅ Should be
   if len(data) > config.MIN_SAMPLES_FOR_ANALYSIS:
       # process data
   ```

2. **Files in Wrong Locations**
   ```bash
   # ❌ Flag these
   test_something.py  # Should be in tests/
   NEW_FEATURE.md    # Should be in docs/
   temp_analysis.py  # Should be removed or in examples/
   ```

3. **Missing Configuration**
   ```python
   # ❌ Flag this
   def new_function(threshold=0.5):  # Hardcoded default
       pass
   
   # ✅ Should be
   def new_function(config=None):
       if config is None:
           config = get_eval_config()
       threshold = config.NEW_THRESHOLD
   ```

4. **Missing Tests**
   ```python
   # ❌ Flag this
   # New functionality added but no tests
   
   # ✅ Should have
   # tests/unit/test_new_functionality.py
   ```

## 🚨 **Common PR Issues**

### **File Organization Issues**

1. **Files in Root Directory**
   ```bash
   # ❌ Common mistake
   touch test_something.py
   touch NEW_FEATURE_SUMMARY.md
   touch temp_analysis.py
   
   # ✅ Should be
   touch tests/unit/test_something.py
   touch docs/implementation/new_feature.md
   touch examples/analysis_example.py
   ```

2. **Duplicate Functionality**
   ```python
   # ❌ Common mistake
   # src/rldk/utils.py - has utility functions
   # src/rldk/helpers.py - has similar utility functions
   
   # ✅ Should consolidate
   # src/rldk/utils.py - all utility functions
   ```

3. **Inconsistent Naming**
   ```bash
   # ❌ Common mistake
   touch fix_stuff.py
   touch temp_analysis.py
   
   # ✅ Should be
   touch src/rldk/analysis/kl_divergence_analyzer.py
   ```

### **Configuration Issues**

1. **Hardcoded Values**
   ```python
   # ❌ Common mistake
   if kl_divergence > 0.1:
       # do something
   
   # ✅ Should be
   if kl_divergence > config.KL_DIVERGENCE_THRESHOLD:
       # do something
   ```

2. **Missing Configuration**
   ```python
   # ❌ Common mistake
   def new_function(threshold=0.5):  # Hardcoded default
       pass
   
   # ✅ Should be
   def new_function(config=None):
       if config is None:
           config = get_eval_config()
       threshold = config.NEW_THRESHOLD
   ```

3. **Inconsistent Config Usage**
   ```python
   # ❌ Common mistake
   # Some functions use config, others don't
   
   # ✅ Should be consistent
   # All functions should use config
   ```

### **Testing Issues**

1. **Missing Tests**
   ```python
   # ❌ Common mistake
   # New functionality added but no tests
   
   # ✅ Should have comprehensive tests
   # tests/unit/test_new_functionality.py
   ```

2. **Tests in Wrong Location**
   ```bash
   # ❌ Common mistake
   touch test_a.py
   touch test_b.py
   
   # ✅ Should be organized
   touch tests/unit/test_a.py
   touch tests/unit/test_b.py
   ```

3. **Inadequate Test Coverage**
   ```python
   # ❌ Common mistake
   def test_new_function():
       result = new_function()
       assert result is not None
   
   # ✅ Should test edge cases
   def test_new_function_with_default_config(self):
       result = new_function()
       assert result is not None
   
   def test_new_function_with_custom_config(self):
       config = create_custom_config()
       result = new_function(config)
       assert result is not None
   
   def test_new_function_edge_cases(self):
       # Test edge cases
       pass
   ```

### **Documentation Issues**

1. **Missing Documentation**
   ```python
   # ❌ Common mistake
   def new_function():
       pass
   
   # ✅ Should have documentation
   def new_function(config=None):
       """
       New function that does something important.
       
       Args:
           config: Configuration object
           
       Returns:
           Result of the operation
       """
       pass
   ```

2. **Documentation in Wrong Place**
   ```bash
   # ❌ Common mistake
   touch NEW_FEATURE_SUMMARY.md
   touch IMPLEMENTATION_NOTES.md
   
   # ✅ Should be in docs/
   touch docs/implementation/new_feature.md
   touch docs/architecture/implementation_notes.md
   ```

3. **Outdated Documentation**
   ```python
   # ❌ Common mistake
   # Documentation says function takes 2 parameters
   # But function actually takes 3 parameters
   
   # ✅ Should keep documentation current
   ```

## 🔧 **Automated Checks**

### **Pre-commit Hooks**

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running pre-commit checks..."

# Check file organization
echo "Checking file organization..."
find . -maxdepth 1 -name "*.py" -not -name "setup.py" -not -name "conftest.py"
if [ $? -eq 0 ]; then
    echo "ERROR: Python files found in root directory"
    exit 1
fi

# Check for hardcoded values
echo "Checking for hardcoded values..."
grep -r "if.*> [0-9]" src/ --include="*.py" | grep -v "config\."
if [ $? -eq 0 ]; then
    echo "ERROR: Hardcoded values found"
    exit 1
fi

# Run tests
echo "Running tests..."
pytest tests/unit/
if [ $? -ne 0 ]; then
    echo "ERROR: Tests failed"
    exit 1
fi

# Check configuration
echo "Checking configuration..."
python -c "from src.rldk.config import validate_all_configs; validate_all_configs()"
if [ $? -ne 0 ]; then
    echo "ERROR: Configuration validation failed"
    exit 1
fi

echo "All checks passed!"
```

### **CI/CD Checks**

```yaml
# .github/workflows/pr-checks.yml
name: PR Checks

on:
  pull_request:
    branches: [ main ]

jobs:
  file-organization:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Check file organization
        run: |
          # Check for files in wrong locations
          if [ -f "test_*.py" ]; then
            echo "ERROR: Test files found in root"
            exit 1
          fi
          
          if [ -f "*_SUMMARY.md" ]; then
            echo "ERROR: Summary files found in root"
            exit 1
          fi

  configuration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Check configuration
        run: |
          # Check for hardcoded values
          grep -r "if.*> [0-9]" src/ --include="*.py" | grep -v "config\."
          if [ $? -eq 0 ]; then
            echo "ERROR: Hardcoded values found"
            exit 1
          fi

  tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: |
          pip install -e .
          pytest tests/
```

## 📊 **PR Metrics**

### **Quality Metrics**

- **File Organization Score**: Percentage of files in correct locations
- **Configuration Score**: Percentage of parameters using configuration
- **Test Coverage**: Percentage of code covered by tests
- **Documentation Coverage**: Percentage of public APIs documented

### **Tracking Metrics**

```python
# scripts/calculate_pr_metrics.py
import os
import re

def calculate_file_organization_score():
    """Calculate percentage of files in correct locations."""
    total_files = 0
    correct_files = 0
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                total_files += 1
                if is_in_correct_location(root, file):
                    correct_files += 1
    
    return (correct_files / total_files) * 100

def calculate_configuration_score():
    """Calculate percentage of parameters using configuration."""
    total_params = 0
    config_params = 0
    
    for root, dirs, files in os.walk('src/'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    content = f.read()
                    
                # Count hardcoded values
                hardcoded = re.findall(r'if.*> [0-9]', content)
                total_params += len(hardcoded)
                
                # Count config usage
                config_usage = re.findall(r'config\.', content)
                config_params += len(config_usage)
    
    return (config_params / (total_params + config_params)) * 100

def calculate_test_coverage():
    """Calculate test coverage percentage."""
    # Implementation would use coverage.py
    pass

def calculate_documentation_coverage():
    """Calculate documentation coverage percentage."""
    # Implementation would analyze docstrings
    pass
```

## 🎯 **Best Practices**

### **For PR Authors**

1. **Plan Before Coding**
   - Understand the full impact of your changes
   - Identify all affected components
   - Plan the file organization
   - Plan the configuration changes

2. **Follow Established Patterns**
   - Use existing code as a template
   - Follow naming conventions
   - Use configuration instead of hardcoded values
   - Include comprehensive tests

3. **Document Everything**
   - Update user documentation
   - Update API documentation
   - Include usage examples
   - Document configuration changes

4. **Test Thoroughly**
   - Write unit tests
   - Write integration tests
   - Test edge cases
   - Verify backward compatibility

### **For Reviewers**

1. **Check the Big Picture**
   - Verify file organization
   - Check configuration usage
   - Ensure consistency across components
   - Verify documentation updates

2. **Look for Common Issues**
   - Hardcoded values
   - Files in wrong locations
   - Missing tests
   - Missing documentation

3. **Provide Constructive Feedback**
   - Explain why changes are needed
   - Provide specific examples
   - Suggest improvements
   - Help with implementation

### **For Maintainers**

1. **Enforce Standards**
   - Use automated checks
   - Require comprehensive PR descriptions
   - Block PRs with organization issues
   - Maintain quality metrics

2. **Provide Guidance**
   - Update guidelines as needed
   - Provide examples of good PRs
   - Help with complex changes
   - Maintain documentation

## 🚀 **Quick Reference**

### **PR Checklist**
- [ ] Files in correct directories
- [ ] No hardcoded values
- [ ] Configuration properly handled
- [ ] Tests included and passing
- [ ] Documentation updated
- [ ] Backward compatibility maintained

### **Common Issues to Avoid**
- Files in root directory
- Hardcoded values
- Missing tests
- Missing documentation
- Inconsistent naming
- Circular dependencies

### **File Locations**
- Source code: `src/rldk/`
- Tests: `tests/`
- Examples: `examples/`
- Documentation: `docs/`
- Configuration: `src/rldk/config/`

---

**Remember: A good PR considers the broader impact, maintains consistency, and follows established patterns. When in doubt, ask for help or review existing similar implementations.**