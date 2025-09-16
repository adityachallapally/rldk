# AI Agent Guidelines for RLDK Development

## 🎯 **Overview**

This document provides comprehensive guidelines for AI agents working on the RLDK (RL Debug Kit) codebase. It addresses common issues like scattered file creation, inconsistent PRs, and lack of broader structural awareness.

## 📁 **Project Structure & File Organization**

### **Core Directory Structure**
```
/workspace/
├── src/rldk/                    # Main source code
│   ├── tracking/               # Experiment tracking
│   ├── forensics/              # PPO analysis and debugging
│   ├── ingest/                 # Data ingestion
│   ├── adapters/               # Framework adapters
│   ├── diff/                   # Run comparison
│   ├── determinism/             # Determinism checking
│   ├── reward/                 # Reward analysis
│   ├── evals/                  # Evaluation suites
│   ├── replay/                 # Seeded replay
│   ├── bisect/                 # Git bisect
│   ├── cards/                  # Trust cards
│   ├── io/                     # I/O utilities
│   ├── config/                 # Configuration management
│   └── integrations/           # Framework integrations
├── tests/                      # Test files
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── e2e/                    # End-to-end tests
├── docs/                       # Documentation
│   ├── getting-started/        # Getting started guides
│   ├── reference/              # API reference
│   └── evals/                  # Evaluation documentation
├── examples/                   # Example scripts and demos
├── scripts/                    # Utility scripts
├── tools/                      # Development tools
├── templates/                  # Template files
├── reference/                  # Reference materials
└── recipes/                    # Usage recipes
```

### **File Placement Rules**

#### **✅ CORRECT File Placement**

**Source Code:**
- All Python modules go in `src/rldk/`
- Tests go in `tests/` with matching directory structure
- Examples go in `examples/`
- Documentation goes in `docs/`

**Configuration Files:**
- All config goes in `src/rldk/config/`
- Environment configs in `src/rldk/config/environments/`
- Preset configs in `src/rldk/config/presets/`

**Documentation:**
- User docs in `docs/`
- API docs in `docs/reference/`
- Implementation summaries in `docs/` (not root)
- README files in appropriate subdirectories

**Test Files:**
- Unit tests in `tests/unit/`
- Integration tests in `tests/integration/`
- E2E tests in `tests/e2e/`
- Test data in `tests/data/`

#### **❌ INCORRECT File Placement**

**Never create these in the root:**
- Random `.py` files (use `src/rldk/` or `examples/`)
- Test files (use `tests/`)
- Documentation files (use `docs/`)
- Configuration files (use `src/rldk/config/`)
- Example scripts (use `examples/`)

**Never create these anywhere:**
- Temporary files (use `/tmp/` or add to `.gitignore`)
- Duplicate files with different names
- Files with unclear purposes

## 🔧 **Development Workflow**

### **Before Making Changes**

1. **Understand the Impact:**
   ```bash
   # Check what files are affected by your change
   git status
   git diff --name-only
   
   # Check dependencies
   grep -r "import.*your_module" src/
   grep -r "from.*your_module" src/
   ```

2. **Check Existing Structure:**
   ```bash
   # Look for similar functionality
   find src/ -name "*similar*" -type f
   find examples/ -name "*similar*" -type f
   ```

3. **Review Configuration:**
   ```bash
   # Check if your change affects config
   grep -r "your_parameter" src/rldk/config/
   ```

### **Making Changes**

#### **1. Configuration Changes**
If your change involves parameters or thresholds:

```python
# ✅ CORRECT: Add to appropriate config file
# In src/rldk/config/evaluation_config.py
class EvaluationConfig:
    # Add your new parameter
    NEW_PARAMETER = 0.5
    
    # Update validation
    def validate(self):
        if self.NEW_PARAMETER < 0:
            raise ValueError("NEW_PARAMETER must be non-negative")
```

```python
# ❌ WRONG: Hardcoded values
if some_value > 0.5:  # This should be config.NEW_PARAMETER
    # do something
```

#### **2. New Features**
```python
# ✅ CORRECT: Proper module structure
# Create src/rldk/new_feature/
# ├── __init__.py
# ├── core.py
# ├── utils.py
# └── config.py

# Update src/rldk/__init__.py
from .new_feature import NewFeature

# Update src/rldk/config/__init__.py
from .new_feature_config import get_new_feature_config
```

#### **3. Tests**
```python
# ✅ CORRECT: Matching test structure
# Create tests/unit/test_new_feature.py
# Create tests/integration/test_new_feature_integration.py

# ❌ WRONG: Test files in wrong locations
# test_new_feature.py in root directory
```

#### **4. Documentation**
```markdown
<!-- ✅ CORRECT: Proper documentation structure -->
<!-- Create docs/new_feature.md -->
<!-- Update docs/index.md -->
<!-- Update README.md if needed -->

<!-- ❌ WRONG: Random markdown files -->
<!-- NEW_FEATURE_SUMMARY.md in root -->
```

### **After Making Changes**

1. **Update Dependencies:**
   ```bash
   # Check if you need to update imports
   grep -r "old_module_name" src/
   
   # Update any affected files
   ```

2. **Update Configuration:**
   ```bash
   # Check if config needs updates
   python -c "from src.rldk.config import validate_all_configs; validate_all_configs()"
   ```

3. **Update Tests:**
   ```bash
   # Run tests to ensure nothing broke
   pytest tests/unit/test_your_module.py
   pytest tests/integration/
   ```

4. **Update Documentation:**
   ```bash
   # Update relevant docs
   # Check if examples need updates
   # Update API documentation
   ```

## 📋 **PR Guidelines**

### **PR Checklist**

Before creating a PR, ensure:

- [ ] **File Organization:**
  - [ ] All files are in correct directories
  - [ ] No temporary or duplicate files
  - [ ] Proper naming conventions followed

- [ ] **Code Quality:**
  - [ ] Uses configuration instead of hardcoded values
  - [ ] Follows existing patterns and conventions
  - [ ] Includes proper error handling
  - [ ] Has type hints and docstrings

- [ ] **Testing:**
  - [ ] Unit tests for new functionality
  - [ ] Integration tests if applicable
  - [ ] All existing tests still pass
  - [ ] Test coverage maintained or improved

- [ ] **Configuration:**
  - [ ] New parameters added to appropriate config files
  - [ ] Configuration validation updated
  - [ ] Environment-specific configs considered

- [ ] **Documentation:**
  - [ ] User documentation updated
  - [ ] API documentation updated
  - [ ] Examples updated if applicable
  - [ ] README updated if needed

- [ ] **Dependencies:**
  - [ ] No circular dependencies introduced
  - [ ] Import statements updated
  - [ ] Requirements updated if needed

- [ ] **Backward Compatibility:**
  - [ ] Existing APIs still work
  - [ ] Configuration changes are backward compatible
  - [ ] Migration path provided if breaking changes

### **PR Description Template**

```markdown
## Summary
Brief description of what this PR does.

## Changes Made
- [ ] Added new feature X
- [ ] Updated configuration Y
- [ ] Fixed bug Z

## Files Changed
- `src/rldk/new_feature/` - New feature implementation
- `src/rldk/config/evaluation_config.py` - Configuration updates
- `tests/unit/test_new_feature.py` - Unit tests
- `docs/new_feature.md` - Documentation

## Configuration Impact
- New parameter: `NEW_PARAMETER` in `EvaluationConfig`
- Default value: `0.5`
- Environment override: `RLDK_NEW_PARAMETER`

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Backward compatibility verified

## Documentation
- [ ] User docs updated
- [ ] API docs updated
- [ ] Examples updated
- [ ] README updated

## Breaking Changes
- None (or list if any)

## Migration Guide
- None (or provide if breaking changes)
```

## 🔍 **Impact Analysis**

### **Before Making Changes**

1. **Identify Affected Components:**
   ```bash
   # Find all imports of the module you're changing
   grep -r "from.*your_module" src/
   grep -r "import.*your_module" src/
   
   # Find all references to functions/classes
   grep -r "your_function" src/
   grep -r "YourClass" src/
   ```

2. **Check Configuration Dependencies:**
   ```bash
   # Find config usage
   grep -r "config\." src/
   grep -r "get_.*_config" src/
   ```

3. **Check Test Dependencies:**
   ```bash
   # Find test dependencies
   grep -r "your_module" tests/
   ```

### **Configuration Impact Analysis**

When changing configuration:

1. **Identify All Config Users:**
   ```python
   # Search for config usage
   grep -r "config\." src/
   grep -r "get_.*_config" src/
   ```

2. **Check Environment Overrides:**
   ```python
   # Check if environment variables are used
   grep -r "os\.environ" src/
   ```

3. **Validate Configuration:**
   ```python
   # Run validation
   from src.rldk.config import validate_all_configs
   issues = validate_all_configs()
   if issues:
       print("Configuration issues found:", issues)
   ```

### **API Impact Analysis**

When changing APIs:

1. **Find All Callers:**
   ```bash
   # Find function calls
   grep -r "your_function(" src/
   grep -r "YourClass(" src/
   ```

2. **Check Import Patterns:**
   ```bash
   # Check import statements
   grep -r "from.*your_module" src/
   grep -r "import.*your_module" src/
   ```

3. **Check Documentation:**
   ```bash
   # Check if API is documented
   grep -r "your_function" docs/
   ```

## 🚨 **Common Mistakes to Avoid**

### **File Organization Mistakes**

1. **❌ Creating files in wrong locations:**
   ```bash
   # WRONG: Test file in root
   touch test_something.py
   
   # CORRECT: Test file in tests directory
   touch tests/unit/test_something.py
   ```

2. **❌ Duplicate functionality:**
   ```python
   # WRONG: Creating similar functionality in multiple places
   # src/rldk/utils.py - has utility functions
   # src/rldk/helpers.py - has similar utility functions
   
   # CORRECT: Consolidate in one place
   # src/rldk/utils.py - all utility functions
   ```

3. **❌ Random naming:**
   ```bash
   # WRONG: Unclear file names
   touch fix_stuff.py
   touch temp_analysis.py
   
   # CORRECT: Clear, descriptive names
   touch src/rldk/analysis/kl_divergence_analyzer.py
   ```

### **Configuration Mistakes**

1. **❌ Hardcoded values:**
   ```python
   # WRONG: Hardcoded thresholds
   if kl_divergence > 0.1:
       # do something
   
   # CORRECT: Configuration-based
   if kl_divergence > config.KL_DIVERGENCE_THRESHOLD:
       # do something
   ```

2. **❌ Missing configuration:**
   ```python
   # WRONG: No configuration for new parameters
   def new_function(threshold=0.5):  # Hardcoded default
       pass
   
   # CORRECT: Configuration-based
   def new_function(config=None):
       if config is None:
           config = get_eval_config()
       threshold = config.NEW_THRESHOLD
   ```

### **Testing Mistakes**

1. **❌ Missing tests:**
   ```python
   # WRONG: No tests for new functionality
   # New feature added but no tests
   
   # CORRECT: Comprehensive tests
   # tests/unit/test_new_feature.py
   # tests/integration/test_new_feature_integration.py
   ```

2. **❌ Tests in wrong location:**
   ```bash
   # WRONG: Test files scattered
   touch test_a.py
   touch test_b.py
   
   # CORRECT: Organized test structure
   touch tests/unit/test_a.py
   touch tests/unit/test_b.py
   ```

### **Documentation Mistakes**

1. **❌ Missing documentation:**
   ```python
   # WRONG: No documentation for new features
   def new_function():
       pass
   
   # CORRECT: Proper documentation
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

2. **❌ Documentation in wrong place:**
   ```bash
   # WRONG: Random documentation files
   touch NEW_FEATURE_SUMMARY.md
   touch IMPLEMENTATION_NOTES.md
   
   # CORRECT: Proper documentation structure
   touch docs/new_feature.md
   touch docs/implementation_notes.md
   ```

## 🛠 **Manual Checks**

### **File Organization Checks**

You can manually check for common file organization issues:

1. **Files in wrong locations**:
   - Look for Python files in root directory
   - Look for test files outside `tests/` directory
   - Look for documentation files outside `docs/` directory

2. **Duplicate functionality**:
   - Search for multiple utility files
   - Search for multiple helper files
   - Search for multiple common files

3. **Configuration usage**:
   - Search for hardcoded numeric values
   - Check if functions use configuration parameters
   - Verify configuration imports are present

### **Dependency Analysis**

You can manually analyze dependencies:

1. **Find imports**:
   - Search for all import statements
   - Check for circular import patterns
   - Look for unused imports

2. **Check dependencies**:
   - Verify import order is correct
   - Check for missing imports
   - Look for potential circular dependencies

## 📚 **Best Practices**

### **1. Always Use Configuration**

```python
# ✅ CORRECT: Configuration-based
from ..config import get_eval_config

def analyze_data(data, config=None):
    if config is None:
        config = get_eval_config()
    
    if len(data) > config.MIN_SAMPLES_FOR_ANALYSIS:
        # process data
        pass

# ❌ WRONG: Hardcoded values
def analyze_data(data):
    if len(data) > 10:  # Hardcoded!
        # process data
        pass
```

### **2. Follow Existing Patterns**

```python
# ✅ CORRECT: Follow existing patterns
class NewAnalyzer:
    def __init__(self, config=None):
        if config is None:
            config = get_eval_config()
        self.config = config
    
    def analyze(self, data):
        # Use self.config for all parameters
        pass

# ❌ WRONG: Inconsistent patterns
def new_analyzer_function(data, threshold=0.5):  # Hardcoded default
    pass
```

### **3. Comprehensive Testing**

```python
# ✅ CORRECT: Comprehensive tests
class TestNewAnalyzer:
    def test_with_default_config(self):
        analyzer = NewAnalyzer()
        result = analyzer.analyze(test_data)
        assert result is not None
    
    def test_with_custom_config(self):
        config = create_custom_config(MIN_SAMPLES=5)
        analyzer = NewAnalyzer(config)
        result = analyzer.analyze(test_data)
        assert result is not None
    
    def test_edge_cases(self):
        # Test edge cases
        pass

# ❌ WRONG: Minimal testing
def test_new_analyzer():
    analyzer = NewAnalyzer()
    result = analyzer.analyze(test_data)
    assert result is not None
```

### **4. Proper Documentation**

```python
# ✅ CORRECT: Comprehensive documentation
class NewAnalyzer:
    """
    Analyzes data using configurable parameters.
    
    This analyzer provides comprehensive analysis of training data
    with configurable thresholds and parameters.
    
    Args:
        config: Configuration object with analysis parameters
        
    Example:
        >>> config = get_eval_config()
        >>> analyzer = NewAnalyzer(config)
        >>> result = analyzer.analyze(training_data)
    """
    
    def analyze(self, data):
        """
        Analyze the provided data.
        
        Args:
            data: Training data to analyze
            
        Returns:
            AnalysisResult: Analysis results
            
        Raises:
            ValueError: If data is invalid
        """
        pass

# ❌ WRONG: Minimal documentation
class NewAnalyzer:
    def analyze(self, data):
        pass
```

### **5. Error Handling**

```python
# ✅ CORRECT: Proper error handling
def analyze_data(data, config=None):
    if config is None:
        config = get_eval_config()
    
    if not data:
        raise ValueError("Data cannot be empty")
    
    if len(data) < config.MIN_SAMPLES_FOR_ANALYSIS:
        raise ValueError(f"Data must have at least {config.MIN_SAMPLES_FOR_ANALYSIS} samples")
    
    try:
        # analysis logic
        result = perform_analysis(data, config)
        return result
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

# ❌ WRONG: No error handling
def analyze_data(data, config=None):
    # No validation
    # No error handling
    result = perform_analysis(data, config)
    return result
```

## 🔄 **Migration Guidelines**

### **When Refactoring Existing Code**

1. **Identify All Usages:**
   ```bash
   # Find all references
   grep -r "old_function" src/
   grep -r "OldClass" src/
   ```

2. **Create Migration Plan:**
   ```python
   # Create backward compatibility
   def old_function(data, threshold=0.5):
       """Deprecated: Use new_function with config instead."""
       import warnings
       warnings.warn("old_function is deprecated, use new_function", DeprecationWarning)
       
       config = create_config_from_old_params(threshold)
       return new_function(data, config)
   ```

3. **Update All Callers:**
   ```python
   # Update imports
   # from .old_module import old_function
   from .new_module import new_function
   
   # Update calls
   # result = old_function(data, threshold=0.5)
   config = get_eval_config()
   result = new_function(data, config)
   ```

4. **Remove Old Code:**
   ```python
   # After migration is complete
   # Remove old_function
   # Remove old_module
   ```

### **Configuration Migration**

1. **Add New Parameters:**
   ```python
   # In config file
   class EvaluationConfig:
       # New parameter
       NEW_PARAMETER = 0.5
       
       # Migration from old parameter
       @property
       def OLD_PARAMETER(self):
           import warnings
           warnings.warn("OLD_PARAMETER is deprecated, use NEW_PARAMETER", DeprecationWarning)
           return self.NEW_PARAMETER
   ```

2. **Update Validation:**
   ```python
   def validate(self):
       if self.NEW_PARAMETER < 0:
           raise ValueError("NEW_PARAMETER must be non-negative")
   ```

3. **Update Documentation:**
   ```markdown
   ## Migration Guide
   
   ### Configuration Changes
   - `OLD_PARAMETER` → `NEW_PARAMETER`
   - Default value: `0.5`
   - Environment override: `RLDK_NEW_PARAMETER`
   ```

## 🎯 **Quality Checklist**

Before submitting any changes, ensure:

- [ ] **File Organization:**
  - [ ] All files in correct directories
  - [ ] No temporary or duplicate files
  - [ ] Proper naming conventions

- [ ] **Code Quality:**
  - [ ] Uses configuration instead of hardcoded values
  - [ ] Follows existing patterns
  - [ ] Includes error handling
  - [ ] Has type hints and docstrings

- [ ] **Testing:**
  - [ ] Unit tests for new functionality
  - [ ] Integration tests if applicable
  - [ ] All existing tests pass
  - [ ] Test coverage maintained

- [ ] **Configuration:**
  - [ ] New parameters in appropriate config files
  - [ ] Configuration validation updated
  - [ ] Environment overrides considered

- [ ] **Documentation:**
  - [ ] User documentation updated
  - [ ] API documentation updated
  - [ ] Examples updated if applicable

- [ ] **Dependencies:**
  - [ ] No circular dependencies
  - [ ] Import statements updated
  - [ ] Requirements updated if needed

- [ ] **Backward Compatibility:**
  - [ ] Existing APIs still work
  - [ ] Configuration changes backward compatible
  - [ ] Migration path provided if breaking changes

## 🚀 **Quick Reference**

### **File Locations**
- Source code: `src/rldk/`
- Tests: `tests/`
- Examples: `examples/`
- Documentation: `docs/`
- Configuration: `src/rldk/config/`

### **Configuration Usage**
```python
from ..config import get_eval_config, get_forensics_config

def your_function(config=None):
    if config is None:
        config = get_eval_config()
    # Use config.PARAMETER_NAME
```

### **Testing Pattern**
```python
# tests/unit/test_your_module.py
import pytest
from src.rldk.your_module import YourClass

class TestYourClass:
    def test_with_default_config(self):
        obj = YourClass()
        result = obj.method(test_data)
        assert result is not None
```

### **Documentation Pattern**
```python
def your_function(config=None):
    """
    Brief description.
    
    Args:
        config: Configuration object
        
    Returns:
        Description of return value
    """
    pass
```

---

**Remember: Every change should consider the broader structure, maintain consistency, and follow established patterns. When in doubt, ask for clarification or review existing similar implementations.**