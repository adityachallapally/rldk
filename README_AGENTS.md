# AI Agent Guidelines - Quick Reference

## 🎯 **Overview**

This is a quick reference guide for AI agents working on the RLDK project. For detailed guidelines, see the comprehensive documents in the root directory.

## 📁 **File Organization Rules**

### **✅ CORRECT Locations**
- **Source code**: `src/rldk/`
- **Tests**: `tests/`
- **Examples**: `examples/`
- **Documentation**: `docs/`
- **Configuration**: `src/rldk/config/`
- **Scripts**: `scripts/`

### **❌ NEVER Create in Root**
- Python files (use `src/rldk/` or `examples/`)
- Test files (use `tests/`)
- Documentation files (use `docs/`)
- Configuration files (use `src/rldk/config/`)
- Example scripts (use `examples/`)
- Temporary files (use `/tmp/` or add to `.gitignore`)

## 🔧 **Configuration Usage**

### **✅ CORRECT Pattern**
```python
from ..config import get_eval_config

def your_function(config=None):
    if config is None:
        config = get_eval_config()
    
    # Use config parameters
    if len(data) > config.MIN_SAMPLES_FOR_ANALYSIS:
        # process data
        pass
```

### **❌ WRONG Pattern**
```python
def your_function():
    if len(data) > 10:  # Hardcoded!
        # process data
        pass
```

## 🧪 **Testing Requirements**

### **✅ CORRECT Test Structure**
```python
# tests/unit/test_your_module.py
import pytest
from src.rldk.your_module import YourClass

class TestYourClass:
    def test_with_default_config(self):
        obj = YourClass()
        result = obj.method(test_data)
        assert result is not None
    
    def test_with_custom_config(self):
        config = create_custom_config()
        obj = YourClass(config)
        result = obj.method(test_data)
        assert result is not None
```

## 📚 **Documentation Requirements**

### **✅ CORRECT Documentation**
```python
def your_function(config=None):
    """
    Brief description of what the function does.
    
    Args:
        config: Configuration object with parameters
        
    Returns:
        Description of return value
        
    Example:
        >>> config = get_eval_config()
        >>> result = your_function(config)
    """
    pass
```

## 🔍 **Before Making Changes**

1. **Check Impact**:
   ```bash
   grep -r "your_module" src/
   grep -r "your_function" src/
   ```

2. **Check Configuration**:
   ```bash
   grep -r "config\." src/
   ```

3. **Check Tests**:
   ```bash
   grep -r "your_module" tests/
   ```

4. **Check Documentation**:
   ```bash
   grep -r "your_feature" docs/
   ```

## 📋 **PR Checklist**

- [ ] Files in correct directories
- [ ] No hardcoded values
- [ ] Configuration properly handled
- [ ] Tests included and passing
- [ ] Documentation updated
- [ ] Backward compatibility maintained
- [ ] No circular dependencies
- [ ] Proper error handling
- [ ] Type hints present
- [ ] Docstrings complete

## 🚨 **Common Mistakes to Avoid**

1. **Files in wrong locations**
2. **Hardcoded values**
3. **Missing tests**
4. **Missing documentation**
5. **Inconsistent naming**
6. **Circular dependencies**
7. **Missing error handling**
8. **Missing type hints**

## 🛠 **Available Tools**

- `./scripts/check_file_organization.sh` - Check file organization
- `./scripts/cleanup_files.sh` - Clean up file organization issues
- `./scripts/validate_configuration.sh` - Validate configuration usage
- `./scripts/analyze_dependencies.sh` - Analyze dependencies

## 📖 **Comprehensive Documentation**

- `agents.md` - Detailed AI agent guidelines
- `PROJECT_STRUCTURE.md` - Project structure guidelines
- `PR_GUIDELINES.md` - Pull request guidelines
- `DEPENDENCY_MANAGEMENT.md` - Dependency management guidelines

## 🚀 **Quick Commands**

```bash
# Check file organization
./scripts/check_file_organization.sh

# Clean up files
./scripts/cleanup_files.sh

# Validate configuration
./scripts/validate_configuration.sh

# Analyze dependencies
./scripts/analyze_dependencies.sh
```

---

**Remember: Always consider the broader impact, maintain consistency, and follow established patterns. When in doubt, ask for clarification or review existing similar implementations.**