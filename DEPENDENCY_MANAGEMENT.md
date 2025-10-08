# Dependency Management & Impact Analysis for RLDK

## 🎯 **Overview**

This document provides comprehensive guidelines for managing dependencies and analyzing the impact of changes in the RLDK project. It addresses common issues like circular dependencies, inconsistent imports, and lack of awareness of how changes affect other components.

## 🔍 **Impact Analysis Framework**

### **Before Making Changes**

1. **Identify Direct Dependencies**
   ```bash
   # Find all imports of the module you're changing
   grep -r "from.*your_module" src/
   grep -r "import.*your_module" src/
   
   # Find all references to functions/classes
   grep -r "your_function" src/
   grep -r "YourClass" src/
   ```

2. **Identify Indirect Dependencies**
   ```bash
   # Find modules that import your module's dependencies
   grep -r "from.*dependency_module" src/
   grep -r "import.*dependency_module" src/
   ```

3. **Check Configuration Dependencies**
   ```bash
   # Find config usage
   grep -r "config\." src/
   grep -r "get_.*_config" src/
   ```

4. **Check Test Dependencies**
   ```bash
   # Find test dependencies
   grep -r "your_module" tests/
   ```

5. **Check Documentation References**
   ```bash
   # Find documentation references
   grep -r "your_feature" docs/
   grep -r "your_function" docs/
   ```

### **Manual Dependency Analysis**

#### **Import Analysis**

You can manually analyze imports and dependencies:

1. **Find all imports**:
   - Search for `from.*import` and `import.*` patterns in source code
   - Look for local imports (starting with `rldk.`)
   - Identify third-party imports

2. **Check for circular imports**:
   - Look for modules that import each other
   - Check for import cycles in the dependency graph
   - Verify import order is correct

3. **Find unused imports**:
   - Check if imported modules are actually used in the code
   - Look for imports that are never referenced
   - Remove unused imports to clean up the code

#### **Dependency Graph Generator**

```python
# scripts/generate_dependency_graph.py
import ast
import os
import networkx as nx
import matplotlib.pyplot as plt

def analyze_dependencies():
    """Generate a dependency graph of the RLDK codebase."""
    G = nx.DiGraph()
    
    # Add all modules as nodes
    for root, dirs, files in os.walk('src/rldk'):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                module_path = os.path.join(root, file)
                module_name = module_path.replace('src/rldk/', '').replace('.py', '').replace('/', '.')
                G.add_node(module_name)
    
    # Add edges for imports
    for root, dirs, files in os.walk('src/rldk'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        tree = ast.parse(f.read())
                    
                    source_module = filepath.replace('src/rldk/', '').replace('.py', '').replace('/', '.')
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                if alias.name.startswith('rldk.'):
                                    target_module = alias.name[5:]  # Remove 'rldk.'
                                    G.add_edge(source_module, target_module)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module and node.module.startswith('rldk.'):
                                target_module = node.module[5:]  # Remove 'rldk.'
                                G.add_edge(source_module, target_module)
                except:
                    continue
    
    return G

def find_circular_dependencies(G):
    """Find circular dependencies in the graph."""
    try:
        cycles = list(nx.simple_cycles(G))
        return cycles
    except:
        return []

def generate_dependency_report():
    """Generate a comprehensive dependency report."""
    G = analyze_dependencies()
    cycles = find_circular_dependencies(G)
    
    print("Dependency Analysis Report")
    print("=" * 50)
    
    print(f"Total modules: {G.number_of_nodes()}")
    print(f"Total dependencies: {G.number_of_edges()}")
    
    if cycles:
        print(f"Circular dependencies found: {len(cycles)}")
        for cycle in cycles:
            print(f"  {' -> '.join(cycle)} -> {cycle[0]}")
    else:
        print("No circular dependencies found")
    
    # Find modules with most dependencies
    print("\nModules with most dependencies:")
    out_degrees = G.out_degree()
    sorted_modules = sorted(out_degrees, key=lambda x: x[1], reverse=True)
    for module, degree in sorted_modules[:10]:
        print(f"  {module}: {degree} dependencies")
    
    # Find modules with most dependents
    print("\nModules with most dependents:")
    in_degrees = G.in_degree()
    sorted_modules = sorted(in_degrees, key=lambda x: x[1], reverse=True)
    for module, degree in sorted_modules[:10]:
        print(f"  {module}: {degree} dependents")

if __name__ == "__main__":
    generate_dependency_report()
```

## 🔄 **Change Impact Analysis**

### **Configuration Changes**

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

4. **Update Documentation**
   ```bash
   # Update configuration documentation
   grep -r "old_parameter" docs/
   # Update references to new parameter names
   ```

### **API Changes**

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

4. **Update Tests**
   ```bash
   # Update test files
   grep -r "your_function" tests/
   ```

### **Module Restructuring**

When restructuring modules:

1. **Identify Dependencies**
   ```bash
   # Find all files that import the module being moved
   grep -r "from.*old_module" src/
   grep -r "import.*old_module" src/
   ```

2. **Update Imports**
   ```python
   # Before restructuring
   from rldk.utils import helper_function
   
   # After restructuring
   from rldk.utils.helpers import helper_function
   ```

3. **Update Tests**
   ```bash
   # Move tests to match new structure
   mv tests/unit/test_old_module.py tests/unit/test_new_module.py
   ```

4. **Update Documentation**
   ```bash
   # Update file paths in documentation
   grep -r "old_module" docs/
   ```

## 🚫 **Common Dependency Issues**

### **Circular Dependencies**

#### **Problem**
```python
# Module A imports Module B
# Module B imports Module A
# This creates a circular dependency

# src/rldk/module_a.py
from rldk.module_b import function_b

def function_a():
    return function_b()

# src/rldk/module_b.py
from rldk.module_a import function_a

def function_b():
    return function_a()
```

#### **Solution**
```python
# Option 1: Move shared functionality to a common module
# src/rldk/common.py
def shared_function():
    pass

# src/rldk/module_a.py
from rldk.common import shared_function

def function_a():
    return shared_function()

# src/rldk/module_b.py
from rldk.common import shared_function

def function_b():
    return shared_function()

# Option 2: Use dependency injection
# src/rldk/module_a.py
def function_a(dependency=None):
    if dependency is None:
        from rldk.module_b import function_b
        dependency = function_b
    return dependency()

# src/rldk/module_b.py
def function_b(dependency=None):
    if dependency is None:
        from rldk.module_a import function_a
        dependency = function_a
    return dependency()
```

### **Import Order Issues**

#### **Problem**
```python
# Importing modules in wrong order can cause issues
# src/rldk/module_a.py
from rldk.module_b import function_b  # module_b not yet loaded
from rldk.module_c import function_c  # module_c depends on module_b
```

#### **Solution**
```python
# Import in dependency order
# src/rldk/module_a.py
from rldk.module_c import function_c  # module_c first
from rldk.module_b import function_b  # then module_b
```

### **Missing Dependencies**

#### **Problem**
```python
# Function uses a module that's not imported
def some_function():
    return json.dumps(data)  # json not imported
```

#### **Solution**
```python
# Add missing import
import json

def some_function():
    return json.dumps(data)
```

## 🔧 **Manual Dependency Management**

### **Dependency Checking**

You can manually check for dependency issues:

1. **Check for circular dependencies**:
   - Look for modules that import each other
   - Check for import cycles in the dependency graph
   - Verify import order is correct

2. **Check for unused imports**:
   - Look for imports that are never used in the code
   - Remove unused imports to clean up the code
   - Verify all imports are necessary

3. **Check for missing imports**:
   - Look for common missing imports like `json`, `os`, `sys`
   - Verify all used modules are imported
   - Check for runtime import errors

### **Import Organization**

You can manually organize imports:

1. **Sort imports by category**:
   - Standard library imports first
   - Third-party imports second
   - Local imports last

2. **Group related imports**:
   - Group imports from the same module
   - Use consistent import style
   - Remove duplicate imports

## 📊 **Dependency Metrics**

### **Quality Metrics**

- **Circular Dependencies**: Number of circular dependency cycles
- **Import Count**: Total number of imports
- **Unused Imports**: Number of unused imports
- **Missing Imports**: Number of missing imports
- **Dependency Depth**: Maximum depth of dependency chain

### **Manual Metrics Tracking**

You can manually track dependency metrics:

1. **Count imports**:
   - Count total imports in each module
   - Track import patterns over time
   - Monitor import complexity

2. **Track changes**:
   - Compare import counts between versions
   - Monitor new dependencies
   - Track removed dependencies

3. **Quality metrics**:
   - Count circular dependencies
   - Track unused imports
   - Monitor missing imports

## 🎯 **Best Practices**

### **1. Minimize Dependencies**

```python
# ✅ GOOD: Minimal dependencies
def process_data(data):
    # Only import what you need
    import json
    return json.dumps(data)

# ❌ BAD: Unnecessary dependencies
import json
import os
import sys
import math
import random
import datetime

def process_data(data):
    return json.dumps(data)  # Only using json
```

### **2. Use Dependency Injection**

```python
# ✅ GOOD: Dependency injection
def analyze_data(data, config=None, logger=None):
    if config is None:
        from rldk.config import get_eval_config
        config = get_eval_config()
    
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    # Use injected dependencies
    pass

# ❌ BAD: Hard dependencies
def analyze_data(data):
    from rldk.config import get_eval_config
    import logging
    
    config = get_eval_config()
    logger = logging.getLogger(__name__)
    # Hard dependencies make testing difficult
    pass
```

### **3. Avoid Circular Dependencies**

```python
# ✅ GOOD: No circular dependencies
# src/rldk/common.py
def shared_function():
    pass

# src/rldk/module_a.py
from rldk.common import shared_function

def function_a():
    return shared_function()

# src/rldk/module_b.py
from rldk.common import shared_function

def function_b():
    return shared_function()

# ❌ BAD: Circular dependency
# src/rldk/module_a.py
from rldk.module_b import function_b

def function_a():
    return function_b()

# src/rldk/module_b.py
from rldk.module_a import function_a

def function_b():
    return function_a()
```

### **4. Organize Imports**

```python
# ✅ GOOD: Organized imports
# Standard library imports
import os
import sys
import json

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from rldk.config import get_eval_config
from rldk.utils import helper_function

# ❌ BAD: Disorganized imports
import json
from rldk.config import get_eval_config
import os
import numpy as np
from rldk.utils import helper_function
import sys
import pandas as pd
```

### **5. Use Type Hints**

```python
# ✅ GOOD: Type hints for dependencies
from typing import Optional, Dict, Any
from rldk.config import EvaluationConfig

def analyze_data(
    data: Dict[str, Any], 
    config: Optional[EvaluationConfig] = None
) -> Dict[str, Any]:
    if config is None:
        from rldk.config import get_eval_config
        config = get_eval_config()
    return {}

# ❌ BAD: No type hints
def analyze_data(data, config=None):
    if config is None:
        from rldk.config import get_eval_config
        config = get_eval_config()
    return {}
```

## 🚀 **Quick Reference**

### **Manual Dependency Analysis**
- Check for circular dependencies by examining import patterns
- Look for unused imports in each module
- Verify all necessary imports are present
- Organize imports by category (stdlib, third-party, local)

### **Common Issues to Avoid**
- Circular dependencies
- Unused imports
- Missing imports
- Import order issues
- Hard dependencies

### **Best Practices**
- Minimize dependencies
- Use dependency injection
- Avoid circular dependencies
- Organize imports
- Use type hints

---

**Remember: Good dependency management is crucial for maintainable code. Always analyze the impact of changes and maintain clean dependency relationships.**