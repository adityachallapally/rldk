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

### **Dependency Analysis Tools**

#### **Import Analysis Script**

```bash
#!/bin/bash
# scripts/analyze_imports.sh

echo "🔍 Analyzing imports and dependencies..."

# Find all imports
echo "All imports in source code:"
grep -r "from.*import\|import.*" src/ --include="*.py" | sort | uniq

# Find circular import patterns
echo "Potential circular imports:"
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
"

# Find unused imports
echo "Potentially unused imports:"
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
for root, dirs, files in os.walk('src/rldk'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            unused = find_unused_imports(filepath)
            if unused:
                print(f'{filepath}: {unused}')
"
```

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

## 🔧 **Dependency Management Tools**

### **Dependency Checker Script**

```bash
#!/bin/bash
# scripts/check_dependencies.sh

echo "🔍 Checking dependencies..."

# Check for circular dependencies
echo "Checking for circular dependencies..."
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

# Check for cycles
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
"

# Check for unused imports
echo "Checking for unused imports..."
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
"

# Check for missing imports
echo "Checking for missing imports..."
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
"
```

### **Import Optimizer Script**

```python
# scripts/optimize_imports.py
import ast
import os
import re

def optimize_imports(filepath):
    """Optimize imports in a Python file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Parse the file
    tree = ast.parse(content)
    
    # Find all imports
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                if node.names:
                    names = [alias.name for alias in node.names]
                    imports.append(f"from {node.module} import {', '.join(names)}")
                else:
                    imports.append(f"from {node.module} import *")
    
    # Sort imports
    stdlib_imports = []
    third_party_imports = []
    local_imports = []
    
    for imp in imports:
        if imp.startswith('import ') or imp.startswith('from '):
            if 'rldk' in imp:
                local_imports.append(imp)
            elif any(stdlib in imp for stdlib in ['os', 'sys', 'json', 'math', 'random', 'datetime']):
                stdlib_imports.append(imp)
            else:
                third_party_imports.append(imp)
    
    # Generate optimized imports
    optimized_imports = []
    if stdlib_imports:
        optimized_imports.extend(sorted(stdlib_imports))
    if third_party_imports:
        optimized_imports.extend(sorted(third_party_imports))
    if local_imports:
        optimized_imports.extend(sorted(local_imports))
    
    return optimized_imports

def optimize_all_imports():
    """Optimize imports in all Python files."""
    for root, dirs, files in os.walk('src/rldk'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                print(f"Optimizing imports in {filepath}")
                optimized = optimize_imports(filepath)
                # Write optimized imports back to file
                # (Implementation would write back to file)

if __name__ == "__main__":
    optimize_all_imports()
```

## 📊 **Dependency Metrics**

### **Quality Metrics**

- **Circular Dependencies**: Number of circular dependency cycles
- **Import Count**: Total number of imports
- **Unused Imports**: Number of unused imports
- **Missing Imports**: Number of missing imports
- **Dependency Depth**: Maximum depth of dependency chain

### **Tracking Script**

```python
# scripts/track_dependency_metrics.py
import ast
import os
import json
from datetime import datetime

def calculate_dependency_metrics():
    """Calculate dependency metrics for the codebase."""
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'circular_dependencies': 0,
        'total_imports': 0,
        'unused_imports': 0,
        'missing_imports': 0,
        'max_dependency_depth': 0,
        'modules': {}
    }
    
    # Analyze each module
    for root, dirs, files in os.walk('src/rldk'):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                filepath = os.path.join(root, file)
                module_name = filepath.replace('src/rldk/', '').replace('.py', '').replace('/', '.')
                
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    tree = ast.parse(content)
                    
                    # Count imports
                    imports = []
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.append(alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.append(node.module)
                    
                    metrics['modules'][module_name] = {
                        'imports': len(imports),
                        'import_list': imports
                    }
                    metrics['total_imports'] += len(imports)
                    
                except Exception as e:
                    print(f"Error analyzing {filepath}: {e}")
    
    # Calculate additional metrics
    # (Implementation would calculate circular dependencies, unused imports, etc.)
    
    return metrics

def save_metrics(metrics, filename='dependency_metrics.json'):
    """Save metrics to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)

def compare_metrics(old_metrics, new_metrics):
    """Compare metrics between two versions."""
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'changes': {}
    }
    
    # Compare total imports
    if old_metrics['total_imports'] != new_metrics['total_imports']:
        comparison['changes']['total_imports'] = {
            'old': old_metrics['total_imports'],
            'new': new_metrics['total_imports'],
            'difference': new_metrics['total_imports'] - old_metrics['total_imports']
        }
    
    # Compare modules
    for module in new_metrics['modules']:
        if module not in old_metrics['modules']:
            comparison['changes'][f'new_module_{module}'] = 'added'
        elif old_metrics['modules'][module]['imports'] != new_metrics['modules'][module]['imports']:
            comparison['changes'][f'module_{module}'] = {
                'old_imports': old_metrics['modules'][module]['imports'],
                'new_imports': new_metrics['modules'][module]['imports'],
                'difference': new_metrics['modules'][module]['imports'] - old_metrics['modules'][module]['imports']
            }
    
    return comparison

if __name__ == "__main__":
    metrics = calculate_dependency_metrics()
    save_metrics(metrics)
    print("Dependency metrics saved to dependency_metrics.json")
```

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

### **Dependency Analysis Commands**
```bash
# Check for circular dependencies
./scripts/check_dependencies.sh

# Analyze imports
./scripts/analyze_imports.sh

# Generate dependency graph
python scripts/generate_dependency_graph.py

# Track dependency metrics
python scripts/track_dependency_metrics.py
```

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