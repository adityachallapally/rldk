"""
Configuration for mutmut mutation testing.
"""

def pre_mutation(context):
    """Pre-mutation hook to skip certain mutations."""
    # Skip mutations that would break the test suite
    if hasattr(context, 'current_source_line'):
        line = context.current_source_line
        # Skip mutations in docstrings
        if '"""' in line or "'''" in line:
            return False
        # Skip mutations in comments
        if line.strip().startswith('#'):
            return False
        # Skip mutations in string literals
        if line.strip().startswith('"') or line.strip().startswith("'"):
            return False
    return True

def post_mutation(context):
    """Post-mutation hook to handle mutation results."""
    pass

# Configuration for mutmut
config = {
    'paths_to_mutate': [
        'src/rldk/utils/seed.py',
        'src/rldk/utils/validation.py',
        'src/rldk/utils/error_handling.py',
        'src/rldk/utils/progress.py',
    ],
    'backup': False,
    'runner': 'python -m pytest tests/unit/ -x --tb=short',
    'tests_dir': 'tests/unit/',
    'pre_mutation': pre_mutation,
    'post_mutation': post_mutation,
}
