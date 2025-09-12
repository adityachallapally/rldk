#!/usr/bin/env python3
"""
Test that ensures no documentation files are read during contract tests.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


class FileOpenMonitor:
    """Monitor file opens to detect documentation access."""
    
    def __init__(self):
        self.opened_files = []
        self.original_open = open
    
    def __enter__(self):
        self.patcher = patch('builtins.open', self.monitored_open)
        self.mock_open = self.patcher.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.patcher.stop()
    
    def monitored_open(self, file_path, *args, **kwargs):
        """Monitor file opens and record documentation access."""
        file_path_str = str(file_path)
        
        # Check if this is a documentation file
        if self.is_doc_file(file_path_str):
            self.opened_files.append(file_path_str)
        
        # Call the original open function
        return self.original_open(file_path, *args, **kwargs)
    
    def is_doc_file(self, file_path: str) -> bool:
        """Check if a file path is a documentation file."""
        file_path_lower = file_path.lower()
        
        # Check for common documentation file patterns
        doc_patterns = [
            '.md',  # Markdown files
            '.rst',  # ReStructuredText files
            'readme',
            'docs/',
            'documentation/',
            'guide',
            'manual',
            'tutorial'
        ]
        
        # Special case: .txt files are only docs if they contain doc-related keywords
        if '.txt' in file_path_lower:
            txt_doc_keywords = ['readme', 'guide', 'manual', 'tutorial', 'documentation']
            if any(keyword in file_path_lower for keyword in txt_doc_keywords):
                return True
            return False
        
        for pattern in doc_patterns:
            if pattern in file_path_lower:
                return True
        
        # Check if file is in a docs directory
        path_parts = Path(file_path).parts
        for part in path_parts:
            if part.lower() in ['docs', 'documentation', 'guides', 'tutorials']:
                return True
        
        return False


def test_no_docs_read_during_baseline_contract():
    """Test that baseline contract generation doesn't read documentation files."""
    # Import the baseline contract test module
    from tests.test_contract_baseline import generate_baseline_contract
    
    # Monitor file opens during contract generation
    with FileOpenMonitor() as monitor:
        try:
            # Generate the baseline contract
            contract = generate_baseline_contract()
            
            # Check if any documentation files were opened
            if monitor.opened_files:
                pytest.fail(
                    f"Documentation files were accessed during contract generation:\n"
                    f"{chr(10).join(monitor.opened_files)}"
                )
            
        except Exception as e:
            # If there was an error, check if it was due to documentation access
            if monitor.opened_files:
                pytest.fail(
                    f"Error occurred and documentation files were accessed:\n"
                    f"Error: {e}\n"
                    f"Files: {chr(10).join(monitor.opened_files)}"
                )
            else:
                # Re-raise the original error if it wasn't related to docs
                raise


def test_no_docs_read_during_symbol_validation():
    """Test that symbol validation doesn't read documentation files."""
    from tests.test_contract_baseline import test_validate_symbols
    
    with FileOpenMonitor() as monitor:
        try:
            # Run symbol validation
            test_validate_symbols()
            
            # Check if any documentation files were opened
            if monitor.opened_files:
                pytest.fail(
                    f"Documentation files were accessed during symbol validation:\n"
                    f"{chr(10).join(monitor.opened_files)}"
                )
            
        except Exception as e:
            if monitor.opened_files:
                pytest.fail(
                    f"Error occurred and documentation files were accessed:\n"
                    f"Error: {e}\n"
                    f"Files: {chr(10).join(monitor.opened_files)}"
                )
            else:
                raise


def test_no_docs_read_during_cli_validation():
    """Test that CLI validation doesn't read documentation files."""
    from tests.test_contract_baseline import test_validate_cli_help
    
    with FileOpenMonitor() as monitor:
        try:
            # Run CLI validation
            test_validate_cli_help()
            
            # Check if any documentation files were opened
            if monitor.opened_files:
                pytest.fail(
                    f"Documentation files were accessed during CLI validation:\n"
                    f"{chr(10).join(monitor.opened_files)}"
                )
            
        except Exception as e:
            if monitor.opened_files:
                pytest.fail(
                    f"Error occurred and documentation files were accessed:\n"
                    f"Error: {e}\n"
                    f"Files: {chr(10).join(monitor.opened_files)}"
                )
            else:
                raise


def test_file_open_monitor_detects_doc_files():
    """Test that the file open monitor correctly detects documentation files."""
    monitor = FileOpenMonitor()
    
    # Test various documentation file patterns
    doc_files = [
        "README.md",
        "docs/api.md",
        "documentation/guide.txt",
        "tutorial/example.rst",
        "src/docs/help.md",
        "project/README.txt"
    ]
    
    non_doc_files = [
        "src/main.py",
        "tests/test_file.py",
        "data/sample.json",
        "config.yaml",
        "requirements.txt"
    ]
    
    # Test doc files
    for doc_file in doc_files:
        assert monitor.is_doc_file(doc_file), f"Should detect {doc_file} as doc file"
    
    # Test non-doc files
    for non_doc_file in non_doc_files:
        assert not monitor.is_doc_file(non_doc_file), f"Should not detect {non_doc_file} as doc file"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])