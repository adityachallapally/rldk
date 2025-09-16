"""Event emission utilities for JSONL logging."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


class EventWriter:
    """Writer for canonical JSONL events."""
    
    def __init__(self, path: Union[str, Path], run_id: Optional[str] = None):
        self.path = Path(path)
        self.run_id = run_id or f"run-{int(datetime.utcnow().timestamp())}"
        
        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open file in append mode with line buffering
        self.file = open(self.path, "a", buffering=1)
    
    def log(self, step: int, name: str, value: float, **kwargs):
        """Log an event to JSONL."""
        event = {
            "time": datetime.utcnow().isoformat() + "Z",
            "step": int(step),
            "name": str(name),
            "value": float(value),
            "run_id": self.run_id
        }
        
        # Add optional fields
        if "tags" in kwargs:
            event["tags"] = kwargs["tags"]
        if "meta" in kwargs:
            event["meta"] = kwargs["meta"]
        
        # Write to file
        self.file.write(json.dumps(event) + "\n")
        self.file.flush()
    
    def close(self):
        """Close the file."""
        if hasattr(self, 'file') and self.file:
            self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()