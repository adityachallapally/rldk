"""Custom adapter for our JSONL training logs."""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from .base import BaseAdapter


class CustomJSONLAdapter(BaseAdapter):
    """Adapter for our custom JSONL training logs."""
    
    def can_handle(self) -> bool:
        """Check if source contains our custom JSONL logs."""
        if not self.source.exists():
            return False
        
        if self.source.is_file():
            return self._is_custom_jsonl_file(self.source)
        elif self.source.is_dir():
            # Check for our custom JSONL log files
            jsonl_files = list(self.source.glob("*.jsonl"))
            return len(jsonl_files) > 0
        
        return False
    
    def _is_custom_jsonl_file(self, file_path: Path) -> bool:
        """Check if a file contains our custom JSONL logs."""
        try:
            if file_path.suffix == '.jsonl':
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        # Check for our custom schema
                        return all(key in data for key in ['global_step', 'reward_scalar', 'loss'])
        except:
            pass
        return False
    
    def load(self) -> pd.DataFrame:
        """Load our custom JSONL logs and convert to standard format."""
        if not self.can_handle():
            raise ValueError(f"Cannot handle source: {self.source}")
        
        metrics = []
        
        if self.source.is_file():
            metrics = self._parse_file(self.source)
        elif self.source.is_dir():
            # Find and parse all JSONL log files
            jsonl_files = list(self.source.glob("*.jsonl"))
            for jsonl_file in jsonl_files:
                metrics.extend(self._parse_file(jsonl_file))
        
        if not metrics:
            raise ValueError(f"No valid custom JSONL metrics found in {self.source}")
        
        # Convert to DataFrame
        df = pd.DataFrame(metrics)
        
        # Ensure required columns exist for the standard format
        required_cols = ['step', 'phase', 'reward_mean', 'reward_std', 'kl_mean', 
                        'entropy_mean', 'clip_frac', 'grad_norm', 'lr', 'loss',
                        'tokens_in', 'tokens_out', 'wall_time', 'seed', 'run_id', 'git_sha']
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        
        return df[required_cols]
    
    def _parse_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse a single custom JSONL log file."""
        metrics = []
        
        try:
            if file_path.suffix == '.jsonl':
                with open(file_path, 'r') as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            metric = self._extract_custom_metric(data, line_num)
                            if metric:
                                metrics.append(metric)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
        
        return metrics
    
    def _extract_custom_metric(self, data: Dict[str, Any], line_num: int) -> Dict[str, Any]:
        """Extract metric from our custom JSONL format."""
        try:
            # Map our custom schema to the expected format
            metric = {
                'step': data.get('global_step', line_num),
                'phase': 'train',  # Default phase
                'reward_mean': data.get('reward_scalar', 0.0),
                'reward_std': 0.0,  # We don't have this, use 0.0 instead of None
                'kl_mean': data.get('kl_to_ref', 0.0),
                'entropy_mean': 0.0,  # We don't have this, use 0.0 instead of None
                'clip_frac': 0.0,  # We don't have this, use 0.0 instead of None
                'grad_norm': 0.0,  # We don't have this, use 0.0 instead of None
                'lr': 0.0,  # We don't have this, use 0.0 instead of None
                'loss': data.get('loss', 0.0),
                'tokens_in': 0,  # We don't have this, use 0 instead of None
                'tokens_out': 0,  # We don't have this, use 0 instead of None
                'wall_time': 0.0,  # We don't have this, use 0.0 instead of None
                'seed': data.get('rng.python', 42),
                'run_id': f"custom_{line_num}",
                'git_sha': "unknown"  # We don't have this, use string instead of None
            }
            
            return metric
        except Exception as e:
            print(f"Error extracting metric from line {line_num}: {e}")
            return None