"""Alert output writers for JSONL and human-readable formats."""
import json
from pathlib import Path
from typing import List
from .engine import Alert


class AlertWriter:
    """Dual-format alert writer."""
    
    def __init__(self, alerts_jsonl_path: str, alerts_txt_path: str):
        self.jsonl_path = Path(alerts_jsonl_path)
        self.txt_path = Path(alerts_txt_path)
        
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.txt_path.parent.mkdir(parents=True, exist_ok=True)
    
    def write_alert(self, alert: Alert) -> None:
        """Write alert to both JSONL and TXT formats."""
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(alert.to_dict()) + "\n")
        
        with self.txt_path.open("a", encoding="utf-8") as f:
            f.write(f"[{alert.event.time}] {alert.rule_id}: {alert.message}\n")
    
    def write_alerts(self, alerts: List[Alert]) -> None:
        """Write multiple alerts to both formats."""
        for alert in alerts:
            self.write_alert(alert)
