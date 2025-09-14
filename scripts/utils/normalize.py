#!/usr/bin/env python3
"""Normalization utilities for golden master testing.

This module provides functions to normalize JSON artifacts by removing
or canonicalizing volatile fields like timestamps, paths, and random values.
"""

import json
import pathlib
import re
from typing import Any, Dict, List, Union

ROOT = str(pathlib.Path(".").resolve())


def _round_floats(obj: Any, places: int = 6) -> Any:
    """Round floating point numbers to a fixed precision."""
    if isinstance(obj, float):
        return round(obj, places)
    if isinstance(obj, list):
        return [_round_floats(x, places) for x in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v, places) for k, v in obj.items()}
    return obj


def normalize_json(obj: Any) -> bytes:
    """Normalize a JSON object by removing volatile fields.

    Args:
        obj: The JSON object to normalize

    Returns:
        Normalized JSON as bytes
    """
    # Round floating point numbers
    obj = _round_floats(obj)

    # Convert to JSON string with sorted keys
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    # Replace absolute paths with placeholders
    s = s.replace(ROOT, "${ROOT}")

    # Replace temporary directory paths
    s = re.sub(r"/tmp/[A-Za-z0-9._-]+", "/tmp/${TMP}", s)
    s = re.sub(r"/var/tmp/[A-Za-z0-9._-]+", "/var/tmp/${TMP}", s)

    # Replace timestamps with canonical values
    s = re.sub(r'"timestamp":\s*"[0-9TZ:\-+.]*"', '"timestamp":"1970-01-01T00:00:00Z"', s)
    s = re.sub(r'"created_at":\s*"[0-9TZ:\-+.]*"', '"created_at":"1970-01-01T00:00:00Z"', s)
    s = re.sub(r'"updated_at":\s*"[0-9TZ:\-+.]*"', '"updated_at":"1970-01-01T00:00:00Z"', s)
    s = re.sub(r'"start_time":\s*"[0-9TZ:\-+.]*"', '"start_time":"1970-01-01T00:00:00Z"', s)
    s = re.sub(r'"end_time":\s*"[0-9TZ:\-+.]*"', '"end_time":"1970-01-01T00:00:00Z"', s)

    # Replace run IDs and experiment IDs
    s = re.sub(r'"run_id":\s*"[A-Za-z0-9._-]*"', '"run_id":"${RUN}"', s)
    s = re.sub(r'"experiment_id":\s*"[A-Za-z0-9._-]*"', '"experiment_id":"${EXP}"', s)
    s = re.sub(r'"session_id":\s*"[A-Za-z0-9._-]*"', '"session_id":"${SESSION}"', s)

    # Replace random seeds
    s = re.sub(r'"seed":\s*[0-9]+', '"seed":0', s)
    s = re.sub(r'"random_seed":\s*[0-9]+', '"random_seed":0', s)

    # Replace process IDs
    s = re.sub(r'"pid":\s*[0-9]+', '"pid":12345', s)
    s = re.sub(r'"process_id":\s*[0-9]+', '"process_id":12345', s)

    # Replace memory addresses and object IDs
    s = re.sub(r'0x[0-9a-fA-F]+', '0x${ADDR}', s)

    # Replace UUIDs
    s = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '${UUID}', s, flags=re.IGNORECASE)

    return s.encode("utf-8")


def normalize_text(text: str) -> str:
    """Normalize text output by replacing volatile fields.

    Args:
        text: The text to normalize

    Returns:
        Normalized text
    """
    # Replace absolute paths
    text = text.replace(ROOT, "${ROOT}")

    # Replace temporary directory paths
    text = re.sub(r"/tmp/[A-Za-z0-9._-]+", "/tmp/${TMP}", text)
    text = re.sub(r"/var/tmp/[A-Za-z0-9._-]+", "/var/tmp/${TMP}", text)

    # Replace timestamps in various formats
    text = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:\d{2}|Z)?', '1970-01-01T00:00:00Z', text)
    text = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d+)?', '1970-01-01 00:00:00', text)

    # Replace run IDs and experiment IDs
    text = re.sub(r'run_[A-Za-z0-9._-]+', 'run_${RUN}', text)
    text = re.sub(r'exp_[A-Za-z0-9._-]+', 'exp_${EXP}', text)

    # Replace UUIDs
    text = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '${UUID}', text, flags=re.IGNORECASE)

    # Replace process IDs
    text = re.sub(r'PID \d+', 'PID 12345', text)
    text = re.sub(r'process \d+', 'process 12345', text)

    return text


def get_normalized_checksum(data: Union[Dict, List, str, bytes]) -> str:
    """Get checksum of normalized data.

    Args:
        data: The data to checksum (will be normalized first)

    Returns:
        SHA256 checksum of normalized data
    """
    import hashlib

    if isinstance(data, (dict, list)):
        normalized = normalize_json(data)
    elif isinstance(data, str):
        normalized = normalize_text(data).encode("utf-8")
    elif isinstance(data, bytes):
        normalized = normalize_text(data.decode("utf-8", errors="ignore")).encode("utf-8")
    else:
        normalized = str(data).encode("utf-8")

    return hashlib.sha256(normalized).hexdigest()


if __name__ == "__main__":
    # Test the normalization
    test_data = {
        "timestamp": "2024-01-01T12:00:00Z",
        "path": "/workspace/tmp/abc123/run_data.jsonl",
        "run_id": "run_xyz789",
        "value": 3.141592653589793,
        "pid": 12345
    }

    print("Original:", json.dumps(test_data, indent=2))
    print("Normalized:", normalize_json(test_data).decode("utf-8"))
    print("Checksum:", get_normalized_checksum(test_data))
