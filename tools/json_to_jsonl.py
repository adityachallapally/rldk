#!/usr/bin/env python3
"""Convert JSON training logs to JSONL format.

This helper script converts aggregated JSON structures into line-delimited JSON
records that are easier to stream. When converting dictionaries you can specify
which key contains the list of records via ``--array-key``. If no key is
provided the entire JSON object is emitted as a single record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, Union

JsonValue = Union[dict, list, str, int, float, bool, None]


def iter_records(data: JsonValue, array_key: str | None = None) -> Iterator[JsonValue]:
    """Yield JSON-serializable records from the loaded data."""
    if array_key:
        if not isinstance(data, dict):
            raise ValueError("--array-key can only be used when the JSON document is an object")
        if array_key not in data:
            raise KeyError(f"Key '{array_key}' not found in JSON document")
        candidate = data[array_key]
        if not isinstance(candidate, list):
            raise TypeError(f"Value for '{array_key}' is not a list")
        data = candidate

    if isinstance(data, list):
        for item in data:
            yield item
    else:
        # Emit the entire structure as a single record
        yield data


def convert_json_to_jsonl(source: Path, destination: Path, array_key: str | None = None) -> None:
    """Convert ``source`` JSON file to JSONL format at ``destination``."""
    with source.open("r", encoding="utf-8") as f:
        data = json.load(f)

    records = list(iter_records(data, array_key=array_key))

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert JSON logs to JSONL format")
    parser.add_argument("source", type=Path, help="Path to the input JSON file")
    parser.add_argument(
        "destination",
        type=Path,
        nargs="?",
        help="Optional output path. Defaults to replacing .json with .jsonl",
    )
    parser.add_argument(
        "--array-key",
        dest="array_key",
        help="Name of the key that contains an array of records to emit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source: Path = args.source
    destination: Path

    if args.destination is not None:
        destination = args.destination
    else:
        if source.suffix.lower() == ".json":
            destination = source.with_suffix(".jsonl")
        else:
            destination = source.parent / f"{source.name}.jsonl"

    convert_json_to_jsonl(source, destination, array_key=args.array_key)
    print(f"Converted {source} -> {destination}")


if __name__ == "__main__":
    main()
