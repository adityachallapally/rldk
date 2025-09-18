"""Flexible data adapters with schema mapping and multiple format support."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
import yaml

from ..monitor.presets import FIELD_MAP_PRESETS, get_field_map_preset
from ..utils.error_handling import AdapterError, ValidationError
from .base import BaseAdapter
from .field_resolver import FieldResolver, SchemaError


class FlexibleDataAdapter(BaseAdapter):
    """Flexible adapter that can handle multiple data formats with schema mapping."""

    SUPPORTED_EXTENSIONS = {'.jsonl', '.json', '.csv', '.parquet'}

    def __init__(
        self,
        source: Union[str, Path],
        field_map: Optional[Dict[str, str]] = None,
        config_file: Optional[Union[str, Path]] = None,
        allow_dot_paths: bool = True,
        required_fields: Optional[List[str]] = None,
        validation_mode: str = "flexible",
        preset: Optional[str] = None,
    ):
        """Initialize the flexible adapter.

        Args:
            source: Path to data file or directory
            field_map: Optional explicit mapping from canonical to actual field names
            config_file: Optional path to YAML/JSON config file with field mapping
            allow_dot_paths: Whether to support nested field access with dot notation
            required_fields: List of required canonical field names (defaults to ['step', 'reward'])
            validation_mode: Validation strictness - 'strict', 'flexible', or 'lenient'
        """
        super().__init__(source)
        self.field_resolver = FieldResolver(allow_dot_paths=allow_dot_paths)
        self.field_map = self._normalize_user_field_map(field_map)
        self.config_file = Path(config_file) if config_file else None
        self.allow_dot_paths = allow_dot_paths
        self.required_fields = required_fields or ['step', 'reward']
        self.validation_mode = validation_mode
        self.preset = preset
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load field map from config file if provided
        if self.config_file and self.config_file.exists():
            self._load_config()

    def _load_config(self) -> None:
        """Load field mapping from config file."""
        try:
            with open(self.config_file) as f:
                if self.config_file.suffix.lower() in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                elif self.config_file.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    raise ValidationError(
                        f"Unsupported config file format: {self.config_file.suffix}",
                        suggestion="Use .yaml, .yml, or .json files",
                        error_code="UNSUPPORTED_CONFIG_FORMAT"
                    )

            if 'field_map' in config:
                normalized = self._normalize_user_field_map(config['field_map'])
                if normalized:
                    self.field_map.update(normalized)
                self.logger.info(f"Loaded field map from {self.config_file}")
            else:
                self.logger.warning(f"Config file {self.config_file} does not contain 'field_map' section")

        except Exception as e:
            raise ValidationError(
                f"Failed to load config file {self.config_file}: {e}",
                suggestion="Check that the config file is valid YAML or JSON",
                error_code="CONFIG_LOAD_FAILED"
            ) from e

    def can_handle(self) -> bool:
        """Check if this adapter can handle the given source."""
        if not self.source.exists():
            return False

        if self.source.is_file():
            return self.source.suffix.lower() in self.SUPPORTED_EXTENSIONS
        elif self.source.is_dir():
            # Check if directory contains supported files
            for ext in self.SUPPORTED_EXTENSIONS:
                if list(self.source.glob(f"*{ext}")):
                    return True
            return False

        return False

    def load(self) -> pd.DataFrame:
        """Load data and convert to standardized format."""
        if not self.can_handle():
            raise AdapterError(
                f"Cannot handle source: {self.source}",
                suggestion=f"Source must be a file or directory with supported extensions: {', '.join(self.SUPPORTED_EXTENSIONS)}",
                error_code="CANNOT_HANDLE_SOURCE"
            )

        self._log_operation("Loading flexible data", {"source": str(self.source)})

        # Load data based on file type
        if self.source.is_file():
            data = self._load_single_file(self.source)
        else:
            data = self._load_directory(self.source)

        if not data:
            raise AdapterError(
                "No data found in source",
                suggestion="Check that the source contains valid data in a supported format",
                error_code="NO_DATA_FOUND"
            )

        df = self._records_to_dataframe(data)

        # Resolve field names and validate schema
        df = self._resolve_and_validate_schema(df)

        converted = self._convert_to_training_metrics(df)

        try:
            from ..ingest.training_metrics_normalizer import standardize_training_metrics

            standardized = standardize_training_metrics(converted)
        except ValidationError as exc:
            raise AdapterError(
                f"Failed to standardize schema for {self.source}: {exc}",
                suggestion="Check that the field mapping resolves 'step' and numeric metrics",
                error_code="SCHEMA_STANDARDIZATION_FAILED",
            ) from exc

        return standardized

    def _load_single_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from a single file."""
        extension = file_path.suffix.lower()

        if extension == '.jsonl':
            return self._load_jsonl(file_path)
        elif extension == '.json':
            return self._load_json(file_path)
        elif extension == '.csv':
            return self._load_csv(file_path)
        elif extension == '.parquet':
            return self._load_parquet(file_path)
        else:
            raise AdapterError(
                f"Unsupported file extension: {extension}",
                suggestion=f"Use one of: {', '.join(self.SUPPORTED_EXTENSIONS)}",
                error_code="UNSUPPORTED_FILE_EXTENSION"
            )

    def _load_directory(self, dir_path: Path) -> List[Dict[str, Any]]:
        """Load data from all supported files in a directory."""
        all_data: List[Dict[str, Any]] = []

        for ext in self.SUPPORTED_EXTENSIONS:
            for file_path in sorted(dir_path.glob(f"*{ext}")):
                try:
                    file_data = self._load_single_file(file_path)
                except Exception as e:
                    self.logger.warning(f"Failed to load %s: %s", file_path, e)
                    continue
                all_data.extend(file_data)

        return all_data

    def _records_to_dataframe(self, records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
        """Convert raw records into a DataFrame with flattened columns."""

        if not records:
            return pd.DataFrame()

        try:
            return pd.json_normalize(records, sep='.')
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug("Failed to normalize records: %s", exc)
            return pd.DataFrame(records)

    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSONL file with streaming support for large files."""
        data = []

        try:
            with open(file_path, encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                        if isinstance(record, dict):
                            data.append(record)
                        else:
                            self.logger.warning(f"Skipping non-dict record at line {line_num}")
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"JSON decode error at line {line_num}: {e}")
                        continue
        except Exception as e:
            raise AdapterError(
                f"Failed to load JSONL file {file_path}: {e}",
                suggestion="Check that the file contains valid JSONL data",
                error_code="JSONL_LOAD_FAILED"
            ) from e

        return data

    def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSON file."""
        try:
            with open(file_path, encoding='utf-8') as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Check if it's a single record or a collection
                if any(key in data for key in ['records', 'data', 'items', 'metrics']):
                    # Try common collection keys
                    for key in ['records', 'data', 'items', 'metrics']:
                        if key in data and isinstance(data[key], list):
                            return data[key]
                # Otherwise treat as single record
                return [data]
            else:
                raise ValidationError(
                    f"JSON file contains unsupported data type: {type(data)}",
                    suggestion="JSON file should contain a list of records or a single record object",
                    error_code="INVALID_JSON_STRUCTURE"
                )
        except Exception as e:
            raise AdapterError(
                f"Failed to load JSON file {file_path}: {e}",
                suggestion="Check that the file contains valid JSON data",
                error_code="JSON_LOAD_FAILED"
            ) from e

    def _load_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load CSV file."""
        try:
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        except Exception as e:
            raise AdapterError(
                f"Failed to load CSV file {file_path}: {e}",
                suggestion="Check that the file contains valid CSV data",
                error_code="CSV_LOAD_FAILED"
            ) from e

    def _load_parquet(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load Parquet file."""
        try:
            df = pd.read_parquet(file_path)
            return df.to_dict('records')
        except Exception as e:
            raise AdapterError(
                f"Failed to load Parquet file {file_path}: {e}",
                suggestion="Check that the file contains valid Parquet data",
                error_code="PARQUET_LOAD_FAILED"
            ) from e

    def _resolve_and_validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resolve field names and validate schema."""
        if df.empty:
            return df

        available_headers = df.columns.tolist()
        effective_field_map = self._build_effective_field_map(available_headers)

        # Check for missing required fields based on validation mode
        missing_fields = self.field_resolver.get_missing_fields(
            self.required_fields, available_headers, effective_field_map
        )

        if missing_fields:
            if self.validation_mode == "strict":
                raise SchemaError(
                    f"Missing required fields: {', '.join(missing_fields)}",
                    missing_fields,
                    available_headers,
                    self.field_resolver,
                )
            elif self.validation_mode == "flexible":
                step_missing = 'step' in missing_fields
                has_metric = any(
                    self.field_resolver.resolve_field(
                        metric, available_headers, effective_field_map
                    )
                    for metric in ['reward', 'score', 'return', 'kl', 'entropy', 'loss'])

                if step_missing:
                    extra_missing = [field for field in missing_fields if field != 'step']
                    message = "Missing required 'step' field in flexible mode"
                    if extra_missing:
                        extras = ", ".join(sorted(extra_missing))
                        message += f"; additional missing fields: {extras}"
                    optional_hints: Dict[str, List[str]] = {}
                    for optional_field in ('kl', 'entropy', 'loss'):
                        if optional_field in missing_fields:
                            continue
                        resolved_optional = self.field_resolver.resolve_field(
                            optional_field,
                            available_headers,
                            effective_field_map,
                        )
                        if resolved_optional:
                            continue
                        suggestions = self.field_resolver.get_suggestions(
                            optional_field,
                            available_headers,
                        )
                        if suggestions:
                            optional_hints[optional_field] = suggestions
                    if optional_hints:
                        hint_lines = [
                            f"  {field}: {', '.join(values)}"
                            for field, values in optional_hints.items()
                        ]
                        message += "\n\nOther recognizable fields:\n" + "\n".join(hint_lines)
                    raise SchemaError(
                        message,
                        missing_fields,
                        available_headers,
                        self.field_resolver,
                    )
                elif not has_metric:
                    raise SchemaError(
                        "No metric fields found in flexible mode (need at least one of: reward, score, return, kl, entropy, loss)",
                        [], available_headers, self.field_resolver
                    )
                elif missing_fields and self.logger:
                    optional_missing = [f for f in missing_fields if f != 'step']
                    if optional_missing:
                        self.logger.warning(f"Missing optional fields in flexible mode: {', '.join(optional_missing)}")
            elif self.validation_mode == "lenient" and missing_fields and self.logger:
                self.logger.warning(f"Missing fields in lenient mode: {', '.join(missing_fields)}")

        resolved_df = df.copy()
        resolved_fields: Dict[str, str] = {}
        resolved_candidates = self._get_resolved_fields(
            available_headers, effective_field_map
        )

        for canonical_name, resolved_name in resolved_candidates.items():
            if self.allow_dot_paths and '.' in resolved_name and resolved_name not in df.columns:
                resolved_series = self._extract_nested_field(df, resolved_name)
            else:
                if resolved_name not in df.columns:
                    continue
                resolved_series = df[resolved_name]

            resolved_df[canonical_name] = resolved_series
            resolved_fields[canonical_name] = resolved_name

        self._log_resolution_summary(
            resolved_fields,
            self.field_resolver.get_missing_fields(
                self.required_fields, available_headers, effective_field_map
            ),
            len(df),
        )

        return resolved_df

    def _convert_to_training_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DataFrame has TrainingMetrics-compatible columns."""

        if df.empty:
            return df

        alias_to_canonical = {
            "reward": "reward_mean",
            "kl": "kl_mean",
            "entropy": "entropy_mean",
            "clipfrac": "clip_frac",
        }

        converted = df.copy()

        for alias, canonical in alias_to_canonical.items():
            if alias in converted.columns:
                converted[canonical] = converted.get(canonical, converted[alias])

        return converted

    def _extract_nested_field(self, df: pd.DataFrame, field_path: str) -> pd.Series:
        """Extract nested field using dot notation.

        Args:
            df: DataFrame containing the data
            field_path: Dot-separated field path (e.g., 'metrics.reward')

        Returns:
            Series with extracted values
        """
        parts = field_path.split('.')
        result = []

        for idx, row in df.iterrows():
            value = row.to_dict()  # Convert Series to dict for nested access
            try:
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        value = None
                        break
                result.append(value)
            except (KeyError, TypeError, AttributeError):
                result.append(None)

        return pd.Series(result, index=df.index)

    def _log_resolution_summary(
        self,
        resolved_fields: Dict[str, str],
        missing_fields: List[str],
        total_records: int,
    ) -> None:
        self.field_resolver.log_resolution_summary(
            resolved_fields, missing_fields, total_records
        )

    def _normalize_user_field_map(
        self, field_map: Optional[Dict[str, str]]
    ) -> Dict[str, str]:
        if not field_map:
            return {}

        canonical_fields = self.field_resolver.get_canonical_fields()

        keys_canonical = sum(
            1 for key in field_map if isinstance(key, str) and key in canonical_fields
        )
        values_canonical = sum(
            1 for value in field_map.values()
            if isinstance(value, str) and value in canonical_fields
        )

        if values_canonical > keys_canonical:
            normalized = {
                value: key
                for key, value in field_map.items()
                if isinstance(key, str) and isinstance(value, str)
            }
        else:
            normalized = {
                key: value
                for key, value in field_map.items()
                if isinstance(key, str) and isinstance(value, str)
            }

        return normalized

    def _build_effective_field_map(
        self, available_headers: Sequence[str]
    ) -> Dict[str, str]:
        effective = dict(self.field_map)

        if self.preset:
            preset_mapping = get_field_map_preset(self.preset)
            if preset_mapping is None:
                available = ", ".join(sorted(FIELD_MAP_PRESETS))
                raise ValidationError(
                    f"Unknown field map preset '{self.preset}'",
                    suggestion=f"Use one of: {available}",
                    error_code="UNKNOWN_FIELD_MAP_PRESET",
                )

            for source_field, canonical in preset_mapping.items():
                if canonical not in self.field_resolver.get_canonical_fields():
                    continue
                if canonical in effective:
                    continue
                if source_field not in available_headers and not (
                    self.allow_dot_paths and '.' in source_field
                ):
                    continue
                effective[canonical] = source_field

        return effective

    def _get_resolved_fields(
        self,
        available_headers: Sequence[str],
        effective_field_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        headers_list = list(available_headers)
        effective = (
            effective_field_map
            if effective_field_map is not None
            else self._build_effective_field_map(headers_list)
        )

        resolved: Dict[str, str] = {}
        for canonical_name in self.field_resolver.get_canonical_fields():
            resolved_name = self.field_resolver.resolve_field(
                canonical_name, headers_list, effective
            )
            if resolved_name:
                resolved[canonical_name] = resolved_name

        return resolved

    def get_metadata(self) -> dict:
        """Get metadata about the loaded data."""
        return {
            "source": str(self.source),
            "field_map": self.field_map,
            "config_file": str(self.config_file) if self.config_file else None,
            "allow_dot_paths": self.allow_dot_paths,
            "required_fields": self.required_fields,
            "validation_mode": self.validation_mode,
            "preset": self.preset,
            "supported_extensions": list(self.SUPPORTED_EXTENSIONS)
        }


class FlexibleJSONLAdapter(FlexibleDataAdapter):
    """Specialized adapter for JSONL files with streaming support."""

    def __init__(
        self,
        source: Union[str, Path],
        field_map: Optional[Dict[str, str]] = None,
        config_file: Optional[Union[str, Path]] = None,
        allow_dot_paths: bool = True,
        required_fields: Optional[List[str]] = None,
        validation_mode: str = "flexible",
        stream_large_files: bool = True,
        preset: Optional[str] = None,
    ):
        """Initialize JSONL adapter with streaming support.

        Args:
            source: Path to JSONL file
            field_map: Optional explicit mapping from canonical to actual field names
            config_file: Optional path to YAML/JSON config file with field mapping
            allow_dot_paths: Whether to support nested field access with dot notation
            required_fields: List of required canonical field names (defaults to ['step', 'reward'])
            validation_mode: Validation strictness - 'strict', 'flexible', or 'lenient'
            stream_large_files: Whether to stream large files instead of loading all at once
        """
        super().__init__(
            source,
            field_map,
            config_file,
            allow_dot_paths,
            required_fields,
            validation_mode,
            preset,
        )
        self.stream_large_files = stream_large_files

    def can_handle(self) -> bool:
        """Check if source is a JSONL file."""
        if not self.source.exists():
            return False

        if self.source.is_file():
            return self.source.suffix.lower() == '.jsonl'
        elif self.source.is_dir():
            return len(list(self.source.glob("*.jsonl"))) > 0

        return False

    def load(self) -> pd.DataFrame:
        """Load JSONL data with optional streaming for large files."""
        if not self.can_handle():
            raise AdapterError(
                f"Cannot handle source: {self.source}",
                suggestion="Source must be a .jsonl file or directory containing .jsonl files",
                error_code="CANNOT_HANDLE_SOURCE"
            )

        # For large files, use streaming approach
        if self.stream_large_files and self.source.is_file():
            file_size = self.source.stat().st_size
            if file_size > 100 * 1024 * 1024:  # 100MB threshold
                self.logger.info(f"Large file detected ({file_size / 1024 / 1024:.1f}MB), using streaming")
                return self._load_streaming()

        return super().load()

    def _load_streaming(self) -> pd.DataFrame:
        """Load large JSONL file using streaming approach."""
        # First pass: determine schema and field mapping
        schema_info = self._analyze_schema()

        # Second pass: load data with resolved schema
        streamed = self._load_with_schema(schema_info)
        converted = self._convert_to_training_metrics(streamed)

        try:
            from ..ingest.training_metrics_normalizer import (
                standardize_training_metrics,
            )

            return standardize_training_metrics(converted)
        except ValidationError as exc:
            raise AdapterError(
                f"Failed to standardize schema for {self.source}: {exc}",
                suggestion="Check that the field mapping resolves 'step' and numeric metrics",
                error_code="SCHEMA_STANDARDIZATION_FAILED",
            ) from exc

    def _analyze_schema(self) -> Dict[str, Any]:
        """Analyze schema from first few records."""
        sample_size = 1000
        sample_data = []

        with open(self.source, encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break

                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        if isinstance(record, dict):
                            sample_data.append(record)
                    except json.JSONDecodeError:
                        continue

        if not sample_data:
            raise AdapterError(
                "No valid records found in JSONL file",
                suggestion="Check that the file contains valid JSONL data",
                error_code="NO_VALID_RECORDS"
            )

        # Analyze field mapping
        sample_df = pd.DataFrame(sample_data)
        available_headers = sample_df.columns.tolist()
        effective_field_map = self._build_effective_field_map(available_headers)

        resolved_fields = self._get_resolved_fields(
            available_headers, effective_field_map
        )
        missing_fields = self.field_resolver.get_missing_fields(
            self.required_fields, available_headers, effective_field_map
        )

        if missing_fields:
            if self.validation_mode == "strict":
                raise SchemaError(
                    f"Missing required fields: {', '.join(missing_fields)}",
                    missing_fields, available_headers, self.field_resolver
                )
            elif self.validation_mode == "flexible":
                step_missing = 'step' in missing_fields
                has_metric = any(
                    self.field_resolver.resolve_field(
                        metric, available_headers, effective_field_map
                    )
                    for metric in ['reward', 'score', 'return', 'kl', 'entropy', 'loss']
                )

                if step_missing:
                    raise SchemaError(
                        "Missing required 'step' field in flexible mode",
                        ['step'], available_headers, self.field_resolver
                    )
                elif not has_metric:
                    raise SchemaError(
                        "No metric fields found in flexible mode (need at least one of: reward, score, return, kl, entropy, loss)",
                        [], available_headers, self.field_resolver
                    )

        self._log_resolution_summary(
            resolved_fields,
            missing_fields,
            len(sample_data),
        )

        return {
            "resolved_fields": resolved_fields,
            "available_headers": available_headers,
            "sample_data": sample_data,
            "effective_field_map": effective_field_map,
        }

    def _load_with_schema(self, schema_info: Dict[str, Any]) -> pd.DataFrame:
        """Load data using pre-analyzed schema."""
        resolved_fields = schema_info["resolved_fields"]
        all_data = []

        with open(self.source, encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        # Extract only the fields we need
                        extracted_record = {}
                        for canonical, actual in resolved_fields.items():
                            if self.allow_dot_paths and '.' in actual:
                                extracted_record[canonical] = self._extract_nested_value(record, actual)
                            else:
                                extracted_record[canonical] = record.get(actual)
                        all_data.append(extracted_record)
                except json.JSONDecodeError:
                    self.logger.warning(f"JSON decode error at line {line_num}")
                    continue

        return pd.DataFrame(all_data)

    def _extract_nested_value(self, record: Dict[str, Any], field_path: str) -> Any:
        """Extract nested value from a record."""
        parts = field_path.split('.')
        value = record

        try:
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            return value
        except (KeyError, TypeError, AttributeError):
            return None
