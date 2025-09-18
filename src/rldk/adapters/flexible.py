"""Flexible data adapters with schema mapping and multiple format support."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml

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
        preset: Optional[str] = None,
        allow_dot_paths: bool = True,
        required_fields: Optional[List[str]] = None,
        validation_mode: str = "flexible"
    ):
        """Initialize the flexible adapter.

        Args:
            source: Path to data file or directory
            field_map: Optional explicit mapping from canonical to actual field names
            config_file: Optional path to YAML/JSON config file with field mapping
            preset: Optional preset name for field mapping
            allow_dot_paths: Whether to support nested field access with dot notation
            required_fields: List of required canonical field names (defaults to ['step', 'reward'])
            validation_mode: Validation strictness - 'strict', 'flexible', or 'lenient'
        """
        super().__init__(source)
        self.field_resolver = FieldResolver(allow_dot_paths=allow_dot_paths)
        self.preset = preset
        self.config_file = Path(config_file) if config_file else None
        self.allow_dot_paths = allow_dot_paths
        self.required_fields = required_fields or ['step', 'reward']
        self.validation_mode = validation_mode
        self.logger = logging.getLogger(self.__class__.__name__)

        from ..ingest.stream_normalizer import _combine_field_maps
        preset_field_map = _combine_field_maps(None, preset) if preset else {}
        
        config_field_map = {}
        if self.config_file and self.config_file.exists():
            config_field_map = self._load_config()
        
        combined_field_map = preset_field_map.copy()
        combined_field_map.update(config_field_map)
        
        if field_map:
            explicit_targets = set(field_map.values())
            combined_field_map = {k: v for k, v in combined_field_map.items() 
                                if v not in explicit_targets}
            combined_field_map.update(field_map)
        
        self.field_map = combined_field_map

    def _load_config(self) -> Dict[str, str]:
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
                self.logger.info(f"Loaded field map from {self.config_file}")
                return config['field_map']
            else:
                self.logger.warning(f"Config file {self.config_file} does not contain 'field_map' section")
                return {}

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

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Resolve field names and validate schema
        df = self._resolve_and_validate_schema(df)

        # Log resolution summary
        resolved_fields = self._get_resolved_fields(df.columns.tolist())
        missing_fields = self.field_resolver.get_missing_fields(
            self.required_fields, df.columns.tolist(), self.field_map
        )
        self.field_resolver.log_resolution_summary(
            resolved_fields, missing_fields, len(df)
        )

        try:
            from ..ingest.training_metrics_normalizer import standardize_training_metrics
            df = standardize_training_metrics(df)
        except Exception as e:
            self.logger.warning(f"Standardization failed, returning raw resolved data: {e}")

        return df

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
        all_data = []

        for ext in self.SUPPORTED_EXTENSIONS:
            files = list(dir_path.glob(f"*{ext}"))
            for file_path in files:
                try:
                    file_data = self._load_single_file(file_path)
                    all_data.extend(file_data)
                except Exception as e:
                    self.logger.warning(f"Failed to load {file_path}: {e}")
                    continue

        return all_data

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
        available_headers = df.columns.tolist()

        if self.field_map:
            df = df.rename(columns=self.field_map)
            available_headers = df.columns.tolist()

        # Check for missing required fields based on validation mode
        missing_fields = self.field_resolver.get_missing_fields(
            self.required_fields, available_headers, self.field_map
        )

        if missing_fields:
            if self.validation_mode == "strict":
                raise SchemaError(
                    f"Missing required fields: {', '.join(missing_fields)}",
                    missing_fields, available_headers, self.field_resolver
                )
            elif self.validation_mode == "flexible":
                step_missing = 'step' in missing_fields
                has_metric = any(self.field_resolver.resolve_field(metric, available_headers, self.field_map)
                               for metric in ['reward', 'score', 'return', 'kl', 'entropy', 'loss'])

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
                elif missing_fields and self.logger:
                    optional_missing = [f for f in missing_fields if f != 'step']
                    if optional_missing:
                        self.logger.warning(f"Missing optional fields in flexible mode: {', '.join(optional_missing)}")
            elif self.validation_mode == "lenient" and missing_fields and self.logger:
                self.logger.warning(f"Missing fields in lenient mode: {', '.join(missing_fields)}")

        resolved_data = {}

        for canonical_name in self.field_resolver.get_canonical_fields():
            if canonical_name in available_headers:
                resolved_data[canonical_name] = df[canonical_name].values
            else:
                resolved_name = self.field_resolver.resolve_field(
                    canonical_name, available_headers, {}
                )
                if resolved_name and resolved_name in df.columns:
                    if self.allow_dot_paths and '.' in resolved_name:
                        nested_series = self._extract_nested_field(df, resolved_name)
                        resolved_data[canonical_name] = nested_series.values
                    else:
                        resolved_data[canonical_name] = df[resolved_name].values

        # Create new DataFrame with canonical column names
        result_df = pd.DataFrame(resolved_data)

        # Ensure required columns exist
        for field in self.required_fields:
            if field not in result_df.columns:
                result_df[field] = None
                self.logger.warning(f"Required field '{field}' not found, filled with None values")

        return result_df

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

    def _get_resolved_fields(self, available_headers: List[str]) -> Dict[str, str]:
        """Get mapping of resolved fields."""
        resolved = {}
        for canonical_name in self.field_resolver.get_canonical_fields():
            if canonical_name in available_headers:
                resolved[canonical_name] = canonical_name
            else:
                resolved_name = self.field_resolver.resolve_field(
                    canonical_name, available_headers, self.field_map
                )
                if resolved_name:
                    resolved[canonical_name] = resolved_name
        return resolved

    def get_metadata(self) -> dict:
        """Get metadata about the loaded data."""
        return {
            "source": str(self.source),
            "preset": self.preset,
            "field_map": self.field_map,
            "config_file": str(self.config_file) if self.config_file else None,
            "allow_dot_paths": self.allow_dot_paths,
            "required_fields": self.required_fields,
            "validation_mode": self.validation_mode,
            "supported_extensions": list(self.SUPPORTED_EXTENSIONS)
        }


class FlexibleJSONLAdapter(FlexibleDataAdapter):
    """Specialized adapter for JSONL files with streaming support."""

    def __init__(
        self,
        source: Union[str, Path],
        field_map: Optional[Dict[str, str]] = None,
        config_file: Optional[Union[str, Path]] = None,
        preset: Optional[str] = None,
        allow_dot_paths: bool = True,
        required_fields: Optional[List[str]] = None,
        validation_mode: str = "flexible",
        stream_large_files: bool = True
    ):
        """Initialize JSONL adapter with streaming support.

        Args:
            source: Path to JSONL file
            field_map: Optional explicit mapping from canonical to actual field names
            config_file: Optional path to YAML/JSON config file with field mapping
            preset: Optional preset name for field mapping
            allow_dot_paths: Whether to support nested field access with dot notation
            required_fields: List of required canonical field names (defaults to ['step', 'reward'])
            validation_mode: Validation strictness - 'strict', 'flexible', or 'lenient'
            stream_large_files: Whether to stream large files instead of loading all at once
        """
        super().__init__(source, field_map, config_file, preset, allow_dot_paths, required_fields, validation_mode)
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
        return self._load_with_schema(schema_info)

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
        original_headers = sample_df.columns.tolist()
        
        if self.field_map:
            sample_df = sample_df.rename(columns=self.field_map)
        available_headers = sample_df.columns.tolist()

        resolved_fields = self._get_resolved_fields(available_headers)
        missing_fields = self.field_resolver.get_missing_fields(
            self.required_fields, available_headers, {}
        )

        if missing_fields:
            if self.validation_mode == "strict":
                raise SchemaError(
                    f"Missing required fields: {', '.join(missing_fields)}",
                    missing_fields, available_headers, self.field_resolver
                )
            elif self.validation_mode == "flexible":
                step_missing = 'step' in missing_fields
                has_metric = any(self.field_resolver.resolve_field(metric, available_headers, self.field_map)
                               for metric in ['reward', 'score', 'return', 'kl', 'entropy', 'loss'])

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

        return {
            "resolved_fields": resolved_fields,
            "available_headers": available_headers,
            "sample_data": sample_data
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
