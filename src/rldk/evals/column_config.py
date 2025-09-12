"""Column configuration utilities for evaluation metrics."""

from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ColumnConfig:
    """Configuration class for evaluation metric column mappings."""
    
    def __init__(self):
        self.configs = {
            "throughput": {
                "primary_column": "events",
                "alternative_columns": ["logs", "event_logs", "training_logs", "metrics", "performance_logs"],
                "fallback_metrics": ["tokens_per_second", "throughput_rate", "processing_speed", 
                                   "inference_speed", "batch_throughput", "tps", "throughput"]
            },
            "toxicity": {
                "primary_column": "output",
                "alternative_columns": ["response", "generated_text", "completion", "text", "generated", "model_output"],
                "fallback_metrics": ["toxicity_score", "harm_score", "safety_score", "danger_score",
                                   "inappropriate_score", "offensive_score", "hate_score"]
            },
            "bias": {
                "primary_column": "output",
                "alternative_columns": ["response", "generated_text", "completion", "text", "generated", "model_output"],
                "fallback_metrics": ["bias_score", "fairness_score", "demographic_bias", "unfairness_score",
                                   "discrimination_score", "stereotype_score", "equity_score"]
            }
        }
    
    def get_config(self, metric_name: str) -> Dict[str, Any]:
        """
        Get column configuration for a specific metric.
        
        Args:
            metric_name: Name of the metric (e.g., "throughput", "toxicity", "bias")
            
        Returns:
            Dictionary with column configuration
        """
        return self.configs.get(metric_name, {})
    
    def set_primary_column(self, metric_name: str, column_name: str) -> None:
        """
        Set the primary column for a metric.
        
        Args:
            metric_name: Name of the metric
            column_name: Name of the primary column
        """
        if metric_name not in self.configs:
            self.configs[metric_name] = {}
        self.configs[metric_name]["primary_column"] = column_name
        logger.info(f"Set primary column for {metric_name} to {column_name}")
    
    def add_alternative_column(self, metric_name: str, column_name: str) -> None:
        """
        Add an alternative column for a metric.
        
        Args:
            metric_name: Name of the metric
            column_name: Name of the alternative column
        """
        if metric_name not in self.configs:
            self.configs[metric_name] = {}
        if "alternative_columns" not in self.configs[metric_name]:
            self.configs[metric_name]["alternative_columns"] = []
        if column_name not in self.configs[metric_name]["alternative_columns"]:
            self.configs[metric_name]["alternative_columns"].append(column_name)
            logger.info(f"Added alternative column {column_name} for {metric_name}")
    
    def add_fallback_metric(self, metric_name: str, fallback_metric: str) -> None:
        """
        Add a fallback metric for a metric.
        
        Args:
            metric_name: Name of the metric
            fallback_metric: Name of the fallback metric
        """
        if metric_name not in self.configs:
            self.configs[metric_name] = {}
        if "fallback_metrics" not in self.configs[metric_name]:
            self.configs[metric_name]["fallback_metrics"] = []
        if fallback_metric not in self.configs[metric_name]["fallback_metrics"]:
            self.configs[metric_name]["fallback_metrics"].append(fallback_metric)
            logger.info(f"Added fallback metric {fallback_metric} for {metric_name}")
    
    def get_evaluation_kwargs(self, metric_name: str, custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get evaluation kwargs for a metric with column configuration.
        
        Args:
            metric_name: Name of the metric
            custom_config: Custom configuration to override defaults
            
        Returns:
            Dictionary of kwargs for the evaluation function
        """
        config = self.get_config(metric_name)
        if custom_config:
            config.update(custom_config)
        
        kwargs = {}
        if "primary_column" in config:
            kwargs["log_column" if metric_name == "throughput" else "output_column"] = config["primary_column"]
        if "alternative_columns" in config:
            kwargs["alternative_columns"] = config["alternative_columns"]
        if "fallback_metrics" in config:
            kwargs["fallback_to_other_metrics"] = True
        
        return kwargs
    
    def detect_columns(self, data_columns: List[str]) -> Dict[str, Dict[str, List[str]]]:
        """
        Detect available columns for each metric in the data.
        
        Args:
            data_columns: List of available column names in the data
            
        Returns:
            Dictionary mapping metric names to their detected columns
        """
        detected = {}
        
        for metric_name, config in self.configs.items():
            detected[metric_name] = {
                "primary_found": False,
                "alternatives_found": [],
                "fallbacks_found": []
            }
            
            # Check primary column
            primary_col = config.get("primary_column")
            if primary_col and primary_col in data_columns:
                detected[metric_name]["primary_found"] = True
            
            # Check alternative columns
            alt_cols = config.get("alternative_columns", [])
            for col in alt_cols:
                if col in data_columns:
                    detected[metric_name]["alternatives_found"].append(col)
            
            # Check fallback metrics
            fallback_metrics = config.get("fallback_metrics", [])
            for metric in fallback_metrics:
                if metric in data_columns:
                    detected[metric_name]["fallbacks_found"].append(metric)
        
        return detected
    
    def suggest_columns(self, data_columns: List[str]) -> Dict[str, List[str]]:
        """
        Suggest column mappings based on available data columns.
        
        Args:
            data_columns: List of available column names in the data
            
        Returns:
            Dictionary mapping metric names to suggested column mappings
        """
        suggestions = {}
        
        for metric_name, config in self.configs.items():
            suggestions[metric_name] = []
            
            # Check if primary column exists
            primary_col = config.get("primary_column")
            if primary_col and primary_col in data_columns:
                suggestions[metric_name].append(f"Use '{primary_col}' as primary column")
            else:
                # Find best alternative
                alt_cols = config.get("alternative_columns", [])
                found_alt = [col for col in alt_cols if col in data_columns]
                if found_alt:
                    suggestions[metric_name].append(f"Use '{found_alt[0]}' as alternative to '{primary_col}'")
                
                # Check for fallback metrics
                fallback_metrics = config.get("fallback_metrics", [])
                found_fallbacks = [col for col in fallback_metrics if col in data_columns]
                if found_fallbacks:
                    suggestions[metric_name].append(f"Use fallback metrics: {found_fallbacks}")
                
                if not found_alt and not found_fallbacks:
                    suggestions[metric_name].append(f"No suitable columns found. Available: {data_columns}")
        
        return suggestions


# Global configuration instance
default_config = ColumnConfig()


def get_column_config(metric_name: str) -> Dict[str, Any]:
    """Get column configuration for a metric."""
    return default_config.get_config(metric_name)


def set_primary_column(metric_name: str, column_name: str) -> None:
    """Set the primary column for a metric."""
    default_config.set_primary_column(metric_name, column_name)


def add_alternative_column(metric_name: str, column_name: str) -> None:
    """Add an alternative column for a metric."""
    default_config.add_alternative_column(metric_name, column_name)


def add_fallback_metric(metric_name: str, fallback_metric: str) -> None:
    """Add a fallback metric for a metric."""
    default_config.add_fallback_metric(metric_name, fallback_metric)


def get_evaluation_kwargs(metric_name: str, custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get evaluation kwargs for a metric with column configuration."""
    return default_config.get_evaluation_kwargs(metric_name, custom_config)


def detect_columns(data_columns: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """Detect available columns for each metric in the data."""
    return default_config.detect_columns(data_columns)


def suggest_columns(data_columns: List[str]) -> Dict[str, List[str]]:
    """Suggest column mappings based on available data columns."""
    return default_config.suggest_columns(data_columns)