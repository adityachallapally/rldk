"""Configuration for visualization and plotting parameters."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization and plotting parameters."""

    # Figure settings
    DEFAULT_FIGSIZE: Tuple[int, int] = (12, 8)
    DEFAULT_DPI: int = 300
    DEFAULT_STYLE: str = "seaborn-v0_8"

    # Subplot settings
    SUBPLOT_ROWS: int = 2
    SUBPLOT_COLS: int = 2
    SUBPLOT_SPACING: float = 0.3

    # Font settings
    TITLE_FONTSIZE: int = 16
    TITLE_FONTWEIGHT: str = "bold"
    LABEL_FONTSIZE: int = 12
    TICK_FONTSIZE: int = 10
    LEGEND_FONTSIZE: int = 10
    TEXT_FONTSIZE: int = 9

    # Color settings
    PRIMARY_COLOR: str = "blue"
    SECONDARY_COLOR: str = "red"
    TERTIARY_COLOR: str = "orange"
    GRID_ALPHA: float = 0.3
    LINE_ALPHA: float = 0.7
    SCATTER_ALPHA: float = 0.6

    # Line settings
    LINE_WIDTH: float = 1.0
    TREND_LINE_WIDTH: float = 2.0
    TREND_LINE_ALPHA: float = 0.8
    TREND_LINE_STYLE: str = "--"

    # Histogram settings
    HISTOGRAM_BINS: int = 30
    HISTOGRAM_ALPHA: float = 0.7
    HISTOGRAM_EDGECOLOR: str = "black"
    HISTOGRAM_LINEWIDTH: float = 0.5

    # Scatter plot settings
    SCATTER_SIZE: int = 20
    SCATTER_MARKER: str = "o"

    # Pie chart settings
    PIE_STARTANGLE: int = 90
    PIE_AUTOPCT: str = "%1.1f%%"

    # Calibration curve settings
    CALIBRATION_BINS: int = 10
    CALIBRATION_MARKER: str = "o-"
    PERFECT_CALIBRATION_STYLE: str = "k--"

    # Statistics text box settings
    STATS_BBOX_STYLE: str = "round"
    STATS_BBOX_FACE_COLOR: str = "white"
    STATS_BBOX_ALPHA: float = 0.8
    STATS_VERTICAL_ALIGNMENT: str = "top"

    # Grid settings
    GRID_ENABLED: bool = True

    # Legend settings
    LEGEND_LOCATION: str = "best"
    LEGEND_FRAME_ALPHA: float = 0.9

    # Axis settings
    AXIS_LABEL_PADDING: float = 12.0
    TICK_LABEL_PADDING: float = 8.0

    # Colorbar settings
    COLORBAR_SHRINK: float = 0.8
    COLORBAR_ASPECT: float = 20

    # Sampling settings for large datasets
    MAX_POINTS_FOR_PLOT: int = 1000
    SAMPLING_RANDOM_STATE: int = 42

    # Output settings
    SAVE_DPI: int = 300
    SAVE_BBOX_INCHES: str = "tight"
    SAVE_FORMAT: str = "png"

    # Error handling
    SHOW_ERROR_MESSAGES: bool = True
    ERROR_MESSAGE_ALPHA: float = 0.8
    ERROR_MESSAGE_COLOR: str = "red"

    # Data validation
    MIN_DATA_POINTS_FOR_PLOT: int = 10
    MIN_DATA_POINTS_FOR_CALIBRATION: int = 10
    MIN_DATA_POINTS_FOR_CORRELATION: int = 10

    # Trend line settings
    TREND_POLYFIT_DEGREE: int = 1
    TREND_COLOR: str = "red"
    TREND_ALPHA: float = 0.8

    # Percentile lines
    PERCENTILE_LINE_COLOR: str = "orange"
    PERCENTILE_LINE_STYLE: str = ":"
    PERCENTILE_LINE_ALPHA: float = 0.6

    # Mean/std lines
    MEAN_LINE_COLOR: str = "red"
    MEAN_LINE_STYLE: str = "--"
    MEAN_LINE_ALPHA: float = 0.8

    STD_LINE_COLOR: str = "orange"
    STD_LINE_STYLE: str = ":"
    STD_LINE_ALPHA: float = 0.6

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

# Default configuration instance
DEFAULT_VISUALIZATION_CONFIG = VisualizationConfig()

# Environment-specific configurations
CONFIGS = {
    "default": DEFAULT_VISUALIZATION_CONFIG,
    "publication": VisualizationConfig(
        DEFAULT_FIGSIZE=(10, 6),
        DEFAULT_DPI=600,
        TITLE_FONTSIZE=14,
        LABEL_FONTSIZE=11,
        TICK_FONTSIZE=9,
        LEGEND_FONTSIZE=9,
        TEXT_FONTSIZE=8,
        SAVE_DPI=600,
    ),
    "presentation": VisualizationConfig(
        DEFAULT_FIGSIZE=(16, 10),
        DEFAULT_DPI=150,
        TITLE_FONTSIZE=20,
        LABEL_FONTSIZE=16,
        TICK_FONTSIZE=14,
        LEGEND_FONTSIZE=14,
        TEXT_FONTSIZE=12,
        SAVE_DPI=150,
    ),
    "web": VisualizationConfig(
        DEFAULT_FIGSIZE=(8, 6),
        DEFAULT_DPI=100,
        TITLE_FONTSIZE=12,
        LABEL_FONTSIZE=10,
        TICK_FONTSIZE=8,
        LEGEND_FONTSIZE=8,
        TEXT_FONTSIZE=7,
        SAVE_DPI=100,
    ),
}

def get_visualization_config(config_name: str = "default") -> VisualizationConfig:
    """Get visualization configuration by name."""
    if config_name not in CONFIGS:
        logger.warning(f"Unknown config name '{config_name}', using default")
        return DEFAULT_VISUALIZATION_CONFIG
    return CONFIGS[config_name]

def create_custom_visualization_config(**kwargs) -> VisualizationConfig:
    """Create a custom visualization configuration with overridden values."""
    config = VisualizationConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown visualization configuration parameter: {key}")
    return config
