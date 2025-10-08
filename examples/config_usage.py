#!/usr/bin/env python3
"""
Example demonstrating how to use the RLDK configuration system.
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import rldk
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rldk.config import ConfigSchema, RLDKSettings, settings
from rldk.config.schemas import AnalysisConfig, LoggingConfig, WandBConfig


def main():
    """Demonstrate configuration usage."""
    print("=== RLDK Configuration System Demo ===\n")

    # 1. Using the global settings instance
    print("1. Global Settings Instance:")
    print(f"   Default output directory: {settings.default_output_dir}")
    print(f"   Log level: {settings.log_level}")
    print(f"   W&B enabled: {settings.wandb_enabled}")
    print(f"   W&B project: {settings.wandb_project}")
    print(f"   Default tolerance: {settings.default_tolerance}")
    print(f"   Default window size: {settings.default_window_size}")
    print()

    # 1.5. Initialize configuration (optional - only needed for logging/directories)
    print("1.5. Initializing Configuration:")
    try:
        settings.initialize()
        print("   ✅ Configuration initialized successfully")
    except PermissionError as e:
        print(f"   ⚠️  Could not create directories (read-only environment): {e}")
    except Exception as e:
        print(f"   ⚠️  Configuration initialization warning: {e}")
    print()

    # 2. Creating a custom settings instance
    print("2. Custom Settings Instance:")
    custom_settings = RLDKSettings(
        log_level="DEBUG",
        wandb_enabled=False,
        default_tolerance=0.05,
        wandb_project="my-custom-project"
    )
    print(f"   Custom log level: {custom_settings.log_level}")
    print(f"   Custom W&B enabled: {custom_settings.wandb_enabled}")
    print(f"   Custom tolerance: {custom_settings.default_tolerance}")
    print(f"   Custom W&B project: {custom_settings.wandb_project}")
    print()

    # 3. Using configuration schemas
    print("3. Configuration Schemas:")

    # Create a logging config
    logging_config = LoggingConfig(
        level="WARNING",
        console=False,
        file=Path("custom.log")
    )
    print(f"   Logging config level: {logging_config.level}")
    print(f"   Logging config console: {logging_config.console}")
    print(f"   Logging config file: {logging_config.file}")

    # Create an analysis config
    analysis_config = AnalysisConfig(
        tolerance=0.02,
        window_size=100,
        batch_size=64
    )
    print(f"   Analysis config tolerance: {analysis_config.tolerance}")
    print(f"   Analysis config window size: {analysis_config.window_size}")
    print(f"   Analysis config batch size: {analysis_config.batch_size}")

    # Create a W&B config
    wandb_config = WandBConfig(
        project="my-experiment",
        enabled=True,
        tags=["experiment", "test"],
        entity="my-entity"
    )
    print(f"   W&B config project: {wandb_config.project}")
    print(f"   W&B config enabled: {wandb_config.enabled}")
    print(f"   W&B config tags: {wandb_config.tags}")
    print(f"   W&B config entity: {wandb_config.entity}")
    print()

    # 4. Using the main configuration schema
    print("4. Main Configuration Schema:")
    main_config = ConfigSchema(
        logging=logging_config,
        analysis=analysis_config,
        wandb=wandb_config
    )

    print(f"   Configuration version: {main_config.version}")
    print(f"   Debug mode: {main_config.environment.debug}")
    print(f"   Performance workers: {main_config.performance.num_workers}")
    print(f"   Visualization style: {main_config.visualization.style}")

    # Get configuration summary
    summary = main_config.get_summary()
    print("\n   Configuration Summary:")
    for key, value in summary.items():
        print(f"     {key}: {value}")
    print()

    # 5. Environment variable configuration
    print("5. Environment Variable Configuration:")
    print("   You can override settings using environment variables with RLDK_ prefix:")
    print("   Example: RLDK_LOG_LEVEL=DEBUG RLDK_WANDB_ENABLED=false python script.py")
    print()

    # 6. Configuration validation
    print("6. Configuration Validation:")
    is_valid = main_config.validate_config()
    print(f"   Configuration is valid: {is_valid}")
    print()

    # 7. Export configuration
    print("7. Export Configuration:")
    config_dict = main_config.to_dict()
    print(f"   Configuration as dict keys: {list(config_dict.keys())}")

    config_json = main_config.to_json()
    print(f"   Configuration JSON length: {len(config_json)} characters")
    print()

    print("=== Demo Complete ===")


if __name__ == "__main__":
    main()
