"""Command-line interface for RL Debug Kit."""

# Import the main app from the modular CLI structure
from .cli.main import app

# Make the app available for direct execution
if __name__ == "__main__":
    app()