"""Common CLI utilities and shared options for RLDK commands."""

import sys
import logging
from pathlib import Path
from typing import NoReturn, Optional, Any, Dict, List
from functools import wraps

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

# Initialize rich console for consistent output
console = Console()

# Exit codes following the specification
class ExitCodes:
    SUCCESS = 0
    INVALID_ARGS = 2
    THRESHOLD_FAILED = 3
    RUNTIME_ERROR = 4
    INTERNAL_ERROR = 5

def die(msg: str, code: int = ExitCodes.RUNTIME_ERROR) -> NoReturn:
    """Print error message and exit with specified code.
    
    Args:
        msg: Error message to display
        code: Exit code to return
    """
    print(f"ERROR: {msg}", file=sys.stderr)
    
    # Add hint based on exit code
    if code == ExitCodes.INVALID_ARGS:
        print("Hint: Use --help for usage information", file=sys.stderr)
    elif code == ExitCodes.RUNTIME_ERROR:
        print("Hint: Use --verbose or --log-file for detailed error information", file=sys.stderr)
    elif code == ExitCodes.THRESHOLD_FAILED:
        print("Hint: Check threshold settings or use --gate mode for CI integration", file=sys.stderr)
    
    sys.exit(code)

def hint(msg: str) -> None:
    """Print a hint message to stderr.
    
    Args:
        msg: Hint message to display
    """
    console.print(f"[yellow]Hint:[/yellow] {msg}", file=sys.stderr)

def success(msg: str) -> None:
    """Print a success message.
    
    Args:
        msg: Success message to display
    """
    console.print(f"[green]✅ {msg}[/green]")

def warning(msg: str) -> None:
    """Print a warning message.
    
    Args:
        msg: Warning message to display
    """
    console.print(f"[yellow]⚠️  {msg}[/yellow]")

def info(msg: str) -> None:
    """Print an info message.
    
    Args:
        msg: Info message to display
    """
    console.print(f"[blue]ℹ️  {msg}[/blue]")

def error(msg: str) -> None:
    """Print an error message.
    
    Args:
        msg: Error message to display
    """
    console.print(f"[red]❌ {msg}[/red]")

def progress_operation(operation_name: str):
    """Decorator to show progress for long-running operations.
    
    Args:
        operation_name: Name of the operation to display
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[blue]{operation_name}[/blue]"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(operation_name, total=None)
                try:
                    result = func(*args, **kwargs)
                    progress.update(task, completed=True)
                    return result
                except Exception as e:
                    progress.update(task, completed=True)
                    raise
        return wrapper
    return decorator

class SharedOptions:
    """Shared options mixin for all CLI commands."""
    
    @staticmethod
    def config_option() -> Optional[str]:
        """Load defaults from a TOML or YAML file if present."""
        return typer.Option(
            None,
            "--config",
            help="Path to configuration file (TOML or YAML)"
        )
    
    @staticmethod
    def json_option() -> bool:
        """Print machine readable output where applicable."""
        return typer.Option(
            False,
            "--json",
            help="Print machine-readable JSON output"
        )
    
    @staticmethod
    def verbose_option() -> bool:
        """Enable verbose logging."""
        return typer.Option(
            False,
            "--verbose", "-v",
            help="Enable verbose logging"
        )
    
    @staticmethod
    def quiet_option() -> bool:
        """Reduce logs to warnings and errors only."""
        return typer.Option(
            False,
            "--quiet",
            help="Reduce logs to warnings and errors only"
        )
    
    @staticmethod
    def log_file_option() -> Optional[str]:
        """Write full logs to specified file."""
        return typer.Option(
            None,
            "--log-file",
            help="Write full logs to specified file"
        )

def setup_logging(
    verbose: bool = False,
    quiet: bool = False,
    log_file: Optional[str] = None
) -> None:
    """Setup logging configuration based on CLI options.
    
    Args:
        verbose: Enable verbose logging
        quiet: Reduce logs to warnings and errors only
        log_file: Write logs to specified file
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    # Configure logging format
    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Setup handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format_string))
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always debug level for file
        file_handler.setFormatter(logging.Formatter(format_string))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True
    )

def validate_file_path(
    file_path: str,
    must_exist: bool = True,
    file_extensions: Optional[List[str]] = None
) -> Path:
    """Validate file path and return Path object.
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must exist
        file_extensions: List of allowed file extensions
        
    Returns:
        Path object if valid
        
    Raises:
        SystemExit: If validation fails
    """
    path = Path(file_path)
    
    if must_exist and not path.exists():
        die(f"File does not exist: {file_path}", ExitCodes.INVALID_ARGS)
    
    if must_exist and not path.is_file():
        die(f"Path is not a file: {file_path}", ExitCodes.INVALID_ARGS)
    
    if file_extensions and path.suffix not in file_extensions:
        die(
            f"Invalid file extension: {path.suffix}. Expected one of: {', '.join(file_extensions)}",
            ExitCodes.INVALID_ARGS
        )
    
    return path

def validate_directory_path(
    dir_path: str,
    must_exist: bool = True,
    create_if_missing: bool = False
) -> Path:
    """Validate directory path and return Path object.
    
    Args:
        dir_path: Path to validate
        must_exist: Whether directory must exist
        create_if_missing: Whether to create directory if missing
        
    Returns:
        Path object if valid
        
    Raises:
        SystemExit: If validation fails
    """
    path = Path(dir_path)
    
    if must_exist and not path.exists():
        if create_if_missing:
            try:
                path.mkdir(parents=True, exist_ok=True)
                info(f"Created directory: {path}")
            except Exception as e:
                die(f"Failed to create directory {path}: {e}", ExitCodes.RUNTIME_ERROR)
        else:
            die(f"Directory does not exist: {dir_path}", ExitCodes.INVALID_ARGS)
    
    if must_exist and not path.is_dir():
        die(f"Path is not a directory: {dir_path}", ExitCodes.INVALID_ARGS)
    
    return path

def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not config_path:
        return {}
    
    path = validate_file_path(config_path, must_exist=True)
    
    try:
        if path.suffix.lower() in ['.yaml', '.yml']:
            import yaml
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}
        elif path.suffix.lower() == '.toml':
            import toml
            return toml.load(path) or {}
        else:
            die(f"Unsupported config file format: {path.suffix}", ExitCodes.INVALID_ARGS)
    except Exception as e:
        die(f"Failed to load config file {config_path}: {e}", ExitCodes.RUNTIME_ERROR)

def handle_cli_exceptions(func):
    """Decorator to handle CLI exceptions consistently.
    
    This decorator catches exceptions and provides consistent error handling
    with appropriate exit codes and user-friendly messages.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user[/yellow]")
            sys.exit(1)
        except typer.Exit:
            # Re-raise typer.Exit to preserve intended exit codes
            raise
        except FileNotFoundError as e:
            die(f"File not found: {e}", ExitCodes.INVALID_ARGS)
        except PermissionError as e:
            die(f"Permission denied: {e}", ExitCodes.RUNTIME_ERROR)
        except ValueError as e:
            die(f"Invalid value: {e}", ExitCodes.INVALID_ARGS)
        except Exception as e:
            die(f"Unexpected error: {e}", ExitCodes.INTERNAL_ERROR)
    
    return wrapper

def print_json_output(data: Dict[str, Any]) -> None:
    """Print data as formatted JSON to stdout.
    
    Args:
        data: Data to print as JSON
    """
    import json
    print(json.dumps(data, indent=2, default=str))

def print_usage_examples(command: str, examples: List[str]) -> None:
    """Print usage examples for a command.
    
    Args:
        command: Command name
        examples: List of example usage strings
    """
    console.print(f"\n[blue]Examples for '{command}':[/blue]")
    for i, example in enumerate(examples, 1):
        console.print(f"  {i}. {example}")

def print_troubleshooting_tips(tips: List[str]) -> None:
    """Print troubleshooting tips.
    
    Args:
        tips: List of troubleshooting tip strings
    """
    console.print("\n[yellow]Troubleshooting tips:[/yellow]")
    for tip in tips:
        console.print(f"  • {tip}")

def format_error_message(error: Exception, context: str = "") -> str:
    """Format error message with context.
    
    Args:
        error: Exception to format
        context: Additional context string
        
    Returns:
        Formatted error message
    """
    msg = str(error)
    if context:
        msg = f"{context}: {msg}"
    return msg

def log_error_with_context(error: Exception, context: str) -> None:
    """Log error with context information.
    
    Args:
        error: Exception to log
        context: Context where error occurred
    """
    logging.error(f"Error in {context}: {error}", exc_info=True)

def check_dependencies(dependencies: List[str]) -> None:
    """Check if required dependencies are available.
    
    Args:
        dependencies: List of dependency names to check
        
    Raises:
        SystemExit: If any dependency is missing
    """
    missing = []
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    
    if missing:
        die(
            f"Missing required dependencies: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}",
            ExitCodes.RUNTIME_ERROR
        )

def with_retry(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry operations with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (2 ** attempt)
                        warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        break
            
            # All retries failed
            die(f"Operation failed after {max_retries} retries: {last_exception}", ExitCodes.RUNTIME_ERROR)
        
        return wrapper
    return decorator

def handle_graceful_degradation(func):
    """Decorator to handle graceful degradation when optional features fail.
    
    This decorator catches exceptions from optional features and logs warnings
    instead of failing the entire operation.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.warning(f"Optional feature failed: {e}")
            return None
    
    return wrapper

def safe_operation(func):
    """Decorator to safely execute operations with proper error handling.
    
    This decorator provides comprehensive error handling for operations that
    might fail due to various reasons (network, file system, etc.).
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Operation failed: {e}", exc_info=True)
            raise
    
    return wrapper