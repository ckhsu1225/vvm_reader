"""
VVM Reader Logging Configuration

This module provides logging configuration for the VVM Reader package.
Users can control logging output through standard Python logging facilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for VVM Reader.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               Can be string or logging constant
        log_file: Optional path to log file
        format_string: Custom format string for log messages
        date_format: Custom date format string

    Returns:
        logging.Logger: Configured logger instance

    Examples:
        # Basic setup with INFO level
        >>> from vvm_reader import setup_logging
        >>> setup_logging()

        # Debug mode for development
        >>> setup_logging(level=logging.DEBUG)

        # Write logs to file
        >>> setup_logging(log_file='/path/to/vvm_reader.log')

        # Custom format
        >>> setup_logging(format_string='[%(levelname)s] %(message)s')

        # Disable logging
        >>> import logging
        >>> logging.getLogger('vvm_reader').disabled = True
    """
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Default format strings
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    if date_format is None:
        date_format = '%Y-%m-%%d %H:%M:%S'

    # Get or create vvm_reader logger
    logger = logging.getLogger('vvm_reader')
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt=date_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_path}")

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        logging.Logger: Logger instance

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("Loading dataset...")
    """
    return logging.getLogger(f'vvm_reader.{name}')


# Initialize default logger with NullHandler (no output by default)
_default_logger = logging.getLogger('vvm_reader')
if not _default_logger.handlers:
    _default_logger.addHandler(logging.NullHandler())
_default_logger.setLevel(logging.WARNING)  # Only warnings and errors by default


# Convenience function to quickly change log level
def set_log_level(level: Union[int, str]) -> None:
    """
    Quickly change the logging level for vvm_reader.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Examples:
        >>> from vvm_reader import set_log_level
        >>> set_log_level('DEBUG')   # Show all messages
        >>> set_log_level('ERROR')   # Only show errors
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.WARNING)

    logger = logging.getLogger('vvm_reader')
    logger.setLevel(level)

    # Update all handlers
    for handler in logger.handlers:
        handler.setLevel(level)
