"""Logging configuration utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure logging for the application.

    Parameters
    ----------
    level : str, default='INFO'
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : Path, optional
        Path to log file. If None, logs only to console.
    format_string : str, optional
        Custom format string. If None, uses default format.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format=format_string,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={level}, log_file={log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Parameters
    ----------
    name : str
        Logger name (typically __name__)

    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(name)
