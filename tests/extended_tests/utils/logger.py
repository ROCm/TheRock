"""Logging configuration using Python's built-in logging module."""

import logging
import sys
from pathlib import Path
from typing import Optional

# Configure logging format
LOG_FORMAT = "[%(asctime)s][%(levelname)-8s]: %(message)s"
TIMESTAMP_FORMAT = "%y-%m-%d %H:%M:%S"

# Create logger
log = logging.getLogger("benchmark")
log.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, TIMESTAMP_FORMAT))
log.addHandler(console_handler)

# Prevent propagation to root logger
log.propagate = False


def set_log_level(level_str: str):
    """Set logging level from string.

    Args:
        level_str: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    level = level_map.get(level_str.upper(), logging.INFO)
    log.setLevel(level)
    console_handler.setLevel(level)


def setup_file_logging(
    log_file: str, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5
):
    """Setup rotating file handler (optional).

    Args:
        log_file: Path to log file
        max_bytes: Max file size before rotation (default: 10MB)
        backup_count: Number of backup files (default: 5)
    """
    from logging.handlers import RotatingFileHandler

    try:
        # Create log directory
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        # Add file handler with rotation
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, TIMESTAMP_FORMAT))
        file_handler.setLevel(log.level)
        log.addHandler(file_handler)
        log.info(f"File logging enabled: {log_file}")
    except Exception as e:
        log.warning(f"Failed to setup file logging: {e}")
