"""Logging module with console output, file logging, and automatic rotation."""

import sys
import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


class Logger:
    """Logger with console and file output, supporting automatic rotation.
    """
    
    def __init__(self, level=logging.INFO, log_file: Optional[str] = None,
                 max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5):
        """Initialize logger.
        
        Args:
            level: Logging level (default: INFO)
            log_file: Path to log file (optional, enables file logging)
            max_bytes: Maximum log file size before rotation (default: 10MB)
            backup_count: Number of backup files to keep (default: 5)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear any existing handlers
        
        # Log format
        self.log_format = '[%(asctime)s][%(levelname)-8s]: %(message)s'
        self.timestamp_format = "%y-%m-%d %H:%M:%S"
        self.formatter = logging.Formatter(fmt=self.log_format, datefmt=self.timestamp_format)
        
        # Console handler
        self.console_handler = logging.StreamHandler(stream=sys.stdout)
        self.console_handler.setFormatter(self.formatter)
        self.console_handler.setLevel(level)
        self.logger.addHandler(self.console_handler)
        
        # File handler (optional with rotation)
        self.file_handler = None
        if log_file:
            self._setup_file_handler(log_file, max_bytes, backup_count)
    
    def _setup_file_handler(self, log_file: str, max_bytes: int, backup_count: int):
        """Setup rotating file handler for log output.
        
        Args:
            log_file: Path to log file
            max_bytes: Max file size before rotation in bytes
            backup_count: Number of backup files to keep
        """
        try:
            # Create log directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create rotating file handler
            self.file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            self.file_handler.setFormatter(self.formatter)
            self.file_handler.setLevel(self.logger.level)
            self.logger.addHandler(self.file_handler)
            
        except Exception as e:
            # If file logging fails, just log to console
            self.logger.warning(f"Failed to setup file logging: {e}")
            self.file_handler = None
    
    def info(self, *args):
        """Log info message."""
        message = ' '.join(str(arg) for arg in args) if len(args) > 1 else str(args[0]) if args else ''
        self.logger.info(message)
    
    def warning(self, *args):
        """Log warning message."""
        message = ' '.join(str(arg) for arg in args) if len(args) > 1 else str(args[0]) if args else ''
        self.logger.warning(message)
    
    def error(self, *args):
        """Log error message."""
        message = ' '.join(str(arg) for arg in args) if len(args) > 1 else str(args[0]) if args else ''
        self.logger.error(message)
    
    def debug(self, *args):
        """Log debug message."""
        message = ' '.join(str(arg) for arg in args) if len(args) > 1 else str(args[0]) if args else ''
        self.logger.debug(message)
    
    def enable_file_logging(self, log_file: str, max_bytes: int = 10 * 1024 * 1024,
                           backup_count: int = 5):
        """Enable file logging with rotation.
        
        Args:
            log_file: Path to log file
            max_bytes: Max file size before rotation in bytes (default: 10MB)
            backup_count: Number of backup files to keep (default: 5)
        """
        if self.file_handler:
            self.logger.warning("File logging already enabled")
            return
        
        self._setup_file_handler(log_file, max_bytes, backup_count)
        if self.file_handler:
            self.logger.info(f"File logging enabled: {log_file}")
            self.logger.info(f"  Max size: {max_bytes / (1024 * 1024):.1f} MB")
            self.logger.info(f"  Backup count: {backup_count}")
    
    def disable_file_logging(self):
        """Disable file logging."""
        if self.file_handler:
            self.logger.removeHandler(self.file_handler)
            self.file_handler.close()
            self.file_handler = None
            self.logger.info("File logging disabled")
    
    def get_log_info(self) -> dict:
        """Get current logging configuration details.
        
        Returns:
            Dictionary with logging configuration info
        """
        info = {
            'level': logging.getLevelName(self.logger.level),
            'console_enabled': self.console_handler is not None,
            'file_enabled': self.file_handler is not None,
        }
        
        if self.file_handler:
            info['log_file'] = self.file_handler.baseFilename
            info['max_bytes'] = self.file_handler.maxBytes
            info['backup_count'] = self.file_handler.backupCount
            
            # Get current log file size
            try:
                info['current_size'] = os.path.getsize(self.file_handler.baseFilename)
                info['current_size_mb'] = info['current_size'] / (1024 * 1024)
            except Exception:
                info['current_size'] = 0
                info['current_size_mb'] = 0.0
        
        return info
    
    def rotate_logs(self):
        """Manually trigger log file rotation (if file logging is enabled)."""
        if self.file_handler:
            self.logger.info("Manually rotating log files...")
            self.file_handler.doRollover()
            self.logger.info("Log rotation complete")
        else:
            self.logger.warning("File logging not enabled, cannot rotate")


# Global logger instance
log = Logger()


def set_log_level(level_str: str):
    """Set global logging level from string.
    
    Args:
        level_str: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    level = level_map.get(level_str.upper(), logging.INFO)
    log.logger.setLevel(level)
    log.console_handler.setLevel(level)


# Import constants
from .constants import Constants

# Separator for output formatting (backward compatibility)
SEPARATOR_LINE = Constants.SEPARATOR_LINE
