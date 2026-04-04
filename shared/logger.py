"""
Centralized logging for the Vexoo Labs AI System.
Provides consistent log formatting across all subsystems.
"""

import logging
import sys
from datetime import datetime


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a configured logger with console output and consistent formatting.
    
    Args:
        name: Logger name (typically module __name__)
        level: Logging level (default: INFO)
    
    Returns:
        Configured logging.Logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Console handler with readable format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def get_file_logger(name: str, log_path: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger that writes to both console and a file.
    
    Args:
        name: Logger name
        log_path: Path to the log file
        level: Logging level
    
    Returns:
        Configured logging.Logger instance
    """
    logger = get_logger(name, level)
    
    # Add file handler
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(level)
    
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
