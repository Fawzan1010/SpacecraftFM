"""Logging configuration for SpacecraftFM."""

import logging
import sys
from pathlib import Path
from loguru import logger


def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
        )
    
    # Configure standard logging to use loguru
    logging.basicConfig(handlers=[logging.NullHandler()], level=log_level)
    logging.getLogger().handlers = [logging.NullHandler()]


def get_logger(name: str) -> object:
    """Get logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logger.bind(name=name)
