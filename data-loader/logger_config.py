import os
import logging
from pathlib import Path

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Set up a logger with consistent formatting and handlers
    
    Args:
        name: Logger name (typically __name__)
        log_file: Optional specific log file path
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Set default log file based on module name if not provided
    if log_file is None:
        module_name = name.split('.')[-1]
        log_file = f"logs/{module_name}.log"
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Get logger and set level
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Only add handlers if they don't exist to prevent duplicate logging
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger 