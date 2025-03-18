import os
import logging
from pathlib import Path

def setup_logger(name, log_level=None):
    """
    Set up a logger with consistent configuration and optional level override
    
    Args:
        name: Name of the logger
        log_level: Optional log level override
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Use environment variable or default to INFO level
    if log_level is None:
        log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    
    # Set level based on string or object
    if isinstance(log_level, str):
        logger.setLevel(getattr(logging, log_level))
    else:
        logger.setLevel(log_level)
    
    # Add context information to logs
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # Ensure we don't add duplicate handlers
    if not logger.handlers:
        # Add file handler
        os.makedirs('logs', exist_ok=True)
        file_handler = logging.FileHandler(f'logs/{name.replace(".", "_")}.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger 