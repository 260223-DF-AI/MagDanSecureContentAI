# logging_config.py
import logging
import os

def setup_logging():
    """
    Configure the main ResearchFlow logger and return it.
    """
    logger = logging.getLogger("researchflow")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if setup_logging() is called multiple times
    if logger.handlers:
        return logger

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # File handler
    file_handler = logging.FileHandler("logs/researchflow.log")
    file_handler.setLevel(logging.INFO)

    # Log format
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
