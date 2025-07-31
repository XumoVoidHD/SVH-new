import logging
import sys
import os
from datetime import datetime

class StreamToLogger:
    """
    File-like object to redirect stdout and stderr to logger.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, message):
        message = message.strip()
        if message:
            self.logger.log(self.log_level, message)

    def flush(self):
        pass  # Required for compatibility

def setup_logger(base_log_dir="logs"):
    # Get current date and time
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create directory: logs/YYYY-MM-DD/
    log_dir = os.path.join(base_log_dir, current_date)
    os.makedirs(log_dir, exist_ok=True)

    # Create log file: YYYY-MM-DD_HH-MM-SS.log
    log_path = os.path.join(log_dir, f"{current_time}.log")

    # Configure logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s')


    # File handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Redirect print() and exceptions to logger
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    return logger
