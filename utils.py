import logging
from datetime import datetime
import os

def ensure_logs_directory():
    """Create logs directory if it doesn't exist."""
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    return logs_dir

def setup_logger():
    """Configure and return the logger instance."""
    logger = logging.getLogger('TexasHoldem')
    logger.setLevel(logging.DEBUG)  # Log level, adjust as needed (DEBUG for development, INFO for production)
    
    # Ensure logs directory exists and get path
    logs_dir = ensure_logs_directory()
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings and above to console to reduce clutter
    
    # Create a file handler to store logs in a file
    log_filename = f"game_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(os.path.join(logs_dir, log_filename))
    file_handler.setLevel(logging.DEBUG)  # Log all levels to file
    
    # Create log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Log the start of a new game session
    logger.info("========== NEW POKER GAME SESSION STARTED ==========")
    
    return logger

# Initialize the logger
logger = setup_logger()

def log_sensitive(message: str, data: dict = None):
    """
    Log sensitive game information for debugging purposes.
    This information is only written to the log file, not displayed in the console.
    
    Args:
        message (str): The main log message
        data (dict, optional): Additional structured data to include in the log
    """
    if data is None:
        data = {}
        
    # Format the data as a string if present
    data_str = " | " + ", ".join([f"{k}={v}" for k, v in data.items()]) if data else ""
    
    # Create the full log message
    full_message = f"SENSITIVE: {message}{data_str}"
    
    # Log at DEBUG level (only goes to file, not console)
    logger.debug(full_message)

# Legacy function - kept for compatibility
def print_bold(text: str):
    print(f"\033[1;32m{text}\033[0m")  # Bold and green text output
    logger.info(text)
