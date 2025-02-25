import logging

# Configure the logger
def setup_logger():
    logger = logging.getLogger('TexasHoldem')
    logger.setLevel(logging.DEBUG)  # Log level, adjust as needed (DEBUG for development, INFO for production)
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Output to console

    # Create a file handler to store logs in a file
    file_handler = logging.FileHandler('game_log.log')
    file_handler.setLevel(logging.INFO)  # Log info level and above to file
    
    # Create log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Initialize the logger
logger = setup_logger()


def print_bold(text: str):
    # print(f"\033[1m{text}\033[0m")  # Bold text output
    print(f"\033[1;32m{text}\033[0m")  # Bold and green text output
