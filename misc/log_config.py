import logging

logging_configured = False  # Flag to check if logging is configured


def configure_logging():
    """
    This function sets up logging to write messages to both a file and the terminal.
    """
    global logging_configured  # Access the global flag

    # Check if logging is not already configured
    if not logging_configured:
        logger = logging.getLogger()  # Get the root logger
        logger.setLevel(logging.INFO)  # Set the default log level to INFO

        # Create a file handler for writing logs to a file
        file_handler = logging.FileHandler("PVGIS_POA_TMY.log", mode="a")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        # Create a stream handler for writing logs to the terminal
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        # Add both handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        # Update the flag to indicate that logging is configured
        logging_configured = True
