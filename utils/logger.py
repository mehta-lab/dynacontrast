import logging
import os

def make_logger(log_dir, logger_name='dynacontrast.log', log_level=20):
    """
    Creates a logger which writes to a file, not to console.

    :param str log_dir: Path to directory where log file will be written
    :param str logger_name: name of the logger instance
    :param int log_level: DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50
    :return logging instance logger
    """
    log_path = os.path.join(log_dir, logger_name)
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(log_format)
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)
    return logger