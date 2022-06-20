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
    logging.basicConfig(filename=log_path,
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=log_level)
    logger = logging.getLogger()
    consoleHandler = logging.StreamHandler()
    logger.addHandler(consoleHandler)
    return logger