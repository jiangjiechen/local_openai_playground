import logging
import os


def init_logger(log_file=None, log_file_level=logging.INFO, from_scratch=False):
    from coloredlogs import ColoredFormatter

    fmt = "[%(asctime)s %(levelname)s] %(message)s"
    log_format = ColoredFormatter(fmt=fmt)
    # log_format = logging.Formatter()
    logger = logging.getLogger()
    logger.setLevel(log_file_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        if from_scratch and os.path.exists(log_file):
            logger.warning('Removing previous log file: %s' % log_file)
            os.remove(log_file)
        path = os.path.dirname(log_file)
        os.makedirs(path, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger


GPT3_NAME_AND_COST = {
    'text-davinci-003': 0.02,
    'text-davinci-002': 0.02,
    # 'code-davinci-002': 0,
    'text-davinci-001': 0.02,
    'davinci': 0.02,
    'text-curie-001': 0.002,
    'curie': 0.002,
    'babbage': 0.0005,
    'ada': 0.0004,
}