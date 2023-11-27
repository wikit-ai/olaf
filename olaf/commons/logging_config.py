import logging

logger = logging.getLogger("olaf")
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setFormatter(
    logging.Formatter("[%(asctime)s] [%(levelname)s] [%(module)s] [%(funcName)s] [%(message)s]")
)
logger.addHandler(logger_stream_handler)
