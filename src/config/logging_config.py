""" Contains a dict with all the logger config. 
There is two handlers : One for the console display, one for a file display
You can change the level of logs by changing the "level" field in any 
handler you want. You can choose between : DEBUG, INFO, WARNING, ERROR, CRITICAL
If you want to change the format of the log, 
refer to : https://docs.python.org/3/library/logging.html#logrecord-attributes, 
and change the formatters format field.
"""
import logging.config
import config.core
import os.path

log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "root": {
        "handlers": ["console", "file"],
        "level": "DEBUG"
    },
    "handlers": {
        "console": {
            "formatter": "std_out",
            "class": "logging.StreamHandler",
            "level": "INFO"
        },
        "file": {
            "formatter": "std_out",
            "class": "logging.FileHandler",
            "level": "INFO",
            "filename": os.path.join(config.core.DATA_PATH, "ontology_learning_logs.log")
        }
    },
    "formatters": {
        "std_out": {
            "format": "%(levelname)s: %(module)s : %(funcName)s : %(message)s : %(asctime)s",
        }
    },
}

logging.config.dictConfig(log_config)
logger = logging.getLogger(__name__)
