import configparser
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__))
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

config = configparser.ConfigParser()
config.read(os.path.join(
    CONFIG_PATH + '/core_configs.ini'))

SPACY_MODEL = config["SPACY_MODEL"]["SPACY_MODEL_NAME"]
CORPUS_PATH = config["DATA"]["CORPUS_PATH"]

OCCURRENCE_THRESHOLD = float(config["LEARN2CONSTRUCT"]["OCCURRENCE_THRESHOLD"])

TOPIC_NUMBER = int(config["LEARN2CONSTRUCT"]["TOPIC_NUMBER"])
