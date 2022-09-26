import configparser
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__))
PROJECT_ROOT_PATH = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_PATH = os.path.join(PROJECT_ROOT_PATH, 'data')
SPACY_PIPELINE_PATH = os.path.join(DATA_PATH, "spacy_pipelines")


config = configparser.ConfigParser()
config.read(os.path.join(
    CONFIG_PATH + '/core_configs.ini'))

SPACY_MODEL = config["SPACY_MODEL"]["SPACY_MODEL_NAME"]
CORPUS_PATH = config["DATA"]["CORPUS_PATH"]

OCCURRENCE_THRESHOLD = float(config["LEARN2CONSTRUCT"]["OCCURRENCE_THRESHOLD"])

TOPIC_NUMBER = int(config["LEARN2CONSTRUCT"]["TOPIC_NUMBER"])
