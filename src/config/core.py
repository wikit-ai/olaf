import configparser
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__))
PROJECT_ROOT_PATH = os.path.join(os.path.dirname(__file__), '..', '..')
DATA_PATH = os.path.join(PROJECT_ROOT_PATH, 'data')
SPACY_PIPELINE_PATH = os.path.join(DATA_PATH, "spacy_pipelines")


configurations_parser = configparser.ConfigParser()
configurations_parser.read(os.path.join(
    CONFIG_PATH + '/core.ini'))

CORPUS_PATH = configurations_parser["CORPUS_DETAILS"]["CORPUS_PATH"]

SPACY_MODEL = configurations_parser["SPACY_MODEL"]["SPACY_MODEL_NAME"]

OCCURRENCE_THRESHOLD = float(
    configurations_parser["LEARN2CONSTRUCT"]["OCCURRENCE_THRESHOLD"])

TOPIC_NUMBER = int(configurations_parser["LEARN2CONSTRUCT"]["TOPIC_NUMBER"])
