"""Interface between data, models and the data preprocessing service."""

import json
from typing import List
import spacy
import os.path
import os
import pathlib
from config import core
import config.logging_config as logging_config
import spacy.language
from data_preprocessing.data_preprocessing_schema import FileTypeDetailsNotFound


def load_json_file(file_path: str, text_field: str) -> List[str]:
    """Load data from a json file. The json file is expected to be a list of json objects
        with at least the `text_field` variable string as a field containing the text. 

    Parameters
    ----------
    file_path : str
        The path to the json file.
    text_field : str
        The name of the json field containing the text to extract.

    Returns
    -------
    List[str]
        The list of text strings.
    """
    with open(file_path, "r", encoding='utf-8') as file:
        file_content = json.load(file)

    texts = [content[text_field] for content in file_content]

    return texts


def load_csv_file(file_path: str, separator: str) -> List[str]:
    """Load data from a csv file. The csv file is expected to be a list of text strings
        separated by a separator character. 

    Parameters
    ----------
    file_path : str
        The path to the csv file.
    separator : str
        The character separating the text strings.

    Returns
    -------
    List[str]
        The list of text strings.
    """
    if separator == "\\n":
        texts = load_text_corpus(file_path)
    else:
        with open(file_path, "r", encoding='utf-8') as file:
            file_content = file.read()
        if separator == "\\t":
            texts = file_content.split("\t")

        else:
            texts = file_content.split(separator)

    return texts


def load_text_file(file_path: str) -> str:
    """Load text for a text file. The text file is expected to contain only one text document.

    Parameters
    ----------
    file_path : str
        The path to the text file.

    Returns
    -------
    str
        The text string
    """
    with open(file_path, "r", encoding='utf-8') as file:
        text = file.read()

    return text


def load_text_corpus(file_path: str) -> List[str]:
    """Load a corpus from on text file. The text file is expected to contain one document text per line.

    Parameters
    ----------
    file_path : str
        The path to the text file.

    Returns
    -------
    List[str]
        The list of text strings.
    """
    with open(file_path, "r", encoding='utf-8') as file:
        texts = file.readlines()

    return texts


def load_corpus() -> List[str]:
    """Loading corpus from directory set in configuration file.
    Selecting field _content_ in documents.

    Returns
    -------
    List(str)
        Corpus as list of documents content.
    """
    corpus = []
    texts = []

    if os.path.isdir(core.CORPUS_PATH):

        for filename in os.listdir(core.CORPUS_PATH):
            file_path = os.path.join(core.CORPUS_PATH, filename)

            if os.path.isfile(file_path):
                try:

                    if filename.split('.')[-1] == "json":
                        if "json_field" in core.configurations_parser["CORPUS_DETAILS"]:
                            json_field = core.configurations_parser["CORPUS_DETAILS"]["json_field"]
                            texts = load_json_file(file_path=file_path,
                                                   text_field=json_field)
                        else:
                            raise FileTypeDetailsNotFound(
                                f"Error while loading file {filename}, JSON field not found in config while loading corpus from folder {core.CORPUS_PATH}")

                    if filename.split('.')[-1] == "csv":
                        if "csv_separator" in core.configurations_parser["CORPUS_DETAILS"]:
                            csv_separator = core.configurations_parser["CORPUS_DETAILS"]["csv_separator"]
                            texts = load_csv_file(file_path=file_path,
                                                  separator=csv_separator)
                        else:
                            raise FileTypeDetailsNotFound(
                                f"Error while loading file {filename}, CSV separator not found in config while loading corpus from folder {core.CORPUS_PATH}")

                    if filename.split('.')[-1] == "txt":
                        texts = load_text_file(file_path=file_path)

                except Exception as e:
                    logging_config.logger.error(
                        f"Could not load file. Trace : {e}")
                    texts = []
                else:
                    logging_config.logger.info(f"File {filename} loaded.")

            else:
                logging_config.logger.error(
                    "File path {file_path} is invalid.")

    elif os.path.isfile(core.CORPUS_PATH):
        filename = pathlib.Path(core.CORPUS_PATH).parts[-1]
        try:

            if filename.split('.')[-1] == "json":
                if "json_field" in core.configurations_parser["CORPUS_DETAILS"]:
                    json_field = core.configurations_parser["CORPUS_DETAILS"]["json_field"]
                    texts = load_json_file(file_path=core.CORPUS_PATH,
                                           text_field=json_field)
                else:
                    raise FileTypeDetailsNotFound(
                        f"JSON field not found in config while loading corpus from file {core.CORPUS_PATH}")

            if filename.split('.')[-1] == "csv":
                if "csv_separator" in core.configurations_parser["CORPUS_DETAILS"]:
                    csv_separator = core.configurations_parser["CORPUS_DETAILS"]["csv_separator"]
                    texts = load_csv_file(file_path=core.CORPUS_PATH,
                                          separator=csv_separator)
                else:
                    raise FileTypeDetailsNotFound(
                        f"CSV separator not found in config while loading corpus from file {core.CORPUS_PATH}")

            if filename.split('.')[-1] == "txt":
                texts = load_text_corpus(core.CORPUS_PATH)

        except Exception as e:
            logging_config.logger.error(
                f"Could not load file. Trace : {e}")
            texts = []
        else:
            logging_config.logger.info(f"File {filename} loaded.")

    else:
        logging_config.logger.error(
            f"Corpus folder or filename {core.CORPUS_PATH} is invalid.")

    corpus.extend(texts)

    return corpus


def load_spacy_model() -> spacy.language.Language:
    """Loading of spacy language model based on the configuration set.

    Returns
    -------
    spacy.language.Language
        Language model loaded.
    """
    try:
        if os.path.isdir(os.path.join(core.SPACY_PIPELINE_PATH, core.SPACY_MODEL)):
            spacy_model = spacy.load(os.path.join(
                core.SPACY_PIPELINE_PATH, core.SPACY_MODEL))
        else:
            spacy_model = spacy.load(core.SPACY_MODEL)
    except Exception as _e:
        spacy_model = None
        logging_config.logger.error(
            "Could not load spacy model. Trace : %s", _e)
    else:
        logging_config.logger.info("Spacy model has been loaded.")

    return spacy_model
