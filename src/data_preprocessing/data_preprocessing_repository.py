"""Interface between data, models and the data preprocessing service."""

import json
from typing import List
import spacy
from os import listdir,path
from config import core
import logging_config

def load_corpus() -> List[str]:
    """Loading corpus from directory set in configuration file.
    Selecting field _content_ in documents.

    Returns
    -------
    List(str)
        Corpus as list of documents content.
    """
    corpus = []
    
    if path.isdir(core.CORPUS_PATH):

        for filename in listdir(core.CORPUS_PATH):
            filepath = path.join(core.CORPUS_PATH, filename)
            if path.isfile(filepath):
                try : 
                    with open(filepath, encoding='utf-8') as file:
                        file_content = json.load(file)
                    corpus.append(file_content['content'])
                except Exception as _e:
                    logging_config.logger.error("Could not load file. Trace : %s", _e)
                else : 
                    logging_config.logger.info("File %s loaded.", filename)
            else : 
                logging_config.logger.error("File path %s is invalid.", filepath)

    else : 
        logging_config.logger.error("Corpus folder %s is invalid.", core.CORPUS_PATH)

    return corpus


def load_spacy_model() -> spacy.language.Language:
    """Loading of spacy language model based on the configuration set.

    Returns
    -------
    spacy.language.Language
        Language model loaded.
    """
    try:
        spacy_model = spacy.load(core.SPACY_MODEL)
    except Exception as _e:
        spacy_model = None
        logging_config.logger.error("Could not load spacy model. Trace : %s", _e)
    else:
        logging_config.logger.info("Spacy model has been loaded.")

    return spacy_model