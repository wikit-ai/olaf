"""Interface between data, models and the data preprocessing service."""

import json
from typing import List
import spacy
from os import listdir,path
from config import core
import logging_config
import re
import spacy.language


def no_split_on_dash_in_words_tokenizer(spacy_model: spacy.language.Language) -> spacy.tokenizer.Tokenizer:
    """Build a tokenizer based on Spacy Language model specifically to be able to extract ngrams for the C-value computation.
        In particular, the tokenizer does not split on dashes ('-') in words.

    Parameters
    ----------
    spacy_model : spacy.language.Language
        The Spacy Language model the tokenizer will be set on.

    Returns
    -------
    spacy.tokenizer.Tokenizer
        The tokenizer
    """

    special_cases = {}
    prefix_re = re.compile(r'''^[\[\("'!#$%&\\\*+,\-./:;<=>?@\^_`\{|~]''')
    suffix_re = re.compile(r'''[\]\)"'!#$%&\\\*+,\-./:;<=>?@\^_`\}|~]$''')
    infix_re = re.compile(r'''(?<=[0-9])[+\-\*^](?=[0-9-])''')
    simple_url_re = re.compile(r'''^https?://''')

    tokenizer = spacy.tokenizer.Tokenizer(spacy_model.vocab, rules=special_cases,
                                          prefix_search=prefix_re.search,
                                          suffix_search=suffix_re.search,
                                          infix_finditer=infix_re.finditer,
                                          url_match=simple_url_re.match)

    return tokenizer


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
