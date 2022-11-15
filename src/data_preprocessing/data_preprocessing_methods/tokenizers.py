import configparser
from typing import Callable
import spacy.language
import spacy.tokenizer
import re
import os.path

import config.logging_config as logging_config


@spacy.registry.tokenizers("no_split_on_dash_in_words_tokenizer")
def create_no_split_on_dash_in_words_tokenizer() -> Callable[[spacy.language.Language], spacy.tokenizer.Tokenizer]:
    def create_tokenizer(spacy_model: spacy.language.Language) -> spacy.tokenizer.Tokenizer:
        """Build a tokenizer based on Spacy Language model.
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

        try:

            tokenizer = spacy.tokenizer.Tokenizer(spacy_model.vocab, rules=special_cases,
                                                  prefix_search=prefix_re.search,
                                                  suffix_search=suffix_re.search,
                                                  infix_finditer=infix_re.finditer,
                                                  url_match=simple_url_re.match)
        except Exception as tokenizer_exception:
            logging_config.logger.error(
                f"Could not create Spacy tokenizer. Trace : {tokenizer_exception}")
        else:
            logging_config.logger.info(
                "Spacy tokenizer 'no_split_on_dash_in_words_tokenizer' created")

        return tokenizer

    return create_tokenizer
