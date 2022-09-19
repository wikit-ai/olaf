import configparser
import spacy.language
import spacy.tokenizer
import re

import config.logging_config as logging_config
from data_preprocessing.data_preprocessing_schema import TokenSelectionPipeline


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

    try:

        tokenizer = spacy.tokenizer.Tokenizer(spacy_model.vocab, rules=special_cases,
                                              prefix_search=prefix_re.search,
                                              suffix_search=suffix_re.search,
                                              infix_finditer=infix_re.finditer,
                                              url_match=simple_url_re.match)
    except Exception as tokenizer_exception:
        logging_config.logger.error(
            "Could not create Spacy tokenizer. Trace : %s", tokenizer_exception)
    else:
        logging_config.logger.info(
            "Spacy tokenizer 'no_split_on_dash_in_words_tokenizer' created")

    return tokenizer


@spacy.language.Language.factory("token_selector")
class TokenSelectorComponent:
    """A Spacy pipeline component setting a custom attribute on the Doc object containing a list of Tokens
        selected using a Token Selection Pipeline object. 

        Class instance attributes
        ----------
        doc_attribute_name : str
            The name to use for the custom attribute on the Doc object containing a list of Tokens
            selected
        token_selection_config : configparser.ConfigParser()
            A python config parser object containing the configuration details for the Token Selection Pipeline 
        token_selector_pipeline : TokenSelectionPipeline
            The Token Selection Pipeline used to select tokens.
    """

    def __init__(self, nlp: spacy.language.Language, name: str, token_selection_config_path: str, doc_attribute_name: str) -> None:
        """Initialize a TokenSelectorComponent instance.
            Make sure that the Doc object has the custom attribute to use for storing selected tokens.

        Parameters
        ----------
        nlp : spacy.language.Language
            The Spacy Language attached to the pipeline. The naming is enforced by 
            Spacy framework (https://spacy.io/usage/processing-pipelines#custom-components)
        name : str
            The component name. The naming is enforced by 
            Spacy framework (https://spacy.io/usage/processing-pipelines#custom-components)
        token_selection_config_path : str
            The path to the Token Selection Pipeline configuration file. 
        doc_attribute_name : str
            The name to use for the custom attribute on the Doc object containing a list of Tokens
            selected
        """
        self.doc_attribute_name = doc_attribute_name
        config = configparser.ConfigParser()
        config.read(token_selection_config_path)
        self.token_selection_config = config
        print(self.token_selection_config)
        self.token_selector_pipeline = TokenSelectionPipeline(
            self.token_selection_config)

        if not spacy.tokens.doc.Doc.has_extension(doc_attribute_name):
            spacy.tokens.doc.Doc.set_extension(doc_attribute_name, default=[])

    def __call__(self, doc: spacy.tokens.doc.Doc) -> spacy.tokens.doc.Doc:
        """Process a Spacy Doc object and set the custom attribute with the selected tokens.
            The method signature is enforced by Spacy framework (https://spacy.io/usage/processing-pipelines#custom-components)

        Parameters
        ----------
        doc : spacy.tokens.doc.Doc
            The Spacy Doc to process

        Returns
        -------
        spacy.tokens.doc.Doc
            The Spacy Doc with the custom attribute updated.
        """

        selected_tokens = doc._.get(self.doc_attribute_name)

        for token in doc:
            select_token = self.token_selector_pipeline(token)

            if select_token:
                selected_tokens.append(token)

        doc._.set(self.doc_attribute_name, selected_tokens)

        return doc
