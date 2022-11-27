import collections
from commons.spacy_processing_tools import build_spans_from_tokens
import functools
import inspect
import re
import spacy.tokens
from typing import Any, Dict, List, Union

# from config.core import PROJECT_ROOT_PATH

import config.logging_config as logging_config
from data_preprocessing.data_preprocessing_schema import (
    TokenSelector,
    TokenSelectorNotFound,
    TokenSelectorParamNotFound,
    TokenSelectorParamTypingProcessNotFound,
    str2type_processes
)
import data_preprocessing.data_preprocessing_methods.token_selectors

"""
    The functions should return True if the token should be kept, False otherwise.
    Specifying the parameter types as annotations is critical since the token selector loading from config
    process relies on these typing annotations.
    If you introduce a new typing annotation make sure you provide the way to process it from a config file.
"""


def select_on_pos(token: spacy.tokens.Token, pos_to_select: List[str]) -> bool:
    """Return true if the Spacy Token POS string is in the pos_to_select list.  

    Parameters
    ----------
    token : spacy.tokens.Token
        The Spacy token to test
    pos_to_select : List[str]
        The list of strings corresponding to the POS tags to keep.

    Returns
    -------
    bool
        Wether the Token POS tag is in pos_to_select or not
    """
    if token.pos_ in pos_to_select:
        return True
    else:
        return False


def select_on_shape_match_pattern(token: spacy.tokens.Token, shape_pattern_to_select: re.Pattern) -> bool:
    """Return true if the Spacy Token Shape string matches the pattern shape_pattern_to_select. 

    Parameters
    ----------
    token : spacy.tokens.Token
        The Spacy token to test
    shape_pattern_to_select : re.Pattern
        A re.Pattern object to test the Token shape against.

    Returns
    -------
    bool
        Wether the Token Shape string matches the pattern shape_pattern_to_select or not
    """
    if (shape_pattern_to_select.match(token.shape_)):
        return True
    else:
        return False


def filter_stopwords(token: spacy.tokens.Token) -> bool:
    """Return True if the Spacy Token is NOT a stopword.

    Parameters
    ----------
    token : spacy.tokens.Token
        The Spacy token to test

    Returns
    -------
    bool
        Wether the Token Shape is NOT a stopword or it is
    """
    return not (token.is_stop)


def filter_punct(token: spacy.tokens.Token) -> bool:
    """Return True if the Spacy Token is NOT a punctuation symbol.

    Parameters
    ----------
    token : spacy.tokens.Token
        The Spacy token to test

    Returns
    -------
    bool
        Wether the Token Shape is NOT a punctuation symbol or it is
    """
    return not (token.is_punct)


def filter_num(token: spacy.tokens.Token) -> bool:
    """Return True if the Spacy Token is NOT a numerical value.

    Parameters
    ----------
    token : spacy.tokens.Token
        The Spacy token to test

    Returns
    -------
    bool
        Wether the Token Shape is NOT a numerical value or it is
    """
    return not (token.like_num)


def filter_url(token: spacy.tokens.Token) -> bool:
    """Return True if the Spacy Token is NOT a url.

    Parameters
    ----------
    token : spacy.tokens.Token
        The Spacy token to test

    Returns
    -------
    bool
        Wether the Token Shape is NOT a url or it is
    """
    return not (token.like_url)


def select_on_occurrence_count(token: Union[spacy.tokens.Token, spacy.tokens.span.Span], treshold: int, occurrence_counts: collections.Counter, on_lemma: bool = False) -> bool:
    """Return true if the Spacy Token Text has an occurrence above a defined treshold.

    Parameters
    ----------
    token : Union[spacy.tokens.Token, spacy.tokens.span.Span]
        The Spacy token or span to test
    treshold : int
        The occurrence treshold below which the Token is not selected 
    occurrence_counts : collections.Counter
        A Counter dictionnary with token texts as keys and their occurrence as value.
    on_lemma : bool
        If true the count is made on lemma attribute. By default it is made on the text attribute. 

    Returns
    -------
    bool
        Wether the token text occurrence is above the defined treshold or not.
    """
    if on_lemma:
        token_value = token.lemma_  
    else:
        token_value = token.text
    token_occurrence = occurrence_counts.get(token_value)
    selected = False
    if token_occurrence is not None:
        if token_occurrence > treshold:
            selected = True
    else : 
        logging_config.logger.warning(f"Token {token_value} was not found in the occurrence counter. Please check that it is a wanted behavior.")
    return selected


class TokenSelectionPipeline:
    """A class to define a Token Selection Pipeline. 
        It sets up  a list of token selector based on a configuration files and run them to define if
        a token should be selected or not.

        Class instance attributes
        ----------
        pipeline_config : configparser.ConfigParser()
            A python config parser object containing the configuration details for the Token Selection Pipeline setup.
        pipeline_name : str
            The Token Selection Pipeline name 
        token_selector_names : List[str]
            A list of strings refering to the token selector functions
    """

    def __init__(self, config: Dict) -> None:
        """Initialize the Token Selection Pipeline

        Parameters
        ----------
        config : configparser.ConfigParser()
             A python config parser object containing the configuration details for the Token Selection Pipeline setup.
        """
        self.pipeline_config = config
        self.pipeline_name = self.pipeline_config['pipeline_name']
        self.token_selector_names = self.pipeline_config['token_selector_names']
        self.token_selectors: List[TokenSelector] = self._load_selectors_from_config(
        )

    def __call__(self, token: spacy.tokens.Token) -> bool:
        """Sequentially run the token selectors. If one of them return False, the method return False.

        Parameters
        ----------
        token : spacy.tokens.Token
            The Spacy token to test

        Returns
        -------
        bool
            Wether the token should be selected or not
        """
        select_token = True
        for token_selector in self.token_selectors:
            if not (token_selector(token)):
                select_token = False
                break

        return select_token

    def _load_selectors_from_config(self) -> List[TokenSelector]:
        """Sets up a list of TokenSelector functions based on a configuration. 
            The TokenSelector functions take a Spacy token as input and return a Boolean
        Returns
        -------
        List[TokenSelector]
            The list of TokenSelector functions

        Raises
        ------
        TokenSelectorNotFound
            An Exception raised if the Token Selector mentionned in the configruation has not been found.
        TokenSelectorParamNotFound
            An Exception raised if a Token Selector parameter value (other than the token one) has not been 
            found in the configuration.  
        TokenSelectorParamTypingProcessNotFound
            An Exception raised if no process has been found to turn the string object into the parameter type.
        """
        # inspect the token_selectors module to get le token selectors implemented and their details
        token_selector_inspect = inspect.getmembers(
            data_preprocessing.data_preprocessing_methods.token_selectors, inspect.isfunction)
        token_selectors_dict = {name: funct for name,
                                funct in token_selector_inspect}

        token_selectors = []

        try:
            # build each token selector based on the names provided in the config
            for token_selector_name in self.token_selector_names:
                token_selector_funct = token_selectors_dict.get(
                    token_selector_name)
                if token_selector_funct is None:
                    raise TokenSelectorNotFound(
                        f"{token_selector_name} token selector not found")
                else:
                    # extract the token selector function parameters and their details
                    token_selector_params = inspect.signature(
                        token_selectors_dict[token_selector_name]).parameters
                    token_selector_params_strings = list(inspect.signature(
                        token_selectors_dict[token_selector_name]).parameters.keys())
                    # we only need to process the other extra parameters
                    token_selector_params_strings.remove('token')

                    params_dict = {}
                    # setup each extra parameters
                    for param_string in token_selector_params_strings:
                        # try to get the parameter value from the config
                        param_value_string = self.pipeline_config[token_selector_name].get(
                            param_string)
                        if param_value_string is None:
                            raise TokenSelectorParamNotFound(
                                f"Parameter {param_string} for token selector {token_selector_name} not found in pipeline config")
                        else:
                            # Turn the string extracted from the configuration into the right type
                            # The functions to process the strings should be specified in the str2type_processes
                            param_str2type_processor = str2type_processes.get(
                                token_selector_params.get(param_string).annotation)
                            if param_str2type_processor is None:
                                raise TokenSelectorParamTypingProcessNotFound(
                                    f"String to type process not found for type {token_selector_params.get(param_string).annotation}")
                            else:
                                params_dict[param_string] = param_str2type_processor(
                                    param_value_string)

                    # use partial functions so that the token selecter mathes the TokenSelector type
                    token_selector = functools.partial(
                        token_selector_funct, **params_dict)
                    token_selectors.append(token_selector)
        except Exception as e:
            logging_config.logger.error(
                f"Trace : {e}")
            token_selectors = []
        else:
            logging_config.logger.info(
                f"Token selectors loaded for pipeline {self.pipeline_name}")

        return token_selectors


@spacy.language.Language.factory(name="token_selector")
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
        make_spans : bool
            Wether or not to turn selected tokens into a list of spans
    """

    def __init__(self, nlp: spacy.language.Language, name: str, token_selector_config: Dict[str, Any]) -> None:
        """Initialize a TokenSelectorComponent instance.
            Make sure that the Doc object has the custom attribute to use for storing selected tokens.

        Parameters
        ----------
        nlp : spacy.language.Language
            The Spacy Language attached to the pipeline. The naming is enforced by 
            Spacy framework (https://spacy.io/usage/processing-pipelines#custom-components)
        name : str
            The component name. The attribute naming is enforced by 
            Spacy framework (https://spacy.io/usage/processing-pipelines#custom-components)
        token_selector_config: Dict[str, Any]
             A Dictionary containing the Token Selection Pipeline configuration details.
        doc_attribute_name : str
            The name to use for the custom attribute on the Doc object containing a list of Tokens
            selected
        make_spans : bool
            Wether or not to turn selected tokens into a list of spans
        """
        self.token_selector_config = token_selector_config
        self.make_spans = self.token_selector_config['make_spans']
        self.doc_attribute_name = self.token_selector_config['doc_attribute_name']
        
        try:
            self.token_selector_pipeline = TokenSelectionPipeline(
                self.token_selector_config)
        except Exception as e:
            logging_config.logger.error(
                f"There has been an issue while building TokenSelectionPipeline from token_selection_config {self.token_selector_config}. Trace : {e}")
        else:
            logging_config.logger.info(
                f"TokenSelectionPipeline linked to custom Doc Spacy attribute {self.doc_attribute_name} instance created ")

        if not spacy.tokens.doc.Doc.has_extension(self.doc_attribute_name):
            spacy.tokens.doc.Doc.set_extension(self.doc_attribute_name, default=[])

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

        if self.make_spans:
            selected_tokens_spans = build_spans_from_tokens(
                token_list=selected_tokens, doc=doc)
            doc._.set(self.doc_attribute_name, selected_tokens_spans)
        else:
            doc._.set(self.doc_attribute_name, selected_tokens)

        return doc
