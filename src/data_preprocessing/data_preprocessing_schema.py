
import configparser
import functools
import inspect
import json
import re
from typing import Callable, Dict, List

import spacy.tokens.doc

import config.logging_config as logging_config
import data_preprocessing.data_preprocessing_methods.token_selectors

"""A Type to define token filters.
    A TokenFilter is a function taking an iterable on spacy documents as input.
    A Token filter sets the selected attribute of the spacy Doc instance to False based on some conditions.
"""
TokenSelector = Callable[[spacy.tokens.Token], bool]

str2type_processes = {
    re.Pattern: lambda pattern_str: re.compile(pattern_str),
    List[str]: lambda l_str: l_str.strip().split(),
    Dict[str, int]: lambda json_str: json.loads(json_str),
    int: lambda int_str: int(int_str)
}


class FileTypeDetailsNotFound(Exception):
    """An Exception to flag when the details specific to a corpus file type is not found.
    """
    pass


class TokenSelectorNotFound(Exception):
    """An Exception to flag when the Token selector has not been found.
    """
    pass


class TokenSelectorParamNotFound(Exception):
    """An Exception to flag when the Token selector parameters details have not been found.
    """
    pass


class TokenSelectorParamTypingProcessNotFound(Exception):
    """An Exception to flag when the Token selector parameter type has no string process defined.
        Or the process has not been found.
    """
    pass


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
            a list of strings refering to the token selector functions
    """

    def __init__(self, config: configparser.ConfigParser) -> None:
        """Initialize the Token Selection Pipeline

        Parameters
        ----------
        config : configparser.ConfigParser()
             A python config parser object containing the configuration details for the Token Selection Pipeline setup.
        """
        self.pipeline_config = config['TOKEN_SELECTION_PIPELINE_CONFIG']
        self.pipeline_name: str = self.pipeline_config['PIPELINE_NAME']
        self.token_selector_names = self.pipeline_config['TOKEN_SELECTOR_NAMES'].strip(
        ).split()
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
                    # we only need to process the other extra paraleters
                    token_selector_params_strings.remove('token')

                    params_dict = {}
                    # setup each extra parameters
                    for param_string in token_selector_params_strings:
                        # try to get the parameter value from the config
                        param_value_string = self.pipeline_config.get(
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
