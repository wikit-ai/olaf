from typing import Any, Callable, Dict, List, Union

import spacy

from ...commons.errors import NotCallableError
from ...commons.logging_config import logger
from .data_preprocessing_schema import DataPreprocessing


class TokenSelectorDataPreprocessing (DataPreprocessing):
    """Preprocess data with token selector method.
    """

    def __init__(self, selector: Callable[[spacy.tokens.Token],bool], parameters: Dict[str, Any] | None = None, options: Dict[str, Any] | None = None) -> None:
        """Initialise token selector data preprocessing pipeline component.

        Parameters
        ----------
        selector : Callable[[spacy.tokens.Token],bool]
            Callable function that implements the token selection criterion.
        parameters : Dict[str, Any] | None, optional
            Parameters are fixed values to be defined when building the pipeline. 
            They are necessary for the component functioning. By default dict().
        options : Dict[str, Any] | None, optional
            Options are tunable parameters which will be updated to optimise the component performance.
            By default dict().

        Raises
        ------
        NotCallableError
            Exception raised when an object is expected callable but is not.
        """
        super().__init__(parameters, options)
        self.corpus = None
        self._token_sequences_doc_attribute = None

        if not(isinstance(selector, Callable)):
            raise NotCallableError(str(selector))
        
        self.token_selector = selector

        self._check_parameters()

    def _check_parameters(self) -> None:
        """Check wether required parameters are given and correct. If this is not the case, suitable default ones are set.
            The parameter to check is `token_sequence_doc_attribute`.
            If not correctly provided, the full doc will be used for the analysis and it will default to `selected_tokens`.
        """
        user_defined_attribute_name = self.parameters.get("token_sequence_doc_attribute")

        if user_defined_attribute_name:
            self._token_sequences_doc_attribute = user_defined_attribute_name
        else:
            self._token_sequences_doc_attribute = "selected_tokens"
            logger.warning(
                """Data preprocessing token sequence attribute not set by the user. 
                By default the token sequence attribute selected_tokens will be used.""")

        if not spacy.tokens.doc.Doc.has_extension(self._token_sequences_doc_attribute):
            spacy.tokens.doc.Doc.set_extension(self._token_sequences_doc_attribute, default=[])


    def _select_tokens(self, tokens: Union[spacy.tokens.doc.Doc,List[spacy.tokens.Token]]) -> List[spacy.tokens.Token] :
        """Select tokens passed as input based on a criterion defined by the token selector function.

        Parameters
        ----------
        tokens : Union[spacy.tokens.doc.Doc,List[spacy.tokens.Token]]
            Tokens to analyse. spaCy doc if a specific attribute is not defined, list of spaCy token instead.

        Returns
        -------
        List[spacy.tokens.Token]
            Token selected by the selector chosen.
        """
        selected_tokens = []
        for token in tokens :
            if self.token_selector(token) :
                selected_tokens.append(token)
        return selected_tokens


    def run(self, pipeline: Any) -> None :
        """Method that is responsible for the execution of the component to preprocess all corpus documents based on a token selector.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline running.
        """

        self.corpus = pipeline.corpus
        for doc in self.corpus :
            selected_tokens = doc._.get(self._token_sequences_doc_attribute)

            if not(selected_tokens) :
                selected_tokens = doc
            selected_tokens = self._select_tokens(selected_tokens)

            doc._.set(self._token_sequences_doc_attribute, selected_tokens)