from dataclasses import dataclass
from typing import Any, Callable, List, Union

import spacy

from ...commons.errors import NotCallableError
from ...commons.logging_config import logger
from .data_preprocessing_schema import DataPreprocessing


@dataclass
class TokenSelectorDataPreprocessingConfig:
    token_sequence_doc_attribute: str = "selected_tokens"

    def __post_init__(self):
        if self.token_sequence_doc_attribute == "selected_tokens":
            logger.warning(
                """Data preprocessing token sequence attribute not set by the user. 
                By default the token sequence attribute selected_tokens will be used.""")

        if not spacy.tokens.Doc.has_extension(self.token_sequence_doc_attribute):
            spacy.tokens.Doc.set_extension(self.token_sequence_doc_attribute, default=[])

class TokenSelectorDataPreprocessing (DataPreprocessing):
    """Preprocess data with token selector method.
    """

    def __init__(self,
                 selector: Callable[[spacy.tokens.Token],bool],
                 config: TokenSelectorDataPreprocessingConfig=TokenSelectorDataPreprocessingConfig()
                ) -> None:
        """Initialise token selector data preprocessing pipeline component.

        Parameters
        ----------
        selector : Callable[[spacy.tokens.Token],bool]
            Callable function that implements the token selection criterion.

        Raises
        ------
        NotCallableError
            Exception raised when an object is expected callable but is not.
        """
        super().__init__(config)
        self.corpus = None
        self._token_sequence_doc_attribute = config.token_sequence_doc_attribute

        if not(isinstance(selector, Callable)):
            raise NotCallableError(str(selector))

        self.token_selector = selector



    def _select_tokens(self,
                       tokens: Union[spacy.tokens.Doc,List[spacy.tokens.Token]]
                    ) -> List[spacy.tokens.Token] :
        """Select tokens passed as input based on a criterion defined by the token selector 
        function.

        Parameters
        ----------
        tokens : Union[spacy.tokens.doc.Doc,List[spacy.tokens.Token]]
            Tokens to analyse. spaCy doc if a specific attribute is not defined, list of spaCy token 
            instead.

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
        """Method that is responsible for the execution of the component to preprocess all corpus 
        documents based on a token selector.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline running.
        """

        self.corpus = pipeline.corpus
        for doc in self.corpus :
            selected_tokens = doc._.get(self._token_sequence_doc_attribute)

            if not(selected_tokens):
                selected_tokens = doc
            selected_tokens = self._select_tokens(selected_tokens)

            doc._.set(self._token_sequence_doc_attribute, selected_tokens)
