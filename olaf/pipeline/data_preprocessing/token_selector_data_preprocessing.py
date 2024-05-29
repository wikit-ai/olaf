from typing import Any, Callable, List, Optional

import spacy

from ..pipeline_schema import Pipeline
from ...commons.errors import NotCallableError
from ...commons.logging_config import logger
from .data_preprocessing_schema import DataPreprocessing


class TokenSelectorDataPreprocessing(DataPreprocessing):
    """Preprocess data with token selector method.

    Attributes
    ----------
    corpus: spacy.tokens.Doc
        spaCy corpus to process.
    token_selector: Callable[[spacy.tokens.Token], bool]
        Callable function that implements the token selection criterion.
    token_sequence_doc_attribute: str, Optional
        Name of the spaCy doc attribute containing the selected tokens, by default "selected_tokens".
    """

    def __init__(
        self,
        selector: Callable[[spacy.tokens.Token], bool],
        token_sequence_doc_attribute: Optional[str] = None,
    ) -> None:
        """Initialise token selector data preprocessing pipeline component.

        Parameters
        ----------
        selector : Callable[[spacy.tokens.Token],bool]
            Callable function that implements the token selection criterion.
        token_sequence_doc_attribute: str, optional
            Name of the spaCy doc attribute containing the selected tokens, by default to "selected_tokens".

        Raises
        ------
        NotCallableError
            Exception raised when an object is expected callable but is not.
        """
        super().__init__()
        self.corpus = None
        self._token_sequence_doc_attribute = token_sequence_doc_attribute

        if self._token_sequence_doc_attribute is None:
            self._token_sequence_doc_attribute = "selected_tokens"
            logger.warning(
                """Data preprocessing token sequence attribute not set by the user. 
                By default the token sequence attribute selected_tokens will be used."""
            )

        if not spacy.tokens.Doc.has_extension(self._token_sequence_doc_attribute):
            spacy.tokens.Doc.set_extension(
                self._token_sequence_doc_attribute, default=[]
            )

        if not (isinstance(selector, Callable)):
            raise NotCallableError(str(selector))

        self.token_selector = selector

    def _select_tokens(
        self, tokens: List[spacy.tokens.Span]
    ) -> List[spacy.tokens.Span]:
        """Select tokens passed as input based on a criterion defined by the token selector
        function.

        Parameters
        ----------
        tokens : List[spacy.tokens.Span]
            Tokens to analyse.

        Returns
        -------
        List[spacy.tokens.Span]
            Tokens selected by the selector chosen.
        """
        selected_tokens = []
        for span in tokens:
            for token in span:
                if self.token_selector(token):
                    selected_tokens.append(token.doc[token.i : token.i + 1])
        return selected_tokens

    def run(self, pipeline: Pipeline) -> None:
        """Method that is responsible for the execution of the component to preprocess all corpus
        documents based on a token selector.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline running.
        """

        self.corpus = pipeline.corpus
        for doc in self.corpus:
            selected_tokens = doc._.get(self._token_sequence_doc_attribute)

            if not (selected_tokens):
                selected_tokens = [doc[:]]
            selected_tokens = self._select_tokens(selected_tokens)

            doc._.set(self._token_sequence_doc_attribute, selected_tokens)
