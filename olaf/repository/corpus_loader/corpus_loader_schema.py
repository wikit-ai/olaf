from abc import ABC, abstractmethod

import spacy

from ...commons.errors import EmptyCorpusError
from ...commons.logging_config import logger


class CorpusLoader(ABC):
    """Component to load a text corpus and encode it into spacy representation.

    Parameters
    ----------
    corpus_path : str
        Path of the text corpus to use.
    """

    def __init__(self, corpus_path: str) -> None:
        """Initialise CorpusLoader instance.

        Parameters
        ----------
        corpus_path : str
            Path of the text corpus to use.
        """
        self.corpus_path = corpus_path

    def __call__(self, spacy_model: spacy.language.Language) -> list[spacy.tokens.Doc]:
        """Convert a list of text to a list of spacy documents.

        Parameters
        ----------
        spacy_model: spacy.language.Language
            The spacy model used to represent text corpus.

        Returns
        -------
        List[spacy.tokens.doc.Doc]
            Corpus represented as a list of spacy documents.

        Raises
        ------
        EmptyCorpusError
            An error raised when the loaded corpus is empty signifying an issue in the loading process.
        """
        text_corpus = self._read_corpus()
        spacy_corpus = []
        for i, spacy_document in enumerate(spacy_model.pipe(text_corpus)):
            try:
                spacy_corpus.append(spacy_document)
            except Exception as _e:
                logger.error(
                    "Could not load content as spacy document. \nTrace : %s.\nDocument : %s.",
                    _e,
                    text_corpus[i]
                    )
            else:
                logger.info(
                    "File content %i converted to spacy document.",
                    i
                )

        if not spacy_corpus:
            raise EmptyCorpusError

        return spacy_corpus

    @abstractmethod
    def _read_corpus(self) -> list[str]:
        """Load documents and convert them as a list of texts.

        Returns
        -------
        List[str]
            Corpus represented as a list of texts.
        """
        ...
