from typing import List

import spacy
import spacy.tokens.doc
import spacy.tokenizer
import spacy.tokens.span
import spacy.language

from data_preprocessing.data_preprocessing_repository import load_corpus, load_spacy_model
import config.logging_config as logging_config


class DataPreprocessing():
    """First basic processing of the corpus.

    Attributes: 
        corpus: List[spacy.tokens.doc.Doc]
            Documents to process
    """

    def __init__(self) -> None:
        self.corpus = []

    def _set_corpus(self) -> None:
        corpus = load_corpus()
        self.corpus = self._get_document_representation(corpus)

    def _get_document_representation(self, corpus: List[str]) -> List[spacy.tokens.doc.Doc]:
        """Convert text to spacy document representation.

        Returns
        -------
        List(spacy.tokens.doc)
            Corpus of spacy document representation.
        """
        self.spacy_model = load_spacy_model()
        corpus_preprocessed = []

        try:
            for spacy_document in self.spacy_model.pipe(corpus):
                corpus_preprocessed.append(spacy_document)
        except Exception as _e:
            logging_config.logger.error(
                f"Could not load content as spacy document. Trace : {_e}")
        else:
            logging_config.logger.info(
                "File content converted to spacy document.")
        return corpus_preprocessed
