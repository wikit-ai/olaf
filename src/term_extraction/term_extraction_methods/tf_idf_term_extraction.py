from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy.tokens.doc

import config.logging_config as logging_config
from term_extraction.term_extraction_schema import TermExtractionResults, TFIDF_ANALYZER


def default_tfidf_token_filter(doc: spacy.tokens.doc.Doc) -> List[str]:
    tokens = [token.text for token in doc]
    return tokens


class TFIDF_TermExtraction:

    def __init__(self, corpus: List[spacy.tokens.doc.Doc], tfidf_agg_type: str = 'MEAN', tfidf_token_filter: Optional[TFIDF_ANALYZER] = None) -> None:
        """Initializer for a TF IDF based term extraction. TF IDF scores are specific to a term in context of a document.
            To compute a score for a term regardless of the document we either:
            - take the maximum TF IDF score for a term in the corpus: `tfidf_agg_type = "MAX"`
            - or compute the mean of the non zero TF IDF scores: `tfidf_agg_type = "MEAN"` (default)

        Parameters
        ----------
        corpus: List[spacy.tokens.doc.Doc]:
            The list of documents composing the corpus.
        tfidf_agg_type: str = 'MEAN': 
            The TF IDF aggreagtion type. Either "MEAN" or "MAX".
        tfidf_token_filter: Optional[TFIDF_ANALYZER]:
            The function to pre select tokens to consider for the TF IDF score computation.
        """
        self.corpus = corpus
        self.tfidf_values = None
        self.tfidf_agg_type = tfidf_agg_type

        if tfidf_token_filter is not None:
            self.tfidf_token_filter = tfidf_token_filter
        else:
            self.tfidf_token_filter = default_tfidf_token_filter
            logging_config.logger.info(
                "No TF IDF token filter provided -- using the default one")

        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer=self.tfidf_token_filter
        )

    @property
    def tfidf_values(self) -> List[TermExtractionResults]:
        """Getter for the self.tfidf_values attribute.

        Returns
        -------
        List[TermExtractionResults]
            The list of terms associated with their TF IDF scores.
        """
        if self._tfidf_values is None:
            self.compute_tfidf()

        return self._tfidf_values

    @tfidf_values.setter
    def tfidf_values(self, value: List[TermExtractionResults]) -> None:
        """Setter for the self.tfidf_values attribute.

        Parameters
        ----------
        value : List[TermExtractionResults]
            The list of terms associated with their TF IDF scores.
        """
        if (value is None) or isinstance(value, list):
            self._tfidf_values = value
        else:
            logging_config.logger.error(
                "Incompatible value type for self._tfidf_values attribute. It should be List[TermExtractionResults]")

    def compute_tfidf(self) -> List[TermExtractionResults]:
        """Method to compute the TF IDF scores.

        Returns
        -------
        List[TermExtractionResults]
            The list of terms associated with their TF IDF scores.
        """
        term_tfidf_res = list()

        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                self.corpus).toarray()
        except Exception as e:
            logging_config.logger.error(
                f"There has been an issue while computing the TF IDF matrix. Trace : {e}")
        else:
            logging_config.logger.info(
                f"TF IDF matrix computed.")

        if self.tfidf_agg_type == "MEAN":
            tfidf_values = tfidf_matrix.sum(
                axis=0) / np.count_nonzero(tfidf_matrix, axis=0)

        elif self.tfidf_agg_type == "MAX":
            tfidf_values = tfidf_matrix.max(
                axis=0)

        for token, idx in self.tfidf_vectorizer.vocabulary_.items():
            term_tfidf_res.append(
                TermExtractionResults(tfidf_values[idx], token))

        term_tfidf_res.sort(key=lambda r: r.score)

        self.tfidf_values = term_tfidf_res
