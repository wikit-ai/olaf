from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import spacy.tokens.doc
import spacy.tokens.span
from sklearn.feature_extraction.text import TfidfVectorizer

from ....commons.errors import OptionError
from ....commons.logging_config import logger
from ....commons.spacy_processing_tools import spacy_span_ngrams
from ....data_container.candidate_term_schema import CandidateTerm
from .term_extraction_schema import TermExtractionPipelineComponent


class TFIDFTermExtraction(TermExtractionPipelineComponent):
    """Extract candidate terms using TF-IDF based scores computed on the corpus.

    Attributes
    ----------
    cts_post_processing_functions: List[Callable[[Set[CandidateTerm]], Set[CandidateTerm]]], optional
        A list of candidate term post processing functions to run after candidate term extraction
        and before assigning the extracted candidate terms to the pipeline, by default None.
    token_sequence_preprocessing : Callable[[spacy.tokens.span.Span],Tuple[str]], optional
        By default None.
    _token_sequences_doc_attribute : str
        The name of the spaCy doc custom attribute containing the sequences of tokens to
        form the corpus for the c-value computation. Default is None which default to the full doc.
    _max_term_token_length : int
        The maximum number of tokens a term can have, by default 1.
    tfidf_agg_type : Union["MEAN", "MAX"]
        The operation to use to aggregate TF-IDF values of candidate terms.
        can be "MEAN" to aggregate by mean values or "MAX" to aggregate by max values, by default "MEAN".
    candidate_term_threshold : float
        The TF-IDF score threshold below which terms will be ignored, by default 0.0.
    _ngram_range : Tuple[int, int]
        The ngram range for the TF-IDF vectorizer.
    _custom_tokenizer : Callable[[str], List[str]]
        Tokenizer for the TF-IDF vectorizer.
    tfidf_vectorizer : sklearn.feature_extraction.text.TfidfVectorizer, optional
        The TF-IDF vectorizer to compute TF-IDF scores.
    """

    def __init__(
        self,
        token_sequence_preprocessing: Optional[
            Callable[[spacy.tokens.span.Span], Tuple[str]]
        ] = None,
        token_sequences_doc_attribute: Optional[str] = None,
        cts_post_processing_functions: Optional[
            List[Callable[[Set[CandidateTerm]], Set[CandidateTerm]]]
        ] = None,
        max_term_token_length: Optional[int] = None,
        tfidf_agg_type: Optional[str] = "MEAN",
        candidate_term_threshold: Optional[float] = None,
        tfidf_vectorizer: Optional[TfidfVectorizer] = None,
    ) -> None:
        """Initialise TF-IDF term extraction pipeline component instance.

        TF IDF scores are specific to a term in context of a document.
        To compute a score for a term regardless of the document we either:
        - take the maximum TF IDF score for a term in the corpus: `tfidf_agg_type = "MAX"`
        - or compute the mean of the non zero TF IDF scores: `tfidf_agg_type = "MEAN"` (default)

        Parameters
        ----------
        token_sequence_preprocessing: Callable[[spacy.tokens.span.Span],Tuple[str]], optional
            A function to preprocess token sequences composing the corpus, by default None.
            The function should return a tuple of token texts.
        token_sequences_doc_attribute : str, optional
            The name of the spaCy doc custom attribute containing the sequences of tokens to
            form the corpus for the c-value computation. Default is None which default to the full doc.
        cts_post_processing_functions: Callable[[Set[CandidateTerm]], Set[CandidateTerm]], optional
            A list of candidate term post processing functions to run after candidate term extraction
            and before assigning the extracted candidate terms to the pipeline, by default None.
        max_term_token_length : int
            The maximum number of tokens a term can have, by default 1.
        tfidf_agg_type : Union["MEAN", "MAX"], optional
            The operation to use to aggregate TF-IDF values of candidate terms.
            can be "MEAN" to aggregate by mean values or "MAX" to aggregate by max values, by default "MEAN".
        candidate_term_threshold : float, optional
            The TF-IDF score threshold below which terms will be ignored, by default 0.0.
        tfidf_vectorizer : sklearn.feature_extraction.text.TfidfVectorizer, optional
            The TF-IDF vectorizer to compute TF-IDF scores.
        """

        super().__init__(cts_post_processing_functions)
        self.token_sequence_preprocessing = token_sequence_preprocessing
        self._token_sequences_doc_attribute = token_sequences_doc_attribute

        self._max_term_token_length = max_term_token_length
        self.tfidf_agg_type = tfidf_agg_type
        self.candidate_term_threshold = candidate_term_threshold

        self._check_parameters()

        self._ngram_range = (1, self._max_term_token_length)
        self._custom_tokenizer = lambda text: [t.strip() for t in text.split()]

        self.tfidf_vectorizer = (
            tfidf_vectorizer
            if tfidf_vectorizer is not None
            else TfidfVectorizer(
                tokenizer=self._custom_tokenizer, ngram_range=self._ngram_range
            )
        )

    def _check_parameters(self) -> None:
        """Check wether required parameters are given and correct. If this is not the case,
        suitable default ones are set.

        For the TFIDFtermExtraction pipeline component the parameter to check are:
        - "token_sequence_doc_attribute": name of the spaCy Doc custom attribute containing
            the token sequences to focus on.
        - token_sequence_preprocessing: the function to use for text preprocessing.
        """
        if user_defined_attribute_name := self._token_sequences_doc_attribute:
            if not spacy.tokens.Doc.has_extension(user_defined_attribute_name):
                logger.warning(
                    """User defined selected token sequence document attribute %s not set on spaCy Doc.
                    By default the system will use the entire content of the document.""",
                    user_defined_attribute_name,
                )

        else:
            logger.warning(
                """Selected token sequence document attribute not set by the user.
                By default the system will use the entire content of the document."""
            )

        if (self.token_sequence_preprocessing is None) or not callable(
            self.token_sequence_preprocessing
        ):
            logger.debug(
                "No preprocessing function provided for the token sequences. Using the default one."
            )
            self.token_sequence_preprocessing = lambda span: [
                token.lower_.strip() for token in span
            ]

        if self._max_term_token_length is None:
            logger.debug(
                "No max token length for extracted terms provided. Defaulting to unigram."
            )
            self._max_term_token_length = 1

        if self.candidate_term_threshold is None:
            logger.debug(
                "No threshold provided for candidate term selection. Defaulting to 0."
            )
            self.candidate_term_threshold = 0

        if self.tfidf_agg_type is None:
            logger.debug(
                "No aggregation type provided for TF-IDF computation. Defaulting to 'MEAN'."
            )
            self.tfidf_agg_type = "MEAN"
        elif self.tfidf_agg_type not in {"MEAN", "MAX"}:
            logger.error("Option tfidf_agg_type should be either 'MEAN', or 'MAX'.")
            raise OptionError(
                component_name=self.__class__,
                option_name="tfidf_agg_type",
                error_type="Wrong value type",
            )

    def optimise(
        self, validation_terms: Set[str], option_values_map: Set[float]
    ) -> None:
        # TODO: how far in the grid search do we go?
        # scikitlearn grid search
        # default to grid search and log a warning
        # enable user defined optimisation function alternative
        """A method to optimise the pipeline component by tuning the options."""
        raise NotImplementedError

    def _check_resources(self) -> None:
        """Method to check that the component has access to all its required resources.

        This pipeline component does not need any access to any external resource.
        """
        logger.info(
            "TF-IDF term extraction pipeline component has no external resource to check."
        )

    def _compute_metrics(self) -> None:
        """A method to compute component performance metrics. It is used by the optimise
        method to update the options.
        """
        raise NotImplementedError

    def get_performance_report(self) -> Dict[str, Any]:
        """A getter for the pipeline component performance report.
            If the component has been optimised, it only returns the best performance.
            Otherwise, it returns the results obtained with the set parameters.

        Returns
        -------
        Dict[str, Any]
            The pipeline component performance report.
        """
        raise NotImplementedError

    def _spaced_term_corpus_occ_map(
        self, token_seqs_spans: Set[spacy.tokens.Span]
    ) -> Dict[str, Set[spacy.tokens.Span]]:
        """Build a mapping between term string to be processed and the spaCy spans
        they were extracted from.

        We extract all the substring from 1 token to self._max_term_token_length in
        each token sequence. Hence, some candidate  terms might be substrings of the
        initial token sequences. We pre-generate all of them to make sure to have the
        mapping with a spaCy span to construct the Candidate Term instance.

        Parameters
        ----------
        token_seqs_spans : Tuple[spacy.tokens.Span]
            The spaCy spans of the token sequences to extract the candidate terms from.

        Returns
        -------
        Dict[str, Set[spacy.tokens.Span]]
            The mapping between term string to be processed spaCy spans they were
            extracted from.
        """
        term_corpus_occ_mapping = defaultdict(set)

        for span in token_seqs_spans:
            preprocessed_span_string = " ".join(self.token_sequence_preprocessing(span))
            # to make sure terms generated by the TF-IDF process are indexed.
            spaced_term = " ".join(self._custom_tokenizer(preprocessed_span_string))
            term_corpus_occ_mapping[spaced_term].add(span)

        return term_corpus_occ_mapping

    def _get_corpus_occurrences(
        self, term: str, term_corpus_occ_mapping: Dict[str, Set[spacy.tokens.Span]]
    ) -> Set[spacy.tokens.Span]:
        """Retrieve spaCy spans corresponding to term.

        Parameters
        ----------
        term : str
            The term to retrieve the corpus occurrences from.
        term_corpus_occ_mapping : Dict[str, Set[spacy.tokens.Span]]
            The mapping of term to corpus occurrences.

        Returns
        -------
        Set[spacy.tokens.Span]
            The set of corpus occurrences.
        """
        term_corpus_occurrences = term_corpus_occ_mapping.get(term, set())

        if len(term_corpus_occurrences) == 0:
            logger.warning("No corpus occurrence found for candidate term %s", term)

        return term_corpus_occurrences

    def _create_ngram_spans(
        self, token_sequences: List[spacy.tokens.Span]
    ) -> List[spacy.tokens.Span]:
        """Extract spans ngrams of size from 1 to self._max_term_token_length.

        Parameters
        ----------
        token_sequences : List[spacy.tokens.Span]
            The spans to extract the ngram spans from.

        Returns
        -------
        List[spacy.tokens.Span]
            The ngram spans
        """
        ngram_spans = []

        for span in token_sequences:
            for gram_size in range(1, min(self._max_term_token_length, len(span)) + 1):
                span_ngrams = spacy_span_ngrams(span=span, gram_size=gram_size)
                ngram_spans.extend(span_ngrams)

        return ngram_spans

    def _extract_token_sequences(
        self, corpus: List[spacy.tokens.Doc]
    ) -> Tuple[spacy.tokens.Span]:
        """Extract token sequences to constitute the corpus.

        Parameters
        ----------
        corpus: List[spacy.tokens.doc.Doc]
            The corpus to extract the candidate terms from.

        Returns
        -------
        Tuple[spacy.tokens.Span]
            The list of extracted sequences.
        """
        token_sequences = []

        if self._token_sequences_doc_attribute:
            for doc in corpus:
                doc_token_sequences = doc._.get(self._token_sequences_doc_attribute)
                token_sequences.extend(doc_token_sequences)
        else:
            for doc in corpus:
                # Extract the document tokens in a span instance for type consistency.
                doc_span = doc[:]
                token_sequences.append(doc_span)

        return tuple(token_sequences)

    def _extract_candidate_terms(self, terms: List[str]) -> Tuple[str]:
        """Compute the TF-IDF score for each term and filter out the ones with a score
        below the threshold.

        Parameters
        ----------
        terms : List[str]
            The list of terms to process.

        Returns
        -------
        Tuple[str]
            The list the selected terms.
        """

        tfidf_values = []

        tfidf_matrix = self.tfidf_vectorizer.fit_transform(terms).toarray()

        if self.tfidf_agg_type == "MEAN":
            tfidf_values = tfidf_matrix.sum(axis=0) / np.count_nonzero(
                tfidf_matrix, axis=0
            )

        elif self.tfidf_agg_type == "MAX":
            tfidf_values = tfidf_matrix.max(axis=0)

        candidate_terms_scores = []
        for term, idx in self.tfidf_vectorizer.vocabulary_.items():
            if tfidf_values[idx] > self.candidate_term_threshold:
                candidate_terms_scores.append((term, tfidf_values[idx]))

        candidate_terms_scores.sort(key=lambda t: t[1])

        candidate_terms = [term_score[0] for term_score in candidate_terms_scores]

        return candidate_terms

    def run(self, pipeline: Any) -> None:
        """Method that is responsible for the execution of the component.

        Parameters
        ----------
        pipeline: Pipeline
            The pipeline to run the component with.
        """

        token_sequences = self._extract_token_sequences(corpus=pipeline.corpus)

        vocabulary_spans = self._create_ngram_spans(token_sequences)

        spaced_term_corpus_occ_map = self._spaced_term_corpus_occ_map(vocabulary_spans)

        corpus_spaced_token_sequences = [
            " ".join([t.strip() for t in self.token_sequence_preprocessing(span)])
            for span in token_sequences
        ]

        extracted_terms = self._extract_candidate_terms(
            terms=corpus_spaced_token_sequences
        )

        candidate_terms = set()
        for extracted_term in extracted_terms:
            term_corpus_occurrences = self._get_corpus_occurrences(
                term=extracted_term,
                term_corpus_occ_mapping=spaced_term_corpus_occ_map,
            )
            candidate_term = CandidateTerm(
                label=extracted_term, corpus_occurrences=term_corpus_occurrences
            )

            candidate_terms.add(candidate_term)

        candidate_terms = self.apply_post_processing(candidate_terms)

        pipeline.candidate_terms.update(candidate_terms)
