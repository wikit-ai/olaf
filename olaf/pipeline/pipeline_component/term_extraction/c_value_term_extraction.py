from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import spacy.tokens.span

from ....algorithm.c_value import Cvalue
from ....commons.errors import OptionError
from ....commons.logging_config import logger
from ....commons.spacy_processing_tools import spacy_span_ngrams
from ....data_container.candidate_term_schema import CandidateTerm
from .term_extraction_schema import TermExtractionPipelineComponent


class CvalueTermExtraction(TermExtractionPipelineComponent):
    """Extract candidate terms using C-value scores computed based on the corpus.

    Attributes
    ----------
    cts_post_processing_functions: List[Callable[[Set[CandidateTerm]], Set[CandidateTerm]]], optional
        A list of candidate term post processing functions to run after candidate term extraction
        and before assigning the extracted candidate terms to the pipeline, by default None.
    _token_sequences_doc_attribute : str, optional
        The name of the spaCy doc custom attribute containing the sequences of tokens to
        form the corpus for the c-value computation. Default is None which default to the full doc.
    _candidate_term_threshold : float, optional
        The c-value score threshold below which terms will be ignored.
    _c_value_threshold : float, optional
        The threshold used during the c-value scores computation process, by defaut 0.0.
    _max_term_token_length : int, optional
        The maximum number of tokens a term can have, by defaut 5.
    """

    def __init__(
        self,
        candidate_term_threshold: Optional[float] = 0.0,
        max_term_token_length: Optional[int] = 5,
        token_sequences_doc_attribute: Optional[str] = None,
        c_value_threshold: Optional[float] = None,
        cts_post_processing_functions: Optional[
            List[Callable[[Set[CandidateTerm]], Set[CandidateTerm]]]
        ] = None,
        stop_token_list: Set[str] = None,
    ) -> None:
        """Initialise C-value term extraction pipeline component instance.

        Parameters
        ----------
        candidate_term_threshold : float, optional
            The c-value score threshold below which terms will be ignored, by defaut 0.0.
        max_term_token_length : int, optional
            The maximum number of tokens a term can have, by defaut 5.
        token_sequences_doc_attribute : str, optional
            The name of the spaCy doc custom attribute containing the sequences of tokens to
            form the corpus for the c-value computation. Default is None which default to the full doc.
        c_value_threshold : float, optional
            The threshold used during the c-value scores computation process.
            Default is None which default to the candidate_term_threshold.
        cts_post_processing_functions: List[Callable[[Set[CandidateTerm]], Set[CandidateTerm]]], optional
            A list of candidate term post processing functions to run after candidate term extraction
            and before assigning the extracted candidate terms to the pipeline, by default None.
        stop_token_list: Set[str], optional
            A set of stop words that should not appear in a term.
            _terms_string_tokens: Tuple[Tuple[str]], by default None.
        """

        super().__init__(cts_post_processing_functions)
        self._token_sequences_doc_attribute = token_sequences_doc_attribute

        self._candidate_term_threshold = candidate_term_threshold
        self._c_value_threshold = c_value_threshold
        self._max_term_token_length = max_term_token_length
        self._stop_token_list = (
            stop_token_list if stop_token_list is not None else set()
        )
        self._check_parameters()

    def _check_parameters(self) -> None:
        if self._token_sequences_doc_attribute is None:
            logger.warning(
                """C-value token sequence attribute not set by the user.
                By default the system will use the entire content of the document."""
            )

        elif not spacy.tokens.Doc.has_extension(self._token_sequences_doc_attribute):
            logger.warning(
                """User defined c-value token sequence attribute %s not set on spaCy Doc.
                    By default the system will use the entire content of the document.""",
                self._token_sequences_doc_attribute,
            )
            self._token_sequences_doc_attribute = None

        if not isinstance(self._candidate_term_threshold, float):
            raise OptionError(
                component_name=self.__class__,
                option_name="threshold",
                error_type="Wrong value type",
            )

        if not isinstance(self._max_term_token_length, int):
            raise OptionError(
                component_name=self.__class__,
                option_name="max_term_token_length",
                error_type="Wrong value type",
            )

        if (self._c_value_threshold is None) or not isinstance(
            self._c_value_threshold, float
        ):
            logger.warning(
                """No Correct value provided for the C-value algorithm threshold. 
                The system will default to the provided C-value candidate term extraction threshold."""
            )
            self._c_value_threshold = self._candidate_term_threshold

    def optimise(
        self, validation_terms: Set[str], option_values_map: Set[float]
    ) -> None:
        # TODO: how far in the grid search do we go?
        # scikitlearn grid search
        # default to grid search and log a warning
        # enable user defined optimisation function alternative
        """A method to optimise the pipeline component by tuning the options."""
        raise NotImplementedError

    def check_resources(self) -> None:
        """Method to check that the component has access to all its required resources.

        This pipeline component does not need any access to any external resource.
        """
        logger.info(
            "C-value term extraction pipeline component has no external resource to check."
        )

    def _compute_metrics(self) -> None:
        """A method to compute component performance metrics.
        It is used by the optimise method to update the options.
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

    def _create_max_length_spans(
        self, token_sequences: List[spacy.tokens.Span]
    ) -> List[spacy.tokens.Span]:
        """Extract spans of size limited to the self._max_term_token_length.

        Parameters
        ----------
        token_sequences : List[spacy.tokens.Span]
            The spans to extract the max length spans from.

        Returns
        -------
        List[spacy.tokens.Span]
            The max length spans
        """
        max_length_spans = []

        for span in token_sequences:
            if len(span) > self._max_term_token_length:
                span_ngrams = spacy_span_ngrams(
                    span=span, gram_size=self._max_term_token_length
                )
                max_length_spans.extend(span_ngrams)
            else:
                max_length_spans.append(span)

        return max_length_spans

    def _extract_token_sequences(
        self, corpus: List[spacy.tokens.doc.Doc]
    ) -> Tuple[spacy.tokens.Span]:
        """Extract token sequences for the C-value algorithm to process them.

        Returns
        -------
        Tuple[spacy.tokens.Span]
            The list of extracted sequences.
        """
        token_sequences = []

        if self._token_sequences_doc_attribute:
            for doc in corpus:
                doc_token_sequences = doc._.get(self._token_sequences_doc_attribute)
                max_length_spans = self._create_max_length_spans(doc_token_sequences)
                token_sequences.extend(max_length_spans)
        else:
            for doc in corpus:
                # Extract the document tokens in a span instance for consistency.
                doc_span = doc[:]
                max_length_spans = self._create_max_length_spans([doc_span])
                token_sequences.extend(max_length_spans)

        return tuple(token_sequences)

    def _spaced_term_corpus_occ_map(
        self, token_seqs_spans: Tuple[spacy.tokens.Span]
    ) -> Dict[str, List[spacy.tokens.Span]]:
        """Build a mapping between term string processed by the c-value algorithm
        and the spaCy spans they were extracted from.

        We extract all the substring in each token sequence because the C-value
        algorithm will create them and return scores for each. Hence, some candidate
        terms might be substrings of the initial token sequences. We pre-generate all
        of them to make sure to have the mapping with a spaCy span to construct the
        Candidate Term instance.

        Parameters
        ----------
        token_seqs_spans : Tuple[spacy.tokens.Span]
            The spaCy spans of the token sequences to extract the candidate terms from.

        Returns
        -------
        Dict[str, List[spacy.tokens.Span]]
            The mapping between term string processed by the c-value algorithm and the
            spaCy spans they were extracted from.
        """
        term_corpus_occ_mapping = defaultdict(list)

        for token_seqs_span in token_seqs_spans:
            for i in range(2, len(token_seqs_span)):
                spans = spacy_span_ngrams(token_seqs_span, i)

                for term_span in spans:
                    spaced_term = " ".join([token.text for token in term_span])
                    term_corpus_occ_mapping[spaced_term].append(term_span)

        return term_corpus_occ_mapping

    def _extract_terms(self, terms: List[str]) -> Tuple[str]:
        """Compute the C-value score for each term and filter out the ones with a
        score below the threshold.

        Parameters
        ----------
        terms : List[str]
            The list of terms to process.

        Returns
        -------
        Tuple[str]
            The list the selected terms.
        """
        c_value = Cvalue(
            corpus_terms=terms,
            max_term_token_length=self._max_term_token_length,
            stop_list=self._stop_token_list,
            c_value_threshold=self._c_value_threshold,
        )

        c_value.compute_c_values()

        candidate_terms = tuple(
            [
                c_value_tuple[1]
                for c_value_tuple in c_value.c_values
                if c_value_tuple[0] >= self._candidate_term_threshold
            ]
        )

        return candidate_terms

    def _get_corpus_occurrences(
        self, term: str, term_corpus_occ_mapping: Dict[str, List[spacy.tokens.Span]]
    ) -> List[spacy.tokens.Span]:
        """Retrieve spaCy spans corresponding to term.

        Parameters
        ----------
        term : str
            The term to retrieve the corpus occurrences from.
        term_corpus_occ_mapping : Dict[str, List[spacy.tokens.Span]]
            The mapping of term to corpus occurrences.

        Returns
        -------
        List[spacy.tokens.Span]
            The list of corpus occurrences.
        """
        term_corpus_occurrences = term_corpus_occ_mapping.get(term, [])

        return term_corpus_occurrences

    def run(self, pipeline: Any) -> None:
        """Method that is responsible for the execution of the component.

        Parameters
        ----------
        pipeline: Pipeline
            The pipeline to run the component with.
        """

        token_sequences = self._extract_token_sequences(corpus=pipeline.corpus)

        spaced_term_corpus_occ_map = self._spaced_term_corpus_occ_map(token_sequences)

        corpus_spaced_token_sequences = [
            " ".join([token.text for token in token_sequence])
            for token_sequence in token_sequences
        ]

        extracted_terms = self._extract_terms(terms=corpus_spaced_token_sequences)

        candidate_terms = set()
        for extracted_term in extracted_terms:
            term_corpus_occurrences = self._get_corpus_occurrences(
                term=extracted_term, term_corpus_occ_mapping=spaced_term_corpus_occ_map
            )
            candidate_term = CandidateTerm(
                label=extracted_term, corpus_occurrences=term_corpus_occurrences
            )

            candidate_terms.add(candidate_term)

        candidate_terms = self.apply_post_processing(candidate_terms)

        pipeline.candidate_terms.update(candidate_terms)
