from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import spacy

from ....commons.logging_config import logger
from ....commons.spacy_processing_tools import select_on_pos
from ....data_container.candidate_term_schema import CandidateTerm
from .term_extraction_schema import TermExtractionPipelineComponent


class POSTermExtraction(TermExtractionPipelineComponent):
    """Extract candidate terms with part-of-speech (POS) tagging.

    Attributes
    ----------
    cts_post_processing_functions: List[Callable[[Set[CandidateTerm]], Set[CandidateTerm]]], optional
        A list of candidate term post processing functions to run after candidate term extraction
        and before assigning the extracted candidate terms to the pipeline. Default to None.
    token_processing: Optional[Callable[[spacy.tokens.Token],str]]=None
        A function to process token.
    _pos_selection: List[str]
        List of POS tags to select in the corpus.
    _token_sequences_doc_attribute: str
        Attribute indicating which sequences to use for processing.
        If None, the entire doc is used.
    parameters: Dict[str, Any], optional
        Parameters are fixed values to be defined when building the pipeline.
        They are necessary for the component functioning. By default dict().
    options: Dict[str, Any], optional
        Options are tunable parameters which will be updated to optimise the component performance.
        By default dict().
    """

    def __init__(
        self,
        token_processing: Optional[Callable[[spacy.tokens.Token], str]] = None,
        cts_post_processing_functions: Optional[
            List[Callable[[Set[CandidateTerm]], Set[CandidateTerm]]]
        ] = None,
        parameters: Dict[str, Any] | None = None,
        options: Dict[str, Any] | None = None,
    ) -> None:
        """Initialise part-of-speech term extraction pipeline component instance.

        Parameters
        ----------
        token_processing: Optional[Callable[[spacy.tokens.Token],str]]
            A function to process token.
        cts_post_processing_functions: List[Callable[[Set[CandidateTerm]], Set[CandidateTerm]]], optional
            A list of candidate term post processing functions to run after candidate term extraction
            and before assigning the extracted candidate terms to the pipeline. Default to None.
        parameters : Dict[str, Any], optional
            Parameters are fixed values to be defined when building the pipeline.
            They are necessary for the component functioning. By default dict().
        options : Dict[str, Any], optional
            Options are tunable parameters which will be updated to optimise the component performance.
            By default dict().
        """
        super().__init__(cts_post_processing_functions, parameters, options)

        if (token_processing is None) or not callable(token_processing):
            logger.warning(
                "No preprocessing function provided for tokens. Using the default one."
            )
            self.token_processing = lambda token: token.lower_
        else:
            self.token_processing = token_processing

        self._pos_selection = None
        self._token_sequences_doc_attribute = None
        self._check_parameters()

    def _check_parameters(self) -> None:
        """Check wether required parameters are given and correct. If this is not the case, suitable default ones are set."""
        user_defined_attribute_name = self.parameters.get(
            "token_sequence_doc_attribute"
        )

        if user_defined_attribute_name:
            if spacy.tokens.Doc.has_extension(user_defined_attribute_name):
                self._token_sequences_doc_attribute = user_defined_attribute_name
            else:
                logger.warning(
                    """User defined POS term extraction token sequence attribute %s not set on spaCy Doc.
                    By default the system will use the entire content of the document.""",
                    user_defined_attribute_name,
                )

        else:
            logger.warning(
                """POS term extraction token sequence attribute not set by the user.
                By default the system will use the entire content of the document."""
            )

        pos_selection = self.parameters.get("pos_selection")

        if pos_selection:
            self._pos_selection = pos_selection
        else:
            logger.warning(
                """POS selection not set by the user.
                By default the system will use the NOUN part-of-speech tag."""
            )
            self._pos_selection = ["NOUN"]

    def _check_resources(self) -> None:
        """Method to check that the component has access to all its required resources.

        This pipeline component does not need any access to any external resource.
        """
        logger.info(
            "POS term extraction pipeline component has no external resource to check."
        )

    def optimise(
        self, validation_terms: Set[str], option_values_map: Set[float]
    ) -> None:
        """A method to optimise the pipeline component by tuning the options."""
        raise NotImplementedError

    def _compute_metrics(self) -> None:
        """A method to compute component performance metrics. It is used by the optimise method to update the options."""
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

    def _extract_token_sequences(
        self, corpus: List[spacy.tokens.Doc]
    ) -> Tuple[spacy.tokens.Span]:
        """Extract token sequences from the corpus.

        Parameters
        ----------
        corpus: List[spacy.tokens.doc.Doc]
            The corpus to extract the sequences from.

        Returns
        -------
        Tuple[spacy.tokens.Span]
            The list of extracted sequences.
        """
        token_sequences = []

        if self._token_sequences_doc_attribute:
            for doc in corpus:
                doc_token_sequences = doc._.get(self._token_sequences_doc_attribute)
                token_sequences.append(doc_token_sequences)
        else:
            for doc in corpus:
                # Extract the document tokens in a span instance for type consistency.
                doc_span = doc[:]
                token_sequences.append(doc_span)

        return tuple(token_sequences)

    def _extract_candidate_tokens(
        self, token_sequences: Tuple[spacy.tokens.Span]
    ) -> Tuple[spacy.tokens.Token]:
        """Extract candidate tokens from token sequences based on POS tagging selection.

        Parameters
        ----------
        token_sequences : Tuple[spacy.tokens.Span]
            Token sequences to extract candidates from.

        Returns
        -------
        Tuple[spacy.tokens.Token]
            Candidate tokens under interest.
        """
        candidate_tokens = []

        for token_sequence in token_sequences:
            for token in token_sequence:
                if select_on_pos(token, self._pos_selection):
                    candidate_tokens.append(token)

        return tuple(candidate_tokens)

    def _build_term_corpus_occ_map(
        self, candidate_tokens: Tuple[spacy.tokens.Token]
    ) -> Dict[str, Set[spacy.tokens.Token]]:
        """Build a mapping between term string to be processed and the spaCy tokens
        they were extracted from.

        Parameters
        ----------
        candidate_tokens : Tuple[spacy.tokens.Token]
            The spaCy token candidates.

        Returns
        -------
        Dict[str, Set[spacy.tokens.Token]]
            The mapping between term string to be processed spaCy tokens they were
            extracted from.
        """
        term_corpus_occ_mapping = defaultdict(set)

        for token in candidate_tokens:
            token_string = self.token_processing(token)
            term_corpus_occ_mapping[token_string].add(token)

        return term_corpus_occ_mapping

    def run(self, pipeline: Any) -> None:
        """Execution of the POS term extraction on the corpus. Pipeline candidate terms are updated.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline running.
        """

        token_sequences = self._extract_token_sequences(corpus=pipeline.corpus)
        candidate_tokens = self._extract_candidate_tokens(token_sequences)
        term_corpus_occ_map = self._build_term_corpus_occ_map(candidate_tokens)

        candidate_terms = set()
        for term_label, term_occurrences in term_corpus_occ_map.items():
            candidate_term = CandidateTerm(
                label=term_label, corpus_occurrences=term_occurrences
            )

            candidate_terms.add(candidate_term)

        candidate_terms = self.apply_post_processing(candidate_terms)

        pipeline.candidate_terms.update(candidate_terms)
