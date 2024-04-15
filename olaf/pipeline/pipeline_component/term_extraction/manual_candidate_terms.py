from typing import Any, Callable, Dict, List, Optional, Set

import spacy.language
from spacy.matcher import PhraseMatcher

from ....commons.errors import ParameterError
from ....commons.logging_config import logger
from ....data_container.candidate_term_schema import CandidateTerm
from .term_extraction_schema import TermExtractionPipelineComponent


class ManualCandidateTermExtraction(TermExtractionPipelineComponent):
    """A pipeline component to manually add candidate terms.

    Attributes
    ----------
    cts_post_processing_functions: List[Callable[[Set[CandidateTerm]], Set[CandidateTerm]]], optional
        A list of candidate term post processing functions to run after candidate term extraction
        and before assigning the extracted candidate terms to the pipeline, by default None.
    ct_label_strings_map: Dict[str, Set[str]], optional
        The mapping of candidate term label and their matching strings.
        Optional only if a custom spaCy phrase matcher is provided.
    phrase_matcher: PhraseMatcher, optional
        The spaCy phrase matcher for new candidate term corpus occurrence matching.
        Default to matching the label provided strings.
    """

    def __init__(
        self,
        cts_post_processing_functions: Optional[
            List[Callable[[Set[CandidateTerm]], Set[CandidateTerm]]]
        ] = None,
        ct_label_strings_map: Optional[Dict[str, Set[str]]] = None,
        phrase_matcher: Optional[PhraseMatcher] = None,
    ) -> None:
        """Initialise ManualCandidateTermExtraction pipeline component instance.

        Parameters
        ----------
        cts_post_processing_functions: List[Callable[[Set[CandidateTerm]], Set[CandidateTerm]]], optional
            A list of candidate term post processing functions to run after candidate term extraction
            and before assigning the extracted candidate terms to the pipeline, by default None.
        ct_label_strings_map: Dict[str, Set[str]], optional
            The mapping of candidate term label and their matching strings.
            Optional only if a custom spaCy phrase matcher is provided.
        phrase_matcher: PhraseMatcher, optional
            The spaCy phrase matcher for new candidate term corpus occurrence matching.
            Default to matching the label provided strings.
        """
        super().__init__(cts_post_processing_functions)

        self.ct_label_strings_map = ct_label_strings_map
        self.phrase_matcher = phrase_matcher

        self._check_parameters()

    def _check_parameters(self) -> None:
        """Check wether required parameters are given and correct.
        If this is not the case, suitable default ones are set or errors are raised.

        Raises
        ------
        ParameterError
            Exception raised when a required parameter is missing or a wrong value is provided.
        """
        if self.ct_label_strings_map is None and self.phrase_matcher is None:
            raise ParameterError(
                component_name="Manual candidate term extraction",
                param_name="ct_label_strings_map or custom_matcher",
                error_type="Missing parameter",
            )

    def optimise(self) -> None:
        """A method to optimise the pipeline component by tuning the options."""
        raise NotImplementedError

    def _check_resources(self) -> None:
        """Method to check that the component has access to all its required resources."""
        logger.info(
            "Manual candidate term extraction pipeline component has no external resources to check."
        )

    def _compute_metrics(self) -> None:
        """A method to compute component performance metrics."""
        raise NotImplementedError

    def get_performance_report(self) -> Dict[str, Any]:
        """A getter for the pipeline component performance report.
            If the component has been optimised, it only returns the best performance.
            Otherwise, it returns the results obtained with the parameters set.

        Returns
        -------
        Dict[str, Any]
            The pipeline component performance report.
        """
        raise NotImplementedError

    def _build_matcher(self, nlp: spacy.language.Language) -> PhraseMatcher:
        """Build the default spaCy phrase matcher using the provided candidate terms label strings.
        The phrase matcher will be case insensitive.

        Parameters
        ----------
        nlp : spacy.language.Language
            The spaCy language model to use for the matcher construction.

        Returns
        -------
        PhraseMatcher
            The constructed phrase matcher.
        """
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

        for label, match_strings in self.ct_label_strings_map.items():
            matcher.add(label, [nlp(string) for string in match_strings])

        return matcher

    def run(self, pipeline: Any) -> None:
        """Execution of the candidate term extraction based manually provided strings.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline running.
        """

        if self.phrase_matcher is None:
            self.phrase_matcher = self._build_matcher(pipeline.spacy_model)

        candidate_terms_index = {}

        for doc in pipeline.corpus:
            matches = self.phrase_matcher(doc, as_spans=True)

            for match in matches:
                if match.label not in candidate_terms_index:
                    candidate_terms_index[match.label] = CandidateTerm(
                        label=pipeline.spacy_model.vocab.strings[match.label],
                        corpus_occurrences={match},
                    )
                else:
                    candidate_terms_index[match.label].add_corpus_occurrences({match})

        candidate_terms = set(candidate_terms_index.values())

        candidate_terms = self.apply_post_processing(candidate_terms)

        pipeline.candidate_terms.update(candidate_terms)
