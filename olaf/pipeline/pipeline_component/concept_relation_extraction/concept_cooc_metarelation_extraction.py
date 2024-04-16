from itertools import combinations
from typing import Any, Callable, Dict, Optional, Set

import spacy

from ....commons.logging_config import logger
from ....commons.spacy_processing_tools import spacy_span_ngrams, spans_overlap
from ....data_container.concept_schema import Concept
from ....data_container.metarelation_schema import Metarelation
from ..pipeline_component_schema import PipelineComponent


class ConceptCoocMetarelationExtraction(PipelineComponent):
    """A pipeline component to extract metarelations based on concept co-occurrence.

    Attributes
    ----------
    metarelation_creation_metric: Callable[[int], bool], optional
        The function to define based on the concept co-occurrence count whether or not to create a
        metarelation, by default co-occurrence count > self.threshold.
    window_size: int, optional
        The token window size to consider for concept co-occurrence. Minimum is 2, by default None.
    threshold: int, optional
        The co-occurrence minimum count threshold for metarelation construction, by default 0.
    scope: str, optional
        The corpus scope to consider. Either 'doc' or 'sent', by default 'doc'.
    metarelation_label: str, optional
        The metarelation label to use, by default 'RELATED_TO'.
    create_symmetric_metarelation: bool, optional
        Whether to create the symmetric metarelation, by default False.
        WARNING! this option can create a lot of metarelation that can easily be created in a later
        process.
    """

    def __init__(
        self,
        custom_metarelation_creation_metric: Optional[Callable[[int], bool]] = None,
        window_size: Optional[int] = None,
        threshold: Optional[int] = None,
        scope: Optional[str] = "doc",
        metarelation_label: Optional[str] = "RELATED_TO",
        create_symmetric_metarelation: Optional[bool] = False,
    ) -> None:
        """Initialise ConceptCoocMetarelationExtraction pipeline component instance.

        Parameters
        ----------
        custom_metarelation_creation_metric: Callable[[int], bool], optional
            The function to define based on the concept co-occurrence count whether or not to
            create a metarelation, by default co-occurrence count > self.threshold.
        window_size: int, optional
            The token window size to consider for concept co-occurrence. Minimum is 2, by default None.
        threshold: int, optional
            The co-occurrence minimum count threshold for metarelation construction, by default 0.
        scope: str, optional
            The corpus scope to consider. Either 'doc' (default) or 'sent'.
        metarelation_label: str, optional
            The metarelation label to use, by default 'RELATED_TO'.
        create_symmetric_metarelation: bool, optional
            Whether to create the symmetric metarelation, by default False.
            WARNING! this option can create a lot of metarelation that can easily be created in a later
            process.
        """
        self.window_size = window_size
        self.threshold = threshold
        self.metarelation_creation_metric = (
            custom_metarelation_creation_metric
            if custom_metarelation_creation_metric is not None
            else lambda freq: freq > self.threshold
        )
        self.scope = scope
        self.metarelation_label = metarelation_label
        self.create_symmetric_metarelation = create_symmetric_metarelation

        self._check_parameters()

    def _check_parameters(self) -> None:
        """Check whether required parameters are given and correct.
        If this is not the case, suitable default ones are set or errors are raised.

        This method affects the self.scope attribute.
        """
        if self.window_size is not None and self.window_size < 2:
            self.window_size = 2
            logger.warning(
                """Wrong window size value. Window size cannot be None or lower than 2.
                            Default to window size = 2.
                        """
            )

        if not self.threshold:
            self.threshold = 0
            logger.warning(
                """Wrong threshold value, cannot be None. Default to threshold = 0."""
            )

        if self.scope not in {"sent", "doc"}:
            self.scope = "doc"
            logger.warning(
                """Wrong scope value. Possible values are 'sent' or 'doc'.
                            Default to scope = 'doc'.
                           """
            )

    def optimise(self) -> None:
        """A method to optimise the pipeline component by tuning the options."""
        raise NotImplementedError

    def _check_resources(self) -> None:
        """Method to check that the component has access to all its required resources."""
        logger.info(
            "Concept co-occurrence metarelation extraction pipeline component has no external resources to check."
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

    def _fetch_concept_occurrences_fragments(
        self, concept: Concept
    ) -> Set[spacy.tokens.Span]:
        """Fetch the concept occurrences corpus fragments.
        The corpus fragments depends on the self.scope and self.window_size attributes.
        Only the corpus fragments containing the concepts are returned.

        Parameters
        ----------
        concept : Concept
            The concept to find the corpus fragments from.

        Returns
        -------
        Set[spacy.tokens.Span]
            The set of corpus fragments containing the concept.
        """
        concept_occ_fragments = set()

        for c_lr in concept.linguistic_realisations:
            for c_corpus_occ in c_lr.corpus_occurrences:
                if self.scope == "sent":
                    c_occ_fragment = c_corpus_occ.sent
                else:
                    c_occ_fragment = c_corpus_occ.doc

                if self.window_size:
                    concept_occ_fragments.update(
                        {
                            span
                            for span in spacy_span_ngrams(
                                c_occ_fragment, self.window_size
                            )
                            if spans_overlap(c_corpus_occ, span)
                        }
                    )
                else:
                    concept_occ_fragments.add(c_occ_fragment)

        return concept_occ_fragments

    def _count_concept_cooccurrence(self, concept1: Concept, concept2: Concept) -> int:
        """Count the concepts co-occurrence in the corpus.
        This count depend on the defined window size and scope.
        Note: if some concepts co-occur multiple times in the same corpus fragment, it will be
        counted as only one co-occurrence.

        Parameters
        ----------
        concept1 : Concept
            The first concept.
        concept2 : Concept
            The second concept.

        Returns
        -------
        int
            The concepts co-occurrence count.
        """
        concept1_fragments = self._fetch_concept_occurrences_fragments(concept1)
        concept2_fragments = self._fetch_concept_occurrences_fragments(concept2)

        concept_cooc_count = len(concept1_fragments & concept2_fragments)

        return concept_cooc_count

    def run(self, pipeline: Any) -> None:
        """Execution of the metarelation extraction based on concept co-occurrence.
        Metarelations are created and added to the pipeline knowledge representation.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline running.
        """
        for concept1, concept2 in combinations(pipeline.kr.concepts, 2):
            concept_cooc_count = self._count_concept_cooccurrence(concept1, concept2)

            if self.metarelation_creation_metric(concept_cooc_count):
                pipeline.kr.metarelations.add(
                    Metarelation(
                        source_concept=concept1,
                        destination_concept=concept2,
                        label=self.metarelation_label,
                    )
                )
                if self.create_symmetric_metarelation:
                    pipeline.kr.metarelations.add(
                        Metarelation(
                            source_concept=concept2,
                            destination_concept=concept1,
                            label=self.metarelation_label,
                        )
                    )
