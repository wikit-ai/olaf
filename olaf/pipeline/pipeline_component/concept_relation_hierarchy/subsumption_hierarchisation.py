from itertools import combinations
from typing import Any, Dict, Set, Optional

from tqdm import tqdm

from ...pipeline_schema import Pipeline
from ....commons.logging_config import logger
from ....data_container.concept_schema import Concept
from ....data_container.metarelation_schema import Metarelation
from ..pipeline_component_schema import PipelineComponent


class SubsumptionHierarchisation(PipelineComponent):
    """Extract hierarchisation metarelations based on subsumption.

    Attributes
    ----------
    threshold : float, optional
        Threshold used to validate the subsumption relation or not, by default 0.5.
    """

    def __init__(self, threshold: Optional[float] = None) -> None:
        """Initialise subsumption hierarchisation instance.

        Parameters
        ----------
        threshold : float, optional
            Threshold used to validate the subsumption relation or not, by default 0.5.
        """
        self.threshold = threshold
        self._check_parameters()

    def _check_parameters(self) -> None:
        """Check whether required parameters are given and correct.
        If this is not the case, suitable default ones are set or errors are raised.

        This method affects the self.scope attribute.
        """
        if not self.threshold:
            self.threshold = 0.5
            logger.warning(
                "No value given for threshold parameter, default will be set to 0.5."
            )
        elif not isinstance(self.threshold, float):
            self.threshold = 0.5
            logger.warning(
                "Incorrect value given for threshold parameter, default will be set to 0.5."
            )

    def optimise(
        self, validation_terms: Set[str], option_values_map: Set[float]
    ) -> None:
        """A method to optimise the pipeline component by tuning the options."""
        raise NotImplementedError

    def check_resources(self) -> None:
        """Method to check that the component has access to all its required resources.

        This pipeline component does not need any access to any external resource.
        """
        logger.info(
            "Subsumption hierarchisation pipeline component has no external resource to check."
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

    def _concept_occurrence_count(self, concept: Concept) -> int:
        """Count the total number of corpus occurrences of the concepts by adding the number of corpus occurrences for each linguistic realisation.

        Parameters
        ----------
        concept : Concept
            Concept to count corpus occurrences from.

        Returns
        -------
        int
            Total number of corpus occurrences of the concept.
        """
        concept_occurrences = 0
        for lr in concept.linguistic_realisations:
            concept_occurrences += len(lr.corpus_occurrences)
        return concept_occurrences

    def _concepts_cooccurrence_count(
        self, concept_1: Concept, concept_2: Concept
    ) -> int:
        """Count number of concepts cooccurrences in the corpus.
        Original corpus occurrences sentences are used.

        Parameters
        ----------
        concept_1 : Concept
            First concept to use for cooccurrence count.
        concept_2 : Concept
            Second concept to use for cooccurrence count.

        Returns
        -------
        int
            Number of cooccurrence between the two concepts.
        """
        concept_1_sent = set()
        for lr in concept_1.linguistic_realisations:
            for co in lr.corpus_occurrences:
                concept_1_sent.add(co.sent)

        concept_2_sent = set()
        for lr in concept_2.linguistic_realisations:
            for co in lr.corpus_occurrences:
                concept_2_sent.add(co.sent)

        concepts_cooccurrence = len(concept_1_sent & concept_2_sent)

        return concepts_cooccurrence

    def _compute_subsumption(self, nb_cooccurrence: int, nb_occurrence: int) -> float:
        """Computation of the subsumption score.

        Parameters
        ----------
        nb_cooccurrence : int
            Number of cooccurrences between concepts.
        nb_occurrence : int
            Number of general concept occurrence.

        Returns
        -------
        float
            Subsumption score.
        """
        if not nb_occurrence == 0:
            subsumption_score = nb_cooccurrence / nb_occurrence
        else:
            subsumption_score = 0
        return subsumption_score

    def _is_sub_hierarchy(self, sub_score: float, inv_sub_score: float) -> bool:
        """Test if there is a subsumption relation.
        The subsumption score must be higher than the threshold defined.
        The subsumption score must be higher than the one calculated by the subsumption in the opposite order.

        Parameters
        ----------
        sub_score : float
            Subsumption score.
        inv_sub_score : float
            Subsumption score in the opposite order.

        Returns
        -------
        bool
            True is there is a subsumption and False otherwise.
        """
        sub_hierarchy = False
        if (sub_score > self.threshold) and (sub_score > inv_sub_score):
            sub_hierarchy = True
        return sub_hierarchy

    def run(self, pipeline: "Pipeline") -> None:
        """Execution of the subsumption hierarchisation process on pipeline concepts.
        Generalisation metarelations are created.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline running.
        """
        concept_pairs = list(combinations(pipeline.kr.concepts, 2))
        for concept_1, concept_2 in tqdm(concept_pairs):
            concept_1_occ = self._concept_occurrence_count(concept_1)
            concept_2_occ = self._concept_occurrence_count(concept_2)
            concepts_cooc = self._concepts_cooccurrence_count(concept_1, concept_2)
            sub_score = self._compute_subsumption(concepts_cooc, concept_1_occ)
            inv_sub_score = self._compute_subsumption(concepts_cooc, concept_2_occ)
            if self._is_sub_hierarchy(sub_score, inv_sub_score):
                metarelation = Metarelation(
                    source_concept=concept_1,
                    destination_concept=concept_2,
                    label="is_generalised_by",
                )
                pipeline.kr.metarelations.add(metarelation)
            elif self._is_sub_hierarchy(inv_sub_score, sub_score):
                metarelation = Metarelation(
                    source_concept=concept_2,
                    destination_concept=concept_1,
                    label="is_generalised_by",
                )
                pipeline.kr.metarelations.add(metarelation)
