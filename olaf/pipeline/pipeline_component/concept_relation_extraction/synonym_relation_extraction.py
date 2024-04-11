from typing import Any, Dict, Optional

from ....commons.candidate_term_tools import group_cts_on_synonyms
from ....commons.logging_config import logger
from ....commons.relation_tools import crs_to_relation, cts_to_crs
from ..pipeline_component_schema import PipelineComponent


class SynonymRelationExtraction(PipelineComponent):
    """Extract relations based on synonyms grouping.

    Attributes
    ----------
    concept_max_distance: int, optional
        The maximum distance between the candidate term and the concept sought.
        Set to 5 by default if not specified.
    scope: str
        Scope used to search concepts. Can be "doc" for the entire document or "sent" for the
        candidate term "sentence".
        Set to "doc" by default if not specified.
    """

    def __init__(
        self,
        concept_max_distance: Optional[int] = 5,
        scope: Optional[str] = "doc"
    ) -> None:
        """Initialise synonym grouping relation extraction instance.

        Parameters
        ----------
        concept_max_distance: int, optional
            The maximum distance between the candidate term and the concept sought.
            Set to 5 by default if not specified.
        scope: str
            Scope used to search concepts. Can be "doc" for the entire document or "sent" for the
            candidate term "sentence".
            Set to "doc" by default if not specified.
        """
        super().__init__()
        self.concept_max_distance = concept_max_distance
        self.scope = scope

        self._check_parameters()

    def _check_parameters(self) -> None:
        """Check whether required parameters are given and correct.
        If this is not the case, suitable default ones are set or errors are raised.

        This method affects the self.scope attribute.
        """
        if self.scope not in {"sent", "doc"}:
            self.scope = "doc"
            logger.warning(
                """Wrong scope value. Possible values are 'sent' or 'doc'. Default to scope = 'doc'."""
            )
        
        if not isinstance(self.concept_max_distance, int):
            self.concept_max_distance = 5
            logger.warning(
                "No value given for concept_max_distance parameter, default will be set to 5."
            )

    def optimise(self) -> None:
        """A method to optimise the pipeline component by tuning the options."""
        logger.info(
            "Synonym grouping concept extraction pipeline component cannot be optimise."
        )

    def _check_resources(self) -> None:
        """Method to check that the component has access to all its required resources."""
        logger.info(
            "Synonym grouping relation extraction pipeline component has no external resources to check."
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

    def run(self, pipeline: Any) -> None:
        """Execution of the synonyms grouping for relation extraction on candidate terms.
        Candidate terms are converted into candidate relations.
        Candidate relations with same synonyms, source and destination concepts are grouped
        together as a new relation. Candidate terms are purged.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline running.
        """

        concepts_labels_map = dict()
        for concept in pipeline.kr.concepts:
            concepts_labels_map[concept.label] = concept

        candidate_relations = cts_to_crs(
            pipeline.candidate_terms,
            concepts_labels_map,
            pipeline.spacy_model,
            self.concept_max_distance,
            self.scope,
        )
        candidate_relation_groups = group_cts_on_synonyms(candidate_relations)

        for cr_group in candidate_relation_groups:
            new_relation = crs_to_relation(cr_group)
            pipeline.kr.relations.add(new_relation)

        pipeline.candidate_terms = set()
