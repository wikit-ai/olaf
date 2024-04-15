from typing import Any, Dict, Optional

from ....commons.logging_config import logger
from ....commons.relation_tools import crs_to_relation, cts_to_crs
from ..pipeline_component_schema import PipelineComponent


class CTsToRelationExtraction(PipelineComponent):
    """A pipeline component to create relations directly from the candidate terms.

    Attributes
    ----------
    parameters: Dict[str, Any]
        Parameters are fixed values to be defined when building the pipeline.
        They are necessary for the component functioning.
    options: Dict[str, Any]
        Options are tunable parameters which will be updated to optimise the component performance.
    concept_max_distance: int, optional
        The maximum distance between the candidate term and the concept sought.
        Set to 5 by default if not specified.
    scope: str
        Scope used to search concepts. Can be "doc" for the entire document or "sent" for
        the candidate term "sentence". Set to "doc" by default if not specified.
    """

    def __init__(
        self,
        concept_max_distance: Optional[int] = None,
        scope: Optional[str] = None,
    ) -> None:
        """Initialise CTsToRelationExtraction pipeline component instance.

        Parameters
        ----------
        parameters : Dict[str, Any], optional
            Parameters are fixed values to be defined when building the pipeline.
            They are necessary for the component functioning, by default None.
        options : Dict[str, Any], optional
            Options are tunable parameters which will be updated to optimise the
            component performance, by default None.
        """
        self.concept_max_distance = concept_max_distance
        self.scope = scope

        self._check_parameters()

    def _check_parameters(self) -> None:
        """Check whether required parameters are given and correct.
        If this is not the case, suitable default ones are set or errors are raised.

        This method affects the self.scope attribute.
        """

        if not isinstance(self.concept_max_distance, int):
            self.concept_max_distance = 5
            logger.warning(
                "No value given for concept_max_distance parameter, default will be set to 5."
            )

        if self.scope not in {"sent", "doc"}:
            self.scope = "doc"
            logger.warning(
                """Wrong scope value. Possible values are 'sent' or 'doc'. Default to scope = 'doc'."""
            )

    def optimise(self) -> None:
        """A method to optimise the pipeline component by tuning the options."""
        raise NotImplementedError

    def _check_resources(self) -> None:
        """Method to check that the component has access to all its required resources."""
        logger.info(
            "Candidate term to relation extraction pipeline component has no external resources to check."
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
        """Execution of the relation extraction directly from existing candidate terms.
        Candidate terms are first converted into candidate relations.
        Then the candidate relations are converted into relations.
        The pipeline candidate terms are consumed.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline running.
        """

        concepts_labels_map = {}
        for concept in pipeline.kr.concepts:
            concepts_labels_map[concept.label] = concept

        candidate_relations = cts_to_crs(
            pipeline.candidate_terms,
            concepts_labels_map,
            pipeline.spacy_model,
            self.concept_max_distance,
            self.scope,
        )

        for cr in candidate_relations:
            pipeline.kr.relations.add(crs_to_relation({cr}))

        pipeline.candidate_terms = set()
