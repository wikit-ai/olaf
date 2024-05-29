from typing import Any, Dict

from ...pipeline_schema import Pipeline
from ....commons.candidate_term_tools import cts_to_concept
from ....commons.logging_config import logger
from ..pipeline_component_schema import PipelineComponent


class CTsToConceptExtraction(PipelineComponent):
    """A pipeline component to create concepts directly from the candidate terms."""

    def __init__(self) -> None:
        """Initialise CTsToConceptExtraction pipeline component instance."""

    def optimise(self) -> None:
        """A method to optimise the pipeline component by tuning the options."""
        raise NotImplementedError

    def check_resources(self) -> None:
        """Method to check that the component has access to all its required resources."""
        logger.info(
            "Candidate term to concept extraction pipeline component has no external resources to check."
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

    def run(self, pipeline: Pipeline) -> None:
        """Execution of the concept extraction directly from existing candidate terms.
        The pipeline candidate terms are consumed.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline running.
        """

        for ct in pipeline.candidate_terms:
            pipeline.kr.concepts.add(cts_to_concept({ct}))

        pipeline.candidate_terms = set()
