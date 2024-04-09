from typing import Any, Dict, Optional

from ....commons.candidate_term_tools import cts_to_concept
from ....commons.logging_config import logger
from ..pipeline_component_schema import PipelineComponent


class CTsToConceptExtraction(PipelineComponent):
    """A pipeline component to create concepts directly from the candidate terms.

    Attributes
    ----------
    parameters: Dict[str, Any]
        Parameters are fixed values to be defined when building the pipeline.
        They are necessary for the component functioning.
    options: Dict[str, Any]
        Options are tunable parameters which will be updated to optimise the component performance.
    """

    def __init__(
        self,
        parameters: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise CTsToConceptExtraction pipeline component instance.

        Parameters
        ----------
        parameters : Dict[str, Any], optional
            Parameters are fixed values to be defined when building the pipeline.
            They are necessary for the component functioning, by default None.
        options : Dict[str, Any], optional
            Options are tunable parameters which will be updated to optimise the
            component performance, by default None.
        """
        super().__init__(parameters, options)

    def optimise(self) -> None:
        """A method to optimise the pipeline component by tuning the options."""
        raise NotImplementedError

    def _check_resources(self) -> None:
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

    def run(self, pipeline: Any) -> None:
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
