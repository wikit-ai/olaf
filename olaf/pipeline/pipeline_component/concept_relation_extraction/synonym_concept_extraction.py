from typing import Any, Dict

from ....commons.candidate_term_tools import cts_to_concept, group_cts_on_synonyms
from ....commons.logging_config import logger
from ..pipeline_component_schema import PipelineComponent

class SynonymConceptExtraction(PipelineComponent):
    """Extract concepts based on synonyms grouping.

    Attributes
    ----------
    parameters: Dict[str, Any]
        Parameters are fixed values to be defined when building the pipeline. 
        This component do not need parameters to run.
    options: Dict[str, Any]
        Options are tunable parameters which will be updated to optimise the component performance.
        This component has no options to optimise.
    """

    def __init__(self, parameters: Dict[str, Any] = None, options: Dict[str, Any] = None) -> None:
        """Initialise synonym grouping concept extraction instance.

        Parameters
        ----------
        parameters : Dict[str, Any], optional
            Parameters used to configure the component, by default None.
        options : Dict[str, Any], optional
            Tunable options to use to optimise the component performance, by default None.
        """
        super().__init__(parameters, options)

    def optimise(self) -> None:
        """A method to optimise the pipeline component by tuning the options.
        """
        logger.info("Synonym grouping concept extraction pipeline component cannot be optimise.")
    
    def _check_resources(self) -> None:
        """Method to check that the component has access to all its required resources.
        """
        logger.info("Synonym grouping concept extraction pipeline component has no external resources to check.")

    def _compute_metrics(self) -> None:
        """A method to compute component performance metrics.
        """
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
        """Execution of the synonyms grouping for concept extraction on candidate terms. 
        Concepts are created and candidate terms are purged.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline running.
        """

        concept_candidates = group_cts_on_synonyms(pipeline.candidate_terms)

        for concept_candidate in concept_candidates :
            new_concept = cts_to_concept(concept_candidate)
            pipeline.kr.concepts.add(new_concept)

        pipeline.candidate_terms = set()
            