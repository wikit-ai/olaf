from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set

from ....commons.errors import NotCallableError
from ....data_container.candidate_term_schema import CandidateTerm
from ..pipeline_component_schema import PipelineComponent


class TermExtractionPipelineComponent(PipelineComponent):
    """A pipeline component schema for term extraction tasks.

    Attributes
    ----------
    cts_post_processing_functions: Optional[List[Callable[[Set[CandidateTerm]], Set[CandidateTerm]]]]
        A list of candidate term post processing functions to run after candidate term extraction
        and before assigning the extracted candidate terms to the pipeline.
    parameters: Dict[str, Any]
        Parameters are fixed values to be defined when building the pipeline.
        They are necessary for the component functioning.
    options: Dict[str, Any]
        Options are tunable parameters which will be updated to optimise the component performance.
    """

    def __init__(
        self,
        cts_post_processing_functions: Optional[
            List[Callable[[Set[CandidateTerm]], Set[CandidateTerm]]]
        ] = None,
        parameters: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise CandidateTermPipelineComponent instance.

        Parameters
        ----------
        cts_post_processing_functions: Optional[List[Callable[[Set[CandidateTerm]], Set[CandidateTerm]]]]
            A list of candidate term post processing functions to after candidate term extraction
            and before assigning the extracted candidate terms to the pipeline.
        parameters : Optional[Dict[str, Any]]
            Parameters are fixed values to be defined when building the pipeline.
            They are necessary for the component functioning.
        options : Optional[Dict[str, Any]]
            Options are tunable parameters which will be updated to optimise the
            component performance.
        """
        super().__init__(parameters, options)

        self.cts_post_processing_functions = cts_post_processing_functions

        if self.cts_post_processing_functions is not None:
            for post_processing_function in self.cts_post_processing_functions:
                if not callable(post_processing_function):
                    raise NotCallableError(
                        function_name=post_processing_function.__str__()
                    )

    @abstractmethod
    def _check_resources(self) -> None:
        """Method to check that the component has access to all its required resources."""

    @abstractmethod
    def optimise(self) -> None:
        """A method to optimise the pipeline component by tuning the options."""

    @abstractmethod
    def _compute_metrics(self) -> None:
        """A method to compute component performance metrics. It is used by the optimise
        method to update the options.
        """

    @abstractmethod
    def get_performance_report(self) -> Dict[str, Any]:
        """A getter for the pipeline component performance report.
            If the component has been optimised, it only returns the best performance.
            Otherwise, it returns the results obtained with the set parameters.

        Returns
        -------
        Dict[str, Any]
            The pipeline component performance report.
        """

    def apply_post_processing(
        self, candidate_terms: Set[CandidateTerm]
    ) -> Set[CandidateTerm]:
        """Apply candidate terms post processing functions.

        Parameters
        ----------
        candidate_terms : Set[CandidateTerm]
            The set of candidate terms to post process.

        Returns
        -------
        Set[CandidateTerm]
            The post processed set of candidate terms.
        """
        if self.cts_post_processing_functions is not None:
            for post_processing_func in self.cts_post_processing_functions:
                candidate_terms = post_processing_func(candidate_terms)
        return candidate_terms

    @abstractmethod
    def run(self, pipeline: Any) -> None:
        """Method that is responsible for the execution of the component.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline running
        """
