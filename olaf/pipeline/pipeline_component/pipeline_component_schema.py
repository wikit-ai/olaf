from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class PipelineComponent(ABC):
    """A pipeline component is part of a pipeline performing one specific task.

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
        """Initialise PipelineComponent instance.

        Parameters
        ----------
        parameters : Optional[Dict[str, Any]]
            Parameters are fixed values to be defined when building the pipeline.
            They are necessary for the component functioning.
        options : Optional[Dict[str, Any]]
            Options are tunable parameters which will be updated to optimise the
            component performance.
        """

        self.parameters = parameters if parameters else dict()
        self.options = options if options else dict()

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

    @abstractmethod
    def run(self, pipeline: Any) -> None:
        """Method that is responsible for the execution of the component.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline running
        """
