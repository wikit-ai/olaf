from abc import ABC, abstractmethod
from typing import Any, Dict


class PipelineComponent(ABC):
    """A pipeline component is part of a pipeline performing one specific task.
    """

    def __init__(
        self,
        #*kwargs
    ) -> None:
        """Initialise PipelineComponent instance.
        """

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
