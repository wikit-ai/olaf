from abc import ABC, abstractmethod
from typing import Any


class DataPreprocessing(ABC):
    """Component specific to the data preprocessing.
        The sequence of data preprocessing tasks should result in a corpus object, i.e., a List[spacy.tokens.doc.Doc].
    """

    def __init__(self, config: Any) -> None:
        """Initialise DataPreprocessing instance.

        Parameters
        ----------
        parameters : Optional[Dict[str, Any]]
            Parameters are fixed values to be defined when building the pipeline. 
            They are necessary for the component functioning.
        options : Optional[Dict[str, Any]]
            Options are tunable parameters which will be updated to optimise the component performance.
        """
        self.config = config

    @abstractmethod
    def run(self, pipeline: Any) -> None:
        """Method that is responsible for the execution of the component.

        Parameters
        ----------
        pipeline : Pipeline
            The running pipeline.
            Type Any is used instead of Pipeline to avoid circular import.
        """
        ...