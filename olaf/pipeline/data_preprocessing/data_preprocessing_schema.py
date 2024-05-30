from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..pipeline_schema import Pipeline


class DataPreprocessing(ABC):
    """Component specific to the data preprocessing.
    The sequence of data preprocessing tasks should result in a corpus object,
    i.e., a List[spacy.tokens.doc.Doc].
    """

    def __init__(self) -> None:
        """Initialise DataPreprocessing instance."""

    @abstractmethod
    def run(self, pipeline: 'Pipeline') -> None:
        """Method that is responsible for the execution of the component.

        Parameters
        ----------
        pipeline : Pipeline
            The running pipeline.
            Type Any is used instead of Pipeline to avoid circular import.
        """
        ...
