from abc import ABC, abstractmethod
from typing import Optional, Set
from uuid import uuid4

from .linguistic_realisation_schema import LinguisticRealisation


class DataContainer(ABC):
    """Data structure for Concept, Relation and Metarelation.

    Parameters
    ----------
    uid : str
        The unique identifier for the container.
    external_uids : Set[str], optional
        External unique identifiers found for the data container.
    label : str
        The human readable label for the data container.
    linguistic_realisations : Set[LinguisticRealisation]
        Instances or realisations of a concept or a relation in the text corpus.
    """

    def __init__(
        self,
        label: Optional[str] = None,
        external_uids: Optional[Set[str]] = None,
        linguistic_realisations: Optional[Set[LinguisticRealisation]] = None,
    ) -> None:
        """Initialise DataContainer instance.

        Parameters
        ----------
        label : str, optional
            The unique identifier for the container, by default None.
        external_uids : Set[str], optional
            External unique identifiers found for the data container, by default None.
        linguistic_realisations : Set[LinguisticRealisation], optional
            Instances or realisations of a concept or a relation in the text corpus,
            by default None.
        """
        self.uid = str(uuid4())
        self.external_uids = external_uids if external_uids is not None else set()
        self.label = label
        self.linguistic_realisations = (
            linguistic_realisations if linguistic_realisations else set()
        )

    @abstractmethod
    def add_linguistic_realisation(
        self, linguistic_realisation: LinguisticRealisation
    ) -> None:
        """Add a new linguistic realisation to the data container.

        Parameters
        ----------
        linguistic_realisation : LinguisticRealisation
            The linguistic realisation instance to add.
        """

    @abstractmethod
    def remove_linguistic_realisation(
        self, linguistic_realisation: LinguisticRealisation
    ) -> None:
        """Delete a linguistic realisation of the data container.

        Parameters
        ----------
        linguistic_realisation : LinguisticRealisation
            The linguistic realisation instance to remove.
        """
