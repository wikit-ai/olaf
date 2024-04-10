from typing import Optional, Set

from .concept_schema import Concept
from .data_container_schema import DataContainer
from .linguistic_realisation_schema import LinguisticRealisation


class Relation(DataContainer):
    """A Relation is the explicit link between two concepts which is the result of an action or
    a state and which could itself be a concept.
    It is oriented and has a label.
    The relation is defined by its triple (source, label, destination).

    Parameters
    ----------
    label : str
        The relation human readable label
    source_concept : Concept, optional
        The source concept in the relation triple, by default None.
    destination_concept : Concept, optional
        The destination concept in the relation triple, by default None.
    external_uids : Set[str]
        External unique identifiers found for the relation, by default None.
    linguistic_realisations : Set[LinguisticRealisation]
        The relation linguistic realisations, i.e. instances of the relation in the text corpus,
        by default None.
    """

    def __init__(
        self,
        label: str,
        source_concept: Optional[Concept] = None,
        destination_concept: Optional[Concept] = None,
        external_uids: Optional[Set[str]] = None,
        linguistic_realisations: Optional[Set[LinguisticRealisation]] = None,
    ) -> None:
        """Initialise Relation instance.

        Parameters
        ----------
        label : str
            The relation human readable label.
        source_concept : Concept, optional
            The source concept in the relation triple, by default None.
        destination_concept : Concept, optional
            The destination concept in the relation triple, by default None.
        external_uids : Set[str], optional
            External unique identifiers found for the relation., by default None.
        linguistic_realisations : Set[LinguisticRealisation], optional
            The relation linguistic realisations, by default None.
        """
        super().__init__(
            label=label,
            external_uids=external_uids,
            linguistic_realisations=linguistic_realisations,
        )
        self.source_concept = source_concept
        self.destination_concept = destination_concept

    def add_linguistic_realisation(
        self, linguistic_realisation: LinguisticRealisation
    ) -> None:
        """Add a new linguistic realisation to the relation.

        Parameters
        ----------
        linguistic_realisation : LinguisticRealisation
            The linguistic realisation instance to add.
        """
        self.linguistic_realisations.add(linguistic_realisation)

    def add_linguistic_realisations(
        self, linguistic_realisations: Set[LinguisticRealisation]
    ) -> None:
        """Add new linguistic realisations to the relation.

        Parameters
        ----------
        linguistic_realisations : Set[LinguisticRealisation]
            The set of linguistic realisation instances to add.
        """
        self.linguistic_realisations.update(linguistic_realisations)

    def remove_linguistic_realisation(
        self, linguistic_realisation: LinguisticRealisation
    ) -> None:
        """Delete a linguistic realisation of the relation.

        Parameters
        ----------
        linguistic_realisation : LinguisticRealisation
            The LinguisticRealisation instance to remove.
        """
        self.linguistic_realisations.remove(linguistic_realisation)
