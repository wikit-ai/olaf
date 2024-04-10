from typing import Optional, Set

from .data_container_schema import DataContainer
from .linguistic_realisation_schema import LinguisticRealisation


class Concept(DataContainer):
    """Concept is a particular DataContainer.
    It denotes a thing in the conceptual word and it is created by concept extraction processes.

    Parameters
    ----------
    uid : str
        The concept unique identifier.
    external_uids : Set[str]
        External unique identifiers found for the concept, by default None.
    label : str
        The concept human readable label.
    linguistic_realisations : Set[LinguisticRealisation]
        The concept linguistic realisations, i.e. instances of the concept in the text corpus,
        by default None.
    """

    def __init__(
        self,
        label: str,
        external_uids: Optional[Set[str]] = None,
        linguistic_realisations: Optional[Set[LinguisticRealisation]] = None,
    ) -> None:
        """Initialise concept instance.

        Parameters
        ----------
        label : str
            The concept human readable label.
        external_uids : Set[str], optional
            External unique identifiers found for the concept, by default None.
        linguistic_realisations : Set[LinguisticRealisation], optional
            The concept linguistic realisations, i.e. instances of the concept in the text corpus,
            by default None.
        """
        super().__init__(
            label=label,
            external_uids=external_uids,
            linguistic_realisations=linguistic_realisations,
        )

    def add_linguistic_realisation(
        self, linguistic_realisation: LinguisticRealisation
    ) -> None:
        """Add a new linguistic realisation to the concept.

        Parameters
        ----------
        linguistic_realisation : LinguisticRealisation
            The linguistic realisation instance to add.
        """

        existing_lr = False
        for lr in self.linguistic_realisations:
            if lr.label == linguistic_realisation.label:
                lr.add_corpus_occurrences(linguistic_realisation.corpus_occurrences)
                existing_lr = True
                break
        if not (existing_lr):
            self.linguistic_realisations.add(linguistic_realisation)

    def remove_linguistic_realisation(
        self, linguistic_realisation: LinguisticRealisation
    ) -> None:
        """Delete a linguistic realisation of the concept.

        Parameters
        ----------
        linguistic_realisation : LinguisticRealisation
            The linguistic realisation instance to remove.
        """
        self.linguistic_realisations.remove(linguistic_realisation)
