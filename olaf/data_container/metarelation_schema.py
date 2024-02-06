from typing import Optional, Set

from rdflib import RDFS

from .concept_schema import Concept
from .linguistic_realisation_schema import LinguisticRealisation
from .relation_schema import Relation

METARELATION_RDFS_OWL_MAP = {
    "is_generalised_by": RDFS.subClassOf
}


class Metarelation(Relation):
    """We distinguish between Relations and Metarelations.
    A Metarelation is a link between concepts that is not an action or a state,
    it is implicitly expressed thanks to syntax or particular formulations.
    Metarelations can correspond (but are not restricted to) to the common taxonomic
    hierarchical relations in  broad sense, i.e., including generic-specific, instance,
    and whole-part.
    It is a Relation and as such is oriented and its label is its type.
    The metarelation is defined by its triple (source, label, destination).

    Parameters
    ----------
    uid : str
        The metarelation unique identifier.
    source_concept : Concept
        The source concept in the metarelation triple.
    destination_concept : Concept
        The destination concept in the metarelation triple.
    label : METARELATION_TYPE
        The metarelation type.
    external_uids : Set[str], optional
        An external unique identifier for the metarelation, by default set().
    linguistic_realisations : Set[LinguisticRealisation]
        The metarelation linguistic realisations, i.e. instances of the metarelation in the
        text corpus, by default None.
    """

    def __init__(
        self,
        source_concept: Concept,
        destination_concept: Concept,
        label: str,
        external_uids: Set[Optional[str]] = None,
        linguistic_realisations: Optional[Set[LinguisticRealisation]] = None,
    ) -> None:
        """Initialise MetaRelation instance.

        Parameters
        ----------
        source_concept : Concept
            The source concept in the metarelation triple.
        destination_concept : Concept
            The destination concept in the metarelation triple.
        label : METARELATION_TYPE
            The metarelation type.
        external_uids: Set[Optional[str]], optional
            An external unique identifier for the metarelation, by default set().
        linguistic_realisations : Optional[Set[LinguisticRealisation]]
            The metarelation linguistic realisations, by default None.
        """
        super().__init__(
            source_concept=source_concept,
            destination_concept=destination_concept,
            label=label,
            external_uids=external_uids,
            linguistic_realisations=linguistic_realisations,
        )
