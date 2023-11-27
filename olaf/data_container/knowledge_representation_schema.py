from dataclasses import dataclass, field
from typing import Set

from .concept_schema import Concept
from .relation_schema import Relation
from .metarelation_schema import Metarelation


@dataclass
class KnowledgeRepresentation:
    """The knowledge representation is the data structure that contains the information learned from the text corpus. 
    It is composed of concepts, relations and metarelations. 

    Parameters
    ----------
    concepts : Set[Concept]
        The set of concepts under interest. 
        Empty set by default if it is initialised without concept.
    relations : Set[Relation]
        The set of relations under interest.
        Empty set by default if it is initialised without relation.
    metarelations : Set[Metarelation]
        The set of metarelations under interest.
        Empty set by default if it is initialised without metarelation.

    """
    concepts: Set[Concept] = field(default_factory=set)
    relations: Set[Relation] = field(default_factory=set)
    metarelations: Set[Metarelation] = field(default_factory=set)
