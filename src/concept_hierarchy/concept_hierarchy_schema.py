from dataclasses import dataclass
from typing import Literal, Set

@dataclass
class RepresentativeTerm:
    """Dataclass which represents the most representative string of a given concept.

    Parameters
    ----------
    value : str
        String most representative of a concept.
    concept_id : str
        UID of the given concept.
    """
    value: str
    concept_id: str

@dataclass
class Concept:
    """Dataclass that contains concept information.

    Parameters
    ----------
    uid : str
        Concept unique id.
    terms : Set(str)
        Set of terms that is to say strings that represent the concept.
    """
    uid : str
    terms : Set(str)

@dataclass
class Relation:
    """Dataclass that contains relation information.

    Parameters
    ----------
    uid : str
        Relation unique id.
    source : str
        Id of the relationship source.
    destination : str
        Id of the relationship destination.
    terms : Set(str)
        Set of terms that is to say strings that represent the concept.
    """
    uid : str
    source : str
    destination : str
    terms : Set(str)

MetaRelationType = Literal["generalisation"]

@dataclass
class MetaRelation:
    """Dataclass that contains meta-relation information.

    Parameters
    ----------
    uid : str
        Meta relation unique id.
    source : str
        Id of the meta relationship source.
    destination : str
        Id of the meta relationship destination.
    relation_type : MetaRelationType
        Type of the meta relation.
    """
    uid : str
    source : str
    destination : str
    relation_type : MetaRelationType

@dataclass
class KR:
    """Dataclass that contains knowledge representation information.
    concepts : Set(Concept)
        Concepts contained in the knowledge representation.
    relations : Set(Relation)
        Relations contained in the knowledge representation.
    meta_relations : Set(MetaRelation)
        Meta relations contained in the knowlegde representation.
    """
    concepts = Set(Concept)
    relations = Set(Relation)
    meta_relation = Set(MetaRelation)