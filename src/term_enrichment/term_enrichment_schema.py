from dataclasses import dataclass, field
from typing import Any,  Dict, List, Optional


@dataclass
class ConceptNetEdgeData:
    """A dataclass to old conceptnet edges data useful for term enrichment

    Attributes
    ----------
    edge_rel_id: str
        The edge realtion ID.
    end_node_concept_id: str
        The concept ID the edge is pointing to.
    start_node_concept_id: str
        The source edge concept ID.
    end_node_label: str
        The concept label the edge is pointing to.
    end_node_lang: str
        The concept language the edge is pointing to.
    end_node_sense_label: Optional[str]
        The concept sense label (from Wordnet) the edge is pointing to.
    start_node_label: str
        The source edge concept label.
    start_node_lang: str
        The source edge concept language.
    start_node_sense_label: Optional[str]
        The source edge concept sense label (from Wordnet).
    """
    edge_rel_id: str
    end_node_concept_id: str
    start_node_concept_id: str

    end_node_label: str
    end_node_lang: str
    end_node_sense_label: Optional[str]

    start_node_label: str
    start_node_lang: str
    start_node_sense_label: Optional[str]


@dataclass
class ConceptNetTermData:
    """A dataclass to old conceptnet data useful for term enrichment

    Attributes
    ----------
    conceptnet_id: str
        The conceptnet term ID (relative URi)
    synonym_edges: List[ConceptNetEdgeData]
        The conceptnet "/r/Synonym" edge objects 
    isa_edges: List[ConceptNetEdgeData]
        The conceptnet "/r/IsA" edge objects
    formof_edges: List[ConceptNetEdgeData]    
        The conceptnet "/r/FormOf" edge objects
    antonym_edges: List[ConceptNetEdgeData]
        The conceptnet "/r/Antonym" edge objects
    """
    conceptnet_id: str
    synonym_edges: List[ConceptNetEdgeData] = field(default_factory=list)
    isa_edges: List[ConceptNetEdgeData] = field(default_factory=list)
    formof_edges: List[ConceptNetEdgeData] = field(default_factory=list)
    antonym_edges: List[ConceptNetEdgeData] = field(default_factory=list)
