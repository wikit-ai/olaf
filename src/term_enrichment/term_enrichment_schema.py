from dataclasses import dataclass, field
from typing import Set


@dataclass
class CandidateTerm:
    """A Dataclass to contain Candidate terms informations.

    Parameters
    ----------
    value: str
        The candidate term string
    synonyms: Set[str]
        The candidate term sysnonyms strings
    hypernyms: Set[str]
        The candidate term hypernyms strings
    hyponyms: Set[str]
        The candidate term hyponyms strings
    antonyms: Set[str]
        The candidate term antonyms strings
    """
    value: str
    synonyms: Set[str] = field(default_factory=set)
    hypernyms: Set[str] = field(default_factory=set)
    hyponyms: Set[str] = field(default_factory=set)
    antonyms: Set[str] = field(default_factory=set)
