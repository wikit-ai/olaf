from dataclasses import dataclass, field
from typing import Set


@dataclass
class CandidateTerm:
    value: str
    synonyms: Set[str] = field(default_factory=set)
    hypernyms: Set[str] = field(default_factory=set)
    hyponyms: Set[str] = field(default_factory=set)
    antonyms: Set[str] = field(default_factory=set)
