from dataclasses import dataclass, field
from typing import Set, Dict
from collections import defaultdict


@dataclass
class CandidateTerm:
    value: str
    enriching_terms: Set[str] = field(default_factory=set)
    source_ids: Dict[str, Set[str]] = field(default_factory=defaultdict(set))
