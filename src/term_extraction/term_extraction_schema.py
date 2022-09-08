from dataclasses import dataclass
from typing import Any, Callable, List


TokenSequenceFilter = Callable[[List[Any]], List[List[str]]]


@dataclass
class CValueResults:
    """A dataclass to contain the C-value information
    """
    c_value: float
    candidate_term: str


@dataclass
class CandidateTermStatTriple:
    """
    This dataclass will store information about the statistical triples from the algorithm in <https://doi.org/10.1007/s007999900023> (section 2.3, page 4):
    For every string a, that is extracted as a candidate term, we create for each of its substrings b, a triple (f(b), t(b), c(b)), 
    where f(b) is the total frequency of b in the corpus, t(b) is the frequency of b as a nested string of candidate terms, c(b) is the 
    number of these longer candidate terms.

    When an instance is created, the count of longer candidate term is initialized to 1 according to the algorithm.
    """
    candidate_term: str
    substring: str
    substring_corpus_frequency: int = 0
    substring_nested_frequency: int = 0
    count_longer_terms: int = 1
