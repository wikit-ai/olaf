from dataclasses import dataclass


class DocAttributeNotFound(Exception):
    """An Exception to flag when a custom attribute on a Spacy Doc has not been found.
    """
    pass


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
    where f(b) is the total occurenc of b in the corpus, t(b) is the occurence of b as a nested string of candidate terms, c(b) is the 
    number of these longer candidate terms.

    When an instance is created, the count of longer candidate term is initialized to 1 according to the algorithm.
    """
    candidate_term: str
    substring: str
    substring_corpus_occurence: int = 0
    substring_nested_occurence: int = 0
    count_longer_terms: int = 1
