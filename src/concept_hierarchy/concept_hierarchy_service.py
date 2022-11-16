import spacy.tokens
from typing import List

from concept_hierarchy.concept_hierarchy_schema import KR
from concept_hierarchy.concept_hierarchy_methods.term_subsumption import TermSubsumption

class Concept_Hierarchy():
    """Step of hierarchisation between concepts.
    Feed meta relation of knowledge representation.
    """

    def __init__(self, corpus : List[spacy.tokens.doc.Doc], kr : KR) -> None:
        self.corpus = corpus
        self.kr = kr

    def term_subsumption(self):
        """Find generalisation relations with term subsumption method.
        """
        generalisation_relations = TermSubsumption(self.corpus,self.kr)
        generalisation_relations.term_subsumption()
