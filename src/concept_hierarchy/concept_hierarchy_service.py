import spacy.tokens
from typing import List

class Concept_Hierarchy():

    def __init__(self, corpus : List[spacy.tokens.doc.Doc]) -> None:
        self.corpus = corpus

    def term_subsumption(self):
        return 
