from typing import List

from commons.ontology_learning_schema import CandidateTerm, KR
from concept_extraction.concept_extraction_methods.group_by_synonyms import GroupBySynonyms

class ConceptExtraction():

    def __init__(self, candidate_terms: List[CandidateTerm], kr: KR = KR()) -> None:
        self.candidate_terms = candidate_terms
        self.kr = kr

    def group_by_synonyms(self):
        group_by_syn = GroupBySynonyms(self.candidate_terms, self.kr)
        group_by_syn()