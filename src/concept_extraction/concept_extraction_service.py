from typing import Any, Dict, List

from commons.ontology_learning_schema import CandidateTerm, KR
from concept_extraction.concept_extraction_methods.group_by_synonyms import GroupBySynonyms
from config.core import config


class ConceptExtraction():

    def __init__(self, candidate_terms: List[CandidateTerm], config: Dict[str, Any] = config['concept_extraction']) -> None:
        self.candidate_terms = candidate_terms
        self.kr = KR()
        self.config = config

    def group_by_synonyms(self) -> None:
        """This method merges candidate terms in a concept if they have value or synonyms in common.
        """
        group_by_syn = GroupBySynonyms(self.candidate_terms, self.kr)
        group_by_syn()
