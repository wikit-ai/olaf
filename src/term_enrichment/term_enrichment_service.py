from typing import List

from term_enrichment.term_enrichment_schema import CandidateTerm
from term_enrichment.term_enrichment_repository import load_candidate_terms_from_file


class TermEnrichment:

    def __init__(self, candidate_terms: List[CandidateTerm] = None) -> None:
        self.candidate_terms = candidate_terms

        if self.candidate_terms is None:
            self.candidate_terms = load_candidate_terms_from_file()
