from typing import List

from term_enrichment.term_enrichment_schema import CandidateTerm
from term_enrichment.term_enrichment_repository import load_candidate_terms_from_file
from term_enrichment.term_enrichment_methods.wordnet_enrichment import WordNetTermEnrichment

from config.core import config


class TermEnrichment:

    def __init__(self, candidate_terms: List[CandidateTerm] = None) -> None:
        self.candidate_terms = candidate_terms
        self.wordnet_enrichment = None

        if self.candidate_terms is None:
            self.candidate_terms = load_candidate_terms_from_file()

    def wordnet_term_enrichment(self) -> None:
        lang = config["term_enrichment"]["wordnet"].get("lang")
        use_domains = config["term_enrichment"]["wordnet"].get("use_domains")
        use_pos = config["term_enrichment"]["wordnet"].get("use_pos")

        self.wordnet_enrichment = WordNetTermEnrichment(
            lang, use_domains, use_pos)

        self.wordnet_enrichment(self.candidate_terms)
