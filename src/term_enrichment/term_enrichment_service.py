from typing import List

from term_enrichment.term_enrichment_schema import CandidateTerm
from term_enrichment.term_enrichment_repository import load_candidate_terms_from_file
from term_enrichment.term_enrichment_methods.wordnet_enrichment import WordNetTermEnrichment

from config.core import config
import config.logging_config as logging_config


class TermEnrichment:
    """A Class to old all implemented term enrichment methods.

    Attributes
    ----------
    candidate_terms: List[CandidateTerm]
        The candidate terms to enrich
    wordnet_term_enricher: WordNetTermEnrichment
        The instance of WordNet Enricher
    """

    def __init__(self, candidate_terms: List[CandidateTerm] = None) -> None:
        """_summary_

        Parameters
        ----------
        candidate_terms : List[CandidateTerm], optional
            The candidate terms to enrich, by default None
        """
        self.candidate_terms = candidate_terms
        self.wordnet_term_enricher = None

        if self.candidate_terms is None:
            self.candidate_terms = load_candidate_terms_from_file()

    def _set_wordnet_term_enricher(self) -> None:
        """A private method to setup the WordNetTermEnrichment instance based on the configuration file.
            Sets attribute self.wordnet_term_enricher
        """
        lang = config["term_enrichment"]["wordnet"].get("lang")
        use_domains = config["term_enrichment"]["wordnet"].get("use_domains")
        use_pos = config["term_enrichment"]["wordnet"].get("use_pos")

        try:
            self.wordnet_term_enricher = WordNetTermEnrichment(
                lang, use_domains, use_pos)
        except Exception as e:
            logging_config.logger.error(
                f"Could not setup attribute wordnet_term_enricher. Trace : {e}")
        else:
            logging_config.logger.info(
                f"Attribute wordnet_term_enricher initialized.")

    def wordnet_term_enrichment(self) -> None:
        """The method to enirch the candidate terms using wordnet term enricher.
        """

        if self.wordnet_term_enricher is None:
            self._set_wordnet_term_enricher()

        self.wordnet_term_enricher(self.candidate_terms)
