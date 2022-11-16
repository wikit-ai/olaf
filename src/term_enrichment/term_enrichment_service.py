from typing import List, Dict, Any

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
    """

    def __init__(self, candidate_terms: List[CandidateTerm] = None, config: Dict[str, Any] = config['term_enrichment']) -> None:
        """_summary_

        Parameters
        ----------
        candidate_terms : List[CandidateTerm], optional
            The candidate terms to enrich, by default None
        """
        self.candidate_terms = candidate_terms
        self.config = config

        if self.candidate_terms is None:
            self.candidate_terms = load_candidate_terms_from_file()

    def wordnet_term_enrichment(self) -> None:
        """The method to enirch the candidate terms using wordnet term enricher.
        """

        try:
            wordnet_enricher_options = self.config['wordnet']
        except KeyError as key_error_exception:
            logging_config.logger.error(
                f"No configuration found for WordNet term enricher. Trace: {key_error_exception}.")

        lang = wordnet_enricher_options.get("lang")
        use_domains = wordnet_enricher_options.get("use_domains")
        use_pos = wordnet_enricher_options.get("use_pos")

        try:
            wordnet_term_enricher = WordNetTermEnrichment(wordnet_enricher_options,
                                                          lang, use_domains, use_pos)
        except Exception as e:
            logging_config.logger.error(
                f"Could not setup attribute wordnet_term_enricher. Trace : {e}")
        else:
            logging_config.logger.info(
                f"Attribute wordnet_term_enricher initialized.")

        wordnet_term_enricher(self.candidate_terms)
