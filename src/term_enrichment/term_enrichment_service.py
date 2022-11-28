from typing import List, Dict, Any

from commons.ontology_learning_schema import CandidateTerm
from term_enrichment.term_enrichment_repository import load_candidate_terms_from_file
from term_enrichment.term_enrichment_methods.wordnet_enrichment import WordNetTermEnrichment
from term_enrichment.term_enrichment_methods.conceptnet_enrichment import ConceptNetTermEnrichment

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
        config: Dict[str, Any]
            The configuration details.
        """
        self.config = config

        if self.config.get("load_candidate_terms_from_file"):
            self.candidate_terms = load_candidate_terms_from_file()
        else:
            self.candidate_terms = candidate_terms

        try:
            assert self.candidate_terms is not None
        except AssertionError as e:
            logging_config.logger.error(
                f"""No candidate terms could be load.
                    Either provide a list of candidate terms as input.
                    Or set the config parameter `term_enrichment.load_candidate_terms_from_file = true` 
                    and provide the adequate file path by setting config parameter `term_enrichment.candidate_terms_path = path/to/your/file.txt`   
                    Trace : {e}
                """)

    def wordnet_term_enrichment(self) -> None:
        """The method to enirch the candidate terms using wordnet term enricher.
        """

        try:
            wordnet_enricher_options = self.config['wordnet']
        except KeyError as key_error_exception:
            logging_config.logger.error(
                f"No configuration found for WordNet term enricher. Trace: {key_error_exception}.")

        try:
            wordnet_term_enricher = WordNetTermEnrichment(
                wordnet_enricher_options)
        except Exception as e:
            logging_config.logger.error(
                f"Could not setup attribute wordnet_term_enricher. Trace : {e}")
        else:
            logging_config.logger.info(
                f"Attribute wordnet_term_enricher initialized.")

        wordnet_term_enricher(self.candidate_terms)

    def conceptnet_term_enrichment(self) -> None:
        """The method to enirch the candidate terms using conceptnet term enricher.
        """

        try:
            conceptnet_enricher_options = self.config['conceptnet']
        except KeyError as key_error_exception:
            logging_config.logger.error(
                f"No configuration found for ConceptNet term enricher. Trace: {key_error_exception}.")

        try:
            conceptnet_term_enricher = ConceptNetTermEnrichment(
                conceptnet_enricher_options)
        except Exception as e:
            logging_config.logger.error(
                f"Could not setup attribute conceptnet_term_enricher. Trace : {e}")
        else:
            logging_config.logger.info(
                f"Attribute conceptnet_term_enricher initialized.")

        conceptnet_term_enricher(self.candidate_terms)
