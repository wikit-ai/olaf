from typing import Any, Dict, List

from commons.ontology_learning_schema import CandidateTerm, KR
from concept_extraction.concept_extraction_methods.conceptnet_concept_extraction import ConceptNetConceptExtraction
from concept_extraction.concept_extraction_methods.group_by_synonyms import GroupBySynonyms
from config.core import config
import config.logging_config as logging_config


class ConceptExtraction:

    def __init__(self, candidate_terms: List[CandidateTerm], configuration: Dict[str, Any] = None) -> None:
        """Main class for concept extraction.

        Parameters
        ----------
        candidate_terms : List[CandidateTerm]
            The list of candidate terms to extract the concepts from.
        config : Dict[str, Any], optional
            The concept extraction configuration details, by default config['concept_extraction']
        """

        if configuration is None :
            self.config = config['concept_extraction']
        else :
            self.config = configuration
        self.candidate_terms = candidate_terms
        self.kr = KR()

    def conceptnet_concept_extraction(self) -> None:
        """Main class for concept extraction.

        Returns
        -------
        KR
            The Knowledge Representation object containing the extracted concepts.
        """
        try:
            assert self.config["conceptnet"] is not None
        except AssertionError as e:
            logging_config.logger.error(
                f"""Config information missing or wrong for ConceptNet extraction. Make sure you provided the right configuration fields:
                    - concept_extraction.conceptnet
                    Trace : {e}
                """)

        conceptnet_extraction = ConceptNetConceptExtraction(
            self.candidate_terms, self.kr, options=self.config["conceptnet"])

        conceptnet_extraction()

    def group_by_synonyms(self) -> None:
        """This method merges candidate terms in a concept if they have value or synonyms in common.
        """
        group_by_syn = GroupBySynonyms(self.candidate_terms, self.kr)
        group_by_syn()
