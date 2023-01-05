from typing import Any, Dict, List

import spacy.language

from commons.ontology_learning_schema import CandidateTerm
import config.logging_config as logging_config

class EmbeddingEnrichment:
    """A class to handle embedding based term enrichment.

    Attributes
    ----------
    candidate_terms: List[CandidateTerm]
        List of candidate terms to process.
    options: Dict[str, Any]
        The parameters to setup the class. 
    nlp_model: 
        Language model used to represente candidate terms value and find synonyms from similarity. 
    """

    def __init__(self, candidate_terms: List[CandidateTerm], options: Dict[str, Any], nlp_model: spacy.language.Language = None) -> None:
        """Initializer for a embedding based term enrichment process.

        Parameters
        ----------
        candidate_terms: List[CandidateTerm]
            List of candidate terms to process.
        options: Dict[str, Any]
            The specific parameters to setup the class.
        nlp_model: 
            Spacy language model used to represente candidate terms value and find synonyms from similarity. Default to None if another model is wanted. 
        """
        self.candidate_terms = candidate_terms
        self.options = options
        self.nlp_model = None
        try:
            assert self.options["model"] is not None
            assert self.options["similarity_threshold"] is not None
        except AssertionError as e:
            logging_config.logger.error(
                f"""Config information missing for embedding term enrichment. Make sure you provided the configuration fields:
                    - term_enrichment.embedding.model
                    - term_enrichment.embedding.similarity_threshold
                    Trace : {e}
                """)
        if self.options.get("model").lower() == "spacy":
            self.nlp_model = nlp_model
        else : 
            logging_config.logger.error (
                f"""Error in config information for term enrichment embedding model. Make sure you provided one of this value in the config file : 
                    - spacy.
                """
            )
       

    def enrich_candidate_term(self, candidate_term: CandidateTerm) -> None:
        """Enrich a candidate term from its value.
        Most similar words in known vocabulary are found with similarity measure. 

        Parameters
        ----------
        candidate_term : CandidateTerm
            The candidate term to enrich. 
        """
        similarity_threshold = self.options.get("similarity_threshold")

        if self.options.get("model").lower() == "spacy" :
            if self.nlp_model.vocab.has_vector(candidate_term.value):
                word_embedding = self.nlp_model(candidate_term.value)
                synonyms = []
                for word in self.nlp_model.vocab : 
                    if ((word_embedding.similarity(word) > similarity_threshold) and not(word.lower_ == candidate_term.value)):
                        synonyms.append(word.lower_)
                candidate_term.synonyms.update(synonyms)
         

    def enrich_candidate_terms(self) -> None:
        """Method that enriches each candidate term in the list one by one. 
        """
        for candidate_term in self.candidate_terms:
            self.enrich_candidate_term(candidate_term)