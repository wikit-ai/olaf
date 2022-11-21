import spacy.tokens
from typing import Any, Dict, List

from commons.ontology_learning_schema import KR
from concept_hierarchy.concept_hierarchy_methods.term_subsumption import TermSubsumption
from config.core import config
import config.logging_config as logging_config


class Concept_Hierarchy():
    """Step of hierarchisation between concepts.
    Feed meta relation of knowledge representation.
    """

    def __init__(self, corpus : List[spacy.tokens.doc.Doc], kr : KR, config: Dict[str, Any] = config['concept_hierarchy']) -> None:
        self.corpus = corpus
        self.kr = kr
        self.config = config

    def term_subsumption(self):
        """Find generalisation relations with term subsumption method.
        """

        try:
            threshold = self.config['term_subsumption']['threshold']
            use_lemma = self.config['tem_subsumption']['use_lemma']
        except KeyError as e:
            logging_config.logger.error(
                f"""Config information missing for Term subsumption. Make sure you provided the configuration fields:
                    - concept_hierarchy.term_subsumption.threshold
                    - concept_hierarchy.term_subsumption.use_lemma
                    Trace : {e}
                """)

        term_subsumption = TermSubsumption(self.corpus, self.kr, threshold,use_lemma)
        term_subsumption()
