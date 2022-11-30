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
            assert self.config['term_subsumption']['algo_type'] is not None
            assert self.config['term_subsumption']['subsumption_threshold'] is not None
            assert self.config['tem_subsumption']['use_lemma'] is not None
            assert self.config['use_span'] is not None
            if self.config['term_subsumption']['algo_type'] == "MEAN":
                assert self.config['term_subsumption']['mean']['high_threshold'] is not None
                assert self.config['term_subsumption']['mean']['low_threshold'] is not None
        except KeyError as e:
            logging_config.logger.error(
                f"""Config information missing for Term subsumption. Make sure you provided the configuration fields:
                    - concept_hierarchy.use_span
                    - concept_hierarchy.term_subsumption.subsumption_threshold
                    - concept_hierarchy.term_subsumption.use_lemma
                    - concept_hierarchy.term_subsumption.algo_type.
                    If you set algo_type to "mean", make sure you provided the configuration fields : 
                    - concept_hierarchy.term_subsumption.mean_high_threshold
                    - concept_hierarchy.term_subsumption.mean_low_threshold
                    Trace : {e}.
                """)
        else : 
            term_sub_options = {
                "algo_type": self.config['term_subsumption']['algo_type'],
                "subsumption_threshold": self.config['term_subsumption']['subsumption_threshold'],
                "use_lemma": self.config['tem_subsumption']['use_lemma'],
                "use_span": self.config['use_span']
            }
            if term_sub_options["algo_type"] == "MEAN":
                term_sub_options["mean_high_threshold"] = self.config['term_subsumption']['mean']['high_threshold']
                term_sub_options["mean_low_threshold"] = self.config['term_subsumption']['mean']['low_threshold']
            term_subsumption = TermSubsumption(self.corpus, self.kr, term_sub_options)
            term_subsumption()
