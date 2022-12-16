import spacy.tokens
from typing import Any, Dict, List

from commons.ontology_learning_schema import KR
from config.core import config
from config import logging_config
from relation_extraction.relation_extraction_methods.on_cooc_with_sep_term_relation_extraction import OnCoocWithSepTermMetaRelationExtraction
from relation_extraction.relation_extraction_methods.on_occurrence_relation_extraction import OnOccurrenceRelationExtraction
from relation_extraction.relation_extraction_methods.on_pos_relation_extraction import OnPosRelationExtraction


class RelationExtraction():

    def __init__(self, corpus: List[spacy.tokens.doc.Doc], kr: KR, configuration: Dict[str, Any] = None) -> None:
        self.corpus = corpus
        self.kr = kr
        if configuration is None:
            self.config = config['relation_extraction']
        else:
            self.config = configuration

    def on_occurence_relation_extraction(self) -> None:
        """Extract relations based on concepts co-occurrence.
        """
        try:
            assert self.config['on_occurrence'].get('use_lemma') is not None
            assert self.config['on_occurrence'].get('threshold') is not None
        except KeyError as e:
            logging_config.logger.error(
                f"""Config information missing for relation extraction based on co-occurrence. Make sure you provided the configuration fields:
                    - relation_extraction.on_occurrence.use_lemma
                    - relation_extraction.on_occurrence.threshold
                    Trace : {e}
                """)
        else:
            options = self.config['on_occurrence']
        relation_extration = OnOccurrenceRelationExtraction(
            self.corpus, self.kr, options)
        relation_extration.on_occurrence_relation_extraction()

    def on_pos_relation_extraction(self) -> None:
        """Extract relations based on pos-tagging.
        """
        try:
            assert self.config['on_pos'].get('use_lemma') is not None
            assert self.config['on_pos'].get('pos_selection') is not None
            assert len(self.config['on_pos'].get('pos_selection')) > 0
        except KeyError as e:
            logging_config.logger.error(
                f"""Config information missing for relation extraction based on co-occurrence. Make sure you provided the configuration fields:
                    - relation_extraction.on_pos.use_lemma
                    - relation_extraction.on_pos.pos_selection.
                    Trace : {e}
                """)
        else:
            options = self.config['on_pos']
        relation_extration = OnPosRelationExtraction(
            self.corpus, self.kr, options)
        relation_extration.on_pos_relation_extraction()

    def on_cooc_with_sep_term_map_meta_rel_extraction(self, spacy_nlp: spacy.language.Language) -> None:
        """Extract meta relations based on cooccurrences of concepts and a mapping of term to meta relation types.
            The method requires the same Spacy Language model as the one used to process the self.corpus attribute.
            The method updates the self.kr attribute.

        Parameters
        ----------
        spacy_nlp : spacy.language.Language
            The same Spacy Language model as the one used to process the self.corpus attribute.
        """
        relation_extraction = OnCoocWithSepTermMetaRelationExtraction(
            corpus=self.corpus,
            kr=self.kr,
            spacy_nlp=spacy_nlp,
            options=self.config["on_occurrence_with_sep_term"]
        )

        relation_extraction.on_cooc_with_sep_term_map_meta_rel_extraction()
