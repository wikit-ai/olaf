from typing import List, Dict, Any
from collections import Counter

import spacy
import spacy.tokens
import spacy.tokenizer
import spacy.language

from config.core import config
import config.logging_config as logging_config
from term_extraction.term_extraction_methods.c_value import Cvalue
from commons.ontology_learning_schema import CandidateTerm
from data_preprocessing.data_preprocessing_methods.token_selectors import select_on_pos, select_on_occurence_count


class Term_Extraction():
    """Second processing of the corpus.
    Finding of terms under interest.

    """

    def __init__(self, corpus: List[spacy.tokens.doc.Doc], config: Dict[str, Any] = config['term_extraction']) -> None:
        self.corpus = corpus
        self.config = config

    def c_value_term_extraction(self) -> List[CandidateTerm]:
        """Returns the list of candidate terms having a c-value score equal or greater to the treshold defined in 
        the configuration field term_extraction.c_value.treshold.

        Returns
        -------
        List[CandidateTerm]
            The list of extracted candidate terms.
        """

        try:
            doc_attribute_name = self.config['selected_tokens_doc_attribute']
            max_size_gram = self.config['c_value']['max_size_gram']
        except KeyError as e:
            logging_config.logger.error(
                f"""Config information missing for C-value. Make sure you provided the configuration fields:
                    - term_extraction.selected_tokens_doc_attribute
                    - term_extraction.c_value.max_size_gram
                    Trace : {e}
                """)

        c_value = Cvalue(self.corpus, doc_attribute_name, max_size_gram)

        # Compute C-values
        try:
            treshold = self.config['c_value']['treshold']
        except KeyError as e:
            logging_config.logger.error(
                f"""Config information missing for C-value. Make sure you provided the configuration field:
                    - term_extraction.c_value.treshold
                    Trace : {e}
                """)

        c_values = c_value.compute_c_values()

        candidate_terms = [
            CandidateTerm(c_val.candidate_term) for c_val in c_values if c_val.c_value >= treshold
        ]

        return candidate_terms

    def _get_doc(self, use_selected_token: bool, doc: spacy.tokens.doc.Doc):
        """Get the doc content of interest for the term extraction process.
        The term extraction can be performed on either the raw source documents or selected parts of each document after token selection process.


        Parameters
        ----------
        doc : spacy.tokens.doc.Doc
            Spacy representation of document

        Returns
        -------
        List[spacy.tokens.Token]
            Attribute of selected tokens if it exists, spacy doc otherwise
        """
        if config['term_extraction']['selected_tokens_doc_attribute']:
            return doc._.get(config['term_extraction']['selected_tokens_doc_attribute'])
        else:
            return doc

    def on_pos_term_extraction(self) -> List[str]:
        """Return unique candidate terms after filtering on pos-tagging labels.

        Returns
        -------
        List[str]
            List of unique validated terms.
        """
        candidate_pos_terms = []

        for doc in self.corpus:
            for token in self._get_doc(doc):
                if select_on_pos(token, config['term_extraction']['on_pos']['pos_selection']):
                    if config['term_extraction']['on_pos']['use_lemma']:
                        candidate_pos_terms.append(token.lemma_)
                    else:
                        candidate_pos_terms.append(token.text)
        unique_candidates = list(set(candidate_pos_terms))

        return unique_candidates

    def on_occurence_term_extraction(self) -> List[str]:
        """Return unique candidate terms with occurence higher than a configured threshold.

        Returns
        -------
        List[str]
            List of unique validated terms.
        """
        candidate_terms = [
            token for doc in self.corpus for token in self._get_doc(doc)]
        candidate_occurence_terms = []

        on_lemma = False

        if config['term_extraction']['on_occurence']['use_lemma']:
            terms = [token.lemma_ for token in candidate_terms]
            on_lemma = True
        else:
            terms = [token.text for token in candidate_terms]

        occurences = Counter(terms)

        for token in candidate_terms:
            if select_on_occurence_count(token, config['term_extraction']['on_occurence']['occurence_threshold'], occurences, on_lemma):
                if on_lemma:
                    candidate_occurence_terms.append(token.lemma_)
                else:
                    candidate_occurence_terms.append(token.text)
        unique_candidates = list(set(candidate_occurence_terms))

        return unique_candidates
