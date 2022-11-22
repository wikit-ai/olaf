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
from data_preprocessing.data_preprocessing_methods.token_selectors import select_on_pos, select_on_occurrence_count


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

    def _get_doc_content_for_term_extraction(self, selected_tokens_doc_attribute:str , doc: spacy.tokens.doc.Doc):
        """Get the doc content of interest for the term extraction process.
        The term extraction can be performed on either the raw source documents or selected parts of each document after token selection process.


        Parameters
        ----------
        selected_tokens_doc_attribute:str
            Name of selected tokens attribute if it exists, empty string otherwise

        doc : spacy.tokens.doc.Doc
            Spacy representation of document

        Returns
        -------
        List[spacy.tokens.Token]
            Attribute of selected tokens if it exists, spacy doc otherwise
        """
        if selected_tokens_doc_attribute is not None:
            content =  doc._.get(selected_tokens_doc_attribute)
        else:
            content = doc
        return content

    def on_pos_term_extraction(self) -> List[CandidateTerm]:
        """Return unique candidate terms after filtering on pos-tagging labels.

        Returns
        -------
        List[CandidateTerm]
            The list of extracted candidate terms.
        """
        candidate_pos_terms = []

        for doc in self.corpus:
            for token in self._get_doc_content_for_term_extraction(self.config['selected_tokens_doc_attribute'],doc):
                if select_on_pos(token, self.config['on_pos']['pos_selection']):
                    if self.config['on_pos']['use_lemma']:
                        candidate_pos_terms.append(token.lemma_)
                    else:
                        candidate_pos_terms.append(token.text)
        unique_candidates = list(set(candidate_pos_terms))

        candidate_terms = [
            CandidateTerm(unique_candidate) for unique_candidate in unique_candidates
        ]

        return candidate_terms

    def on_occurrence_term_extraction(self) -> List[CandidateTerm]:
        """Return unique candidate terms with occurrence higher than a configured threshold.

        Returns
        -------
        List[CandidateTerm]
            The list of extracted candidate terms.
        """
        candidate_terms = [
            token for doc in self.corpus for token in self._get_doc_content_for_term_extraction(self.config['selected_tokens_doc_attribute'],doc)]
        candidate_occurrence_terms = []

        on_lemma = False

        if self.config['on_occurrence']['use_lemma']:
            terms = [token.lemma_ for token in candidate_terms]
            on_lemma = True
        else:
            terms = [token.text for token in candidate_terms]

        occurrences = Counter(terms)

        for token in candidate_terms:
            if select_on_occurrence_count(token, self.config['on_occurrence']['occurrence_threshold'], occurrences, on_lemma):
                if on_lemma:
                    candidate_occurrence_terms.append(token.lemma_)
                else:
                    candidate_occurrence_terms.append(token.text)
        unique_candidates = list(set(candidate_occurrence_terms))

        candidate_terms = [
            CandidateTerm(unique_candidate) for unique_candidate in unique_candidates
        ]

        return candidate_terms
