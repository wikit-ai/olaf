from typing import Any, Dict, List, Optional, Set
from collections import Counter

import spacy
import spacy.tokens
import spacy.tokenizer
import spacy.language

from config.core import config
import config.logging_config as logging_config
from term_extraction.term_extraction_methods.candidate_terms_post_filters import str2candidateTermFilter
from term_extraction.term_extraction_methods.c_value import Cvalue
from term_extraction.term_extraction_methods.tf_idf_term_extraction import TFIDF_TermExtraction, TFIDF_ANALYZER
from commons.ontology_learning_schema import CandidateTerm
from data_preprocessing.data_preprocessing_methods.token_selectors import select_on_pos, select_on_occurrence_count


class TermExtraction():
    """Second processing of the corpus.
    Finding of terms under interest.

    """

    def __init__(self, corpus: List[spacy.tokens.doc.Doc], configuration: Dict[str, Any] = None) -> None:
        self.corpus = corpus
        if configuration is None:
            self.config = config['term_extraction']
        else:
            self.config = configuration

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

        c_values = c_value.c_values

        candidate_terms = [
            CandidateTerm(c_val.candidate_term) for c_val in c_values if c_val.score >= treshold
        ]

        return candidate_terms

    def _get_doc_content_for_term_extraction(self, selected_tokens_doc_attribute: str, doc: spacy.tokens.doc.Doc):
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
            content = doc._.get(selected_tokens_doc_attribute)
        else:
            content = doc
        return content

    def on_pos_term_extraction(self) -> List[CandidateTerm]:
        """Return unique candidate terms after filtering on pos-tagging labels.
        Not working for span.

        Returns
        -------
        List[CandidateTerm]
            The list of extracted candidate terms.
        """

        if self.config.get("use_span"):
            candidate_terms = []
            logging_config.logger.error(
                f"Could not extract spans with pos tagging. Update configuration file or use an other method.")
        else:
            candidate_pos_terms = []
            for doc in self.corpus:
                for token in self._get_doc_content_for_term_extraction(self.config['selected_tokens_doc_attribute'], doc):
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
        if self.config.get("use_span") and (self.config.get('selected_tokens_doc_attribute') is None):
            candidate_terms = []
            logging_config.logger.error(
                f"Could not extract spans on occurence without specific attribute on doc. Update configuration file or use an other method.")

        else:
            if self.config.get("use_span"):
                terms_of_interest = [span for doc in self.corpus for span in doc._.get(
                    self.config.get('selected_tokens_doc_attribute'))]

            else:
                terms_of_interest = [
                    token for doc in self.corpus for token in self._get_doc_content_for_term_extraction(self.config.get('selected_tokens_doc_attribute'), doc)]

            candidate_occurrence_terms = []

            on_lemma = False
            if self.config['on_occurrence']['use_lemma']:
                terms = [term.lemma_ for term in terms_of_interest]
                on_lemma = True
            else:
                terms = [term.text for term in terms_of_interest]

            occurrences = Counter(terms)

            for term in terms_of_interest:
                if select_on_occurrence_count(term, self.config['on_occurrence']['occurrence_threshold'], occurrences, on_lemma):
                    if on_lemma:
                        candidate_occurrence_terms.append(term.lemma_)
                    else:
                        candidate_occurrence_terms.append(term.text)

            unique_candidates = list(set(candidate_occurrence_terms))
            candidate_terms = [
                CandidateTerm(unique_candidate) for unique_candidate in unique_candidates
            ]

        return candidate_terms

    def tfidf_term_extraction(self, tfidf_token_filter: Optional[TFIDF_ANALYZER] = None) -> List[CandidateTerm]:
        """Returns the list of candidate terms having a tfidf score equal or greater to the treshold defined in 
        the configuration field term_extraction.tfidf.treshold.


         Parameters
        ----------
        tfidf_token_filter: Optional[TFIDF_ANALYZER] = None
            The function to preprocess tokens before computing TF IDF scores.

        Returns
        -------
        List[CandidateTerm]
            The list of extracted candidate terms.
        """
        try:
            treshold = self.config['tfidf']['treshold']
        except KeyError as e:
            logging_config.logger.error(
                f"""Config information missing for TF IDF term extraction. Make sure you provided the configuration field:
                    - term_extraction.tfidf.treshold
                    Trace : {e}
                """)

        candidate_terms = list()

        tfidf_agg_type = self.config["tfidf"].get("tfidf_agg_type", "MEAN")
        tfidf_term_extraction = TFIDF_TermExtraction(
            self.corpus, tfidf_agg_type, tfidf_token_filter)

        tfidf_term_extract_res = tfidf_term_extraction.tfidf_values

        candidate_terms = [
            CandidateTerm(term_extraction_res.candidate_term) for term_extraction_res in tfidf_term_extract_res if term_extraction_res.score >= treshold
        ]

        return candidate_terms

    def post_filter_candidate_terms_on_tokens_presence(self,
                                                       candidate_terms: List[CandidateTerm],
                                                       filter_type: str,
                                                       filtering_tokens: Set[str]
                                                       ) -> List[CandidateTerm]:
        """Post process a list of extracted candidate terms to filter them based on hard rules 
            and a set of string that should not appear in specific locations of the candidate term value.

        Parameters
        ----------
        candidate_terms: List[CandidateTerm]
            List of tokens to preprocess

        filter_type: str
            The string refering to the filter function.

        filtering_tokens: Set[str]
            The set of tokens string to filter on.

        Returns
        -------
        List[CandidateTerm]
            The list of preprocessed candidate terms.
        """

        filtered_candidate_terms = list()

        if filter_type not in str2candidateTermFilter.keys():
            logging_config.logger.error(
                f"""Filter type does not exists. Available filter types are:
                    - on_first_token
                    - on_last_token
                    - if_token_in_term
                """)
        else:
            filter = str2candidateTermFilter[filter_type]

            filtered_candidate_terms = filter(
                candidate_terms, filtering_tokens)

        return filtered_candidate_terms
