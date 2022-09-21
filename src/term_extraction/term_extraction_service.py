from typing import Iterable, List, Dict

import spacy
import spacy.tokens
import spacy.tokenizer
import spacy.language

from config import core
import config.logging_config as logging_config
from term_extraction.term_extraction_methods.c_value import Cvalue


class Term_Extraction():
    """Second processing of the corpus.
    Finding of terms under interest.

    """

    def __init__(self) -> None:
        pass

    def c_value(self, corpus: List[spacy.tokens.doc.Doc], tokenSequences_doc_attribute_name: str, max_size_gram: int) -> Cvalue:
        self.c_value = Cvalue(
            corpus, tokenSequences_doc_attribute_name, max_size_gram)
        return self.c_value

    def on_pos_token_filtering(self, corpus: List[List[spacy.tokens.token.Token]], token_pos_filter: List[str]) -> List[Dict[str, int]]:
        """Return unique candidate terms after filtering on pos-tagging labels.
        Candidate terms are lemmatized and put into lowercase.

        Parameters
        ----------
        corpus : List[List[spacy.tokens.token.Token]]
            Cleaned corpus.
        token_pos_filter : List[str]
            Pos-tagging filters to apply.

        Returns
        -------
        List[Dict[str,int]]
            List of unique candidate terms lemmatized and their occurrences.
        """
        candidate_terms = []
        try:
            for document in corpus:
                for token in document:
                    if token.pos_ in token_pos_filter:
                        candidate_terms.append(token.lemma_.lower())
        except Exception as _e:
            logging_config.logger.error(
                "Could not filter and lemmatize spacy tokens. Trace : %s", _e)
        else:
            logging_config.logger.info(
                "List of tokens filtered and lemmatized.")
        unique_candidate_terms = list(set(candidate_terms))
        count_candidate_terms = [{"term": term, "occurrence": candidate_terms.count(
            term)} for term in unique_candidate_terms]
        return count_candidate_terms

    def frequency_filtering(self, count_candidate_terms: List[Dict[str, int]]) -> List[str]:
        """Return candidate terms with frequency higher than a configured threshold.

        Parameters
        ----------
        count_candidate_terms : List[Dict[str,int]]
            List of unique candidate terms and their occurrences.

        Returns
        -------
        List[str]
            Candidate terms extracted.
        """
        nb_term_candidates = len(count_candidate_terms)
        validated_terms = []
        if nb_term_candidates > 0:
            term_occurrence = []
            try:
                for candidate in count_candidate_terms:
                    term_occurrence.append(
                        {"term": candidate['term'], "occurrence": candidate['occurrence']})

                validated_terms = [
                    term['term'] for term in term_occurrence if term['occurrence'] > core.OCCURRENCE_THRESHOLD]
            except Exception as _e:
                logging_config.logger.error(
                    "Could not filter candidate terms by occurrence. Trace : %s", _e)
            else:
                logging_config.logger.info(
                    "List of tokens filtered by occurrence.")
        else:
            logging_config.logger.error("No term candidate found.")
            validated_terms = None
        return validated_terms


term_extraction = Term_Extraction()
