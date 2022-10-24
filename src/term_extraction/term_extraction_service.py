from typing import Iterable, List, Dict
from collections import Counter

import spacy
import spacy.tokens
import spacy.tokenizer
import spacy.language

from config.core import config
import config.logging_config as logging_config
from term_extraction.term_extraction_methods.c_value import Cvalue
from data_preprocessing.data_preprocessing_methods.token_selectors import select_on_pos,select_on_occurence_count


class Term_Extraction():
    """Second processing of the corpus.
    Finding of terms under interest.

    """

    def __init__(self, corpus: List[spacy.tokens.doc.Doc]) -> None:
        self.corpus = corpus

    def c_value_term_extraction(self, tokenSequences_doc_attribute_name: str, max_size_gram: int) -> Cvalue:
        """Computes the C-value score for candidate terms attached to the Doc by the custom 
            attribute named tokenSequences_doc_attribute_name.

        Parameters
        ----------
        tokenSequences_doc_attribute_name : str
            The name of the custom attribute storing the Document parts to consider for the C-value computation.
        max_size_gram : int
            The maximum number of words a candidate term can have.

        Returns
        -------
        Cvalue
            The class containing the c-value scores.
        """
        self.c_value = Cvalue(
            self.corpus, tokenSequences_doc_attribute_name, max_size_gram)
        return self.c_value

    def _get_doc(self,doc: spacy.tokens.doc.Doc) :
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
        if config['term_extraction']['tokens_attribute']: 
            return doc._.get(config['term_extraction']['tokens_attribute'])
        else : 
            return doc

    def on_pos_term_extraction(self) -> List[str]:
        """Return unique candidate terms after filtering on pos-tagging labels.

        Parameters
        ----------
        on_lemma : bool
            If true, the output is the lemma of token. By defaut, the output is the text.
        Returns
        -------
        List[str]
            List of unique validated terms.
        """
        candidate_pos_terms = []

        for doc in self.corpus:
            for token in self._get_doc(doc) : 
                if select_on_pos(token,config['term_extraction']['on_pos']['pos_selection']):
                    if config['term_extraction']['use_lemma']['use_lemma'] :
                        candidate_pos_terms.append(token.lemma_)
                    else : 
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
        candidate_terms = [token for doc in self.corpus for token in self._get_doc(doc)]
        candidate_occurence_terms = []

        on_lemma = False

        if config['term_extraction']['use_lemma']['use_lemma']:
            terms = [token.lemma_ for token in candidate_terms]
            on_lemma = True
        else : 
            terms = [token.text for token in candidate_terms]
        
        occurences = Counter(terms)

        for token in candidate_terms : 
            if select_on_occurence_count(token,config['term_extraction']['on_occurence']['occurence_threshold'],occurences,on_lemma):
                if on_lemma :
                    candidate_occurence_terms.append(token.lemma_)
                else : 
                    candidate_occurence_terms.append(token.text)
        unique_candidates = list(set(candidate_occurence_terms))
        
        return unique_candidates