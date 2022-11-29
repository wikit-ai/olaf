import numpy as np
import spacy.tokens
import spacy.vocab
from statistics import mean
from typing import Any, Dict, List
import uuid

from commons.ontology_learning_schema import Concept, KR, MetaRelation
from concept_hierarchy.concept_hierarchy_schema import RepresentativeTerm
import config.logging_config as logging_config


class TermSubsumption():
    """Algorithm that find generalisation meta relations with subsumption method.
    If the option use_lemma is set to true, the algorithm will only consider lemmatisation of the corpus.
    """

    def __init__(self, corpus: List[spacy.tokens.doc.Doc], kr: KR, options: Dict[str, Any]) -> None:
        """Initialisation.

        Parameters
        ----------
        corpus : List[spacy.tokens.doc.Doc]
            Corpus used to find generalisation relations.
        kr : KR
            Existing knowledge representation of the corpus.
        options : Dict[str, Any]
            Options needed for term subsumption algorithm that is to say configuration for span, lemma and threshold value.
        """
        self.corpus = corpus
        self.kr = kr
        self.representative_terms = []
        self.terms_count = {}
        self.options = options

        try:
            self.representative_terms = self._get_representative_terms()
        except Exception as _e:
            logging_config.logger.error(f"Could not set representative terms. Trace : {_e}.")
        else:
            logging_config.logger.info(f"Representative terms set.")

        try:
            self.terms_count = self._get_terms_count()
        except Exception as _e: 
            logging_config.logger.error(f"Could not set terms count. Trace: {_e}.")
        else : 
            logging_config.logger.info("Terms count set.")

    def __call__(self) -> None:
        """The method directly update the knowledge representation"""
        self.term_subsumption()

    def _get_representative_terms(self) -> List[RepresentativeTerm]:
        """Get one string per concept. This string is the best text representation of the concept among all its terms.

        Returns
        -------
        List[RepresentativeTerm]
            List of one representing term by concept.
        """
        representative_terms = []
        for concept in self.kr.concepts:
            if len(concept.terms) == 1:
                term = list(concept.terms)[0]
            else:
                try:
                    term = self._get_most_representative_term(concept)
                except Exception as _e:
                    term = None
                    logging_config.logger.error(f"Could not get most representative term for concept {concept.uid}. Trace: {_e}.")
                else : 
                    logging_config.logger.info(f"Most representative term found for concept {concept.uid}.")
            
            if term is not None :
                representative_term_value = term
                representative_term_concept_id = concept.uid
                representative_term = RepresentativeTerm(representative_term_value, representative_term_concept_id)
                representative_terms.append(representative_term)
        return representative_terms

    def _get_terms_count(self) -> Dict[str,int]:
        """Get for each terms the number of documents containing the term.

        Returns
        -------
        Dict[str,int]
            Dict with terms as keys and their count as values.
        """
        terms_count = {}
        for term in self.representative_terms:
            terms_count[term.value] = self._count_doc_with_term(term.value)
        return terms_count

    def _compute_similarity_between_tokens(self, term1_vector: np.ndarray, term2_vector: np.ndarray) -> float:
        """Compute similarity between two vectors.

        Parameters
        ----------
        term1_vector : np.ndarray
            Vector of the first term.
        term2_vector : np.ndarray
            Vector of the second term.

        Returns
        -------
        float
            Similarity score between two vectors.
        """
        denominator = np.linalg.norm(term1_vector) * np.linalg.norm(term2_vector)
        if denominator == 0.0 :
            logging_config.logger.error(f"ZeroDivisionError, could not compute similarity.")
            similarity = 0.0
        else : 
            similarity = np.dot(term1_vector, term2_vector) / denominator
            similarity = similarity.item()
        return similarity

    def _get_representative_vector(self, term: str, vocab: spacy.vocab.Vocab) -> np.ndarray:
        """Get representative vector of an expression.
        If the expression contains only one word, it directly gets the vector of the word from the vocabulary.
        If the expression contains multiple words, it computes the mean from each word's representative vector. 

        Parameters
        ----------
        term : str
            Term to be represented as a vector.
        vocab : spacy.vocab.Vocab
            Vocabulary containing all term vectors.

        Returns
        -------
        np.ndarray
            Representative vector of the wanted expression.
        """
        term_nb_words = len(term.split())
        if self.options.get('use_span') and (term_nb_words > 1):
            term_words = term.split()
            term_vectors = []
            for word in term_words : 
                term_vectors.append(vocab.get_vector(word))
                if vocab.has_vector(word) == False :
                    logging_config.logger.warning(f"Term {term} has no representative vector in the spacy vocabulary.")
            term_vector = np.mean(term_vectors, axis=0)
        else : 
            term_vector = vocab.get_vector(term)
            if vocab.has_vector(term) == False :
                logging_config.logger.warning(f"Term {term} has no representative vector in the spacy vocabulary.")
        return term_vector


    def _get_most_representative_term(self, concept: Concept) -> str:
        """Try to find the most representative term of a concept. For each term it computes the average of similarity with other terms. The term with the higher average similarity is used as most representative term.

        Parameters
        ----------
        concept : Concept
            Concept to analyze.

        Returns
        -------
        str
            Most representative term in concept terms.
        """
        concept_terms = list(concept.terms)
        vocab = self.corpus[0].vocab
        if len(vocab) > 0:
            terms_mean_similarity = []
            for i,term in enumerate (concept_terms):
                term_vector = self._get_representative_vector(term, vocab)
                other_terms = self._find_other_terms(i, concept_terms)
                
                term_similarities = []
                for other_term in other_terms:
                    terms_similarity = self._compute_similarity_between_tokens(term_vector, self._get_representative_vector(other_term,vocab))
                    term_similarities.append(terms_similarity)
                terms_mean_similarity.append(mean(term_similarities))
            index_most_representative_term = np.argmax(terms_mean_similarity)
            most_representative_term = concept_terms[index_most_representative_term]
        else : 
            logging_config.logger.error(f"Spacy vocabulary not loaded.")
            raise ValueError
        return most_representative_term

    def _check_term_in_doc(self, term: str, doc_words: List[str]) -> bool:
        """Test if a term in present or not in a document content.

        Parameters
        ----------
        term : str
            Term to be tested as present or not in document.
        doc_words : List[str]
            Document representation as list of words lemmatized or not.

        Returns
        -------
        bool
            True in term is in doc and false otherwise.
        """
        term_in_doc = False
        term_nb_words = len(term.split())
        if self.options.get('use_span') and (term_nb_words > 1):
            expression_to_test = " " + " ".join(doc_words) + " "
            term_to_find = " " + term + " "
            term_in_doc = term_to_find in expression_to_test
        else : 
            term_in_doc = term in doc_words
        return term_in_doc


    def _count_doc_with_term(self, term: str) -> int:
        """Count the number of documents containing the term in parameter.

        Parameters
        ----------
        term : str
            Representative term to analyse.

        Returns
        -------
        int
            Number of doc containing term.
        """
        count = 0
        for doc in self.corpus:
            if self.options.get('use_lemma'):
                doc_words = [token.lemma_ for token in doc]
            else : 
                doc_words = [token.text for token in doc]
            if self._check_term_in_doc(term, doc_words):
                count += 1
        return count

    def _count_doc_with_both_terms(self, term1: str, term2: str) -> int:
        """Count the number of documents containing both terms in parameter.

        Parameters
        ----------
        term1 : str
            First representative term to analyse.
        term2 : str
            Second representative term to analyse.

        Returns
        -------
        int
            Number of doc containing both terms.
        """
        count = 0
        for doc in self.corpus:
            if self.options.get('use_lemma'):
                doc_words = [token.lemma_ for token in doc]
            else : 
                doc_words = [token.text for token in doc]
            if self._check_term_in_doc(term1,doc_words) and self._check_term_in_doc(term2,doc_words):
                count += 1
        return count

    def _compute_subsumption(self, nb_doc_cooccurrence: int, nb_doc_occurrence: int) -> float:
        """Compute subsumption score between two terms.

        Parameters
        ----------
        nb_doc_cooccurrence : int
            Number of docs that contains both terms.
        nb_doc_occurrence : int
            Number of docs that contains the most specific term.

        Returns
        -------
        float
            Score of generalisation between term1 and term2.
        """
        if not(nb_doc_occurrence == 0):
            subsumption_score = nb_doc_cooccurrence/nb_doc_occurrence
        else :
            subsumption_score = 0
            logging_config.logger.warning(f"nb_doc_occurrence as value 0. Check that this behavior is expected.")
        return subsumption_score

    def _verify_threshold(self, subsumption_score: float, inverse_subsumption_score: float) -> bool:
        """Verify that terms respect generalisation rules (x more general than y) that is to say :
        - subsumption (P(x,y)) > t (a given threshold)
        - subsumption (P(x,y)) > inverse_subsumption (P(y,x))

        Parameters
        ----------
        subsumption_score : float
            Number of docs containing both terms divided by number of docs containing the most specialised term.
        inverse_subsumption_score : float
            Number of docs containing both terms divided by number of docs containing the most general term.

        Returns
        -------
        bool
            True if rules are respected, that is to say there is a generalisation relation. False otherwise.
        """
        if (subsumption_score > self.options.get('threshold')) and (subsumption_score > inverse_subsumption_score):
            return True
        else:
            return False

    def _find_other_terms(self, term_index: int, terms: List[Any]) -> List[Any]:
        """Duplicate list without the specified index.

        Parameters
        ----------
        term_index : int
            Index of non-wanted object.

        terms: List[Any]
            List of terms or representative terms.

        Returns
        -------
        List[Any]
            List built.
        """
        if (term_index < 0) or (term_index >= len(terms)):
            logging_config.logger.error(f"Could not find other terms. You should check the term index.")
            other_terms = []
        else : 
            other_terms = [term for term in terms[: term_index] + terms[term_index +1 :]]
        return other_terms

    def _create_generalisation_relation(self, source_concept_id : str, destination_concept_id : str) -> MetaRelation:
        """Create generalition relation.

        Parameters
        ----------
        source : str
            Uid of the source concept in the relation.
        destination : str
            Uid of the destination concept in the relation.

        Returns
        -------
        MetaRelation
            Generalisation relation built between concepts.
        """
        meta_relation_uid = uuid.uuid4()
        meta_relation_source_concept_id = source_concept_id
        meta_relation_destination_concept_id = destination_concept_id
        meta_relation_type = "generalisation"
        meta_relation = MetaRelation(meta_relation_uid, meta_relation_source_concept_id, meta_relation_destination_concept_id, meta_relation_type)
        return meta_relation

    def term_subsumption(self):
        """Find generalisation relations between concepts from term representation via term subsumption method.
        """
        for i, term in enumerate(self.representative_terms):
            for pair_term in self._find_other_terms(i, self.representative_terms):
                score_cooccurrence = self._count_doc_with_both_terms(term.value, pair_term.value)
                subsumption_score = self._compute_subsumption(score_cooccurrence, self.terms_count[pair_term.value])
                inverse_subsumption_score = self._compute_subsumption(score_cooccurrence, self.terms_count[term.value])
                if self._verify_threshold(subsumption_score, inverse_subsumption_score):
                    meta_relation = self._create_generalisation_relation(term.concept_id, pair_term.concept_id)
                    self.kr.meta_relations.add(meta_relation)
