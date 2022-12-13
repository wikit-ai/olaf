from itertools import combinations
import numpy as np
import spacy.tokens
import spacy.vocab
from statistics import mean
from tqdm import tqdm
from typing import Any, Dict, List, Tuple
import uuid

from commons.ontology_learning_schema import Concept, KR, MetaRelation
from commons.ontology_learning_utils import check_term_in_content
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
        self.pair_terms_count = {}
        self.options = options

        try:
            self.representative_terms = self._get_representative_terms()
        except Exception as _e:
            logging_config.logger.error(
                f"Could not set representative terms. Trace : {_e}.")
        else:
            logging_config.logger.info(f"Representative terms set.")

        try:
            self._compute_terms_cout()
        except Exception as _e:
            logging_config.logger.error(
                f"Could not set terms count and pair terms count. Trace: {_e}.")
        else:
            logging_config.logger.info("Terms count and pair terms count set.")


    def _get_representative_terms(self) -> List[RepresentativeTerm]:
        """Get one string per concept. This string is the best text representation of the concept among all its terms.

        Returns
        -------
        List[RepresentativeTerm]
            List of one representing term by concept.
        """
        representative_terms = []
        if self.options.get('algo_type') == "MEAN":
            for concept in tqdm(self.kr.concepts):
                representative_terms += [RepresentativeTerm(term, concept.uid) for term in concept.terms]
        else:
            for concept in tqdm(self.kr.concepts):
                if len(concept.terms) < 3:
                    term = list(concept.terms)[0]
                else:
                    try:
                        term = self._get_most_representative_term(concept)
                    except Exception as _e:
                        term = None
                        logging_config.logger.error(
                            f"Could not get most representative term for concept {concept.uid}. Trace: {_e}.")
                    else:
                        logging_config.logger.info(
                            f"Most representative term found for concept {concept.uid}.")

                if term is not None:
                    representative_terms.append(RepresentativeTerm(term, concept.uid))
        return representative_terms

    def _compute_terms_cout(self) -> None:
        """Compute terms occurrence and terms pair occurrence in the corpus.
        """
        terms_pairs = list(combinations([term.value for term in self.representative_terms], 2))
        for doc in tqdm(self.corpus):
            if self.options.get('use_lemma'):
                doc_words = [token.lemma_ for token in doc]
            else:
                doc_words = [token.text for token in doc]
            for term in self.representative_terms:
                if not(term.value in self.terms_count.keys()):
                    self.terms_count[term.value] = 0
                if check_term_in_content(term.value, doc_words):
                    self.terms_count[term.value] = self.terms_count.get(term.value) + 1
            for terms_pair in terms_pairs:
                if not(terms_pair in self.pair_terms_count.keys()):
                    self.pair_terms_count[terms_pair] = 0
                if check_term_in_content(terms_pair[0], doc_words) and check_term_in_content(terms_pair[1], doc_words):
                    self.pair_terms_count[terms_pair] = self.pair_terms_count.get(terms_pair, 0) + 1
                

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
        denominator = np.linalg.norm(
            term1_vector) * np.linalg.norm(term2_vector)
        if denominator == 0.0:
            logging_config.logger.error(
                f"ZeroDivisionError, could not compute similarity.")
            similarity = 0.0
        else:
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
        term_words = term.strip().split()
        if self.options.get('use_span') and (len(term_words) > 1):
            term_vectors = []
            for word in term_words:
                term_vectors.append(vocab.get_vector(word))
                if vocab.has_vector(word) == False:
                    logging_config.logger.warning(
                        f"Term {term} has no representative vector in the spacy vocabulary.")
            term_vector = np.mean(term_vectors, axis=0)
        else:
            term_vector = vocab.get_vector(term)
            if vocab.has_vector(term) == False:
                logging_config.logger.warning(
                    f"Term {term} has no representative vector in the spacy vocabulary.")
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
            for i, term in enumerate(concept_terms):
                term_vector = self._get_representative_vector(term, vocab)
                other_terms = self._find_other_terms(i, concept_terms)

                term_similarities = []
                for other_term in other_terms:
                    terms_similarity = self._compute_similarity_between_tokens(
                        term_vector, self._get_representative_vector(other_term, vocab))
                    term_similarities.append(terms_similarity)
                terms_mean_similarity.append(mean(term_similarities))
            index_most_representative_term = np.argmax(terms_mean_similarity)
            most_representative_term = concept_terms[index_most_representative_term]
        else:
            logging_config.logger.error(f"Spacy vocabulary not loaded.")
            raise ValueError
        return most_representative_term

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
        if not (nb_doc_occurrence == 0):
            subsumption_score = nb_doc_cooccurrence/nb_doc_occurrence
        else:
            subsumption_score = 0
            logging_config.logger.warning(
                f"nb_doc_occurrence as value 0. Check that this behavior is expected.")
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
        if (subsumption_score > self.options.get('subsumption_threshold')) and (subsumption_score > inverse_subsumption_score):
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
            logging_config.logger.error(
                f"Could not find other terms. You should check the term index.")
            other_terms = []
        else:
            other_terms = [
                term for term in terms[: term_index] + terms[term_index + 1:]]
        return other_terms

    def _compute_general_words_percentage(self, general_concept_rt: List[RepresentativeTerm], specialized_concept_rt: List[RepresentativeTerm]) -> float:
        """Compute the generalisation score between the general concept and the specialized concept.
        The score is the proportion of terms in the general concept that subsumes terms in the specialized concept.

        Parameters
        ----------
        general_concept_rt : List[RepresentativeTerm]
            Representative terms of the general concept.
        specialized_concept_rt : List[RepresentativeTerm]
            Representative terms of the specialized concept.

        Returns
        -------
        float
            Generalisation score.
        """
        total_count = 0
        sub_count = 0
        for gen_term in general_concept_rt:
            for spe_term in specialized_concept_rt:
                total_count += 1
                score_cooccurrence = self.pair_terms_count.get(
                    (gen_term.value, spe_term.value), self.pair_terms_count.get((spe_term.value, gen_term.value)))
                sub_score = self._compute_subsumption(
                    score_cooccurrence, self.terms_count[spe_term.value])
                inverse_sub_score = self._compute_subsumption(
                    score_cooccurrence, self.terms_count[gen_term.value])
                if self._verify_threshold(sub_score, inverse_sub_score):
                    sub_count += 1
        return sub_count/total_count

    def _check_concept_more_general(self, score_gen_concept_1: float, score_gen_concept_2: float) -> bool:
        """Verify if a first concept is more general than a second concept according to term subsumption rules based on thresholds.

        Parameters
        ----------
        score_gen_concept_1 : float
            Generalisation score of the first concept.
        score_gen_concept_2 : float
            Generalisation score of the second concept.

        Returns
        -------
        bool
            True is the first concept is more general, False otherwise.
        """
        validity = False
        if score_gen_concept_1 > self.options.get('mean_high_threshold') > self.options.get('mean_low_threshold') > score_gen_concept_2:
            validity = True
        return validity

    def _create_generalisation_relation(self, source_concept_id: str, destination_concept_id: str) -> None:
        """Create generalition relation and add it to the knowledge representation.

        Parameters
        ----------
        source : str
            Uid of the source concept in the relation.
        destination : str
            Uid of the destination concept in the relation.
        """
        meta_relation_uid = str(uuid.uuid4())
        meta_relation_source_concept_id = source_concept_id
        meta_relation_destination_concept_id = destination_concept_id
        meta_relation_type = "generalisation"
        meta_relation = MetaRelation(meta_relation_uid, meta_relation_source_concept_id,
                                     meta_relation_destination_concept_id, meta_relation_type)

        self.kr.meta_relations.add(meta_relation)

    def term_subsumtion(self) -> None:
        """The method directly update the knowledge representation"""
        if self.options.get("algo_type") == "UNIQUE":
            self.term_subsumption_unique()
        elif self.options.get("algo_type") == "MEAN":
            self.term_subsumption_mean()
        else:
            logging_config.logger.error(
                f"Invalid value for algo_type option, must be 'UNIQUE' or 'MEAN'.")


    def term_subsumption_unique(self) -> None:
        """Find generalisation relations between concepts via term subsumption method.
        This method uses one unique representative word by concept.
        """
        representative_terms_pairs = combinations(self.representative_terms, 2)
        for term_source, term_destination in tqdm(representative_terms_pairs):
            score_cooccurrence = self.pair_terms_count.get((term_source.value, term_destination.value), self.pair_terms_count.get((term_destination.value, term_source.value)))
            subsumption_score = self._compute_subsumption(score_cooccurrence, self.terms_count[term_destination.value])
            inverse_subsumption_score = self._compute_subsumption(score_cooccurrence, self.terms_count[term_source.value])
            if self._verify_threshold(subsumption_score, inverse_subsumption_score):
                self._create_generalisation_relation(term_source.concept_id, term_destination.concept_id)
            elif self._verify_threshold(inverse_subsumption_score, subsumption_score):
                self._create_generalisation_relation(term_destination.concept_id, term_source.concept_id)

    def term_subsumption_mean(self) -> None:
        """Find generalisation relations between concepts via term subsumption method.
        This method computes for each concept the percentage of words generalising words in another concept.
        """
        concept_pairs = list(combinations(self.kr.concepts, 2))
        for pair in tqdm(concept_pairs):
            concept_representative_terms = list(filter(lambda rt: rt.concept_id == pair[0].uid, self.representative_terms))
            other_concept_representative_terms = list(filter(lambda rt: rt.concept_id == pair[1].uid, self.representative_terms))
            score_concept_gen = self._compute_general_words_percentage(concept_representative_terms, other_concept_representative_terms)
            score_other_concept_gen = self._compute_general_words_percentage(other_concept_representative_terms, concept_representative_terms)
            if self._check_concept_more_general(score_concept_gen, score_other_concept_gen):
                self._create_generalisation_relation(pair[0].uid, pair[1].uid)
            elif self._check_concept_more_general(score_other_concept_gen, score_concept_gen):
                self._create_generalisation_relation(pair[1].uid, pair[0].uid)