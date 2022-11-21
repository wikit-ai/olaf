import numpy as np
import spacy.tokens
from statistics import mean
from typing import Any,Dict,List
import uuid

from commons.ontology_learning_schema import Concept, KR, MetaRelation
from concept_hierarchy.concept_hierarchy_schema import RepresentativeTerm
import config.logging_config as logging_config


class TermSubsumption():
    """Algorithm that find generalisation meta relations with subsumption method.
    """

    def __init__(self, corpus: List[spacy.tokens.doc.Doc], kr : KR, threshold : float, use_lemma: bool) -> None:
        """Initialisation.

        Parameters
        ----------
        corpus : List[spacy.tokens.doc.Doc]
            Corpus used to find generalisation relations.
        kr : KR
            Existing knowledge representation of the corpus.
        threshold : float
            Validation threshold for subsumption score
        use_lemma : boolean
            Define if term are identified in doc from lemmas or from raw text values.
        """
        self.corpus = corpus
        self.kr = kr
        self.representative_terms = []
        self.terms_count = {}
        self.threshold = threshold
        self.use_lemma = use_lemma

        try : 
            self.representative_terms = self._get_representative_terms()
        except Exception as _e:
            logging_config.logger.error("Could not set representative terms. Trace : %s", _e)
        else:
            logging_config.logger.info("Representative terms set.")

        try : 
            self.terms_count = self._get_terms_count()
        except Exception as _e: 
            logging_config.logger.error("Could not set terms count. Trace: %s", _e)
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
            else : 
                try : 
                    term = self._get_most_representative_term(concept)
                except Exception as _e: 
                    term = None
                    logging_config.logger.error("Could not get most representative term. Trace: %s", _e)
                else : 
                    logging_config.logger.info("Most representative term found.")
            
            if term is not None :
                representative_term_value = term
                representative_term_concept_id = concept.uid
                representative_term = RepresentativeTerm(representative_term_value,representative_term_concept_id)
                representative_terms.append(representative_term)
        return representative_terms

    def _get_terms_count(self) -> Dict[str,int]:
        """Get for each terms the number of documents contaning the term.

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
        try : 
            similarity = np.dot(term1_vector, term2_vector) / (np.linalg.norm(term1_vector) * np.linalg.norm(term2_vector))
            # Cast result from numpy.float32 to float
            similarity = similarity.item()
        except Exception as _e:
            logging_config.logger.error(f"Could not compute similarity. Trace : {_e}")
            similarity = 0
        return similarity
    
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
        vocab = self.corpus[0].vocab
        if len(vocab)>0 :
            terms_mean_similarity = []
            for i,term in enumerate (concept.terms):
                term_vector = vocab.get_vector(term)
                term_similarities = [self._compute_similarity_between_tokens(term_vector,vocab.get_vector(other_term)) for other_term in self._find_other_terms(i,concept.terms)]
                terms_mean_similarity.append(mean(term_similarities))
            index_most_representative_term = np.argmax(terms_mean_similarity)
            most_representative_term = list(concept.terms)[index_most_representative_term]
        else : 
            logging_config.logger.error("Spacy vocabulary not loaded.")
            raise ValueError
        return most_representative_term

    def _count_doc_with_term(self, term: str) -> int:
        """Count the number of documents contaning the term in parameter.

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
            if self.use_lemma:
                doc_words = [token.lemma_ for token in doc]
            else : 
                doc_words = doc.text.split()
            if term in doc_words:
                count +=1
        return count

    def _count_doc_with_both_terms(self, term1: str, term2: str) -> int:
        """Count the number of documents contaning both terms in parameter.

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
            if self.use_lemma:
                doc_words = [token.lemma_ for token in doc]
            else : 
                doc_words = doc.text.split()
            if (term1 in doc_words) and (term2 in doc_words):
                count +=1
        return count

    def _compute_subsumption(self,nb_doc_coocurence: int, nb_doc_occurence: int) -> float:
        """Compute subsumption score between two terms.

        Parameters
        ----------
        nb_doc_coocurence : int
            Number of docs that contains both terms.
        nb_doc_occurence : int
            Number of docs that contains the most specific term.

        Returns
        -------
        float
            Score of generalisation between term1 and term2.
        """
        if not(nb_doc_occurence == 0):
            subsumption_score = nb_doc_coocurence/nb_doc_occurence
        else :
            subsumption_score = 0
            raise ZeroDivisionError
        return subsumption_score

    def _verify_threshold(self,subsumption: float, inverse_subsumption: float) -> bool:
        """Verify that terms respect generalisation rules (x more general than y) that is to say :
        - subsumption (P(x,y)) > t (a given threshold)
        - subsumption (P(x,y)) > inverse_subsumption (P(y,x))

        Parameters
        ----------
        subsumption : float
            Number of docs containing both terms divided by number of docs containing the most specialised term.
        inverse_subsumption : float
            Number of docs containing both terms divided by number of docs containing the most general term.

        Returns
        -------
        bool
            True if rules are respected, that is to say there is a generalisation relation. False otherwise.
        """
        if (subsumption > self.threshold) and (subsumption > inverse_subsumption):
            return True
        else : 
            return False
    
    def _find_other_terms(self, term_index: int, terms: List[Any]) -> List[Any]:
        """Duplicate list without the specified index.

        Parameters
        ----------
        term_index : int
            Index of non-wanted object.

        Returns
        -------
        List[Any]
            List built.
        """
        if term_index >= len(terms):
            logging_config.logger.error("Could not find other terms. You should check the term index.")
            raise IndexError

        try : 
            if not(type(terms)==List):
                terms = list(terms)
            other_terms = [term for term in terms[:term_index]+terms[term_index+1:]]
        except IndexError as _e: 
            other_terms = []
            logging_config.logger.error("Could not find other terms. You should check the term index. Trace: %s", _e)
        return other_terms

    def _create_generalisation_relation(self,source_id : str, destination_id : str) -> MetaRelation:
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
        meta_relation_source = source_id
        meta_relation_destination = destination_id
        meta_relation_type = "generalisation"
        meta_relation = MetaRelation(meta_relation_uid,meta_relation_source,meta_relation_destination,meta_relation_type)
        return meta_relation

    def term_subsumption(self):
        """Find generalisation relations between concepts from term representation via term subsumption method.
        """
        for i,term in enumerate (self.representative_terms):
            for pair_term in self._find_other_terms(i, self.representative_terms):
                score_coocurence = self._count_doc_with_both_terms(term.value,pair_term.value)
                subsumption = self._compute_subsumption(score_coocurence,self.terms_count[pair_term.value])
                inverse_subsumption = self._compute_subsumption(score_coocurence,self.terms_count[term.value])
                if self._verify_threshold(subsumption,inverse_subsumption):
                    meta_relation = self._create_generalisation_relation(term.concept_id,pair_term.concept_id)
                    self.kr.meta_relations.add(meta_relation)