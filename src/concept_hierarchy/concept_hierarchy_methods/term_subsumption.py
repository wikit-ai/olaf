from concept_hierarchy.concept_hierarchy_schema import Concept, KR, MetaRelation, RepresentativeTerm
import spacy.tokens
from typing import Dict,List
import uuid

from config.core import config

class TermSubsumption():
    """Algorithm that find generalisation meta relations with subsumption method.
    """

    def __init__(self, corpus: List[spacy.tokens.doc.Doc], kr : KR) -> None:
        """Initialisation.

        Parameters
        ----------
        corpus : List[spacy.tokens.doc.Doc]
            Corpus used to find generalisation relations.
        kr : KR
            Existing knowledge representation of the corpus.
        """
        self.corpus = corpus
        self.kr = kr
        self.representative_terms = self._set_representative_terms()
        self.terms_count = self._set_terms_count()

    def _set_representative_terms(self) -> None:
        """Set one string per concept. This string is the best text representation of the concept among all its terms.
        """
        for concept in self.kr.concepts:
            if len(concept.terms) == 1:
                term = list(concept.terms)[0]
            else : 
                term = self._get_most_representative_term(concept)
            representative_term = RepresentativeTerm()
            representative_term.value = term
            representative_term.concept_id = concept.uid
            self.representative_terms.append(representative_term)

    def _set_terms_count(self):
        """Set for each terms the number of documents contaning the term.
        """
        for term in self.representative_terms:
            self.terms_count[term.value] = self._count_doc_with_term(term.value)

    def _get_most_representative_term(self, concept: Concept) -> str:
        return ""

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
            if term in doc:
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
            if (term1 in doc) and (term2 in doc):
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
        subsumption_score = nb_doc_coocurence/nb_doc_occurence
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
        if (subsumption > config['concept_hierachy']['term_subsumption']['threshold']) and (subsumption > inverse_subsumption):
            return True
        else : 
            return False
    
    def _find_other_terms(self, term_index: int) -> List[Dict[str,str]]:
        """Duplicate the representative terms list but without the object at index given as parameter.

        Parameters
        ----------
        term_index : int
            Index of non-wanted object.

        Returns
        -------
        List[Sict[str,str]]
            List of representative terms built.
        """
        return [term for term in self.representative_terms[:term_index]+self.representative_terms[term_index+1:]]

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
        meta_relation = MetaRelation()
        meta_relation.uid = uuid.uuid4()
        meta_relation.source = source_id
        meta_relation.destination = destination_id
        meta_relation.relation_type = "generalisation"
        return meta_relation

    def term_subsumption(self):
        """Find generalisation relations between concepts from term representation via term subsumption method.
        """
        for i,term in enumerate (self.representative_terms):
            for pair_term in self._find_other_terms(i):
                score_coocurence = self._count_doc_with_both_terms(term.string_value,pair_term.string_value)
                subsumption = self._compute_subsumption(score_coocurence,self.terms_count[pair_term.string_value])
                inverse_subsumption = self._compute_subsumption(score_coocurence,self.terms_count[term.string_value])
                if self._verify_threshold(subsumption,inverse_subsumption):
                    meta_relation = self._create_generalisation_relation(term.concept_id,pair_term.concept_id)
                    self.kr.add(meta_relation)