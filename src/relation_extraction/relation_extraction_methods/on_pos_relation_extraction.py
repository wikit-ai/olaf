from collections import Counter
import numpy as np
import spacy.tokens
from typing import Any, Dict, List, Tuple, Union
import uuid

from commons.ontology_learning_schema import CandidateTerm, Concept, KR, Relation
from commons.ontology_learning_utils import check_term_in_content
from term_extraction.term_extraction_service import TermExtraction

class OnPosRelationExtraction():

    def __init__(self, corpus: List[spacy.tokens.doc.Doc], kr: KR, options: Dict[str, Any]) -> None:
        """Initialize on occurrence relation extraction.

        Parameters
        ----------
        corpus : List[spacy.tokens.doc.Doc]
            Corpus preprocessed with spacy.
        kr : KR
            Knowledge representation previously created and used.
        options : Dict[str, Any]
            Class options.
        """
        self.corpus = corpus
        self.kr = kr
        self.options = options
        if self.corpus[0].has_annotation('SENT_START'):
            if not spacy.tokens.span.Span.has_extension('concepts'):
                spacy.tokens.span.Span.set_extension('concepts', default = [])
        else : 
            if not spacy.tokens.doc.Doc.has_extension('concepts'):
                spacy.tokens.doc.Doc.set_extension('concepts', default = [])

    def _check_token_distance_limit(self, index: int, len_content: int) -> bool:
        """Check if conditions on token distance limit are respected or not, that is to say the candidate respects the distance between the concept in front and the concept behind.
        Condition 1 : Current index must be lower than the limit distance so that the candidate is not too far from the first concept.
        Condition 2 : Distance between the candidate and the second concept must be lower than the limit distance so that the candidate is not too far from the second concept.

        Parameters
        ----------
        index : int
            Current index.
        len_content : int
            Length of the content.

        Returns
        -------
        bool
            True is token distance limit is respected, false otherwise.
        """
        conditions_limit = [
            index < self.options.get('token_distance_limit'),
            len_content - 1 - index < self.options.get('token_distance_limit')
        ]
        token_distance_valid = all(conditions_limit) 
        return token_distance_valid

    def _pos_selection_between_concepts(self, content: List[spacy.tokens.token.Token]) -> List[str]:
        """Find relation terms between two concepts based on pos-tagging.

        Parameters
        ----------
        content : List[spacy.tokens.token.Token]
            Content in which relation terms based on pos-tagging are searched.

        Returns
        -------
        List[str]
            List of relation terms based on pos-tagging.
        """
        pos_selection_terms = []
        for i,token in enumerate (content):
            if (self.options.get('token_distance_limit') is not None) and not(self._check_token_distance_limit(i, len(content))):
                break
            elif (token.pos_ in self.options.get('pos_selection')):
                if self.options.get('use_lemma'):
                    pos_selection_terms.append(token.lemma_)
                else :
                    pos_selection_terms.append(token.text)
        return pos_selection_terms

    def _find_concepts(self, content: Union[spacy.tokens.doc.Doc,spacy.tokens.span.Span]) -> List[Dict[str,Any]]:
        """Find concepts in content (all document or sentence).

        Parameters
        ----------
        content : Union[spacy.tokens.doc.Doc,spacy.tokens.span.Span]
            Content in which concepts are searched.

        Returns
        -------
        List[Dict[str,Any]]
            List of concept with their index in the content.
        """
        content_concepts = []
        if self.options.get('use_lemma'):
            words = [token.lemma_ for token in content]
        else : 
            words = [token.text for token in content]
        for concept in self.kr.concepts:
            for term in concept.terms:
                if check_term_in_content(term, words):
                    concept_index = words.index(term.strip().split()[-1])
                    content_concepts.append({"concept":concept, "index":concept_index})           
        return content_concepts
    
    def _find_relation_triplets(self, content: Union[spacy.tokens.doc.Doc,spacy.tokens.span.Span]) -> List[Tuple[Concept, str, Concept]]:
        """Find concepts in content (all document or sentence) and try to find relation between them. 

        Parameters
        ----------
        content : Union[spacy.tokens.doc.Doc,spacy.tokens.span.Span]
            Content in which relations are searched.

        Returns
        -------
        List[Tuple[Concept, str, Concept]]
            List of relation triplets containing the source concept, relation term and destination concept.
        """
        relation_triplets = []
        concepts_with_index = self._find_concepts(content)
        if len(concepts_with_index) > 1:
            ranked_by_index_concepts = sorted(concepts_with_index, key=lambda x: x['index'])
            for i in range (len(ranked_by_index_concepts) - 1):
                pos_selection_terms = self._pos_selection_between_concepts(content[ranked_by_index_concepts[i]['index'] + 1 : ranked_by_index_concepts[i + 1]['index']])
                if len(pos_selection_terms) > 0:
                    for pos_selection_term in pos_selection_terms:
                        relation_triplets.append((ranked_by_index_concepts[i]['concept'], pos_selection_term, ranked_by_index_concepts[i + 1]['concept']))
        return relation_triplets

    def _find_candidate_relations_in_corpus(self) -> List[Tuple[Concept, str, Concept]]:
        """Iterate on corpus to find relations between concepts based on pos tagging.

        Returns
        -------
        List[Tuple[Concept, str, Concept]]
            List of relation triplets containing the source concept, relation term and destination concept.
        """
        candidate_relations = []
        for doc in self.corpus:
            if doc.has_annotation('SENT_START'):
                for sentence in doc.sents:
                    new_candidate_relations = self._find_relation_triplets(sentence)
                    candidate_relations += new_candidate_relations
            else :
                new_candidate_relations = self._find_relation_triplets(doc)
                candidate_relations += new_candidate_relations
        return candidate_relations

    def _validate_relations(self, candidate_relations: List[Tuple[Concept, str, Concept]]) -> List[Tuple[Concept, str, Concept]]:
        """Validate candidates relations based on occurrence threshold if it is defined. 
        Otherwise, keep only one example of redundant relations.

        Parameters
        ----------
        candidate_relations : List[Tuple[Concept, str, Concept]]
            List of relation triplets containing the source concept, relation term and destination concept.

        Returns
        -------
        List[Tuple[Concept, str, Concept]]
            List of validated relation triplet which contains source concept, relation term and destination concept.
        """
        if self.options.get('occurrence_threshold') is not None:
            occurrence_threshold = self.options.get('occurrence_threshold')
            relations_counter = Counter(candidate_relations)
            counter_keys = relations_counter.keys()
            unique_relations = list(filter(lambda relation: relations_counter[relation] >= occurrence_threshold, counter_keys))
        else : 
            unique_relations = list(set(candidate_relations))
        return unique_relations

    def _create_relation(self, relation_triplet: Tuple[Concept, str, Concept]) -> None:
        """Create new relation in the knowledge representation.

        Parameters
        ----------
        relation_triplet : Tuple[Concept, str, Concept]
            Relation triplet containig the source concept, relation term and destination concept.
        """
        relation_uid = str(uuid.uuid4())
        relation_concept_source_uid = relation_triplet[0].uid
        relation_terms = set()
        relation_terms.add(relation_triplet[1])
        relation_concept_destination_uid = relation_triplet[2].uid
        new_relation = Relation(relation_uid, relation_concept_source_uid, relation_concept_destination_uid, relation_terms)
        self.kr.relations.add(new_relation)

    def on_pos_relation_extraction(self) -> None:
        """Find relations between existing concepts in the corpus.
        """
        candidate_relations = self._find_candidate_relations_in_corpus()
        validated_relations = self._validate_relations(candidate_relations)
        for relation in validated_relations:
            self._create_relation(relation)
                        