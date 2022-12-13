from itertools import chain, combinations
import spacy.tokens
from typing import Any, Dict, List, Tuple
import uuid

from commons.ontology_learning_schema import Concept, KR, MetaRelation

class OnOccurrenceRelationExtraction():

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
        if not spacy.tokens.span.Span.has_extension('concepts'):
            spacy.tokens.span.Span.set_extension('concepts', default = [])
        if not spacy.tokens.doc.Doc.has_extension('concepts'):
            spacy.tokens.doc.Doc.set_extension('concepts', default = [])

    def _label_doc_with_concept(self) -> None:
        """Create attribute in spacy sentence that contains list of concepts found in the sentence.
        """
        for doc in self.corpus:
            if doc.has_annotation('SENT_START'):
                for sentence in doc.sents :
                    sentence._.concepts = self._find_concepts_in_sentence(sentence)
            else :
                doc._.concepts = self._find_concepts_in_sentence(doc)

    def _find_concepts_in_sentence(self, sentence: spacy.tokens.span.Span) -> List[Concept]:
        """Find concepts in a sentence.

        Parameters
        ----------
        sentence : spacy.tokens.span.Span
            Sentence to analyze.

        Returns
        -------
        List[Concept]
            List of concept identified in a sentence.
        """
        if self.options.get('use_lemma'):
            sentence_words = [token.lemma_ for token in sentence]
        else: 
            sentence_words = [token.text for token in sentence]
        concepts_in_sentence = []
        for concept in self.kr.concepts:
            conditions = [self._term_in_sentence(concept_term, sentence_words) for concept_term in concept.terms]
            if any(conditions) :
                concepts_in_sentence.append(concept)
        return concepts_in_sentence

    def _term_in_sentence(self, term:str, sentence: List[str]) -> bool:
        """Check if a term is in a sentence.
        For term with multiple words, all words must be in the sentence and indexes must follow each other.

        Parameters
        ----------
        term : str
            Term to find.
        sentence : List[str]
            Sentence to analyze.

        Returns
        -------
        bool
            True if the term is in a sentence, false otherwise.
        """
        term_words = term.strip().split()
        term_presence = True
        if term_words[0] in sentence:
            term_index = sentence.index(term_words[0])
            for term in term_words[1:]:
                if (term in sentence) and (sentence.index(term) == term_index+1):
                    term_index += 1
                else:
                    term_presence = False  
                    break           
        else:
            term_presence = False
        return term_presence

    def _compute_concept_cooccurrence(self) -> Dict[Tuple[str], int]:
        """Count concept combination occurrences.

        Returns
        -------
        Dict[Tuple[str], int]
            Dictionnary with tuple of concept combinaison as keys and cooccurrence count as value.
        """
        concepts_in_sentences = []
        for doc in self.corpus:
            if doc.has_annotation('SENT_START'):
                for sent in doc.sents :
                    concepts_in_sentences.append(sent._.get('concepts'))
            else:
                concepts_in_sentences.append(doc._.get('concepts'))

        unique_concepts = set(chain.from_iterable(concepts_in_sentences))
        all_pairs = list(combinations(unique_concepts, 2))
        concepts_cooccurrence = {pair: len([x for x in concepts_in_sentences if set(pair) <= set(x)]) for pair in all_pairs}
        return concepts_cooccurrence

    def _create_relatedto_relation(self, concept_pair: Tuple[Concept]) -> None:
        """Create meta relation concept 

        Parameters
        ----------
        concept_pair : Tuple[Concept]
            Pair of related concepts.
        """
        new_relation = MetaRelation(str(uuid.uuid4()), concept_pair[0].uid, concept_pair[1].uid, "related_to")
        self.kr.meta_relations.add(new_relation)

    def on_occurrence_relation_extraction(self) -> None:
        """Find meta relations based on concept cooccurrence in corpus.
        """
        self._label_doc_with_concept()
        concepts_cooccurrence = self._compute_concept_cooccurrence()
        occurrence_threshold = self.options.get("threshold")
        for concept_pair, score in concepts_cooccurrence.items() :
            if score > occurrence_threshold:
                self._create_relatedto_relation(concept_pair)
