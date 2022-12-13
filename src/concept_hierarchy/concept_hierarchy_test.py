from numpy import mean as np_mean
import spacy
import unittest
import uuid

from concept_hierarchy.concept_hierarchy_schema import RepresentativeTerm
from concept_hierarchy.concept_hierarchy_methods.term_subsumption import TermSubsumption
from commons.ontology_learning_schema import Concept, MetaRelation, KR

class TestTermSubsumption(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        corpus = [
            "Première phrase de test",
            "J'adore faire des tests",
            "Aide moi à écrire des premières phrases de test"
        ]
        corpus_preprocessed = []
        self.spacy_model = spacy.load("fr_core_news_md")
        for spacy_document in self.spacy_model.pipe(corpus):
            corpus_preprocessed.append(spacy_document)
        kr = KR()
        self.concept_phrase_span = Concept(uuid.uuid4(), {"premier phrase"})
        self.concept_phrase = Concept(uuid.uuid4(), {"phrase"})
        self.concept_test = Concept(uuid.uuid4(), {"test"})
        kr.concepts.add(self.concept_phrase_span)
        kr.concepts.add(self.concept_phrase)
        kr.concepts.add(self.concept_test)

        options = {
            'subsumption_threshold' : 0.45,
            'use_lemma' : True,
            'use_span' : True,
            'mean_high_threshold' : 0.5,
            'mean_low_threshold' : 0.3
        }

        self.term_sub = TermSubsumption(corpus_preprocessed, kr, options)


    def test_get_representative_terms(self) -> None:
        self.term_sub.options['use_span'] = True
        representative_terms = set()
        representative_terms.add(RepresentativeTerm("premier phrase", self.concept_phrase_span.uid))
        representative_terms.add(RepresentativeTerm("phrase", self.concept_phrase.uid))
        representative_terms.add(RepresentativeTerm("test", self.concept_test.uid))
        self.assertSetEqual(representative_terms, set(self.term_sub._get_representative_terms()))

        concept_multiple_terms = Concept(uuid.uuid4(), {"téléphone portable", "mobile", "smartphone", "portable"})
        self.term_sub.kr.concepts.add(concept_multiple_terms)
        representative_terms.add(RepresentativeTerm("téléphone portable", concept_multiple_terms.uid))
        self.assertSetEqual(representative_terms, set(self.term_sub._get_representative_terms()))

        self.term_sub.options['use_span'] = False
        representative_terms.remove(RepresentativeTerm("téléphone portable", concept_multiple_terms.uid))
        representative_terms.add(RepresentativeTerm("smartphone", concept_multiple_terms.uid))
        self.assertSetEqual(representative_terms, set(self.term_sub._get_representative_terms()))

        self.term_sub.kr.concepts.remove(concept_multiple_terms)

    def test_compute_terms_cout(self) -> None:
        self.term_sub.options['use_span'] = True
        self.term_sub.options['use_lemma'] = True
        terms_count_span = {"premier phrase":2, "phrase": 2, "test":3}
        pair_terms_count = {('test', 'phrase'): 2, ('test', 'premier phrase'): 2, ('phrase', 'premier phrase'): 2, ('phrase','test'): 2, ('premier phrase', 'test'): 2, ('premier phrase', 'phrase'): 2}
        self.term_sub.terms_count.clear()
        self.term_sub.pair_terms_count.clear()
        self.term_sub._compute_terms_cout()
        self.assertDictEqual(terms_count_span, self.term_sub.terms_count)
        for pair_terms in self.term_sub.pair_terms_count:
            self.assertIn(pair_terms, pair_terms_count)

        self.term_sub.options['use_lemma'] = False
        terms_count_span = {"premier phrase":0, "phrase": 1, "test":2}
        pair_terms_count = {('test', 'phrase'): 1, ('phrase', 'test'): 1, ('test', 'premier phrase'): 0, ('premier phrase', 'test'): 0, ('phrase', 'premier phrase'): 0, ('premier phrase', 'phrase'): 0}
        self.term_sub.terms_count.clear()
        self.term_sub.pair_terms_count.clear()
        self.term_sub._compute_terms_cout()
        self.assertDictEqual(terms_count_span, self.term_sub.terms_count)
        for pair_terms in self.term_sub.pair_terms_count:
            self.assertIn(pair_terms, pair_terms_count)

        self.term_sub.options['use_span'] = False
        self.term_sub.terms_count.clear()
        self.term_sub.pair_terms_count.clear()
        self.term_sub._compute_terms_cout()
        self.assertDictEqual(terms_count_span, self.term_sub.terms_count)
        for pair_terms in self.term_sub.pair_terms_count:
            self.assertIn(pair_terms, pair_terms_count)

        self.term_sub.options['use_lemma'] = True
        self.term_sub.options['use_span'] = False
        terms_count_span = {"premier phrase":2, "phrase": 2, "test":3}
        pair_terms_count = {('test', 'phrase'): 2, ('test', 'premier phrase'): 2, ('phrase', 'premier phrase'): 2, ('phrase','test'): 2, ('premier phrase', 'test'): 2, ('premier phrase', 'phrase'): 2}
        self.term_sub.terms_count.clear()
        self.term_sub.pair_terms_count.clear()
        self.term_sub._compute_terms_cout()
        self.assertDictEqual(terms_count_span, self.term_sub.terms_count)
        for pair_terms in self.term_sub.pair_terms_count:
            self.assertIn(pair_terms, pair_terms_count)

        self.term_sub.options['use_span'] = True

    def test_compute_similarity_between_tokens(self) -> None:
        telephone = self.spacy_model("téléphone")
        mobile = self.spacy_model("mobile")
        self.assertEqual(round(telephone.similarity(mobile), 5), round(self.term_sub._compute_similarity_between_tokens(telephone.vector, mobile.vector), 5))

    def test_get_representative_vector(self) -> None:
        self.term_sub.options['use_span'] = True
        one_word = self.spacy_model("phrase")
        self.assertListEqual(list(one_word.vector), list(self.term_sub._get_representative_vector("phrase", self.term_sub.corpus[0].vocab)))
        
        multi_words = self.spacy_model("première phrase")
        self.assertListEqual(list(multi_words.vector), list(self.term_sub._get_representative_vector("première phrase", self.term_sub.corpus[0].vocab)))
    
        self.term_sub.options['use_span'] = False
        self.assertListEqual(list(one_word.vector), list(self.term_sub._get_representative_vector("phrase", self.term_sub.corpus[0].vocab)))
        self.assertListEqual([0]*300, list(self.term_sub._get_representative_vector("première phrase", self.term_sub.corpus[0].vocab)))

    def test_get_most_representative_terms(self) -> None:
        self.term_sub.options['use_span'] = True
        concept_multiple_terms_span = Concept(uuid.uuid4(), {"téléphone portable", "mobile", "smartphone","portable"})
        self.assertEqual(self.term_sub._get_most_representative_term(concept_multiple_terms_span), "téléphone portable")

        self.term_sub.options['use_span'] = False
        self.assertEqual(self.term_sub._get_most_representative_term(concept_multiple_terms_span), "smartphone")

    def test_verify_threshold(self) -> None:
        self.assertTrue(self.term_sub._verify_threshold(0.8, 0.4))
        self.assertFalse(self.term_sub._verify_threshold(0.4, 0.2))
        self.assertFalse(self.term_sub._verify_threshold(0.6, 0.9))
        self.assertFalse(self.term_sub._verify_threshold(0.2, 0.4))

    def test_find_other_terms(self) -> None:
        rep_terms = self.term_sub.representative_terms
        self.assertListEqual(self.term_sub._find_other_terms(0, self.term_sub.representative_terms),[rep_terms[1], rep_terms[2]])
        self.assertListEqual(self.term_sub._find_other_terms(4, self.term_sub.representative_terms),[])
        self.assertListEqual(self.term_sub._find_other_terms(1, self.term_sub.representative_terms), [rep_terms[0], rep_terms[2]])

    def test_compute_subsumption(self) -> None:
        cooccurrence = 10
        occurrence = 5
        sub_score = 2
        self.assertEqual(self.term_sub._compute_subsumption(cooccurrence, occurrence), sub_score)

        wrong_occurrence = 0
        self.assertEqual(self.term_sub._compute_subsumption(cooccurrence, wrong_occurrence), 0)

    def test_create_generalisation_relation(self) -> None:
        source_id = uuid.uuid4()
        destination_id = uuid.uuid4()
        nb_meta_relation_before = len(self.term_sub.kr.meta_relations)
        self.term_sub._create_generalisation_relation(source_id, destination_id)
        self.assertEqual(len(self.term_sub.kr.meta_relations), nb_meta_relation_before + 1)

    def test_term_subsumption_unique(self) -> None:
        pass

    def test_check_concept_more_general(self) -> None:
        self.assertTrue(self.term_sub._check_concept_more_general(0.7, 0.2))
        self.assertFalse(self.term_sub._check_concept_more_general(0.7, 0.4))
        self.assertFalse(self.term_sub._check_concept_more_general(0.5, 0.2))
        self.assertFalse(self.term_sub._check_concept_more_general(0.3, 0.6))

    def test_compute_general_words_percentage(self) -> None:
        pass
    
    def test_term_subsumption_mean(self) -> None:
        pass

if __name__ == '__main__':
    unittest.main()