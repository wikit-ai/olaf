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


    def test_get_representative_terms(self):
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

    def test_get_terms_count(self):
        self.term_sub.options['use_span'] = True
        self.term_sub.options['use_lemma'] = True
        terms_count_span = {"premier phrase":2, "phrase": 2, "test":3}
        self.assertDictEqual(terms_count_span, self.term_sub._get_terms_count())
        self.term_sub.options['use_lemma'] = False
        terms_count_span = {"premier phrase":0, "phrase": 1, "test":2}
        self.assertDictEqual(terms_count_span, self.term_sub._get_terms_count())

        self.term_sub.options['use_span'] = False
        self.assertDictEqual(terms_count_span, self.term_sub._get_terms_count())
        self.term_sub.options['use_lemma'] = True
        terms_count_span = {"premier phrase":0, "phrase": 2, "test":3}
        self.assertDictEqual(terms_count_span, self.term_sub._get_terms_count())

        self.term_sub.options['use_span'] = True

    def test_compute_similarity_between_tokens(self):
        telephone = self.spacy_model("téléphone")
        mobile = self.spacy_model("mobile")
        self.assertEqual(round(telephone.similarity(mobile), 5), round(self.term_sub._compute_similarity_between_tokens(telephone.vector, mobile.vector), 5))

    def test_get_representative_vector(self):
        self.term_sub.options['use_span'] = True
        one_word = self.spacy_model("phrase")
        self.assertListEqual(list(one_word.vector), list(self.term_sub._get_representative_vector("phrase", self.term_sub.corpus[0].vocab)))
        
        multi_words = self.spacy_model("première phrase")
        self.assertListEqual(list(multi_words.vector), list(self.term_sub._get_representative_vector("première phrase", self.term_sub.corpus[0].vocab)))
    
        self.term_sub.options['use_span'] = False
        self.assertListEqual(list(one_word.vector), list(self.term_sub._get_representative_vector("phrase", self.term_sub.corpus[0].vocab)))
        self.assertListEqual([0]*300, list(self.term_sub._get_representative_vector("première phrase", self.term_sub.corpus[0].vocab)))

    def test_get_most_representative_terms(self):
        self.term_sub.options['use_span'] = True
        concept_multiple_terms_span = Concept(uuid.uuid4(), {"téléphone portable", "mobile", "smartphone","portable"})
        self.assertEqual(self.term_sub._get_most_representative_term(concept_multiple_terms_span), "téléphone portable")

        self.term_sub.options['use_span'] = False
        self.assertEqual(self.term_sub._get_most_representative_term(concept_multiple_terms_span), "smartphone")
    
    def test_check_term_in_doc(self):
        doc_words = ["première", "phrase", "de", "test"]
        self.term_sub.options['use_span'] = True
        self.assertTrue(self.term_sub._check_term_in_doc("test", doc_words))
        self.assertTrue(self.term_sub._check_term_in_doc("première phrase", doc_words))
        self.assertTrue(self.term_sub._check_term_in_doc("phrase", doc_words))

        self.term_sub.options['use_span'] = False
        self.assertTrue(self.term_sub._check_term_in_doc("test", doc_words))
        self.assertFalse(self.term_sub._check_term_in_doc("première phrase", doc_words))
        self.assertTrue(self.term_sub._check_term_in_doc("phrase", doc_words))

    def test_count_doc_with_term(self):
        self.term_sub.options['use_span'] = True
        self.term_sub.options['use_lemma'] = True
        self.assertEqual(self.term_sub._count_doc_with_term("test"), 3)
        self.assertEqual(self.term_sub._count_doc_with_term("premier phrase"), 2)
        self.assertEqual(self.term_sub._count_doc_with_term("luciolle"), 0)

        self.term_sub.options['use_lemma'] = False
        self.assertEqual(self.term_sub._count_doc_with_term("test"), 2)
        self.assertEqual(self.term_sub._count_doc_with_term("premier phrase"), 0)
        self.assertEqual(self.term_sub._count_doc_with_term("luciolle"), 0)

        self.term_sub.options['use_span'] = False
        self.term_sub.options['use_lemma'] = True
        self.assertEqual(self.term_sub._count_doc_with_term("test"), 3)
        self.assertEqual(self.term_sub._count_doc_with_term("premier phrase"), 0)
        self.assertEqual(self.term_sub._count_doc_with_term("luciolle"), 0)

        self.term_sub.options['use_lemma'] = False
        self.assertEqual(self.term_sub._count_doc_with_term("test"), 2)
        self.assertEqual(self.term_sub._count_doc_with_term("premier phrase"), 0)
        self.assertEqual(self.term_sub._count_doc_with_term("luciolle"), 0)

    def test_count_doc_with_both_terms(self):
        self.term_sub.options['use_span'] = True
        self.term_sub.options['use_lemma'] = True
        self.assertEqual(self.term_sub._count_doc_with_both_terms("premier phrase", "test"), 2)
        self.assertEqual(self.term_sub._count_doc_with_both_terms("phrase", "test"), 2)
        self.assertEqual(self.term_sub._count_doc_with_both_terms("luciolle", "test"), 0)

        self.term_sub.options['use_lemma'] = False
        self.assertEqual(self.term_sub._count_doc_with_both_terms("premier phrase", "test"), 0)
        self.assertEqual(self.term_sub._count_doc_with_both_terms("phrase", "test"), 1)
        self.assertEqual(self.term_sub._count_doc_with_both_terms("luciolle", "test"), 0)

        self.term_sub.options['use_span'] = False
        self.term_sub.options['use_lemma'] = True
        self.assertEqual(self.term_sub._count_doc_with_both_terms("premier phrase", "test"), 0)
        self.assertEqual(self.term_sub._count_doc_with_both_terms("phrase", "test"), 2)
        self.assertEqual(self.term_sub._count_doc_with_both_terms("luciolle", "test"), 0)

        self.term_sub.options['use_lemma'] = False
        self.assertEqual(self.term_sub._count_doc_with_both_terms("premier phrase", "test"), 0)
        self.assertEqual(self.term_sub._count_doc_with_both_terms("phrase", "test"), 1)
        self.assertEqual(self.term_sub._count_doc_with_both_terms("luciolle", "test"), 0)

    def test_verify_threshold(self):
        self.assertTrue(self.term_sub._verify_threshold(0.8, 0.4))
        self.assertFalse(self.term_sub._verify_threshold(0.4, 0.2))
        self.assertFalse(self.term_sub._verify_threshold(0.6, 0.9))
        self.assertFalse(self.term_sub._verify_threshold(0.2, 0.4))

    def test_find_other_terms(self):
        rep_terms = self.term_sub.representative_terms
        self.assertListEqual(self.term_sub._find_other_terms(0, self.term_sub.representative_terms),[rep_terms[1], rep_terms[2]])
        self.assertListEqual(self.term_sub._find_other_terms(4, self.term_sub.representative_terms),[])
        self.assertListEqual(self.term_sub._find_other_terms(1, self.term_sub.representative_terms), [rep_terms[0], rep_terms[2]])

    def test_compute_subsumption(self):
        cooccurrence = 10
        occurrence = 5
        sub_score = 2
        self.assertEqual(self.term_sub._compute_subsumption(cooccurrence, occurrence), sub_score)

        wrong_occurrence = 0
        self.assertEqual(self.term_sub._compute_subsumption(cooccurrence, wrong_occurrence), 0)


    def test_create_generalisation_relation(self):
        source_id = uuid.uuid4()
        destination_id = uuid.uuid4()
        relation_type = "generalisation"
        test_meta_relation = self.term_sub._create_generalisation_relation(source_id, destination_id)
        self.assertIsInstance(test_meta_relation, MetaRelation)
        self.assertEqual(test_meta_relation.source_concept_id, source_id)
        self.assertEqual(test_meta_relation.destination_concept_id, destination_id)
        self.assertEqual(test_meta_relation.relation_type, relation_type)

    def test_term_subsumption_unique(self):
        self.term_sub.options['use_span'] = True
        self.term_sub.term_subsumption_unique()
        test_kr = KR()
        test_kr.concepts.add(Concept(self.concept_phrase_span.uid, self.concept_phrase_span.terms))
        test_kr.concepts.add(Concept(self.concept_phrase.uid, self.concept_phrase.terms))
        test_kr.concepts.add(Concept(self.concept_test.uid, self.concept_test.terms))
        test_meta_relation_id_0 = list(self.term_sub.kr.meta_relations)[0].uid
        test_meta_relation_dest_0 = list(self.term_sub.kr.meta_relations)[0].destination_concept_id 
        test_meta_relation_id_1 = list(self.term_sub.kr.meta_relations)[1].uid
        test_meta_relation_dest_1 = list(self.term_sub.kr.meta_relations)[1].destination_concept_id
        test_kr.meta_relations.add(MetaRelation(test_meta_relation_id_0, self.concept_test.uid, test_meta_relation_dest_0, "generalisation"))
        test_kr.meta_relations.add(MetaRelation(test_meta_relation_id_1, self.concept_test.uid, test_meta_relation_dest_1, "generalisation"))
        self.assertEqual(self.term_sub.kr, test_kr)

        self.term_sub.options['use_span'] = False
        self.term_sub.kr.meta_relations.clear()
        self.term_sub.term_subsumption_unique()
        test_kr = KR()
        test_kr.concepts.add(Concept(self.concept_phrase_span.uid, self.concept_phrase_span.terms))
        test_kr.concepts.add(Concept(self.concept_phrase.uid, self.concept_phrase.terms))
        test_kr.concepts.add(Concept(self.concept_test.uid, self.concept_test.terms))
        test_meta_relation_id = list(self.term_sub.kr.meta_relations)[0].uid
        test_kr.meta_relations.add(MetaRelation(test_meta_relation_id, self.concept_test.uid, self.concept_phrase.uid, "generalisation"))
        self.assertEqual(self.term_sub.kr, test_kr)

    def test_check_concept_more_general(self):
        self.assertTrue(self.term_sub._check_concept_more_general(0.7, 0.2))
        self.assertFalse(self.term_sub._check_concept_more_general(0.7, 0.4))
        self.assertFalse(self.term_sub._check_concept_more_general(0.5, 0.2))
        self.assertFalse(self.term_sub._check_concept_more_general(0.3, 0.6))

    def test__compute_general_words_percentage(self):
        pass
    
    def test_term_subsumption_mean(self):
        pass

if __name__ == '__main__':
    unittest.main()