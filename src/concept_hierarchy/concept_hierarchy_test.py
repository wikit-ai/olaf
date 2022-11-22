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
            "Aide moi à écrire des phrases de test"
        ]
        corpus_preprocessed = []
        spacy_model = spacy.load("fr_core_news_md")
        for spacy_document in spacy_model.pipe(corpus):
                corpus_preprocessed.append(spacy_document)
        kr = KR()
        self.concept_phrase = Concept(uuid.uuid4(),{"phrase"})
        self.concept_test = Concept(uuid.uuid4(),{"test"})
        kr.concepts.add(self.concept_phrase)
        kr.concepts.add(self.concept_test)

        threshold = 0.45
        use_lemma = True
        self.term_sub = TermSubsumption(corpus_preprocessed,kr,threshold,use_lemma)

    def test_get_representative_terms(self):
        representative_terms = set()
        representative_terms.add(RepresentativeTerm("phrase",self.concept_phrase.uid))
        representative_terms.add(RepresentativeTerm("test",self.concept_test.uid))
        self.assertSetEqual(representative_terms, set(self.term_sub._get_representative_terms()))

        concept_multiple_terms = Concept(uuid.uuid4(),{"téléphone","mobile","portable","smartphone"})
        self.term_sub.kr.concepts.add(concept_multiple_terms)
        representative_terms.add(RepresentativeTerm("smartphone",concept_multiple_terms.uid))
        self.assertSetEqual(representative_terms,set(self.term_sub._get_representative_terms()))

        self.term_sub.kr.concepts.remove(concept_multiple_terms)

    def test_get_terms_count(self):
        terms_count = {"phrase":2,"test":3}
        self.assertDictEqual(terms_count, self.term_sub.terms_count)
        self.term_sub.use_lemma = False
        terms_count = {"phrase":1,"test":2}
        self.assertDictEqual(terms_count, self.term_sub._get_terms_count())

    def test_compute_similarity_between_tokens(self):
        spacy_model = spacy.load("fr_core_news_sm")
        telephone = spacy_model("téléphone")
        mobile = spacy_model("mobile")
        self.assertEqual(round(telephone.similarity(mobile),5), round(self.term_sub._compute_similarity_between_tokens(telephone.vector,mobile.vector),5))

    def test_get_most_representative_terms(self):
        concept_multiple_terms = Concept(uuid.uuid4(),{"téléphone","mobile","portable","smartphone"})
        self.assertEqual(self.term_sub._get_most_representative_term(concept_multiple_terms),"smartphone")
    
    def test_count_doc_with_term(self):
        self.term_sub.use_lemma = True
        self.assertEqual(self.term_sub._count_doc_with_term("test"),3)
        self.assertEqual(self.term_sub._count_doc_with_term("phrase"),2)
        self.assertEqual(self.term_sub._count_doc_with_term("luciolle"),0)

        self.term_sub.use_lemma = False
        self.assertEqual(self.term_sub._count_doc_with_term("test"),2)
        self.assertEqual(self.term_sub._count_doc_with_term("phrase"),1)
        self.assertEqual(self.term_sub._count_doc_with_term("luciolle"),0)

    def test_count_doc_with_both_terms(self):
        self.term_sub.use_lemma = True
        self.assertEqual(self.term_sub._count_doc_with_both_terms("phrase","test"),2)
        self.assertEqual(self.term_sub._count_doc_with_both_terms("luciolle","test"),0)

        self.term_sub.use_lemma = False
        self.assertEqual(self.term_sub._count_doc_with_both_terms("phrase","test"),1)
        self.assertEqual(self.term_sub._count_doc_with_both_terms("luciolle","test"),0)

    def test_verify_threshold(self):
        self.assertTrue(self.term_sub._verify_threshold(0.8,0.4))
        self.assertFalse(self.term_sub._verify_threshold(0.4,0.2))
        self.assertFalse(self.term_sub._verify_threshold(0.6,0.9))
        self.assertFalse(self.term_sub._verify_threshold(0.2,0.4))

    def test_find_other_terms(self):
        rep_terms = self.term_sub.representative_terms
        self.assertListEqual(self.term_sub._find_other_terms(0, self.term_sub.representative_terms),[rep_terms[1]])

        with self.assertRaises(IndexError):
            self.term_sub._find_other_terms(3, self.term_sub.representative_terms)

        new_rep_term = RepresentativeTerm("brouillon",uuid.uuid4())
        self.term_sub.representative_terms.append(new_rep_term)
        self.assertListEqual(self.term_sub._find_other_terms(1, self.term_sub.representative_terms),[rep_terms[0],rep_terms[2]])
        self.term_sub.representative_terms.remove(new_rep_term)

    def test_compute_subsumption(self):
        cooccurrence = 10
        occurrence = 5
        sub_score = 2
        self.assertEqual(self.term_sub._compute_subsumption(cooccurrence,occurrence),sub_score)

        wrong_occurrence = 0
        with self.assertRaises(ZeroDivisionError):
            self.term_sub._compute_subsumption(cooccurrence,wrong_occurrence)


    def test_create_generalisation_relation(self):
        source_id = uuid.uuid4()
        destination_id = uuid.uuid4()
        relation_type = "generalisation"
        test_meta_relation = self.term_sub._create_generalisation_relation(source_id,destination_id)
        self.assertIsInstance(test_meta_relation, MetaRelation)
        self.assertEqual(test_meta_relation.source,source_id)
        self.assertEqual(test_meta_relation.destination,destination_id)
        self.assertEqual(test_meta_relation.relation_type,relation_type)

    def test_term_subsumption(self):
        self.term_sub()
        test_kr = KR()
        test_kr.concepts.add(Concept(self.concept_phrase.uid,self.concept_phrase.terms))
        test_kr.concepts.add(Concept(self.concept_test.uid,self.concept_test.terms))
        test_meta_relation_id = list(self.term_sub.kr.meta_relations)[0].uid
        test_kr.meta_relations.add(MetaRelation(test_meta_relation_id,self.concept_test.uid,self.concept_phrase.uid,"generalisation"))

        self.assertEqual(self.term_sub.kr,test_kr)

if __name__ == '__main__':
    unittest.main()