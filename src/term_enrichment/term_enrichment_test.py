import requests
import unittest

from commons.ontology_learning_repository import conceptnet_api_fetch_term, get_paginated_conceptnet_edges
from commons.ontology_learning_schema import CandidateTerm
from term_enrichment.term_enrichment_methods.conceptnet_enrichment import ConceptNetTermEnrichment
from term_enrichment.term_enrichment_methods.wordnet_enrichment import WordNetTermEnrichment
from term_enrichment.term_enrichment_schema import ConceptNetTermData


class TestWordNetTermEnrichment(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        self.raw_terms = [
            "screw",
            "bolts",
            "nuts",
            "screwdriver",
            "circuit breaker"
        ]
        self.candidate_terms = [CandidateTerm(term) for term in self.raw_terms]

        self.wordnet_term_enrich_options = {
            "wordnet_domain_path": "demo_data/wordnet_domains.txt",
            "lang": "en",
            "use_domains": True,
            "use_pos": True,
            "enrichment_domains": None,
            "enrichment_domains_file": "demo_data/wn_enrichment_domains.txt",
            "synset_pos": ["NOUN"]
        }

    def test_get_enrichment_domains(self) -> None:
        pass

    def test_check_domains_exist(self) -> None:
        pass

    def test_get_wordnet_pos(self) -> None:
        pass

    def test_get_domains_for_synset(self) -> None:
        pass

    def test_find_wordnet_domains(self) -> None:
        pass

    def test_filter_synsets_on_domains(self) -> None:
        pass

    def test_enrich_candidate_term(self) -> None:
        pass

    def test_enrich_candidate_terms(self) -> None:
        pass

    def test_get_lemmas_texts(self) -> None:
        pass

    def test_get_term_synonyms(self) -> None:
        pass

    def test_get_term_hypernyms(self) -> None:
        pass

    def test_get_term_hyponyms(self) -> None:
        pass

    def test_get_term_antonyms(self) -> None:
        pass

    def test_get_term_wordnet_synsets(self) -> None:
        pass

    def test_get_synset_hypernyms(self) -> None:
        pass

    def test_get_synset_hyponyms(self) -> None:
        pass


class TestConceptNetTermEnrichment(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        self.test_options = {
            "lang": "en",
            "api_resp_batch_size": 10
        }

        self.conceptnet_term_enricher = ConceptNetTermEnrichment(
            self.test_options)

        self.conceptnet_screw_data = self.conceptnet_term_enricher._get_term_conceptnet_data(
            "screw")
        self.air_pump_conceptnet = requests.get(
            "https://api.conceptnet.io/c/en/air_pump").json()

    def test_get_term_conceptnet_data(self) -> None:
        conceptnet_ion_pump_data = self.conceptnet_term_enricher._get_term_conceptnet_data(
            "ion_pump")

        self.assertIsInstance(conceptnet_ion_pump_data, ConceptNetTermData)
        self.assertEqual(
            conceptnet_ion_pump_data.conceptnet_id, "/c/en/ion_pump")

        conceptnet_error_data = self.conceptnet_term_enricher._get_term_conceptnet_data(
            "912")

        self.assertIsNone(conceptnet_error_data)

    def test_get_paginated_conceptnet_edges(self) -> None:
        air_pump_edges = get_paginated_conceptnet_edges(
            self.air_pump_conceptnet["view"], self.test_options["api_resp_batch_size"])

        self.assertGreater(len(air_pump_edges), 0)
        self.assertEqual(air_pump_edges[0]["@type"], "Edge")

    def test_filter_edges(self) -> None:
        air_pump_conceptnet_edges = self.conceptnet_term_enricher._filter_edges(
            self.air_pump_conceptnet["edges"])

        relations_2_keep = {"/r/Synonym", "/r/IsA", "/r/FormOf", "/r/Antonym"}

        for edge in air_pump_conceptnet_edges:
            self.assertEqual(edge.start_node_lang,
                             self.test_options["lang"])
            self.assertEqual(edge.end_node_lang,
                             self.test_options["lang"])
            self.assertIn(edge.edge_rel_id, relations_2_keep)

    def test_get_term_synonyms_from_syn_edges(self) -> None:
        screw_synonyms = self.conceptnet_term_enricher._get_term_synonyms_from_syn_edges(
            self.conceptnet_screw_data.synonym_edges)

        self.assertIn("screw propeller", screw_synonyms)

    def test_get_term_synonyms_from_formof_edges(self) -> None:
        screw_synonyms = self.conceptnet_term_enricher._get_term_synonyms_from_formof_edges(
            self.conceptnet_screw_data.formof_edges)

        self.assertIn("screws", screw_synonyms)

    def test_get_term_hypernyms_from_isa_edges(self) -> None:
        screw_hypernyms = self.conceptnet_term_enricher._get_term_hypernyms_from_isa_edges(self.conceptnet_screw_data.conceptnet_id,
                                                                                           self.conceptnet_screw_data.isa_edges)

        self.assertIn("tool", screw_hypernyms)

    def test_get_term_antonyms_from_anto_edges(self) -> None:
        screw_hyponyms = self.conceptnet_term_enricher._get_term_hyponyms_from_isa_edges(self.conceptnet_screw_data.conceptnet_id,
                                                                                         self.conceptnet_screw_data.isa_edges)

        self.assertIn("setscrew", screw_hyponyms)

    def test_get_term_antonyms_from_anto_edges(self) -> None:
        screw_antonyms = self.conceptnet_term_enricher._get_term_antonyms_from_anto_edges(self.conceptnet_screw_data.conceptnet_id,
                                                                                          self.conceptnet_screw_data.antonym_edges)

        self.assertIn("unscrew", screw_antonyms)


if __name__ == '__main__':
    unittest.main()
