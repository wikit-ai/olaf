from typing import Any, Dict, Set

import pytest

from olaf.data_container.candidate_term_schema import CandidateTerm
from olaf.repository.knowledge_source.conceptnet_kg import ConceptNetKnowledgeResource


@pytest.fixture(scope="session")
def air_pump_c_term_texts() -> Set[str]:
    term_texts = {"air pump", "vacuum pump"}
    return term_texts


class TestDefaultConceptNetKG:
    @pytest.fixture(scope="class")
    def default_conceptnet_kg(self) -> ConceptNetKnowledgeResource:
        params = {}

        kg = ConceptNetKnowledgeResource(**params)

        return kg

    @pytest.fixture(scope="class")
    def conceptnet_api_response(self, default_conceptnet_kg) -> Dict[str, Any]:
        api_response = default_conceptnet_kg._conceptnet_api_fetch_term(
            term_conceptnet_text="rocks", lang="en", batch_size=100
        )
        return api_response

    def test_conceptnet_api_fetch_term(self, conceptnet_api_response) -> None:
        conceptnet_term_edges = conceptnet_api_response.get("edges", [])

        assert len(conceptnet_term_edges) > 0

    def test_get_concept_uris_from_edges(
        self, conceptnet_api_response, default_conceptnet_kg
    ) -> None:
        conceptnet_term_edges = conceptnet_api_response.get("edges", [])

        concept_uris = default_conceptnet_kg._get_concept_uris_from_edges(
            edges=conceptnet_term_edges
        )

        assert len(concept_uris) > 0
        assert "http://fr.wiktionary.org/wiki/rocks" in concept_uris

    def test_get_term_conceptnet_external_uris(self, default_conceptnet_kg) -> None:
        ion_pump_uris = default_conceptnet_kg._get_term_conceptnet_external_uris(
            term_conceptnet_text="ion_pump"
        )

        rocks_uris = default_conceptnet_kg._get_term_conceptnet_external_uris(
            term_conceptnet_text="rocks"
        )

        assert len(ion_pump_uris) > 0
        assert len(rocks_uris) > 0
        assert "http://fr.wiktionary.org/wiki/rocks" in rocks_uris

    def test_get_term_conceptnet_external_uris_unknown_term(
        self, default_conceptnet_kg
    ) -> None:
        unknown_term_uris = default_conceptnet_kg._get_term_conceptnet_external_uris(
            term_conceptnet_text="non existing term"
        )

        assert len(unknown_term_uris) == 0

    def test_get_paginated_conceptnet_edges(self, default_conceptnet_kg) -> None:
        api_response = default_conceptnet_kg._conceptnet_api_fetch_term(
            term_conceptnet_text="motor", lang="en", batch_size=10
        )

        next_edges = default_conceptnet_kg._get_paginated_conceptnet_edges(
            conceptnet_view_res=api_response["view"], batch_size=500
        )

        assert len(next_edges) > 0

    def test_match_external_concepts(
        self, default_conceptnet_kg, air_pump_c_term_texts
    ) -> None:
        c_term_concept_uris = default_conceptnet_kg.match_external_concepts(
            matching_terms=air_pump_c_term_texts
        )

        direct_uris_conditions = [
            "http://wordnet-rdf.princeton.edu/wn31/102695372-n" in c_term_concept_uris,
            "http://en.wiktionary.org/wiki/air_pump" in c_term_concept_uris,
        ]
        synonyms_uris_conditions = [
            "http://dbpedia.org/resource/Vacuum_pump" in c_term_concept_uris,
            "http://wikidata.dbpedia.org/resource/Q745837" in c_term_concept_uris,
        ]

        assert len(c_term_concept_uris) > 0
        assert all(direct_uris_conditions)
        assert all(synonyms_uris_conditions)


class TestConceptNetKGParams:
    @pytest.fixture(scope="class")
    def custom_conceptnet_kg(self) -> ConceptNetKnowledgeResource:
        params = {
            "check_sources": True,
            "validation_sources": {"dbpedia.org", "en.wiktionary.org"},
        }

        kg = ConceptNetKnowledgeResource(**params)

        return kg

    @pytest.fixture(scope="class")
    def conceptnet_api_response(self, custom_conceptnet_kg) -> Dict[str, Any]:
        api_response = custom_conceptnet_kg._conceptnet_api_fetch_term(
            term_conceptnet_text="vacuum_pump", lang="en", batch_size=100
        )
        return api_response

    def test_filter_edges_on_sources(
        self, custom_conceptnet_kg, conceptnet_api_response
    ) -> None:
        filtered_edges = custom_conceptnet_kg._filter_edges_on_sources(
            conceptnet_api_response.get("edges")
        )

        concept_uris = custom_conceptnet_kg._get_concept_uris_from_edges(filtered_edges)

        assert "http://dbpedia.org/resource/Vacuum_pump" in concept_uris
        assert "http://wikidata.dbpedia.org/resource/Q745837" not in concept_uris

    def test_match_external_concepts(self, custom_conceptnet_kg) -> None:
        c_term_concept_uris = custom_conceptnet_kg.match_external_concepts(
            matching_terms={"air pump"}
        )

        assert "http://wikidata.dbpedia.org/resource/Q745837" not in c_term_concept_uris
        assert "http://en.wiktionary.org/wiki/air_pump" in c_term_concept_uris
        assert (
            "http://wordnet-rdf.princeton.edu/wn31/102695372-n"
            not in c_term_concept_uris
        )


class TestConceptNetKGFrench:
    @pytest.fixture(scope="class")
    def fr_conceptnet_kg(self) -> ConceptNetKnowledgeResource:
        params = {"lang": "fr"}

        kg = ConceptNetKnowledgeResource(**params)

        return kg

    @pytest.fixture(scope="class")
    def conceptnet_api_response(self, fr_conceptnet_kg) -> Dict[str, Any]:
        api_response = fr_conceptnet_kg._conceptnet_api_fetch_term(
            term_conceptnet_text="vacuum_pump", lang="en", batch_size=100
        )
        return api_response

    def test_conceptnet_api_fetch_term(self, conceptnet_api_response) -> None:
        conceptnet_term_edges = conceptnet_api_response.get("edges", [])

        assert len(conceptnet_term_edges) > 0

    def test_get_term_conceptnet_external_uris(self, fr_conceptnet_kg) -> None:
        pompe_uris = fr_conceptnet_kg._get_term_conceptnet_external_uris(
            term_conceptnet_text="pompe"
        )

        assert len(pompe_uris) > 0
        assert "http://fr.dbpedia.org/resource/Pompe" in pompe_uris
