from typing import Any, Dict, Set

import pytest
import spacy.tokens

from olaf.data_container.candidate_term_schema import CandidateTerm
from olaf.data_container.enrichment_schema import Enrichment
from olaf.pipeline.pipeline_component.candidate_term_enrichment.knowledge_based_enrichment import (
    KnowledgeBasedCTermEnrichment,
)
from olaf.pipeline.pipeline_schema import Pipeline
from olaf.repository.knowledge_source.knowledge_source_schema import KnowledgeSource


@pytest.fixture(scope="session")
def c_terms_spacy_doc(en_sm_spacy_model) -> spacy.tokens.Doc:
    c_terms_text = "car bicycle bike cycle tandem velocipede cycle wine drink beer"
    c_terms_doc = en_sm_spacy_model(c_terms_text)
    return c_terms_doc


@pytest.fixture(scope="session")
def c_term_bike_enrich(c_terms_spacy_doc) -> CandidateTerm:
    candidate_term = CandidateTerm(
        label="bike",
        corpus_occurrences={c_terms_spacy_doc[2]},
        enrichment=Enrichment(synonyms={"bicycle", "cycle"}, antonyms={"not_bicycle"}),
    )
    return candidate_term


@pytest.fixture(scope="session")
def c_term_bicycle(c_terms_spacy_doc) -> CandidateTerm:
    candidate_term = CandidateTerm(
        label="bicycle",
        corpus_occurrences={c_terms_spacy_doc[1]},
    )
    return candidate_term


@pytest.fixture(scope="session")
def c_term_wine_enrich(c_terms_spacy_doc) -> CandidateTerm:
    candidate_term = CandidateTerm(
        label="wine",
        corpus_occurrences={c_terms_spacy_doc[7]},
        enrichment=Enrichment(synonyms={"drink", "beer"}, antonyms={"water"}),
    )
    return candidate_term


@pytest.fixture(scope="session")
def candidate_terms(
    c_term_bicycle,
    c_term_bike_enrich,
    c_term_wine_enrich,
) -> Set[CandidateTerm]:
    c_terms = {
        c_term_wine_enrich,
        c_term_bicycle,
        c_term_bike_enrich,
    }
    return c_terms


@pytest.fixture(scope="session")
def pipeline(candidate_terms, en_sm_spacy_model) -> Pipeline:
    pipeline = Pipeline(spacy_model=en_sm_spacy_model, corpus=[])
    pipeline.candidate_terms = candidate_terms
    return pipeline


class MockKnowledgeSource(KnowledgeSource):
    def __init__(self, parameters: Dict[str, Any] | None = None) -> None:
        super().__init__(parameters)

    def _check_parameters(self) -> None:
        raise NotImplementedError

    def _check_resources(self) -> None:
        pass

    def match_external_concepts(self, matching_terms: Set[str]) -> Set[str]:
        raise NotImplementedError

    def fetch_terms_synonyms(self, terms: Set[str]) -> Set[str]:
        terms_synonyms = set()
        for term in terms:
            if term == "beer":
                terms_synonyms.add("potion magique")

            if term == "bike":
                terms_synonyms.add("ecological travel mean")

            if term == "wine":
                terms_synonyms.add("frenchy drink")

        return terms_synonyms

    def fetch_terms_antonyms(self, terms: Set[str]) -> Set[str]:
        terms_antonyms = set()
        for term in terms:
            if term == "beer":
                terms_antonyms.add("panache")

            if term == "water":
                terms_antonyms.add("alcohol")

        return terms_antonyms

    def fetch_terms_hypernyms(self, terms: Set[str]) -> Set[str]:
        terms_hypernyms = set()
        for term in terms:
            if term == "velocipede":
                terms_hypernyms.add("bike")

            if term == "bike":
                terms_hypernyms.add("two wheel vehicle")

            if term == "wine":
                terms_hypernyms.add("grapefruit juice")

        return terms_hypernyms

    def fetch_terms_hyponyms(self, terms: Set[str]) -> Set[str]:
        terms_hyponyms = set()
        for term in terms:
            if term == "bike":
                terms_hyponyms.add("monocycle")

            if term == "beer":
                terms_hyponyms.add("IPA")

        return terms_hyponyms


@pytest.fixture(scope="session")
def mock_knowledge_source() -> KnowledgeSource:
    kg = MockKnowledgeSource()
    return kg


class TestKnowledgeBasedCTsEnrichmentNoSynonyms:
    @pytest.fixture(scope="class")
    def kg_based_cts_enrichmment_no_syn(
        self, mock_knowledge_source
    ) -> KnowledgeBasedCTermEnrichment:
        params = {"use_synonyms": False, "enrichment_kinds": {"antonyms", "hypernyms"}}
        cts_enrichmment = KnowledgeBasedCTermEnrichment(
            mock_knowledge_source, parameters=params
        )
        return cts_enrichmment

    def test_pipeline_cts(self, kg_based_cts_enrichmment_no_syn, pipeline) -> None:
        kg_based_cts_enrichmment_no_syn.run(pipeline)

        assert len(pipeline.candidate_terms) > 0

        for ct in pipeline.candidate_terms:
            if ct.label == "wine":
                assert "potion magique" not in ct.enrichment.synonyms
                assert "frenchy drink" not in ct.enrichment.synonyms
                assert "grapefruit juice" in ct.enrichment.hypernyms
                assert len(ct.enrichment.synonyms) == 2
                assert len(ct.enrichment.hyponyms) == 0
                assert len(ct.enrichment.antonyms) == 1

            if ct.label == "bike":
                assert "ecological travel mean" not in ct.enrichment.synonyms
                assert "two wheel vehicle" in ct.enrichment.hypernyms
                assert len(ct.enrichment.hyponyms) == 0
                assert len(ct.enrichment.antonyms) == 1

            if ct.label == "bicycle":
                assert len(ct.enrichment.hyponyms) == 0
                assert len(ct.enrichment.antonyms) == 0
                assert len(ct.enrichment.hypernyms) == 0
                assert len(ct.enrichment.synonyms) == 0


class TestKnowledgeBasedCTsEnrichment:
    @pytest.fixture(scope="class")
    def kg_based_cts_enrichmment(
        self, mock_knowledge_source
    ) -> KnowledgeBasedCTermEnrichment:
        params = {"enrichment_kinds": {"antonyms", "hypernyms"}}
        cts_enrichmment = KnowledgeBasedCTermEnrichment(
            mock_knowledge_source, parameters=params
        )
        return cts_enrichmment

    def test_pipeline_cts(self, kg_based_cts_enrichmment, pipeline) -> None:
        kg_based_cts_enrichmment.run(pipeline)

        for ct in pipeline.candidate_terms:
            if ct.label == "wine":
                assert "potion magique" not in ct.enrichment.synonyms
                assert "frenchy drink" not in ct.enrichment.synonyms
                assert "panache" in ct.enrichment.antonyms
                assert len(ct.enrichment.hyponyms) == 0

            if ct.label == "bike":
                assert "ecological travel mean" not in ct.enrichment.synonyms
                assert len(ct.enrichment.hyponyms) == 0
                assert len(ct.enrichment.antonyms) == 1

            if ct.label == "bicycle":
                assert len(ct.enrichment.hyponyms) == 0
                assert len(ct.enrichment.antonyms) == 0
                assert len(ct.enrichment.hypernyms) == 0
                assert len(ct.enrichment.synonyms) == 0


class TestKnowledgeBasedCTsEnrichmentSyn:
    @pytest.fixture(scope="class")
    def kg_based_cts_enrichmment(
        self, mock_knowledge_source
    ) -> KnowledgeBasedCTermEnrichment:
        cts_enrichmment = KnowledgeBasedCTermEnrichment(mock_knowledge_source)
        return cts_enrichmment

    def test_pipeline_cts(self, kg_based_cts_enrichmment, pipeline) -> None:
        kg_based_cts_enrichmment.run(pipeline)

        for ct in pipeline.candidate_terms:
            if ct.label == "wine":
                assert "potion magique" in ct.enrichment.synonyms
                assert "frenchy drink" in ct.enrichment.synonyms
                assert len(ct.enrichment.hyponyms) == 0
                assert len(ct.enrichment.antonyms) == 2

            if ct.label == "bike":
                assert "ecological travel mean" in ct.enrichment.synonyms
                assert len(ct.enrichment.hyponyms) == 0
                assert len(ct.enrichment.antonyms) == 1

            if ct.label == "bicycle":
                assert len(ct.enrichment.hyponyms) == 0
                assert len(ct.enrichment.antonyms) == 0
                assert len(ct.enrichment.hypernyms) == 0
                assert len(ct.enrichment.synonyms) == 0
