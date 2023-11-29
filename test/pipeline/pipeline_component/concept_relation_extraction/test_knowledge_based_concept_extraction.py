from typing import Any, Dict, Set

import pytest
import spacy.tokens

from olaf.data_container.candidate_term_schema import CandidateTerm
from olaf.data_container.enrichment_schema import Enrichment
from olaf.pipeline.pipeline_component.concept_relation_extraction.knowledge_based_concept_extraction import (
    KnowledgeBasedConceptExtraction,
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
def c_term_tandem_enrich(c_terms_spacy_doc) -> CandidateTerm:
    candidate_term = CandidateTerm(
        label="tandem",
        corpus_occurrences={c_terms_spacy_doc[4]},
        enrichment=Enrichment(synonyms={"velocipede", "cycle"}),
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
    c_term_tandem_enrich,
    c_term_bike_enrich,
    c_term_wine_enrich,
) -> Set[CandidateTerm]:
    c_terms = {
        c_term_wine_enrich,
        c_term_bicycle,
        c_term_tandem_enrich,
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
        """Check wether required parameters are given and correct. If this is not the case,
        suitable default ones are set.
        """
        raise NotImplementedError

    def _check_resources(self) -> None:
        pass

    def match_external_concepts(self, matching_terms: Set[str]) -> Set[str]:
        """Method to fetch concepts matching the candidate term.

        Parameters
        ----------
        term : CandidateTerm
            The candidate term the method should look concepts for.

        Returns
        -------
        Set[str]
            The concept(s) external UIDs found matching the candidate term.
        """
        concepts_uids = set()

        if "bike" in matching_terms:
            concepts_uids.update({"some_bike_uid", "another_bike_uid"})
        elif "wine" in matching_terms:
            concepts_uids.add("wine_uri")

        return concepts_uids

    def fetch_terms_synonyms(self, terms: Set[str]) -> Set[str]:
        raise NotImplementedError

    def fetch_terms_antonyms(self, terms: Set[str]) -> Set[str]:
        raise NotImplementedError

    def fetch_terms_hypernyms(self, terms: Set[str]) -> Set[str]:
        raise NotImplementedError

    def fetch_terms_hyponyms(self, terms: Set[str]) -> Set[str]:
        raise NotImplementedError


@pytest.fixture(scope="session")
def mock_knowledge_source() -> KnowledgeSource:
    kg = MockKnowledgeSource()
    return kg


class TestKnowledgeBasedConceptExtraction:
    @pytest.fixture(scope="class")
    def kg_based_concept_extraction(
        self, mock_knowledge_source
    ) -> KnowledgeBasedConceptExtraction:
        concept_extraction = KnowledgeBasedConceptExtraction(mock_knowledge_source)
        return concept_extraction

    def test_pipeline_cts(self, kg_based_concept_extraction, pipeline) -> None:
        kg_based_concept_extraction.run(pipeline)

        concepts_ext_udis = set()

        for concept in pipeline.kr.concepts:
            concepts_ext_udis.update(concept.external_uids)

        conditions = [
            "some_bike_uid" in concepts_ext_udis,
            "another_bike_uid" in concepts_ext_udis,
            "wine_uri" in concepts_ext_udis,
        ]

        assert len(pipeline.kr.concepts) == 2
        assert all(conditions)

        assert len(pipeline.candidate_terms) == 0


class TestKnowledgeBasedConceptExtractionNoMerge:
    @pytest.fixture(scope="class")
    def kg_based_concept_extraction_no_merge_syn(
        self, mock_knowledge_source
    ) -> KnowledgeBasedConceptExtraction:
        params = {"merge_ct_on_synonyms": False}
        concept_extraction = KnowledgeBasedConceptExtraction(
            mock_knowledge_source, parameters=params
        )
        return concept_extraction

    def test_pipeline_cts_empty(
        self, kg_based_concept_extraction_no_merge_syn, pipeline
    ) -> None:
        kg_based_concept_extraction_no_merge_syn.run(pipeline)

        concepts_ext_udis = set()

        for concept in pipeline.kr.concepts:
            concepts_ext_udis.update(concept.external_uids)

        conditions = [
            "some_bike_uid" in concepts_ext_udis,
            "another_bike_uid" in concepts_ext_udis,
            "wine_uri" in concepts_ext_udis,
        ]

        assert len(pipeline.kr.concepts) == 2
        assert all(conditions)

        assert len(pipeline.candidate_terms) == 0
