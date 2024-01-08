from typing import Any, Dict, Set

import pytest
from spacy.tokens import Doc

from olaf import Pipeline
from olaf.commons.llm_tools import LLMGenerator
from olaf.data_container import CandidateTerm, Enrichment
from olaf.pipeline.pipeline_component.candidate_term_enrichment import (
    LLMBasedTermEnrichment,
)


class MockLLMGenerator(LLMGenerator):
    def __init__(self) -> None:
        pass

    def check_resources(self) -> None:
        pass

    def generate_text(self, prompt: Any) -> str:
        text = """{"synonyms":["syn1", "syn2"],"hypernyms":["hyper1", "hyper2"],"hyponyms":["hypo1", "hypo2"],"antonyms":["anto1", "anto2"]}"""
        return text


class MockWrongLLMGenerator(LLMGenerator):
    def __init__(self) -> None:
        pass

    def check_resources(self) -> None:
        pass

    def generate_text(self, prompt: Any) -> str:
        return '"synonyms, hypernyms, hyponyms, antonyms"'


@pytest.fixture(scope="session")
def corpus(en_sm_spacy_model) -> Doc:
    text = "This is a test. I will try the prompt."
    return [en_sm_spacy_model(text)]


@pytest.fixture(scope="session")
def llm_generator() -> LLMGenerator:
    return MockLLMGenerator()


@pytest.fixture(scope="session")
def wrong_llm_generator() -> LLMGenerator:
    return MockWrongLLMGenerator()


@pytest.fixture(scope="session")
def pipeline(en_sm_spacy_model, corpus) -> Pipeline:
    return Pipeline(spacy_model=en_sm_spacy_model, corpus=corpus)


@pytest.fixture(scope="session")
def llm_term_enrichment(llm_generator) -> LLMBasedTermEnrichment:
    return LLMBasedTermEnrichment(llm_generator=llm_generator)


@pytest.fixture(scope="session")
def wrong_llm_term_enrichment(wrong_llm_generator) -> LLMBasedTermEnrichment:
    return LLMBasedTermEnrichment(llm_generator=wrong_llm_generator)


@pytest.fixture(scope="function")
def ct1() -> CandidateTerm:
    return CandidateTerm("enrichment", set())


@pytest.fixture(scope="function")
def ct2() -> CandidateTerm:
    return CandidateTerm(
        label="test",
        corpus_occurrences=set(),
        enrichment=Enrichment(
            synonyms=set(["syn0"]),
            hypernyms=set(["hyper0"]),
            hyponyms=set(["hypo0"]),
            antonyms=set(["anto0"]),
        ),
    )


def test_enrich_candidate_term(ct1, ct2, llm_term_enrichment) -> None:
    assert ct1.enrichment is None
    llm_term_enrichment._enrich_cterm(ct1)
    assert ct1.enrichment is not None
    assert len(ct1.enrichment.synonyms) == 2
    assert len(ct1.enrichment.hypernyms) == 2
    assert len(ct1.enrichment.hyponyms) == 2
    assert len(ct1.enrichment.antonyms) == 2

    assert ct2.enrichment is not None
    assert len(ct2.enrichment.synonyms) == 1
    assert len(ct2.enrichment.hypernyms) == 1
    assert len(ct2.enrichment.hyponyms) == 1
    assert len(ct2.enrichment.antonyms) == 1
    llm_term_enrichment._enrich_cterm(ct2)
    assert len(ct2.enrichment.synonyms) == 3
    assert len(ct2.enrichment.hypernyms) == 3
    assert len(ct2.enrichment.hyponyms) == 3
    assert len(ct2.enrichment.antonyms) == 3


def test_not_working_enrich_candidate_term(ct1, ct2, wrong_llm_term_enrichment) -> None:
    assert ct1.enrichment is None
    wrong_llm_term_enrichment._enrich_cterm(ct1)
    assert ct1.enrichment is None

    assert ct2.enrichment is not None
    assert len(ct2.enrichment.synonyms) == 1
    assert len(ct2.enrichment.hypernyms) == 1
    assert len(ct2.enrichment.hyponyms) == 1
    assert len(ct2.enrichment.antonyms) == 1
    wrong_llm_term_enrichment._enrich_cterm(ct2)
    assert ct2.enrichment is not None
    assert len(ct2.enrichment.synonyms) == 1
    assert len(ct2.enrichment.hypernyms) == 1
    assert len(ct2.enrichment.hyponyms) == 1
    assert len(ct2.enrichment.antonyms) == 1
