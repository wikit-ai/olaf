from typing import Any, Dict, Set

import pytest
from spacy.tokens import Doc

from olaf import Pipeline
from olaf.commons.llm_tools import LLMGenerator
from olaf.data_container import CandidateTerm
from olaf.pipeline.pipeline_component.term_extraction import LLMTermExtraction


class MockLLMGenerator(LLMGenerator):
    def __init__(self) -> None:
        super().__init__()

    def check_resources(self) -> None:
        pass

    def generate_text(self, prompt: Any) -> str:
        return '["test","prompt"]'


class MockWrongLLMGenerator(LLMGenerator):
    def __init__(self) -> None:
        super().__init__()

    def check_resources(self) -> None:
        pass

    def generate_text(self, prompt: Any) -> str:
        return '"test, prompt"'


@pytest.fixture(scope="session")
def doc(en_sm_spacy_model) -> Doc:
    text = "This is a test. I will try the prompt."
    return en_sm_spacy_model(text)


@pytest.fixture(scope="session")
def llm_generator() -> LLMGenerator:
    lg = MockLLMGenerator()
    return lg


@pytest.fixture(scope="session")
def wrong_llm_generator() -> LLMGenerator:
    return MockWrongLLMGenerator()


@pytest.fixture(scope="session")
def pipeline(en_sm_spacy_model, doc) -> Pipeline:
    return Pipeline(spacy_model=en_sm_spacy_model, corpus=[doc])


@pytest.fixture(scope="session")
def llm_term_extraction(llm_generator) -> LLMTermExtraction:
    return LLMTermExtraction(llm_generator=llm_generator)


@pytest.fixture(scope="session")
def wrong_llm_term_extraction(wrong_llm_generator) -> LLMTermExtraction:
    return LLMTermExtraction(llm_generator=wrong_llm_generator)


@pytest.fixture(scope="session")
def cts_index(en_sm_spacy_model) -> Dict[str, CandidateTerm]:
    cts_index = {}
    cts_index["test"] = CandidateTerm(
        label="test", corpus_occurrences={en_sm_spacy_model("test")[:]}
    )
    cts_index["prompt"] = CandidateTerm(
        label="prompt", corpus_occurrences={en_sm_spacy_model("prompt")[:]}
    )
    return cts_index


def test_generate_candidate_terms(
    doc, llm_term_extraction, wrong_llm_term_extraction
) -> None:
    ct_labels = llm_term_extraction._generate_candidate_terms(doc)
    assert isinstance(ct_labels, Set)
    assert len(ct_labels) == 2
    assert "test" in ct_labels
    assert "prompt" in ct_labels

    wrong_ct_labels = wrong_llm_term_extraction._generate_candidate_terms(doc)
    assert isinstance(wrong_ct_labels, Set)
    assert len(wrong_ct_labels) == 0


def test_update_candidate_terms(doc, llm_term_extraction, cts_index) -> None:
    empty_index = {}
    llm_term_extraction._update_candidate_terms(doc, {"test", "prompt"}, empty_index)
    assert len(empty_index) == 2
    for label, ct in empty_index.items():
        assert len(ct.corpus_occurrences) == 1
        assert (label == "test") or (label == "prompt")

    llm_term_extraction._update_candidate_terms(doc, {"test", "prompt"}, cts_index)
    assert len(cts_index) == 2
    for label, ct in cts_index.items():
        assert len(ct.corpus_occurrences) == 2
        assert (label == "test") or (label == "prompt")


def test_run_component(pipeline, llm_term_extraction) -> None:
    assert len(pipeline.candidate_terms) == 0
    llm_term_extraction.run(pipeline)
    assert len(pipeline.candidate_terms) == 2
