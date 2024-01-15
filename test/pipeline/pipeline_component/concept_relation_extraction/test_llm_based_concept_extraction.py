from typing import Any, Dict, List, Set

import pytest
from spacy.tokens import Doc

from olaf import Pipeline
from olaf.commons.llm_tools import LLMGenerator
from olaf.data_container import CandidateTerm
from olaf.pipeline.pipeline_component.concept_relation_extraction import (
    LLMBasedConceptExtraction,
)


class MockLLMGenerator(LLMGenerator):
    def __init__(self) -> None:
        pass

    def check_resources(self) -> None:
        pass

    def generate_text(self, prompt: Any) -> str:
        return '[["water", "sparkling water"], ["wine"]]'


@pytest.fixture(scope="session")
def llm_generator() -> LLMGenerator:
    lg = MockLLMGenerator()
    return lg


@pytest.fixture(scope="session")
def llm_concept_extraction(llm_generator) -> LLMBasedConceptExtraction:
    return LLMBasedConceptExtraction(llm_generator=llm_generator)


@pytest.fixture(scope="session")
def small_context_llm_concept_extraction(llm_generator) -> LLMBasedConceptExtraction:
    return LLMBasedConceptExtraction(
        llm_generator=llm_generator, doc_context_max_len=40
    )


@pytest.fixture(scope="session")
def corpus(en_sm_spacy_model) -> List[Doc]:
    texts = [
        "I like drinking wine when eating pizza.",
        "What do you prefer between water, sparkling water and wine ?",
        "Sparkling water is just water with some gas.",
    ]
    return list(en_sm_spacy_model.pipe(texts))


@pytest.fixture(scope="session")
def cterms(corpus) -> Set[CandidateTerm]:
    cterms = [
        CandidateTerm(
            label="wine", corpus_occurrences={corpus[0][3:4], corpus[1][10:11]}
        ),
        CandidateTerm(
            label="water", corpus_occurrences={corpus[1][5:6], corpus[2][4:5]}
        ),
        CandidateTerm(
            label="sparkling water", corpus_occurrences={corpus[1][7:9], corpus[2][0:2]}
        ),
    ]
    return set(cterms)


@pytest.fixture(scope="session")
def doc_count(corpus) -> Dict[Doc, int]:
    return {corpus[1]: 3, corpus[2]: 2, corpus[0]: 1}


@pytest.fixture(scope="session")
def pipeline(corpus, en_sm_spacy_model, cterms) -> Pipeline:
    pipeline = Pipeline(
        corpus=corpus,
        spacy_model=en_sm_spacy_model,
    )
    pipeline.candidate_terms = cterms
    return pipeline


def test_create_doc_count(cterms, doc_count, llm_concept_extraction) -> None:
    doc_count_pred = llm_concept_extraction._create_doc_count(cterms)
    assert doc_count_pred == doc_count


def test_generate_doc_context(
    doc_count, llm_concept_extraction, small_context_llm_concept_extraction
) -> None:
    context = " ".join([doc.text for doc in doc_count.keys()]) + " "

    context_pred = llm_concept_extraction._generate_doc_context(doc_count)
    assert context_pred == context

    small_context_pred = small_context_llm_concept_extraction._generate_doc_context(
        doc_count
    )
    assert small_context_pred == context[:40]


def test_convert_llm_output_to_cc(llm_concept_extraction, cterms) -> None:
    cterm_index = {cterm.label: cterm for cterm in cterms}
    correct_output = '[["water", "sparkling water"], ["wine"]]'
    concept_candidates = llm_concept_extraction._convert_llm_output_to_cc(
        correct_output, cterm_index
    )
    assert len(concept_candidates) == 2
    for cc_group in concept_candidates:
        if len(cc_group) == 1:
            assert cc_group.pop() == cterm_index["wine"]
        else:
            for cc in cc_group:
                assert (cc == cterm_index["water"]) or (
                    cc == cterm_index["sparkling water"]
                )

    wrong_output = "water, sparkling water, wine"
    empty_concept_candidates = llm_concept_extraction._convert_llm_output_to_cc(
        wrong_output, cterm_index
    )
    assert len(empty_concept_candidates) == 0


def test_run_component(pipeline, llm_concept_extraction) -> None:
    assert len(pipeline.candidate_terms) == 3
    assert len(pipeline.kr.concepts) == 0
    llm_concept_extraction.run(pipeline)
    assert len(pipeline.candidate_terms) == 0
    assert len(pipeline.kr.concepts) == 2
    for concept in pipeline.kr.concepts:
        if concept.label == "wine":
            assert len(concept.linguistic_realisations) == 1
        else:
            assert (concept.label == "water") or (concept.label == "sparkling water")
            assert len(concept.linguistic_realisations) == 2
