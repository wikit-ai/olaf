from typing import Any, Dict, List, Set

import pytest
from spacy.tokens import Doc

from olaf import Pipeline
from olaf.commons.llm_tools import LLMGenerator
from olaf.data_container import CandidateTerm
from olaf.pipeline.pipeline_component.concept_relation_extraction import (
    LLMBasedRelationExtraction,
)


class MockLLMGenerator(LLMGenerator):
    def __init__(self) -> None:
        pass

    def check_resources(self) -> None:
        pass

    def generate_text(self, prompt: Any) -> str:
        return '[["like", "prefer"], ["eat"]]'


@pytest.fixture(scope="session")
def llm_generator() -> LLMGenerator:
    lg = MockLLMGenerator()
    return lg


@pytest.fixture(scope="session")
def llm_relation_extraction(llm_generator) -> LLMBasedRelationExtraction:
    return LLMBasedRelationExtraction(llm_generator=llm_generator)


@pytest.fixture(scope="session")
def small_context_llm_relation_extraction(llm_generator) -> LLMBasedRelationExtraction:
    return LLMBasedRelationExtraction(
        llm_generator=llm_generator, doc_context_max_len=40
    )


@pytest.fixture(scope="session")
def corpus(en_sm_spacy_model) -> List[Doc]:
    texts = [
        "I like drinking wine when eating pizza.",
        "What do you prefer between wine, water and sparkling water ?",
        "Sparkling water is just water with some gas.",
    ]
    return list(en_sm_spacy_model.pipe(texts))


@pytest.fixture(scope="session")
def cterms(corpus) -> Set[CandidateTerm]:
    cterms = [
        CandidateTerm(label="like", corpus_occurrences={corpus[0][1:2]}),
        CandidateTerm(label="prefer", corpus_occurrences={corpus[1][3:4]}),
        CandidateTerm(label="eat", corpus_occurrences={corpus[0][5:6]}),
    ]
    return set(cterms)


@pytest.fixture(scope="session")
def doc_count(corpus) -> Dict[Doc, int]:
    return {corpus[0]: 2, corpus[1]: 1}


@pytest.fixture(scope="session")
def pipeline(corpus, en_sm_spacy_model, cterms) -> Pipeline:
    pipeline = Pipeline(
        corpus=corpus,
        spacy_model=en_sm_spacy_model,
    )
    pipeline.candidate_terms = cterms
    return pipeline


def test_create_doc_count(cterms, doc_count, llm_relation_extraction) -> None:
    doc_count_pred = llm_relation_extraction._create_doc_count(cterms)
    assert doc_count_pred == doc_count


def test_generate_doc_context(
    doc_count, llm_relation_extraction, small_context_llm_relation_extraction
) -> None:
    context = " ".join([doc.text for doc in doc_count.keys()]) + " "

    context_pred = llm_relation_extraction._generate_doc_context(doc_count)
    assert context_pred == context

    small_context_pred = small_context_llm_relation_extraction._generate_doc_context(
        doc_count
    )
    assert small_context_pred == context[:40]


def test_convert_llm_output_to_cc(llm_relation_extraction, cterms) -> None:
    cterm_index = {cterm.label: cterm for cterm in cterms}
    correct_output = '[["like", "prefer"], ["eat"]]'
    relation_candidates = llm_relation_extraction._convert_llm_output_to_rc(
        correct_output, cterm_index
    )
    assert len(relation_candidates) == 2
    for cc_group in relation_candidates:
        if len(cc_group) == 1:
            assert cc_group.pop() == cterm_index["eat"]
        else:
            for cc in cc_group:
                assert (cc == cterm_index["like"]) or (cc == cterm_index["prefer"])

    wrong_output = "like, prefer, eat"
    empty_relation_candidates = llm_relation_extraction._convert_llm_output_to_rc(
        wrong_output, cterm_index
    )
    assert len(empty_relation_candidates) == 0


def test_run_component(pipeline, llm_relation_extraction) -> None:
    assert len(pipeline.candidate_terms) == 3
    assert len(pipeline.kr.relations) == 0
    llm_relation_extraction.run(pipeline)
    assert len(pipeline.candidate_terms) == 0
    assert len(pipeline.kr.relations) == 2
    for relation in pipeline.kr.relations:
        if relation.label == "eat":
            assert len(relation.linguistic_realisations) == 1
        else:
            assert (relation.label == "like") or (relation.label == "prefer")
            assert len(relation.linguistic_realisations) == 2
