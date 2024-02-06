from typing import Any, List, Set

import pytest
from spacy.tokens import Doc

from olaf import Pipeline
from olaf.commons.llm_tools import LLMGenerator
from olaf.data_container.concept_schema import Concept
from olaf.data_container.linguistic_realisation_schema import LinguisticRealisation
from olaf.pipeline.pipeline_component.concept_relation_hierarchy import (
    LLMBasedHierarchisation,
)


class MockLLMGenerator(LLMGenerator):
    def __init__(self) -> None:
        pass

    def check_resources(self) -> None:
        pass

    def generate_text(self, prompt: Any) -> str:
        return '[["dog","is_generalised_by","mammal"],["mammal", "is_generalised", "animal"]]'


@pytest.fixture(scope="session")
def llm_generator() -> LLMGenerator:
    lg = MockLLMGenerator()
    return lg


@pytest.fixture(scope="session")
def llm_concept_hierarchy(llm_generator) -> LLMBasedHierarchisation:
    return LLMBasedHierarchisation(llm_generator=llm_generator)


@pytest.fixture(scope="session")
def small_context_llm_concept_hierarchy(llm_generator) -> LLMBasedHierarchisation:
    return LLMBasedHierarchisation(llm_generator=llm_generator, doc_context_max_len=10)


@pytest.fixture(scope="session")
def corpus(en_sm_spacy_model) -> List[Doc]:
    texts = [
        "In the all list of animals, we can find dogs.",
        "Mammals are a specific class of animals.",
        "Dogs are a king of mammals.",
    ]
    return list(en_sm_spacy_model.pipe(texts))


@pytest.fixture(scope="session")
def c_dog(corpus) -> Concept:
    c_dog = Concept(label="dog")
    c_dog.add_linguistic_realisation(
        LinguisticRealisation("dog", {corpus[0][10:11], corpus[2][0:1]})
    )
    return c_dog


@pytest.fixture(scope="session")
def c_mammal(corpus) -> Concept:
    c_mammal = Concept(label="mammal")
    c_mammal.add_linguistic_realisation(
        LinguisticRealisation("mammal", {corpus[1][0:1], corpus[2][5:6]})
    )
    return c_mammal


@pytest.fixture(scope="session")
def c_animal(corpus) -> Concept:
    c_animal = Concept(label="animal")
    c_animal.add_linguistic_realisation(
        LinguisticRealisation("animal", {corpus[0][5:6], corpus[1][6:7]})
    )
    c_animal.add_linguistic_realisation(LinguisticRealisation("beast"))
    return c_animal


@pytest.fixture(scope="session")
def c_wine() -> Concept:
    c_wine = Concept(label="wine")
    return c_wine


@pytest.fixture(scope="session")
def pipeline(corpus, en_sm_spacy_model, c_dog, c_mammal, c_animal, c_wine) -> Pipeline:
    pipeline = Pipeline(
        corpus=corpus,
        spacy_model=en_sm_spacy_model,
    )
    pipeline.kr.concepts.add(c_dog)
    pipeline.kr.concepts.add(c_mammal)
    pipeline.kr.concepts.add(c_animal)
    pipeline.kr.concepts.add(c_wine)
    return pipeline


def test_generate_doc_context(corpus, small_context_llm_concept_hierarchy) -> None:
    assert (
        small_context_llm_concept_hierarchy._generate_doc_context({corpus[0]})
        == corpus[0].text[:10]
    )


def test_concepts_description(pipeline, llm_concept_hierarchy) -> None:
    concepts_description = llm_concept_hierarchy._create_concepts_description(
        pipeline.kr.concepts
    )
    assert "Concepts:" in concepts_description
    assert "animal (beast)" in concepts_description
    assert "wine" in concepts_description
    assert "mammal" in concepts_description
    assert "dog" in concepts_description


def test_find_concept_by_label(pipeline, llm_concept_hierarchy, c_animal, c_mammal):
    assert (
        llm_concept_hierarchy._find_concept_by_label("animal", pipeline.kr.concepts)
        == c_animal
    )
    assert (
        llm_concept_hierarchy._find_concept_by_label("mammal", pipeline.kr.concepts)
        == c_mammal
    )
    assert (
        llm_concept_hierarchy._find_concept_by_label("flower", pipeline.kr.concepts)
        is None
    )


def test_create_metarelations(
    pipeline, llm_concept_hierarchy, c_dog, c_mammal, c_animal
) -> None:
    llm_output = (
        '[["dog","is_generalised_by","mammal"],["mammal", "is_generalised", "animal"]]'
    )
    metarelations = llm_concept_hierarchy._create_metarelations(
        llm_output, pipeline.kr.concepts
    )

    assert len(metarelations) == 2
    for meta in metarelations:
        assert meta.label == "is_generalised_by"
        if meta.source_concept == c_dog:
            assert meta.destination_concept == c_mammal
        else:
            assert meta.source_concept == c_mammal
            assert meta.destination_concept == c_animal

    wrong_llm_output = "This is a wrong llm output."
    no_metarelations = llm_concept_hierarchy._create_metarelations(
        wrong_llm_output, pipeline.kr.concepts
    )
    assert len(no_metarelations) == 0


def test_run(pipeline, llm_concept_hierarchy) -> None:
    assert len(pipeline.kr.metarelations) == 0
    llm_concept_hierarchy.run(pipeline)
    assert len(pipeline.kr.metarelations) == 2
