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
        return "1"


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
    return c_animal


@pytest.fixture(scope="session")
def c_wine() -> Concept:
    c_wine = Concept(label="wine")
    return c_wine


@pytest.fixture(scope="session")
def pipeline(corpus, en_sm_spacy_model, c_dog, c_mammal, c_animal) -> Pipeline:
    pipeline = Pipeline(
        corpus=corpus,
        spacy_model=en_sm_spacy_model,
    )
    pipeline.kr.concepts.add(c_dog)
    pipeline.kr.concepts.add(c_mammal)
    pipeline.kr.concepts.add(c_animal)
    return pipeline


def test_find_concept_cooc(
    c_dog, c_mammal, c_animal, c_wine, llm_concept_hierarchy, corpus
) -> None:
    assert llm_concept_hierarchy._find_concept_cooc(c_dog, c_mammal) == {corpus[2]}
    assert llm_concept_hierarchy._find_concept_cooc(c_dog, c_animal) == {corpus[0]}
    assert llm_concept_hierarchy._find_concept_cooc(c_mammal, c_animal) == {corpus[1]}
    assert llm_concept_hierarchy._find_concept_cooc(c_dog, c_wine) == set()


def test_generate_doc_context(corpus, small_context_llm_concept_hierarchy) -> None:
    assert (
        small_context_llm_concept_hierarchy._generate_doc_context({corpus[0]})
        == corpus[0].text[:10]
    )


def test_create_metarelation(c_dog, c_animal, llm_concept_hierarchy) -> None:
    metarelation = llm_concept_hierarchy._create_metarelation("1", c_dog, c_animal)
    assert metarelation.source_concept == c_animal
    assert metarelation.destination_concept == c_dog

    metarelation = llm_concept_hierarchy._create_metarelation("2", c_dog, c_animal)
    assert metarelation.source_concept == c_dog
    assert metarelation.destination_concept == c_animal

    metarelation = llm_concept_hierarchy._create_metarelation("3", c_dog, c_animal)
    assert metarelation is None

    metarelation = llm_concept_hierarchy._create_metarelation("123", c_dog, c_animal)
    assert metarelation is None


def test_run(pipeline, llm_concept_hierarchy) -> None:
    assert len(pipeline.kr.metarelations) == 0
    llm_concept_hierarchy.run(pipeline)
    assert len(pipeline.kr.metarelations) == 3
