from typing import List, Set

import pytest
import spacy.tokens

from olaf.data_container.candidate_term_schema import CandidateTerm
from olaf.data_container.concept_schema import Concept
from olaf.data_container.knowledge_representation_schema import KnowledgeRepresentation
from olaf.data_container.linguistic_realisation_schema import LinguisticRealisation
from olaf.pipeline.pipeline_component.concept_relation_extraction.candidate_terms_to_relations import (
    CTsToRelationExtraction,
)
from olaf.pipeline.pipeline_schema import Pipeline


@pytest.fixture(scope="session")
def spacy_corpus(en_sm_spacy_model) -> List[spacy.tokens.Doc]:
    corpus = [
        "I do not eat meet.",
        "Cats eat mouses. ",
        "Dogs can eat very litte mouses too.",
        "Cats and dogs eat mouses.",
        "Cats eat. Dogs too.",
    ]
    spacy_corpus = list(en_sm_spacy_model.pipe(corpus))
    return spacy_corpus


@pytest.fixture(scope="session")
def candidate_terms(spacy_corpus) -> Set[CandidateTerm]:
    candidate_term = CandidateTerm(
        label="eat",
        corpus_occurrences={
            spacy_corpus[0][3:4],
            spacy_corpus[1][1:2],
            spacy_corpus[2][2:3],
            spacy_corpus[3][3:4],
            spacy_corpus[4][1:2],
        },
    )
    return {candidate_term}


@pytest.fixture(scope="session")
def concepts() -> Set[Concept]:
    concepts = set()
    c_dog = Concept(
        label="dog", linguistic_realisations={LinguisticRealisation(label="dogs")}
    )
    concepts.add(c_dog)
    c_cat = Concept(
        label="cat", linguistic_realisations={LinguisticRealisation(label="cats")}
    )
    concepts.add(c_cat)
    c_mouse = Concept(
        label="mouse", linguistic_realisations={LinguisticRealisation(label="mouses")}
    )
    concepts.add(c_mouse)
    return concepts


@pytest.fixture(scope="function")
def pipeline(en_sm_spacy_model, spacy_corpus, concepts, candidate_terms) -> Pipeline:
    pipeline = Pipeline(spacy_model=en_sm_spacy_model, corpus=spacy_corpus)
    pipeline.candidate_terms = candidate_terms
    pipeline.kr = KnowledgeRepresentation()
    pipeline.kr.concepts.update(concepts)
    return pipeline


def test_cts_to_relation_default_parameters(pipeline) -> None:
    cts_to_relation = CTsToRelationExtraction()
    assert cts_to_relation.concept_max_distance == 5
    assert cts_to_relation.scope == "doc"
    assert len(pipeline.candidate_terms) == 1
    cts_to_relation.run(pipeline)
    assert len(pipeline.candidate_terms) == 0
    assert len(pipeline.kr.relations) == 4
    for relation in pipeline.kr.relations:
        if relation.source_concept is None:
            assert len(relation.linguistic_realisations.pop().corpus_occurrences) == 1
        elif relation.source_concept.label == "cat":
            if relation.destination_concept.label == "mouse":
                assert (
                    len(relation.linguistic_realisations.pop().corpus_occurrences) == 2
                )
            else:
                assert relation.destination_concept.label == "dog"
                assert (
                    len(relation.linguistic_realisations.pop().corpus_occurrences) == 1
                )
        else:
            assert relation.source_concept.label == "dog"
            assert relation.destination_concept.label == "mouse"
            assert len(relation.linguistic_realisations.pop().corpus_occurrences) == 2


def test_cts_to_relation(pipeline) -> None:
    cts_to_relation = CTsToRelationExtraction(
        concept_max_distance=2, 
        scope="sent"
    )
    assert cts_to_relation.concept_max_distance == 2
    assert cts_to_relation.scope == "sent"
    assert len(pipeline.candidate_terms) == 1
    cts_to_relation.run(pipeline)
    assert len(pipeline.candidate_terms) == 0
    assert len(pipeline.kr.relations) == 3
    for relation in pipeline.kr.relations:
        if relation.source_concept is None:
            assert len(relation.linguistic_realisations.pop().corpus_occurrences) == 3
        elif relation.source_concept.label == "cat":
            if relation.destination_concept.label == "mouse":
                assert (
                    len(relation.linguistic_realisations.pop().corpus_occurrences) == 1
                )
            else:
                assert relation.destination_concept.label == "dog"
                assert (
                    len(relation.linguistic_realisations.pop().corpus_occurrences) == 1
                )
        else:
            assert relation.source_concept.label == "dog"
            assert relation.destination_concept.label == "mouse"
            assert len(relation.linguistic_realisations.pop().corpus_occurrences) == 1
