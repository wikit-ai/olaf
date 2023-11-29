from typing import Set

import pytest

from olaf.data_container.candidate_term_schema import CandidateTerm
from olaf.data_container.concept_schema import Concept
from olaf.data_container.enrichment_schema import Enrichment
from olaf.data_container.knowledge_representation_schema import KnowledgeRepresentation
from olaf.data_container.linguistic_realisation_schema import LinguisticRealisation
from olaf.pipeline.pipeline_component.concept_relation_extraction.synonym_relation_extraction import (
    SynonymRelationExtraction,
)
from olaf.pipeline.pipeline_schema import Pipeline


@pytest.fixture(scope="session")
def corpus(en_sm_spacy_model):
    texts = [
        "Cats eat mouses.",
        "Dogs can eat little mouses too.",
        "I do not know if cats can devour mouses.",
        "I will devour my pizza.",
        "You should not eat so fast",
        "I like bike.",
    ]
    corpus = list(en_sm_spacy_model.pipe(texts))
    return corpus


@pytest.fixture(scope="session")
def candidate_terms(corpus) -> Set[CandidateTerm]:
    candidate_terms = set()

    candidate_terms.add(
        CandidateTerm(
            label="eat",
            corpus_occurrences={corpus[0][1:2], corpus[1][2:3], corpus[4][3:4]},
        )
    )
    candidate_terms.add(
        CandidateTerm(
            label="devour",
            corpus_occurrences={corpus[2][7:8], corpus[3][2:3]},
            enrichment=Enrichment({"eat"}),
        )
    )
    return candidate_terms


@pytest.fixture(scope="session")
def c_cat() -> Concept:
    c_cat = Concept(
        label="cat", linguistic_realisations={LinguisticRealisation(label="cats")}
    )
    return c_cat


@pytest.fixture(scope="session")
def c_mouse() -> Concept:
    c_mouse = Concept(
        label="mouse", linguistic_realisations={LinguisticRealisation(label="mouses")}
    )
    return c_mouse


@pytest.fixture(scope="session")
def c_dog() -> Concept:
    c_dog = Concept(
        label="dog", linguistic_realisations={LinguisticRealisation(label="dogs")}
    )
    return c_dog


@pytest.fixture(scope="session")
def pipeline(
    en_sm_spacy_model, candidate_terms, corpus, c_cat, c_dog, c_mouse
) -> Pipeline:
    pipeline = Pipeline(spacy_model=en_sm_spacy_model, corpus=corpus)
    pipeline.candidate_terms = candidate_terms
    pipeline.kr = KnowledgeRepresentation()
    pipeline.kr.concepts.update({c_cat, c_dog, c_mouse})
    return pipeline


def test_synonym_relation_extraction(pipeline, c_cat, c_dog, c_mouse) -> None:
    synonym_grouping = SynonymRelationExtraction()
    synonym_grouping.run(pipeline)
    assert len(pipeline.kr.relations) == 3
    for relation in pipeline.kr.relations:
        assert relation.label in ["eat", "devour"]
        if relation.source_concept == c_cat:
            assert relation.destination_concept == c_mouse
            assert len(relation.linguistic_realisations) == 2
        elif relation.source_concept == c_dog:
            assert relation.destination_concept == c_mouse
            assert len(relation.linguistic_realisations) == 1
        else:
            assert relation.source_concept is None
            assert relation.destination_concept is None
            assert len(relation.linguistic_realisations) == 2
