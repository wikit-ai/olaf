from typing import Set

import pytest
import spacy

from olaf.data_container.candidate_term_schema import CandidateTerm
from olaf.pipeline.pipeline_component.concept_relation_extraction.candidate_terms_to_concepts import (
    CTsToConceptExtraction,
)
from olaf.pipeline.pipeline_schema import Pipeline


@pytest.fixture(scope="session")
def spacy_nlp():
    spacy_model = spacy.load(
        "en_core_web_sm",
        exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
    )
    return spacy_model


@pytest.fixture(scope="session")
def c_terms_spacy_doc(spacy_nlp) -> spacy.tokens.Doc:
    c_terms_text = "car bicycle bike cycle tandem velocipede cycle wine drink beer"
    c_terms_doc = spacy_nlp(c_terms_text)
    return c_terms_doc


@pytest.fixture(scope="session")
def c_term_bike(c_terms_spacy_doc) -> CandidateTerm:
    candidate_term = CandidateTerm(
        label="bike",
        corpus_occurrences={c_terms_spacy_doc[2]},
    )
    return candidate_term


@pytest.fixture(scope="session")
def c_term_bicycle(c_terms_spacy_doc) -> CandidateTerm:
    candidate_term = CandidateTerm(
        label="bicycle",
        corpus_occurrences={
            c_terms_spacy_doc[1],
            c_terms_spacy_doc[3],
            c_terms_spacy_doc[6],
        },
    )
    return candidate_term


@pytest.fixture(scope="session")
def c_term_tandem(c_terms_spacy_doc) -> CandidateTerm:
    candidate_term = CandidateTerm(
        label="tandem",
        corpus_occurrences={c_terms_spacy_doc[4], c_terms_spacy_doc[5]},
    )
    return candidate_term


@pytest.fixture(scope="session")
def c_term_wine(c_terms_spacy_doc) -> CandidateTerm:
    candidate_term = CandidateTerm(
        label="wine",
        corpus_occurrences={c_terms_spacy_doc[7], c_terms_spacy_doc[8]},
    )
    return candidate_term


@pytest.fixture(scope="session")
def candidate_terms(
    c_term_bicycle,
    c_term_tandem,
    c_term_bike,
    c_term_wine,
) -> Set[CandidateTerm]:
    c_terms = {
        c_term_wine,
        c_term_bicycle,
        c_term_tandem,
        c_term_bike,
    }
    return c_terms


@pytest.fixture(scope="session")
def pipeline(candidate_terms, spacy_nlp) -> Pipeline:
    pipeline = Pipeline(spacy_model=spacy_nlp, corpus=[])
    pipeline.candidate_terms = candidate_terms
    return pipeline


def test_ct_to_concepts_extraction(pipeline) -> None:
    ct_to_concepts_extract = CTsToConceptExtraction()
    ct_to_concepts_extract.run(pipeline)

    assert len(pipeline.candidate_terms) == 0
    assert len(pipeline.kr.concepts) == 4

    concept_index = {concept.label: concept for concept in pipeline.kr.concepts}

    c_bicycle_lr = concept_index["bicycle"].linguistic_realisations.pop()
    c_bike_lr = concept_index["bike"].linguistic_realisations.pop()

    assert len(c_bicycle_lr.corpus_occurrences) == 3
    assert len(c_bike_lr.corpus_occurrences) == 1
