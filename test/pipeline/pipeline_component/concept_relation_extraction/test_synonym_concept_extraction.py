from typing import Set

import pytest
import spacy

from olaf.data_container.candidate_term_schema import CandidateTerm
from olaf.data_container.enrichment_schema import Enrichment
from olaf.pipeline.pipeline_component.concept_relation_extraction.synonym_concept_extraction import (
    SynonymConceptExtraction,
)
from olaf.pipeline.pipeline_schema import Pipeline


@pytest.fixture(scope="session")
def spacy_model():
    spacy_model = spacy.load(
        "en_core_web_sm",
        exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
    )
    return spacy_model


@pytest.fixture(scope="session")
def set_candidates(spacy_model) -> Set[CandidateTerm]:
    candidate_terms = set()

    candidate_terms.add(
        CandidateTerm(
            label="bicycle",
            corpus_occurrences={spacy_model("bicycle")[:]},
            enrichment=Enrichment({"bike", "cycle"}),
        )
    )
    candidate_terms.add(
        CandidateTerm(
            label="wine",
            corpus_occurrences={spacy_model("wine")[:]},
            enrichment=Enrichment({"drink", "beer"}),
        )
    )
    candidate_terms.add(
        CandidateTerm(
            label="tandem",
            corpus_occurrences={spacy_model("tandem")[:]},
            enrichment=Enrichment({"velocipede", "cycle"}),
        )
    )
    candidate_terms.add(
        CandidateTerm(
            label="duo",
            corpus_occurrences={spacy_model("duo")[:]},
            enrichment=Enrichment({"tandem"}),
        )
    )
    candidate_terms.add(
        CandidateTerm(
            label="cycling",
            corpus_occurrences={spacy_model("cycling")[:]},
            enrichment=Enrichment({"bike"}),
        )
    )
    candidate_terms.add(
        CandidateTerm(
            label="drink",
            corpus_occurrences={spacy_model("drink")[:]},
            enrichment=Enrichment({"water"}),
        )
    )
    candidate_terms.add(
        CandidateTerm(
            label="other",
            corpus_occurrences={spacy_model("other")[:]},
            enrichment=Enrichment({"new"}),
        )
    )

    return candidate_terms


@pytest.fixture(scope="session")
def pipeline(spacy_model, set_candidates) -> Pipeline:
    pipeline = Pipeline(spacy_model=spacy_model, corpus=[])
    pipeline.candidate_terms = set_candidates
    return pipeline


def test_run(pipeline):
    synonym_grouping = SynonymConceptExtraction()
    synonym_grouping.run(pipeline)
    assert len(pipeline.kr.concepts) == 3
    for concept in pipeline.kr.concepts:
        assert len(concept.linguistic_realisations) in [2, 4, 7]

        if len(concept.linguistic_realisations) == 2:
            conditions = [
                ct.label in ["other", "new"] for ct in concept.linguistic_realisations
            ]
        if len(concept.linguistic_realisations) == 4:
            conditions = [
                ct.label in ["drink", "wine", "water", "beer"]
                for ct in concept.linguistic_realisations
            ]
            assert all(conditions)

        if len(concept.linguistic_realisations) == 7:
            conditions = [
                ct.label
                in [
                    "cycling",
                    "bicycle",
                    "duo",
                    "tandem",
                    "bike",
                    "cycle",
                    "velocipede",
                ]
                for ct in concept.linguistic_realisations
            ]
            assert all(conditions)
