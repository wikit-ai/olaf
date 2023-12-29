import pytest

from olaf.commons.logging_config import logger
from olaf.data_container.candidate_term_schema import CandidateTerm
from olaf.pipeline.pipeline_component.candidate_term_enrichment import (
    SemanticBasedEnrichment,
)
from olaf.pipeline.pipeline_schema import Pipeline


@pytest.fixture(scope="session")
def c_term_bicycle(en_md_spacy_model) -> CandidateTerm:
    candidate_term = CandidateTerm(
        label="bicycle",
        corpus_occurrences={en_md_spacy_model("bicycle")[0:1]},
    )
    return candidate_term


@pytest.fixture(scope="function")
def pipeline(c_term_bicycle, en_md_spacy_model) -> Pipeline:
    pipeline = Pipeline(spacy_model=en_md_spacy_model, corpus=[])
    pipeline.candidate_terms = {c_term_bicycle}
    return pipeline


@pytest.fixture(scope="session")
def wrong_pipeline(c_term_bicycle, en_sm_spacy_model) -> Pipeline:
    pipeline = Pipeline(spacy_model=en_sm_spacy_model, corpus=[])
    pipeline.candidate_terms = {c_term_bicycle}
    return pipeline


def test_not_working_semantic_based_enrichment(wrong_pipeline, caplog) -> None:
    semantic_enrichment = SemanticBasedEnrichment()
    assert semantic_enrichment.threshold == 0.9
    semantic_enrichment.run(wrong_pipeline)
    assert "No vectors loaded with the spaCy model." in caplog.text


def test_default_semantic_based_enrichment(pipeline) -> None:
    semantic_enrichment = SemanticBasedEnrichment()
    assert semantic_enrichment.threshold == 0.9
    semantic_enrichment.run(pipeline)
    assert len(pipeline.candidate_terms.pop().enrichment.synonyms) == 1


def test_config_semantic_based_enrichment(pipeline) -> None:
    semantic_enrichment = SemanticBasedEnrichment(threshold=0.7)
    assert semantic_enrichment.threshold == 0.7
    semantic_enrichment.run(pipeline)
    assert len(pipeline.candidate_terms.pop().enrichment.synonyms) == 6
