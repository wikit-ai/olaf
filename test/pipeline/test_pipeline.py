import pytest

from olaf.commons.errors import PipelineCorpusInitialisationError
from olaf.commons.spacy_processing_tools import is_not_stopword
from olaf.pipeline.data_preprocessing.token_selector_data_preprocessing import (
    TokenSelectorDataPreprocessing,
)
from olaf.pipeline.pipeline_component.concept_relation_extraction.synonym_concept_extraction import (
    SynonymConceptExtraction,
)
from olaf.pipeline.pipeline_component.term_extraction.pos_term_extraction import (
    POSTermExtraction,
)
from olaf.pipeline.pipeline_schema import Pipeline


@pytest.fixture(scope="function")
def corpus(en_sm_spacy_model):
    text_corpus = ["I like bike.", "I love beer.", "I eat pasta and pizza."]
    corpus = list(en_sm_spacy_model.pipe(text_corpus))
    return corpus


@pytest.fixture(scope="function")
def pipeline(en_sm_spacy_model, corpus) -> Pipeline:
    pipeline = Pipeline(spacy_model=en_sm_spacy_model, corpus=corpus)
    return pipeline


@pytest.fixture(scope="function")
def empty_pipeline(en_sm_spacy_model, corpus) -> Pipeline:
    empty_pipeline = Pipeline(spacy_model=en_sm_spacy_model, corpus=corpus)
    return empty_pipeline


@pytest.fixture(scope="function")
def extraction_pipeline(
    en_sm_spacy_model, corpus, preprocessing, component
) -> Pipeline:
    extraction_pipeline = Pipeline(spacy_model=en_sm_spacy_model, corpus=corpus)
    extraction_pipeline.add_preprocessing_component(preprocessing)
    extraction_pipeline.add_pipeline_component(component)
    return extraction_pipeline


@pytest.fixture(scope="function")
def concept_pipeline(
    en_sm_spacy_model, corpus, preprocessing, component, concept_component
) -> Pipeline:
    concept_pipeline = Pipeline(spacy_model=en_sm_spacy_model, corpus=corpus)
    concept_pipeline.add_preprocessing_component(preprocessing)
    concept_pipeline.add_pipeline_component(component)
    concept_pipeline.add_pipeline_component(concept_component)
    return concept_pipeline


@pytest.fixture(scope="function")
def preprocessing() -> TokenSelectorDataPreprocessing:
    preprocessing = TokenSelectorDataPreprocessing(selector=is_not_stopword)
    return preprocessing


@pytest.fixture(scope="function")
def component() -> POSTermExtraction:
    parameters = {"token_sequences_doc_attribute": "selected_tokens"}
    component = POSTermExtraction(**parameters)
    return component


@pytest.fixture(scope="function")
def concept_component() -> SynonymConceptExtraction:
    concept_component = SynonymConceptExtraction()
    return concept_component


def test_pipeline_initialisation_error(en_sm_spacy_model) -> None:
    with pytest.raises(PipelineCorpusInitialisationError):
        pipeline = Pipeline(spacy_model=en_sm_spacy_model)


def test_add_and_remove_preprocessing(preprocessing, pipeline) -> None:
    assert len(pipeline.preprocessing_components) == 0
    pipeline.add_preprocessing_component(preprocessing)
    assert len(pipeline.preprocessing_components) == 1
    pipeline.remove_preprocessing_component(preprocessing)
    assert len(pipeline.preprocessing_components) == 0


def test_add_and_remove_component(component, pipeline) -> None:
    assert len(pipeline.pipeline_components) == 0
    pipeline.add_pipeline_component(component)
    assert len(pipeline.pipeline_components) == 1
    pipeline.remove_pipeline_component(component)
    assert len(pipeline.pipeline_components) == 0


def test_running_empty_pipeline(empty_pipeline) -> None:
    assert len(empty_pipeline.candidate_terms) == 0
    assert len(empty_pipeline.kr.concepts) == 0
    empty_pipeline.run()
    assert len(empty_pipeline.candidate_terms) == 0
    assert len(empty_pipeline.kr.concepts) == 0


def test_running_extraction_pipeline(extraction_pipeline) -> None:
    assert len(extraction_pipeline.candidate_terms) == 0
    assert len(extraction_pipeline.kr.concepts) == 0
    extraction_pipeline.run()
    assert len(extraction_pipeline.candidate_terms) == 4
    assert len(extraction_pipeline.kr.concepts) == 0


def test_running_concept_pipeline(concept_pipeline) -> None:
    assert len(concept_pipeline.candidate_terms) == 0
    assert len(concept_pipeline.kr.concepts) == 0
    concept_pipeline.run()
    assert len(concept_pipeline.candidate_terms) == 0
    assert len(concept_pipeline.kr.concepts) == 4
