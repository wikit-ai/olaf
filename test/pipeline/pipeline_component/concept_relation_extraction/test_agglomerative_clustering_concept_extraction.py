from typing import Any, Dict

import pytest

from olaf import Pipeline
from olaf.commons.errors import OptionError, ParameterError
from olaf.data_container import CandidateTerm, KnowledgeRepresentation
from olaf.pipeline.pipeline_component.concept_relation_extraction import (
    AgglomerativeClusteringConceptExtraction,
)


@pytest.fixture(scope="session")
def default_parameters() -> Dict[str, Any]:
    parameters = {}
    return parameters


@pytest.fixture(scope="session")
def default_options() -> Dict[str, Any]:
    options = {}
    return options


@pytest.fixture(scope="session")
def wrong_option_nb_clusters() -> Dict[str, Any]:
    options = {"nb_clusters": "deux"}
    return options


@pytest.fixture(scope="session")
def wrong_option_metric() -> Dict[str, Any]:
    options = {"metric": 2}
    return options


@pytest.fixture(scope="session")
def wrong_option_linkage() -> Dict[str, Any]:
    options = {"linkage": 2}
    return options


@pytest.fixture(scope="session")
def wrong_option_distance_threshold() -> Dict[str, Any]:
    options = {"distance_threshold": "deux"}
    return options


@pytest.fixture(scope="session")
def wrong_parameter_embedding_model() -> Dict[str, Any]:
    params = {"embedding_model": 2}
    return params


@pytest.fixture(scope="session")
def good_parameters() -> Dict[str, Any]:
    params = {"embedding_model": "all-mpnet-base-v2"}
    return params


@pytest.fixture(scope="session")
def good_options() -> Dict[str, Any]:
    options = {
        "nb_clusters": None,
        "metric": "cosine",
        "linkage": "average",
        "distance_threshold": 0.2,
    }
    return options


@pytest.fixture(scope="session")
def pipeline(en_sm_spacy_model) -> Pipeline:
    doc = en_sm_spacy_model("car bike bicycle")

    pipeline = Pipeline(spacy_model=en_sm_spacy_model, corpus=[doc])
    pipeline.kr = KnowledgeRepresentation()
    candidate_terms_set = set()
    candidate_terms_set.add(CandidateTerm(label="car", corpus_occurrences={doc[0]}))
    candidate_terms_set.add(CandidateTerm(label="bike", corpus_occurrences={doc[1]}))
    candidate_terms_set.add(CandidateTerm(label="bicycle", corpus_occurrences={doc[2]}))
    pipeline.candidate_terms = candidate_terms_set
    return pipeline


class TestAgglomerativeClusteringExtractionParameters:
    def test_default_values(self, default_parameters, default_options):
        agglo_clustering = AgglomerativeClusteringConceptExtraction(
            **default_parameters, **default_options
        )
        assert agglo_clustering._nb_clusters == 2
        assert agglo_clustering._metric == "cosine"
        assert agglo_clustering._linkage == "average"
        assert agglo_clustering._distance_threshold is None
        assert agglo_clustering._embedding_model == "all-mpnet-base-v2"

    def test_wrong_option_nb_clusters(self, wrong_option_nb_clusters):
        with pytest.raises(OptionError):
            AgglomerativeClusteringConceptExtraction(**wrong_option_nb_clusters)

    def test_wrong_option_metric(self, wrong_option_metric):
        with pytest.raises(OptionError):
            AgglomerativeClusteringConceptExtraction(**wrong_option_metric)

    def test_wrong_option_linkage(self, wrong_option_linkage):
        with pytest.raises(OptionError):
            AgglomerativeClusteringConceptExtraction(**wrong_option_linkage)

    def test_wrong_option_distance_threshold(self, wrong_option_distance_threshold):
        with pytest.raises(OptionError):
            AgglomerativeClusteringConceptExtraction(
                **wrong_option_distance_threshold
            )

    def test_wrong_parameter_embedding_model(self, wrong_parameter_embedding_model):
        with pytest.raises(ParameterError):
            AgglomerativeClusteringConceptExtraction(
                **wrong_parameter_embedding_model
            )


class TestAgglomerativeClusteringExtractionProcess:
    def test_run(self, good_parameters, good_options, pipeline):
        agglo = AgglomerativeClusteringConceptExtraction(
            **good_parameters, **good_options
        )
        agglo.run(pipeline)

        concepts = list(pipeline.kr.concepts)

        assert len(concepts) == 2

        if len(concepts[0].linguistic_realisations) == 1:
            assert concepts[0].label == "car"
            assert len(concepts[1].linguistic_realisations) == 2
            assert (concepts[1].label == "bike") or (concepts[1].label == "bicycle")

        else:
            assert len(concepts[0].linguistic_realisations) == 2
            assert (concepts[0].label == "bike") or (concepts[0].label == "bicycle")
            assert len(concepts[1].linguistic_realisations) == 1
            assert concepts[1].label == "car"

        assert len(pipeline.candidate_terms) == 0
