from typing import Any, Dict

import pytest

from olaf.commons.errors import OptionError, ParameterError
from olaf.data_container import (
    CandidateTerm,
    Concept,
    KnowledgeRepresentation,
    LinguisticRealisation,
)
from olaf.pipeline.pipeline_component.concept_relation_extraction import (
    AgglomerativeClusteringRelationExtraction,
)
from olaf.pipeline.pipeline_schema import Pipeline


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
        "distance_threshold": 0.4,
    }
    return options


@pytest.fixture(scope="session")
def person_concept() -> Concept:
    person = Concept(
        label="person", linguistic_realisations=set([LinguisticRealisation("person")])
    )
    return person


@pytest.fixture(scope="session")
def vegetable_concept() -> Concept:
    vegetable = Concept(
        label="vegetable",
        linguistic_realisations=set([LinguisticRealisation("vegetables")]),
    )
    return vegetable


@pytest.fixture(scope="session")
def meat_concept() -> Concept:
    meat = Concept(
        label="meat", linguistic_realisations=set([LinguisticRealisation("meat")])
    )
    return meat


@pytest.fixture(scope="session")
def pipeline(
    person_concept, vegetable_concept, meat_concept, en_sm_spacy_model
) -> Pipeline:
    corpus = [
        "Let's dance.",
        "A person can eat both vegetables and meat.",
        "This person has food with some vegetables.",
    ]
    docs = list(en_sm_spacy_model.pipe(corpus))

    pipeline = Pipeline(spacy_model=en_sm_spacy_model, corpus=docs)
    pipeline.kr = KnowledgeRepresentation()
    candidate_terms_set = set()
    candidate_terms_set.add(
        CandidateTerm(label="dance", corpus_occurrences=set([docs[0][2:3]]))
    )
    candidate_terms_set.add(
        CandidateTerm(
            label="eat",
            corpus_occurrences=set([docs[1][3:4]]),
        )
    )
    candidate_terms_set.add(
        CandidateTerm(label="have food", corpus_occurrences=set([docs[2][2:4]]))
    )
    pipeline.candidate_terms = candidate_terms_set
    pipeline.kr.concepts = set([person_concept, vegetable_concept, meat_concept])
    return pipeline


class TestAgglomerativeClusteringExtractionParameters:
    def test_default_values(self, default_parameters, default_options):
        agglo_clustering = AgglomerativeClusteringRelationExtraction(
            parameters=default_parameters, options=default_options
        )
        assert agglo_clustering._nb_clusters == 2
        assert agglo_clustering._metric == "cosine"
        assert agglo_clustering._linkage == "average"
        assert agglo_clustering._distance_threshold is None
        assert agglo_clustering._embedding_model == "all-mpnet-base-v2"

    def test_wrong_option_nb_clusters(self, wrong_option_nb_clusters):
        with pytest.raises(OptionError):
            AgglomerativeClusteringRelationExtraction(options=wrong_option_nb_clusters)

    def test_wrong_option_metric(self, wrong_option_metric):
        with pytest.raises(OptionError):
            AgglomerativeClusteringRelationExtraction(options=wrong_option_metric)

    def test_wrong_option_linkage(self, wrong_option_linkage):
        with pytest.raises(OptionError):
            AgglomerativeClusteringRelationExtraction(options=wrong_option_linkage)

    def test_wrong_option_distance_threshold(self, wrong_option_distance_threshold):
        with pytest.raises(OptionError):
            AgglomerativeClusteringRelationExtraction(
                options=wrong_option_distance_threshold
            )

    def test_wrong_parameter_embedding_model(self, wrong_parameter_embedding_model):
        with pytest.raises(ParameterError):
            AgglomerativeClusteringRelationExtraction(
                parameters=wrong_parameter_embedding_model
            )


class TestAgglomerativeClusteringExtractionProcess:
    def test_run(
        self,
        good_parameters,
        good_options,
        pipeline,
        person_concept,
        vegetable_concept,
        meat_concept,
    ):
        agglo = AgglomerativeClusteringRelationExtraction(
            parameters=good_parameters, options=good_options
        )
        agglo.run(pipeline)

        relations = list(pipeline.kr.relations)
        assert len(relations) == 3

        for relation in relations:
            if relation.label == "dance":
                assert len(relation.linguistic_realisations) == 1
                assert relation.source_concept is None
                assert relation.destination_concept is None
            else:
                assert relation.source_concept == person_concept
                assert (relation.label == "eat") or (relation.label == "have food")
                if relation.destination_concept == vegetable_concept:
                    assert len(relation.linguistic_realisations) == 2
                else:
                    assert relation.destination_concept == meat_concept
                    assert len(relation.linguistic_realisations) == 1

        assert len(pipeline.candidate_terms) == 0
