import json
import os
import tempfile

import pytest

from olaf.repository.serialiser import KRJSONSerialiser


@pytest.fixture(scope="module")
def kr_json() -> dict:
    kr_json_serialised = {
            "concepts": [
                {
                    "concept_id": 1423639871008,
                    "label": "Country",
                    "lrs": [
                        {
                            "label": "country",
                            "co_texts": [
                                "country"
                            ]
                        }
                    ]
                },
                {
                    "concept_id": 1423639870528,
                    "label": "Mozzarella",
                    "lrs": [
                        {
                            "label": "mozzarella",
                            "co_texts": [
                                "Mozzarella"
                            ]
                        }
                    ]
                },
                {
                    "concept_id": 1423639868512,
                    "label": "Pepperoni Sausage",
                    "lrs": [
                        {
                            "label": "pepperoni sausage",
                            "co_texts": [
                                "pepperoni sausage"
                            ]
                        }
                    ]
                },
                {
                    "concept_id": 1423639871104,
                    "label": "Cheese",
                    "lrs": [
                        {
                            "label": "cheese",
                            "co_texts": [
                                "cheese"
                            ]
                        }
                    ]
                },
                {
                    "concept_id": 1423639868128,
                    "label": "Tomato",
                    "lrs": [
                        {
                            "label": "tomato",
                            "co_texts": [
                                "tomato"
                            ]
                        }
                    ]
                },
                {
                    "concept_id": 1423639870192,
                    "label": "Non Vegetarian Pizza",
                    "lrs": [
                        {
                            "label": "non vegetarian pizza",
                            "co_texts": [
                                "non vegetarian pizza"
                            ]
                        }
                    ]
                },
                {
                    "concept_id": 1423639868224,
                    "label": "America",
                    "lrs": [
                        {
                            "label": "america",
                            "co_texts": [
                                "America"
                            ]
                        }
                    ]
                },
                {
                    "concept_id": 1423639870336,
                    "label": "Pizza",
                    "lrs": [
                        {
                            "label": "pizza",
                            "co_texts": [
                                "pizza"
                            ]
                        }
                    ]
                },
                {
                    "concept_id": 1423639870864,
                    "label": "Cheesy Pizza",
                    "lrs": [
                        {
                            "label": "cheesy pizza",
                            "co_texts": [
                                "cheesy pizza"
                            ]
                        }
                    ]
                },
                {
                    "concept_id": 1423639869856,
                    "label": "American",
                    "lrs": [
                        {
                            "label": "american",
                            "co_texts": [
                                "American"
                            ]
                        }
                    ]
                },
                {
                    "concept_id": 1423639870384,
                    "label": "Topping",
                    "lrs": [
                        {
                            "label": "topping",
                            "co_texts": []
                        }
                    ]
                }
            ],
            "relations": [
                {
                    "source_concept_id": 1423639869856,
                    "destination_concept_id": 1423639868512,
                    "label": "has ingredient",
                    "lrs": []
                },
                {
                    "source_concept_id": 1423639869856,
                    "destination_concept_id": 1423639870528,
                    "label": "has ingredient",
                    "lrs": []
                },
                {
                    "source_concept_id": 1423639869856,
                    "destination_concept_id": 1423639868224,
                    "label": "has country of origin",
                    "lrs": []
                },
                {
                    "source_concept_id": 1423639869856,
                    "destination_concept_id": 1423639868128,
                    "label": "has ingredient",
                    "lrs": []
                },
                {
                    "source_concept_id": None,
                    "destination_concept_id": None,
                    "label": "has base",
                    "lrs": []
                }
            ],
            "metarelations": [
                {
                    "source_concept_id": 1423639869856,
                    "destination_concept_id": 1423639870192,
                    "label": "has kind",
                    "lrs": []
                },
                {
                    "source_concept_id": 1423639871104,
                    "destination_concept_id": 1423639870384,
                    "label": "is_generalised_by",
                    "lrs": []
                },
                {
                    "source_concept_id": 1423639868512,
                    "destination_concept_id": 1423639870384,
                    "label": "is_generalised_by",
                    "lrs": []
                },
                {
                    "source_concept_id": 1423639870528,
                    "destination_concept_id": 1423639871104,
                    "label": "is_generalised_by",
                    "lrs": []
                },
                {
                    "source_concept_id": 1423639869856,
                    "destination_concept_id": 1423639870864,
                    "label": "is_generalised_by",
                    "lrs": []
                },
                {
                    "source_concept_id": 1423639868224,
                    "destination_concept_id": 1423639871008,
                    "label": "is_generalised_by",
                    "lrs": []
                },
                {
                    "source_concept_id": 1423639869856,
                    "destination_concept_id": 1423639870336,
                    "label": "is_generalised_by",
                    "lrs": []
                },
                {
                    "source_concept_id": 1423639870528,
                    "destination_concept_id": 1423639870384,
                    "label": "is_generalised_by",
                    "lrs": []
                }
            ],
            "rdf_graph": None
        }
    
    return kr_json_serialised

@pytest.fixture(scope="module")
def kr_serialisation_path(kr_json) -> str:
    test_kr_json_fn = "test_kr_json_serialisation.json"

    with tempfile.TemporaryDirectory() as newpath:
        old_cwd = os.getcwd()
        os.chdir(newpath)
        with open(test_kr_json_fn, "w", encoding="utf8") as json_file:
            json.dump(kr_json, json_file)

        yield os.path.join(newpath, test_kr_json_fn)
        os.chdir(old_cwd)

@pytest.fixture(scope="module")
def kr_json_serialiser() -> KRJSONSerialiser:
    kr_serialiser = KRJSONSerialiser()
    return kr_serialiser

@pytest.fixture(scope="module")
def kr_concept_index(kr_json_serialiser, kr_json, american_pizza_pipeline) -> dict:
    c_index = {}
    concepts = kr_json_serialiser.load_concepts_from_json(
        concepts_json=kr_json["concepts"],
        concepts_idx=c_index,
        pipeline=american_pizza_pipeline
    )
    return c_index

def test_load(kr_json_serialiser, american_pizza_ex_kr, kr_serialisation_path, american_pizza_pipeline) -> None:
    kr_json_serialiser.load(pipeline=american_pizza_pipeline, file_path=kr_serialisation_path)

    assert american_pizza_pipeline.kr is not None
    assert len(american_pizza_pipeline.kr.concepts) == len(american_pizza_ex_kr.concepts)
    assert all([len(c.linguistic_realisations) for c in american_pizza_pipeline.kr.concepts])
    assert len(american_pizza_pipeline.kr.relations) == len(american_pizza_ex_kr.relations)
    assert len(american_pizza_pipeline.kr.metarelations) == len(american_pizza_ex_kr.metarelations)

def test_serialise(kr_json_serialiser, american_pizza_ex_kr) -> None:
    test_kr_json_fn = "test_kr_json_serialisation.json"

    with tempfile.TemporaryDirectory() as newpath:
        test_kr_json_file_path = os.path.join(newpath, test_kr_json_fn)
        kr_json_serialiser.serialise(
            kr=american_pizza_ex_kr, file_path=test_kr_json_file_path
        )

        assert os.path.exists(test_kr_json_file_path)

        with open(test_kr_json_file_path, "r", encoding="utf8") as json_file:
            kr_json_dict = json.load(json_file)

        assert len(kr_json_dict["concepts"]) == len(american_pizza_ex_kr.concepts)
        assert len(kr_json_dict["relations"]) == len(american_pizza_ex_kr.relations)
        assert len(kr_json_dict["metarelations"]) == len(american_pizza_ex_kr.metarelations)

def test_load_concepts_from_json(kr_json_serialiser, kr_json, american_pizza_pipeline, american_pizza_ex_kr) -> None:
    c_index = {}
    concepts = kr_json_serialiser.load_concepts_from_json(
        concepts_json=kr_json["concepts"],
        concepts_idx=c_index,
        pipeline=american_pizza_pipeline
    )

    assert len(concepts) == len(american_pizza_ex_kr.concepts)
    assert all([len(c.linguistic_realisations) for c in concepts])
    assert len(c_index) == len(concepts)

def test_load_relations_from_json(kr_json_serialiser, kr_json, american_pizza_pipeline, american_pizza_ex_kr, kr_concept_index) -> None:
    relations = kr_json_serialiser.load_relations_from_json(
        relations_json=kr_json["relations"],
        concepts_idx=kr_concept_index,
        pipeline=american_pizza_pipeline
    )

    assert len(relations) == len(american_pizza_ex_kr.relations)

def test_load_metarelations_from_json(kr_json_serialiser, kr_json, american_pizza_pipeline, american_pizza_ex_kr, kr_concept_index) -> None:
    metarelations = kr_json_serialiser.load_metarelations_from_json(
        metarelations_json=kr_json["metarelations"],
        concepts_idx=kr_concept_index,
        pipeline=american_pizza_pipeline
    )

    assert len(metarelations) == len(american_pizza_ex_kr.metarelations)

def test_get_concepts_json(kr_json_serialiser, american_pizza_ex_kr) -> None:
    concepts_json = kr_json_serialiser.get_concepts_json(kr=american_pizza_ex_kr)

    assert len(concepts_json) == len(american_pizza_ex_kr.concepts)
    assert all([d.get("concept_id", False) for d in concepts_json])

def test_get_metarelations_json(kr_json_serialiser, american_pizza_ex_kr) -> None:
    metarelations_json = kr_json_serialiser.get_metarelations_json(kr=american_pizza_ex_kr)

    assert len(metarelations_json) == len(american_pizza_ex_kr.metarelations)
    assert all([d.get("source_concept_id", False) for d in metarelations_json])
    assert all([d.get("destination_concept_id", False) for d in metarelations_json])

# TODO: test kr with relation without any source or destination concepts
def test_get_relations_json(kr_json_serialiser, american_pizza_ex_kr) -> None:
    relations_json = kr_json_serialiser.get_relations_json(kr=american_pizza_ex_kr)

    assert len(relations_json) == len(american_pizza_ex_kr.relations)

def test_build_cos_from_strings(kr_json_serialiser, en_sm_spacy_model, american_cheesy_pizza_doc) -> None:
    corpus_occ_texts = ["pizza", "american"]

    test_cos = kr_json_serialiser.build_cos_from_strings(
        co_texts=corpus_occ_texts,
        spacy_model=en_sm_spacy_model,
        docs=[american_cheesy_pizza_doc]
    )

    assert len(test_cos) == 8
