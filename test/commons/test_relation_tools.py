from typing import Set

import pytest

from olaf.commons.relation_tools import crs_to_relation, cts_to_crs
from olaf.data_container.candidate_term_schema import CandidateRelation, CandidateTerm
from olaf.data_container.concept_schema import Concept
from olaf.data_container.linguistic_realisation_schema import LinguisticRealisation


@pytest.fixture(scope="session")
def cr_like(en_sm_spacy_model):
    sent1 = en_sm_spacy_model("I like bike.")
    sent2 = en_sm_spacy_model("You look like my sister.")
    cr_like = CandidateRelation(label="like", corpus_occurrences={sent1[1], sent2[2]})
    return cr_like


@pytest.fixture(scope="session")
def cr_love(en_sm_spacy_model):
    sent1 = en_sm_spacy_model("I love bike.")
    cr_love = CandidateRelation(label="love", corpus_occurrences={sent1[1]})
    return cr_love


@pytest.fixture(scope="session")
def c_cat() -> Concept:
    c_cat = Concept(
        label="cat", linguistic_realisations={LinguisticRealisation(label="cat")}
    )
    return c_cat


@pytest.fixture(scope="session")
def c_mouse() -> Concept:
    c_mouse = Concept(
        label="mouse",
        linguistic_realisations={
            LinguisticRealisation(label="mouse"),
            LinguisticRealisation(label="grey rat"),
        },
    )
    return c_mouse


@pytest.fixture(scope="session")
def c_dog() -> Concept:
    c_dog = Concept(
        label="dog", linguistic_realisations={LinguisticRealisation(label="dog")}
    )
    return c_dog


@pytest.fixture(scope="session")
def cr_eat(en_sm_spacy_model, c_cat, c_mouse):
    sent1 = en_sm_spacy_model("Cats eat grey rat.")
    sent2 = en_sm_spacy_model("My cat eat also mouse.")
    cr_eat = CandidateRelation(
        label="eat",
        corpus_occurrences={
            (sent1[0], sent1[1], sent1[2:3]),
            (sent2[1], sent2[2], sent2[4]),
        },
        source_concept=c_cat,
        destination_concept=c_mouse,
    )
    return cr_eat


@pytest.fixture(scope="session")
def ct_eat(en_sm_spacy_model) -> Set[CandidateTerm]:
    sent1 = en_sm_spacy_model("Cat eat grey rat.")
    sent2 = en_sm_spacy_model("My cat eat also little mouse.")
    sent3 = en_sm_spacy_model("I eat spinach.")
    ct_eat = CandidateTerm(
        label="eat", corpus_occurrences={sent1[1:2], sent2[2:3], sent3[1:2]}
    )
    return ct_eat


@pytest.fixture(scope="session")
def ct_like(en_sm_spacy_model) -> Set[CandidateTerm]:
    sent1 = en_sm_spacy_model("I like bike.")
    sent2 = en_sm_spacy_model("You look like my sister.")
    ct_like = CandidateTerm(label="like", corpus_occurrences={sent1[1:2], sent2[2:3]})
    return ct_like


@pytest.fixture(scope="session")
def ct_pairs_eat(en_sm_spacy_model) -> Set[CandidateTerm]:
    sent = en_sm_spacy_model("Cat dog eat mouse.")
    ct_pairs_eat = CandidateTerm(label="eat", corpus_occurrences={sent[2:3]})
    return {ct_pairs_eat}


def test_crs_to_relation(cr_like, cr_love, cr_eat, c_cat, c_mouse) -> None:
    rel1 = crs_to_relation({cr_like, cr_love})
    rel2 = crs_to_relation({cr_eat})

    assert rel1.label in {"like", "love"}
    assert len(rel1.linguistic_realisations) == 2

    assert rel2.label == "eat"
    assert len(rel2.linguistic_realisations) == 1
    assert rel2.source_concept == c_cat
    assert rel2.destination_concept == c_mouse


def test_cts_to_crs(ct_eat, ct_like, c_cat, c_mouse, en_sm_spacy_model) -> None:
    cts = {ct_eat, ct_like}
    concepts_labels_map = {}
    concepts_labels_map["cat"] = c_cat
    concepts_labels_map["mouse"] = c_mouse
    crs = cts_to_crs(cts, concepts_labels_map, en_sm_spacy_model, 2, "doc")

    assert len(crs) == 3
    for cr in crs:
        if cr.label == "eat":
            assert (cr.source_concept == c_cat) or (cr.source_concept is None)
            assert (cr.destination_concept == c_mouse) or (
                cr.destination_concept is None
            )
            if cr.source_concept is None:
                assert len(cr.corpus_occurrences) == 2
            else:
                assert len(cr.corpus_occurrences) == 1
        else:
            assert cr.label == "like"
            assert cr.source_concept is None
            assert cr.destination_concept is None
            assert len(cr.corpus_occurrences) == 2


def test_cts_to_crs_pairs(
    ct_pairs_eat, c_cat, c_dog, c_mouse, en_sm_spacy_model
) -> None:
    concepts_labels_map = {}
    concepts_labels_map["cat"] = c_cat
    concepts_labels_map["mouse"] = c_mouse
    concepts_labels_map["dog"] = c_dog
    crs = cts_to_crs(ct_pairs_eat, concepts_labels_map, en_sm_spacy_model, 3, "doc")
    assert len(crs) == 2
    for cr in crs:
        assert cr.destination_concept.label == "mouse"
        assert cr.source_concept.label == "cat" or cr.source_concept.label == "dog"
