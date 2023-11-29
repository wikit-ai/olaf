from typing import List, Set

import pytest
import spacy.tokens

from olaf.commons.candidate_term_tools import (
    build_cts_from_strings,
    cts_have_common_synonyms,
    cts_to_concept,
    filter_cts_on_first_token_in_term,
    filter_cts_on_last_token_in_term,
    filter_cts_on_token_in_term,
    find_synonym_candidates,
    group_cts_on_synonyms,
    split_cts_on_token,
)
from olaf.data_container.candidate_term_schema import CandidateTerm
from olaf.data_container.enrichment_schema import Enrichment


@pytest.fixture(scope="session")
def candidate_term_bike() -> CandidateTerm:
    candidate_term = CandidateTerm(label="bike", corpus_occurrences=set())
    return candidate_term


@pytest.fixture(scope="session")
def candidate_term_bicycle() -> CandidateTerm:
    candidate_term = CandidateTerm(
        label="bicycle",
        corpus_occurrences=set(),
        enrichment=Enrichment({"bike", "cycle"}),
    )
    return candidate_term


@pytest.fixture(scope="session")
def candidate_term_tandem() -> CandidateTerm:
    candidate_term = CandidateTerm(
        label="tandem",
        corpus_occurrences=set(),
        enrichment=Enrichment({"velocipede", "cycle"}),
    )
    return candidate_term


@pytest.fixture(scope="session")
def candidate_term_wine() -> CandidateTerm:
    candidate_term = CandidateTerm(
        label="wine", corpus_occurrences=set(), enrichment=Enrichment({"drink", "beer"})
    )
    return candidate_term


@pytest.fixture(scope="session")
def candidate_terms(en_sm_spacy_model) -> Set[CandidateTerm]:
    candidate_terms = set()
    candidate_terms.add(
        CandidateTerm(label="bike", corpus_occurrences={en_sm_spacy_model("bike")[:]})
    ),
    candidate_terms.add(
        CandidateTerm(
            label="bicycle", corpus_occurrences={en_sm_spacy_model("bicycle")[:]}
        )
    )
    return candidate_terms


@pytest.fixture(scope="session")
def list_candidates(en_sm_spacy_model) -> List[CandidateTerm]:
    candidate_terms = []

    candidate_terms.append(
        CandidateTerm(
            label="bicycle",
            corpus_occurrences={en_sm_spacy_model("bicycle")[:]},
            enrichment=Enrichment({"bike", "cycle"}),
        )
    )
    candidate_terms.append(
        CandidateTerm(
            label="other",
            corpus_occurrences={en_sm_spacy_model("other")[:]},
            enrichment=Enrichment({"new"}),
        )
    )
    candidate_terms.append(
        CandidateTerm(
            label="wine",
            corpus_occurrences={en_sm_spacy_model("wine")[:]},
            enrichment=Enrichment({"drink", "beer"}),
        )
    )
    candidate_terms.append(
        CandidateTerm(
            label="duo",
            corpus_occurrences={en_sm_spacy_model("duo")[:]},
            enrichment=Enrichment({"tandem"}),
        )
    )
    candidate_terms.append(
        CandidateTerm(
            label="drink",
            corpus_occurrences={en_sm_spacy_model("drink")[:]},
            enrichment=Enrichment({"water"}),
        )
    )
    candidate_terms.append(
        CandidateTerm(
            label="tandem",
            corpus_occurrences={en_sm_spacy_model("tandem")[:]},
            enrichment=Enrichment({"velocipede", "cycle"}),
        )
    )
    candidate_terms.append(
        CandidateTerm(
            label="cycling",
            corpus_occurrences={en_sm_spacy_model("cycling")[:]},
            enrichment=Enrichment({"bike"}),
        )
    )
    return candidate_terms


@pytest.fixture(scope="session")
def set_candidates(en_sm_spacy_model) -> Set[CandidateTerm]:
    candidate_terms = set()

    candidate_terms.add(
        CandidateTerm(
            label="bicycle",
            corpus_occurrences={en_sm_spacy_model("bicycle")[:]},
            enrichment=Enrichment({"bike", "cycle"}),
        )
    )
    candidate_terms.add(
        CandidateTerm(
            label="wine",
            corpus_occurrences={en_sm_spacy_model("wine")[:]},
            enrichment=Enrichment({"drink", "beer"}),
        )
    )
    candidate_terms.add(
        CandidateTerm(
            label="tandem",
            corpus_occurrences={en_sm_spacy_model("tandem")[:]},
            enrichment=Enrichment({"velocipede", "cycle"}),
        )
    )
    candidate_terms.add(
        CandidateTerm(
            label="duo",
            corpus_occurrences={en_sm_spacy_model("duo")[:]},
            enrichment=Enrichment({"tandem"}),
        )
    )
    candidate_terms.add(
        CandidateTerm(
            label="cycling",
            corpus_occurrences={en_sm_spacy_model("cycling")[:]},
            enrichment=Enrichment({"bike"}),
        )
    )
    candidate_terms.add(
        CandidateTerm(
            label="drink",
            corpus_occurrences={en_sm_spacy_model("drink")[:]},
            enrichment=Enrichment({"water"}),
        )
    )
    candidate_terms.add(
        CandidateTerm(
            label="other",
            corpus_occurrences={en_sm_spacy_model("other")[:]},
            enrichment=Enrichment({"new"}),
        )
    )
    return candidate_terms


@pytest.fixture(scope="session")
def raw_corpus() -> List[str]:
    corpus = [
        "Some bike with a fixed size wheel of diameter 2.",
        "A sentence without anything interesting.",
        "We are also talking about a bike and a fixed size wheel here!",
    ]
    return corpus


@pytest.fixture(scope="session")
def corpus_docs(raw_corpus, en_sm_spacy_model) -> List[spacy.tokens.Doc]:
    corpus = [doc for doc in en_sm_spacy_model.pipe(raw_corpus)]
    return corpus


@pytest.fixture(scope="session")
def c_term_bike_with() -> CandidateTerm:
    candidate_term = CandidateTerm(
        label="bike with",
        corpus_occurrences={},
    )
    return candidate_term


@pytest.fixture(scope="session")
def c_term_bike_with_more() -> CandidateTerm:
    candidate_term = CandidateTerm(
        label="bike with a fixed size wheel",
        corpus_occurrences={},
    )
    return candidate_term


@pytest.fixture(scope="session")
def c_term_wheel_diameter() -> CandidateTerm:
    candidate_term = CandidateTerm(
        label="wheel of diameter 2",
        corpus_occurrences={},
    )
    return candidate_term


@pytest.fixture(scope="session")
def c_term_with_a_fixed_size() -> CandidateTerm:
    candidate_term = CandidateTerm(
        label="with a fixed size",
        corpus_occurrences={},
    )
    return candidate_term


@pytest.fixture(scope="session")
def c_term_fixed_size_wheel() -> CandidateTerm:
    candidate_term = CandidateTerm(
        label="fixed size wheel",
        corpus_occurrences={},
    )
    return candidate_term


@pytest.fixture(scope="session")
def candidate_terms_for_post_processing(
    c_term_bike_with,
    c_term_bike_with_more,
    c_term_wheel_diameter,
    c_term_with_a_fixed_size,
    c_term_fixed_size_wheel,
) -> Set[CandidateTerm]:
    c_terms = {
        c_term_bike_with,
        c_term_bike_with_more,
        c_term_wheel_diameter,
        c_term_with_a_fixed_size,
        c_term_fixed_size_wheel,
    }
    return c_terms


def test_synonym_verification_on_label(candidate_term_bike, candidate_term_bicycle):
    assert cts_have_common_synonyms(candidate_term_bike, candidate_term_bicycle)
    assert cts_have_common_synonyms(candidate_term_bicycle, candidate_term_bike)


def test_synonym_verification_on_enrichment(
    candidate_term_bicycle, candidate_term_tandem
):
    assert cts_have_common_synonyms(candidate_term_bicycle, candidate_term_tandem)
    assert cts_have_common_synonyms(candidate_term_tandem, candidate_term_bicycle)


def test_not_synonym_candidates(candidate_term_bicycle, candidate_term_wine):
    assert not (cts_have_common_synonyms(candidate_term_bicycle, candidate_term_wine))
    assert not (cts_have_common_synonyms(candidate_term_wine, candidate_term_bicycle))


def test_concept_creation(candidate_terms):
    created_concept = cts_to_concept(candidate_terms)
    labels = ["bike", "bicycle"]
    assert created_concept.label in labels
    assert len(created_concept.linguistic_realisations) == 2
    for lr in created_concept.linguistic_realisations:
        assert lr.label in labels
        assert list(lr.corpus_occurrences)[0].text in labels


def test_find_common_syn_from_ref_term(list_candidates):
    common_candidates = set()
    ref_term = list_candidates.pop(0)
    common_candidates.add(ref_term)
    find_synonym_candidates(ref_term, list_candidates, common_candidates)
    assert len(common_candidates) == 4
    assert len(list_candidates) == 3
    conditions = [
        ct.label in ["cycling", "bicycle", "duo", "tandem"] for ct in common_candidates
    ]
    assert all(conditions)


def test_group_ct_on_synonyms(set_candidates):
    common_groups = group_cts_on_synonyms(set_candidates)
    assert len(common_groups) == 3
    for group in common_groups:
        assert len(group) in [1, 2, 4]

        if len(group) == 1:
            assert group.pop().label == "other"

        if len(group) == 2:
            conditions = [ct.label in ["drink", "wine"] for ct in group]
            assert all(conditions)

        if len(group) == 4:
            conditions = [
                ct.label in ["cycling", "bicycle", "duo", "tandem"] for ct in group
            ]
            assert all(conditions)


def test_filter_cts_on_first_token_in_term(candidate_terms_for_post_processing) -> None:
    filtered_cts = filter_cts_on_first_token_in_term(
        candidate_terms=candidate_terms_for_post_processing,
        filtering_tokens={"with", "of"},
    )

    filtered_ct_labels = {ct.label for ct in filtered_cts}

    assert len(filtered_cts) == 4
    assert "with a fixed size" not in filtered_ct_labels
    assert "fixed size wheel" in filtered_ct_labels


def test_filter_cts_on_last_token_in_term(candidate_terms_for_post_processing) -> None:
    filtered_cts = filter_cts_on_last_token_in_term(
        candidate_terms=candidate_terms_for_post_processing,
        filtering_tokens={"with", "of"},
    )

    filtered_ct_labels = {ct.label for ct in filtered_cts}

    assert len(filtered_cts) == 4
    assert "bike with" not in filtered_ct_labels
    assert "fixed size wheel" in filtered_ct_labels


def test_filter_cts_on_token_in_term(candidate_terms_for_post_processing) -> None:
    filtered_cts = filter_cts_on_token_in_term(
        candidate_terms=candidate_terms_for_post_processing,
        filtering_tokens={"with", "of"},
    )

    filtered_ct_labels = {ct.label for ct in filtered_cts}

    assert filtered_ct_labels == {"fixed size wheel"}


def test_build_cts_from_strings(en_sm_spacy_model, corpus_docs) -> None:
    cts = build_cts_from_strings(
        ct_label_strings={"fixed size", "anything interesting", "bike"},
        spacy_model=en_sm_spacy_model,
        docs=corpus_docs,
    )

    cts_index = {ct.label: ct for ct in cts}

    assert len(cts) == 3
    assert len(cts_index["fixed size"].corpus_occurrences) == 2
    assert len(cts_index["bike"].corpus_occurrences) == 2


def test_split_cts_on_token(
    candidate_terms_for_post_processing, en_sm_spacy_model, corpus_docs
) -> None:
    cts = split_cts_on_token(
        candidate_terms=candidate_terms_for_post_processing,
        splitting_tokens={"with", "of"},
        spacy_model=en_sm_spacy_model,
        docs=corpus_docs,
    )

    cts_index = {ct.label: ct for ct in cts}

    assert len(cts) == 6
    assert len(cts_index["bike"].corpus_occurrences) == 2
