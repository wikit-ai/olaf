from collections import Counter
from typing import Dict, List, Tuple

import pytest

from olaf.algorithm import Cvalue


@pytest.fixture(scope="module")
def c_value_expected_scores() -> Dict[str, int]:
    return {
        "ADENOID CYSTIC BASAL CELL CARCINOMA": 11.6096,
        "CYSTIC BASAL CELL CARCINOMA": 12.0,
        "ULCERATED BASAL CELL CARCINOMA": 14.0,
        "RECURRENT BASAL CELL CARCINOMA": 10.0,
        "CIRCUMSCRIBED BASAL CELL CARCINOMA": 6.0,
        "BASAL CELL CARCINOMA": 1550.0933
    }


@pytest.fixture(scope="module")
def dummy_corpus_terms() -> List[str]:
    doc_counts = {
        "ADENOID CYSTIC BASAL CELL CARCINOMA": 5,
        "CYSTIC BASAL CELL CARCINOMA": 6,
        "ULCERATED BASAL CELL CARCINOMA": 7,
        "RECURRENT BASAL CELL CARCINOMA": 5,
        "CIRCUMSCRIBED BASAL CELL CARCINOMA": 3,
        "BASAL CELL CARCINOMA": 958
    }

    corpus_terms = []
    for term, count in doc_counts.items():
        corpus_terms.extend([term] * count)

    return corpus_terms


@pytest.fixture(scope="module")
def my_c_value(dummy_corpus_terms) -> Cvalue:
    c_value = Cvalue(
        corpus_terms=dummy_corpus_terms,
        max_term_token_length=5
    )

    return c_value


@pytest.fixture(scope="module")
def my_c_value_computed(dummy_corpus_terms) -> Cvalue:
    c_value = Cvalue(
        corpus_terms=dummy_corpus_terms,
        max_term_token_length=5
    )

    c_value.compute_c_values()

    return c_value


class TestCvalueAttributes:

    def test_c_values_setter_error(self, my_c_value) -> None:
        with pytest.raises(AttributeError):
            my_c_value.c_values = [(12.5, "some bad c-val")]

    def test_candidate_terms_setter_error(self, my_c_value) -> None:
        with pytest.raises(AttributeError):
            my_c_value.candidate_terms = ["some bad candidate term"]

    def test_terms_string_tokens_is_tuple(self, my_c_value) -> None:
        assert isinstance(my_c_value._terms_string_tokens, tuple)

    def test_terms_counter_is_counter(self, my_c_value) -> None:
        assert isinstance(my_c_value._terms_counter, Counter)

    def test_max_term_token_length(self, my_c_value) -> None:
        assert my_c_value.max_term_token_length == 5

    def test_max_term_token_length_is_max_term_length(self, dummy_corpus_terms) -> None:
        no_max_token_length_c_value = Cvalue(
            corpus_terms=dummy_corpus_terms
        )
        assert no_max_token_length_c_value.max_term_token_length == 5

    @pytest.mark.parametrize("max_token_length", [0, 1])
    def test_max_term_token_length_error(self, dummy_corpus_terms, max_token_length) -> None:
        with pytest.raises(AttributeError):
            bad_max_token_length_c_value = Cvalue(
                corpus_terms=dummy_corpus_terms,
                max_term_token_length=max_token_length
            )


class TestCvalueExtractAndCountTermsStringTokens:
    @pytest.mark.parametrize("expected_term_string",
                             [
                                 "ADENOID CYSTIC BASAL CELL CARCINOMA",
                                 "CYSTIC BASAL CELL CARCINOMA",
                                 "ULCERATED BASAL CELL CARCINOMA",
                                 "RECURRENT BASAL CELL CARCINOMA",
                                 "CIRCUMSCRIBED BASAL CELL CARCINOMA",
                                 "BASAL CELL CARCINOMA",
                                 "ADENOID CYSTIC BASAL CELL",
                                 "ADENOID CYSTIC BASAL",
                                 "ADENOID CYSTIC",
                                 "CYSTIC BASAL",
                                 "BASAL CELL",
                                 "CELL CARCINOMA",
                                 "ULCERATED BASAL",
                                 "ULCERATED BASAL CELL",
                                 "CIRCUMSCRIBED BASAL",
                                 "CIRCUMSCRIBED BASAL CELL"
                             ]
                             )
    def test_terms_counter_keys(self, my_c_value, expected_term_string) -> None:
        assert tuple(expected_term_string.split()
                     ) in my_c_value._terms_counter.keys()

    @pytest.mark.parametrize("term_string,count",
                             [
                                 ("ADENOID CYSTIC BASAL CELL CARCINOMA", 5),
                                 ("CYSTIC BASAL CELL CARCINOMA", 11),
                                 ("ULCERATED BASAL CELL CARCINOMA", 7),
                                 ("RECURRENT BASAL CELL CARCINOMA", 5),
                                 ("CIRCUMSCRIBED BASAL CELL CARCINOMA", 3),
                                 ("BASAL CELL CARCINOMA", 984)
                             ]
                             )
    def test_terms_counter_values(self, my_c_value, term_string, count) -> None:
        assert my_c_value._terms_counter[tuple(term_string.split())] == count

    @pytest.mark.parametrize("expected_term_string",
                             [
                                 "ADENOID CYSTIC BASAL CELL CARCINOMA",
                                 "CYSTIC BASAL CELL CARCINOMA",
                                 "ULCERATED BASAL CELL CARCINOMA",
                                 "RECURRENT BASAL CELL CARCINOMA",
                                 "CIRCUMSCRIBED BASAL CELL CARCINOMA",
                                 "BASAL CELL CARCINOMA",
                                 "ADENOID CYSTIC BASAL CELL",
                                 "ADENOID CYSTIC BASAL",
                                 "ADENOID CYSTIC",
                                 "CYSTIC BASAL",
                                 "BASAL CELL",
                                 "CELL CARCINOMA",
                                 "ULCERATED BASAL",
                                 "ULCERATED BASAL CELL",
                                 "CIRCUMSCRIBED BASAL",
                                 "CIRCUMSCRIBED BASAL CELL"
                             ]
                             )
    def test_terms_string_tokens_values(self, my_c_value, expected_term_string) -> None:
        assert tuple(expected_term_string.split()
                     ) in my_c_value._terms_string_tokens


class TestCvalueOrderTermsStringTokens:
    def test_order_terms_string_tokens_token_length_order(self, my_c_value) -> None:
        substrings_token_lengths = [
            len(substring_tokens) for substring_tokens in my_c_value._terms_string_tokens
        ]

        assert sorted(substrings_token_lengths,
                      reverse=True) == substrings_token_lengths


class TestCvalueExtractTermSubstringsTokens:

    @pytest.fixture(scope="class")
    def test_term_substrings_tokens(self, my_c_value) -> Tuple[Tuple[str]]:
        test_term_tokens = tuple("ADENOID CYSTIC BASAL CELL CARCINOMA".split())
        test_term_substrings_tokens = my_c_value._extract_term_substrings_tokens(
            test_term_tokens)
        return test_term_substrings_tokens

    @pytest.mark.parametrize("expected_substring",
                             [
                                 "ADENOID CYSTIC BASAL CELL",
                                 "CYSTIC BASAL CELL CARCINOMA",
                                 "ADENOID CYSTIC BASAL",
                                 "CYSTIC BASAL CELL",
                                 "BASAL CELL CARCINOMA",
                                 "ADENOID CYSTIC",
                                 "CYSTIC BASAL",
                                 "BASAL CELL",
                                 "CELL CARCINOMA"
                             ]
                             )
    def test_extract_term_substrings(self, expected_substring, test_term_substrings_tokens) -> None:
        assert tuple(expected_substring.split()) in test_term_substrings_tokens

    def test_extract_term_substrings_tokens_order(self, test_term_substrings_tokens) -> None:
        substrings_token_lengths = [
            len(substring_tokens) for substring_tokens in test_term_substrings_tokens
        ]

        assert sorted(substrings_token_lengths,
                      reverse=True) == substrings_token_lengths


class TestCvalueComputation:

    @pytest.mark.parametrize("candidate_term",
                             [
                                 "ADENOID CYSTIC BASAL CELL CARCINOMA",
                                 "CYSTIC BASAL CELL CARCINOMA",
                                 "ULCERATED BASAL CELL CARCINOMA",
                                 "RECURRENT BASAL CELL CARCINOMA",
                                 "CIRCUMSCRIBED BASAL CELL CARCINOMA",
                                 "BASAL CELL CARCINOMA"
                             ]
                             )
    def test_candidate_terms_values(self, my_c_value_computed, candidate_term) -> None:

        assert candidate_term in my_c_value_computed.candidate_terms

    def test_c_values(self, my_c_value_computed, c_value_expected_scores) -> None:
        for c_val_tuple in my_c_value_computed.c_values:
            if c_value_expected_scores.get(c_val_tuple[1]) is not None:
                assert c_value_expected_scores.get(
                    c_val_tuple[1]) == round(c_val_tuple[0], 4)

    def test_c_values_order(self, my_c_value_computed) -> None:

        assert tuple(sorted(my_c_value_computed.c_values, key=lambda e: e[0],
                            reverse=True)) == my_c_value_computed.c_values

    def test_candidate_terms_order(self, my_c_value_computed) -> None:
        ordered_c_terms = tuple([c_val[1]
                                for c_val in my_c_value_computed.c_values])
        assert my_c_value_computed.candidate_terms == ordered_c_terms
