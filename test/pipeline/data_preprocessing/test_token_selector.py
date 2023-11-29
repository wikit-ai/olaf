from typing import Any, Callable, Dict, List

import pytest
import spacy.tokens

from olaf.commons.errors import NotCallableError
from olaf.commons.spacy_processing_tools import (
    is_not_num,
    is_not_punct,
    is_not_stopword,
    is_not_url,
)
from olaf.pipeline.data_preprocessing.token_selector_data_preprocessing import (
    TokenSelectorDataPreprocessing,
)
from olaf.pipeline.pipeline_schema import Pipeline


@pytest.fixture(scope="session")
def default_parameters() -> Dict[str, Any]:
    parameters = {}
    return parameters


@pytest.fixture(scope="session")
def token_attribute_parameters() -> Dict[str, Any]:
    parameters = {"token_sequence_doc_attribute": "selected_tokens"}
    return parameters


@pytest.fixture(scope="session")
def not_callable_function() -> bool:
    return True


@pytest.fixture(scope="session")
def selection_function() -> Callable[[spacy.tokens.Token], bool]:
    def func(token: spacy.tokens.Token) -> bool:
        return True

    return func


@pytest.fixture(scope="session")
def example_sentence(en_sm_spacy_model) -> spacy.tokens.Doc:
    example_sentence = en_sm_spacy_model(
        "I use spaCy https://spacy.io/ 2 times and I like it."
    )
    return example_sentence


@pytest.fixture(scope="session")
def example_sentence_no_num(example_sentence) -> List[spacy.tokens.Token]:
    return [token for token in example_sentence if not (token.like_num)]


@pytest.fixture(scope="session")
def example_sentence_no_punct(example_sentence) -> List[spacy.tokens.Token]:
    return [token for token in example_sentence if not (token.is_punct)]


@pytest.fixture(scope="session")
def example_sentence_no_stop_word(example_sentence) -> List[spacy.tokens.Token]:
    return [token for token in example_sentence if not (token.is_stop)]


@pytest.fixture(scope="session")
def example_sentence_no_url(example_sentence) -> List[spacy.tokens.Token]:
    return [token for token in example_sentence if not (token.like_url)]


@pytest.fixture(scope="session")
def example_sentence_no_num_url(example_sentence) -> List[spacy.tokens.Token]:
    return [
        token
        for token in example_sentence
        if (not (token.like_url) and not (token.like_num))
    ]


@pytest.fixture(scope="session")
def pipeline(en_sm_spacy_model, example_sentence) -> Pipeline:
    pipeline = Pipeline(en_sm_spacy_model, corpus=[example_sentence])
    return pipeline


@pytest.fixture(scope="session")
def pipeline_with_doc_attribute(en_sm_spacy_model, example_sentence) -> Pipeline:
    no_url_tokens = [token for token in example_sentence if not (token.like_url)]
    example_sentence._.set("selected_tokens", no_url_tokens)
    pipeline = Pipeline(en_sm_spacy_model, corpus=[example_sentence])
    return pipeline


class TestTokenSelectorDataPreprocessingParameters:
    def test_not_callable_selector(self, default_parameters, not_callable_function):
        with pytest.raises(NotCallableError):
            TokenSelectorDataPreprocessing(
                not_callable_function, parameters=default_parameters
            )

    def test_default_parameters(self, default_parameters, selection_function):
        token_selector = TokenSelectorDataPreprocessing(
            selection_function, parameters=default_parameters
        )
        assert token_selector._token_sequences_doc_attribute == "selected_tokens"

    def test_token_selector_attribute(
        self, token_attribute_parameters, selection_function
    ):
        token_selector = TokenSelectorDataPreprocessing(
            selection_function, parameters=token_attribute_parameters
        )
        assert token_selector._token_sequences_doc_attribute == "selected_tokens"


class TestTokenSelectorFunctions:
    def test_num_selector_true(
        self, token_attribute_parameters, example_sentence, example_sentence_no_num
    ):
        token_selector = TokenSelectorDataPreprocessing(
            is_not_num, token_attribute_parameters
        )
        selected_tokens = token_selector._select_tokens(example_sentence)
        assert len(selected_tokens) == len(example_sentence_no_num)
        assert all(
            [
                token_pred.text == token_true.text
                for token_pred, token_true in zip(
                    selected_tokens, example_sentence_no_num
                )
            ]
        )

    def test_punct_selector_true(
        self, token_attribute_parameters, example_sentence, example_sentence_no_punct
    ):
        token_selector = TokenSelectorDataPreprocessing(
            is_not_punct, token_attribute_parameters
        )
        selected_tokens = token_selector._select_tokens(example_sentence)
        assert len(selected_tokens) == len(example_sentence_no_punct)
        assert all(
            [
                token_pred.text == token_true.text
                for token_pred, token_true in zip(
                    selected_tokens, example_sentence_no_punct
                )
            ]
        )

    def test_stop_word_selector_true(
        self,
        token_attribute_parameters,
        example_sentence,
        example_sentence_no_stop_word,
    ):
        token_selector = TokenSelectorDataPreprocessing(
            is_not_stopword, token_attribute_parameters
        )
        selected_tokens = token_selector._select_tokens(example_sentence)
        assert len(selected_tokens) == len(example_sentence_no_stop_word)
        assert all(
            [
                token_pred.text == token_true.text
                for token_pred, token_true in zip(
                    selected_tokens, example_sentence_no_stop_word
                )
            ]
        )

    def test_url_selector_true(
        self, token_attribute_parameters, example_sentence, example_sentence_no_url
    ):
        token_selector = TokenSelectorDataPreprocessing(
            is_not_url, token_attribute_parameters
        )
        selected_tokens = token_selector._select_tokens(example_sentence)
        assert len(selected_tokens) == len(example_sentence_no_url)
        assert all(
            [
                token_pred.text == token_true.text
                for token_pred, token_true in zip(
                    selected_tokens, example_sentence_no_url
                )
            ]
        )


class TestTokenSelectorRun:
    def test_run_without_doc_attribute(
        self, token_attribute_parameters, pipeline, example_sentence_no_num
    ):
        token_selector = TokenSelectorDataPreprocessing(
            is_not_num, token_attribute_parameters
        )
        token_selector.run(pipeline)
        selected_tokens = token_selector.corpus[0]._.get(
            token_selector._token_sequences_doc_attribute
        )
        assert len(selected_tokens) == len(example_sentence_no_num)
        assert all(
            [
                token_pred.text == token_true.text
                for token_pred, token_true in zip(
                    selected_tokens, example_sentence_no_num
                )
            ]
        )

    def test_run_selected_tokens_doc_attribute(
        self,
        token_attribute_parameters,
        pipeline_with_doc_attribute,
        example_sentence_no_num_url,
    ):
        token_selector = TokenSelectorDataPreprocessing(
            is_not_num, token_attribute_parameters
        )
        token_selector.run(pipeline_with_doc_attribute)
        selected_tokens = token_selector.corpus[0]._.get(
            token_selector._token_sequences_doc_attribute
        )
        assert len(selected_tokens) == len(example_sentence_no_num_url)
        assert all(
            [
                token_pred.text == token_true.text
                for token_pred, token_true in zip(
                    selected_tokens, example_sentence_no_num_url
                )
            ]
        )
