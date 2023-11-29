from typing import Any, Callable, Dict, List

import pytest
<<<<<<< b0cf7c8b8ff034d867d3f0378f8731d3db73b215
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
=======
import spacy

from olaf.commons.errors import NotCallableError
from olaf.commons.spacy_processing_tools import (is_not_num, is_not_punct,
                                                 is_not_stopword, is_not_url)
from olaf.pipeline.data_preprocessing.token_selector_data_preprocessing import (
    TokenSelectorDataPreprocessing, TokenSelectorDataPreprocessingConfig)
>>>>>>> 5f44d722072b9d41dce1367c3ad9d632b4dd9d43
from olaf.pipeline.pipeline_schema import Pipeline


@pytest.fixture(scope="session")
def default_config() -> TokenSelectorDataPreprocessingConfig:
    config = TokenSelectorDataPreprocessingConfig()
    return config


@pytest.fixture(scope="session")
<<<<<<< b0cf7c8b8ff034d867d3f0378f8731d3db73b215
def token_attribute_parameters() -> Dict[str, Any]:
    parameters = {"token_sequence_doc_attribute": "selected_tokens"}
    return parameters
=======
def token_attribute_config() -> TokenSelectorDataPreprocessingConfig:
    config = TokenSelectorDataPreprocessingConfig(token_sequence_doc_attribute="custom_selected_tokens")
    return config
>>>>>>> 5f44d722072b9d41dce1367c3ad9d632b4dd9d43


@pytest.fixture(scope="session")
def not_callable_function() -> bool:
    return True


@pytest.fixture(scope="session")
<<<<<<< b0cf7c8b8ff034d867d3f0378f8731d3db73b215
def selection_function() -> Callable[[spacy.tokens.Token], bool]:
    def func(token: spacy.tokens.Token) -> bool:
=======
def selection_function() -> Callable[[spacy.tokens.Token], bool] :
    def func(token: spacy.tokens.Token) -> bool :
>>>>>>> 5f44d722072b9d41dce1367c3ad9d632b4dd9d43
        return True

    return func

<<<<<<< b0cf7c8b8ff034d867d3f0378f8731d3db73b215

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

=======
@pytest.fixture(scope="session")
def example_sentence() -> spacy.tokens.Doc :
    spacy_model = spacy.load("en_core_web_sm")
    spacy_sentence = spacy_model("I use spaCy https://spacy.io/ 2 times and I like it.")
    return spacy_sentence

@pytest.fixture(scope="session")
def example_sentence_no_num() -> List[spacy.tokens.Token] :
    spacy_model = spacy.load("en_core_web_sm")
    spacy_sentence = spacy_model("I use spaCy https://spacy.io/ 2 times and I like it.")
    return [token for token in spacy_sentence if not(token.like_num)]

@pytest.fixture(scope="session")
def example_sentence_no_punct() -> List[spacy.tokens.Token] :
    spacy_model = spacy.load("en_core_web_sm")
    spacy_sentence = spacy_model("I use spaCy https://spacy.io/ 2 times and I like it.")
    return [token for token in spacy_sentence if not(token.is_punct)]

@pytest.fixture(scope="session")
def example_sentence_no_stop_word() -> List[spacy.tokens.Token] :
    spacy_model = spacy.load("en_core_web_sm")
    spacy_sentence = spacy_model("I use spaCy https://spacy.io/ 2 times and I like it.")
    return [token for token in spacy_sentence if not(token.is_stop)]

@pytest.fixture(scope="session")
def example_sentence_no_url() -> List[spacy.tokens.Token] :
    spacy_model = spacy.load("en_core_web_sm")
    spacy_sentence = spacy_model("I use spaCy https://spacy.io/ 2 times and I like it.")
    return [token for token in spacy_sentence if not(token.like_url)]

@pytest.fixture(scope="session")
def example_sentence_no_num_url() -> List[spacy.tokens.Token] :
    spacy_model = spacy.load("en_core_web_sm")
    spacy_sentence = spacy_model("I use spaCy https://spacy.io/ 2 times and I like it.")
    return [token for token in spacy_sentence if (not(token.like_url) and not(token.like_num))]
>>>>>>> 5f44d722072b9d41dce1367c3ad9d632b4dd9d43

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
<<<<<<< b0cf7c8b8ff034d867d3f0378f8731d3db73b215
def pipeline_with_doc_attribute(en_sm_spacy_model, example_sentence) -> Pipeline:
    no_url_tokens = [token for token in example_sentence if not (token.like_url)]
    example_sentence._.set("selected_tokens", no_url_tokens)
    pipeline = Pipeline(en_sm_spacy_model, corpus=[example_sentence])
=======
def pipeline_with_doc_attribute() -> Pipeline :
    spacy_model = spacy.load("en_core_web_sm")
    spacy_sentence = spacy_model("I use spaCy https://spacy.io/ 2 times and I like it.")
    no_url_tokens = [token for token in spacy_sentence if not(token.like_url)]
    spacy_sentence._.set("custom_selected_tokens", no_url_tokens)
    pipeline = Pipeline(spacy_model, corpus=[spacy_sentence])    
>>>>>>> 5f44d722072b9d41dce1367c3ad9d632b4dd9d43
    return pipeline


<<<<<<< b0cf7c8b8ff034d867d3f0378f8731d3db73b215
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
=======
    def test_not_callable_selector(self, default_config, not_callable_function):
        with pytest.raises(NotCallableError) :
            TokenSelectorDataPreprocessing(not_callable_function, config=default_config)

    def test_default_config(self, default_config, selection_function):
        token_selector = TokenSelectorDataPreprocessing(selection_function, config=default_config)
        assert token_selector._token_sequence_doc_attribute == default_config.token_sequence_doc_attribute

    def test_token_selector_attribute(self,token_attribute_config, selection_function):
        token_selector = TokenSelectorDataPreprocessing(selection_function, config=token_attribute_config)
        assert token_selector._token_sequence_doc_attribute == token_attribute_config.token_sequence_doc_attribute
    
>>>>>>> 5f44d722072b9d41dce1367c3ad9d632b4dd9d43


<<<<<<< b0cf7c8b8ff034d867d3f0378f8731d3db73b215
class TestTokenSelectorFunctions:
    def test_num_selector_true(
        self, token_attribute_parameters, example_sentence, example_sentence_no_num
    ):
        token_selector = TokenSelectorDataPreprocessing(
            is_not_num, token_attribute_parameters
        )
=======
    def test_num_selector_true(self, token_attribute_config, example_sentence, example_sentence_no_num):
        token_selector = TokenSelectorDataPreprocessing(is_not_num, config=token_attribute_config)
>>>>>>> 5f44d722072b9d41dce1367c3ad9d632b4dd9d43
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

<<<<<<< b0cf7c8b8ff034d867d3f0378f8731d3db73b215
    def test_punct_selector_true(
        self, token_attribute_parameters, example_sentence, example_sentence_no_punct
    ):
        token_selector = TokenSelectorDataPreprocessing(
            is_not_punct, token_attribute_parameters
        )
=======
    def test_punct_selector_true(self, token_attribute_config, example_sentence, example_sentence_no_punct):
        token_selector = TokenSelectorDataPreprocessing(is_not_punct, config=token_attribute_config)
>>>>>>> 5f44d722072b9d41dce1367c3ad9d632b4dd9d43
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

<<<<<<< b0cf7c8b8ff034d867d3f0378f8731d3db73b215
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
=======
    def test_stop_word_selector_true(self, token_attribute_config, example_sentence, example_sentence_no_stop_word):
        token_selector = TokenSelectorDataPreprocessing(is_not_stopword, config=token_attribute_config)
        selected_tokens = token_selector._select_tokens(example_sentence)
        assert len(selected_tokens) == len(example_sentence_no_stop_word)
        assert all([token_pred.text == token_true.text for token_pred, token_true in zip(selected_tokens, example_sentence_no_stop_word)])
    
    def test_url_selector_true(self, token_attribute_config, example_sentence, example_sentence_no_url):
        token_selector = TokenSelectorDataPreprocessing(is_not_url, config=token_attribute_config)
>>>>>>> 5f44d722072b9d41dce1367c3ad9d632b4dd9d43
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
<<<<<<< b0cf7c8b8ff034d867d3f0378f8731d3db73b215
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
=======

    def test_run_without_doc_attribute(self, token_attribute_config, pipeline, example_sentence_no_num):
        token_selector = TokenSelectorDataPreprocessing(is_not_num, config=token_attribute_config)
        token_selector.run(pipeline)
        selected_tokens = token_selector.corpus[0]._.get(token_selector._token_sequence_doc_attribute)
        assert len(selected_tokens) == len(example_sentence_no_num)
        assert all([token_pred.text == token_true.text for token_pred, token_true in zip(selected_tokens, example_sentence_no_num)])
        
    def  test_run_selected_tokens_doc_attribute(self, token_attribute_config, pipeline_with_doc_attribute, example_sentence_no_num_url):
        token_selector = TokenSelectorDataPreprocessing(is_not_num, config=token_attribute_config)
        token_selector.run(pipeline_with_doc_attribute)
        selected_tokens = token_selector.corpus[0]._.get(token_selector._token_sequence_doc_attribute)
>>>>>>> 5f44d722072b9d41dce1367c3ad9d632b4dd9d43
        assert len(selected_tokens) == len(example_sentence_no_num_url)
        assert all(
            [
                token_pred.text == token_true.text
                for token_pred, token_true in zip(
                    selected_tokens, example_sentence_no_num_url
                )
            ]
        )
