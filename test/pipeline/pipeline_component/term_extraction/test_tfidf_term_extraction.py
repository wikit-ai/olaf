from typing import Any, Dict, List

import pytest
import spacy.tokens

from olaf.pipeline.pipeline_component.term_extraction.tfidf_term_extraction import (
    TFIDFTermExtraction,
)
from olaf.pipeline.pipeline_schema import Pipeline

custom_token_sequence_doc_attribute = "custom_token_sequence_doc_attribute"

if not spacy.tokens.Doc.has_extension(custom_token_sequence_doc_attribute):
    spacy.tokens.Doc.set_extension(custom_token_sequence_doc_attribute, default=[])


@pytest.fixture(scope="session")
def example_corpus(en_sm_spacy_model) -> List[spacy.tokens.doc.Doc]:
    corpus_texts = [
        "This is the first document.",
        "This is the second second document.",
        "And the third one.",
        "Is this the first document?",
    ]

    corpus = [en_sm_spacy_model(text) for text in corpus_texts]

    for doc in corpus:
        doc._.set(custom_token_sequence_doc_attribute, [doc[:]])

    return corpus


@pytest.fixture(scope="session")
def example_pipeline(en_sm_spacy_model, example_corpus) -> Pipeline:
    pipeline = Pipeline(spacy_model=en_sm_spacy_model, corpus=example_corpus)

    return pipeline


@pytest.fixture(scope="session")
def example_params() -> Dict[str, Any]:
    parameters = {"token_sequence_doc_attribute": custom_token_sequence_doc_attribute}
    return parameters


@pytest.fixture(scope="session")
def example_options() -> Dict[str, Any]:
    options = {"threshold": 0.35, "max_term_token_length": 2, "tfidf_agg_type": "MAX"}
    return options


@pytest.fixture(scope="session")
def custom_tfidf_term_extraction(
    example_params, example_options
) -> TFIDFTermExtraction:
    term_extraction = TFIDFTermExtraction(
        token_sequence_preprocessing=lambda span: [
            token.lower_.strip() for token in span
        ],
        **example_params,
        **example_options,
    )
    return term_extraction


class TestParamsOptionsDefaults:
    @pytest.fixture(scope="class")
    def default_tfidf_term_extraction(self) -> TFIDFTermExtraction:
        term_extraction = TFIDFTermExtraction()
        return term_extraction

    def test_default_token_sequences_doc_attribute(
        self, default_tfidf_term_extraction
    ) -> None:
        assert default_tfidf_term_extraction._token_sequences_doc_attribute is None

    def test_default_token_sequence_preprocessing(
        self, default_tfidf_term_extraction
    ) -> None:
        assert callable(default_tfidf_term_extraction.token_sequence_preprocessing)

    def test_default_max_term_token_length(self, default_tfidf_term_extraction) -> None:
        assert default_tfidf_term_extraction._max_term_token_length == 1

    def test_default_tfidf_agg_type(self, default_tfidf_term_extraction) -> None:
        assert default_tfidf_term_extraction.tfidf_agg_type == "MEAN"

    def test_default_candidate_term_threshold(
        self, default_tfidf_term_extraction
    ) -> None:
        assert default_tfidf_term_extraction.candidate_term_threshold == 0

    def test_default_ngram_range(self, default_tfidf_term_extraction) -> None:
        assert default_tfidf_term_extraction._ngram_range == (1, 1)


class TestCustomParamsOptions:
    def test_custom_token_sequences_doc_attribute(
        self, custom_tfidf_term_extraction
    ) -> None:
        assert (
            custom_tfidf_term_extraction._token_sequences_doc_attribute
            == "custom_token_sequence_doc_attribute"
        )

    def test_custom_max_term_token_length(self, custom_tfidf_term_extraction) -> None:
        assert custom_tfidf_term_extraction._max_term_token_length == 2

    def test_custom_tfidf_agg_type(self, custom_tfidf_term_extraction) -> None:
        assert custom_tfidf_term_extraction.tfidf_agg_type == "MAX"

    def test_custom_candidate_term_threshold(
        self, custom_tfidf_term_extraction
    ) -> None:
        assert custom_tfidf_term_extraction.candidate_term_threshold == 0.35

    def test_custom_ngram_range(self, custom_tfidf_term_extraction) -> None:
        assert custom_tfidf_term_extraction._ngram_range == (1, 2)


class TestTFIDFprocess:
    @pytest.fixture(scope="class")
    def small_corpus(self, example_corpus) -> List[spacy.tokens.doc.Doc]:
        two_docs_corpus = example_corpus[:2]
        return two_docs_corpus

    def test_extract_token_sequences(
        self, custom_tfidf_term_extraction, small_corpus
    ) -> None:
        token_sequences = custom_tfidf_term_extraction._extract_token_sequences(
            corpus=small_corpus
        )
        token_sequences_texts = {span.text for span in token_sequences}

        expected_texts = {
            "This is the first document.",
            "This is the second second document.",
        }

        assert token_sequences_texts == expected_texts

    def test_create_ngram_spans(
        self, custom_tfidf_term_extraction, small_corpus
    ) -> None:
        token_sequences = custom_tfidf_term_extraction._extract_token_sequences(
            corpus=small_corpus
        )

        vocabulary_spans = custom_tfidf_term_extraction._create_ngram_spans(
            token_sequences
        )
        vocabulary_spans_texts = {span.text for span in vocabulary_spans}

        expected_texts = {
            "This is",
            "is the",
            "the first",
            "first document",
            "document.",
            "This",
            "is",
            "the",
            "first",
            "document",
            "document",
            ".",
            "the second",
            "second second",
            "second document",
            "second",
        }

        assert vocabulary_spans_texts == expected_texts

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_spaced_term_corpus_occ_map(
        self, custom_tfidf_term_extraction, example_pipeline
    ) -> None:
        custom_tfidf_term_extraction.run(pipeline=example_pipeline)
        candidate_terms_text = {term.label for term in example_pipeline.candidate_terms}

        token_sequences = custom_tfidf_term_extraction._extract_token_sequences(
            corpus=example_pipeline.corpus
        )
        vocabulary_spans = custom_tfidf_term_extraction._create_ngram_spans(
            token_sequences
        )
        spaced_term_corpus_occ_map = (
            custom_tfidf_term_extraction._spaced_term_corpus_occ_map(vocabulary_spans)
        )

        spaced_term_corpus_occ_map_keys = set(spaced_term_corpus_occ_map)

        assert candidate_terms_text <= spaced_term_corpus_occ_map_keys

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_tfidf_run(
        self, example_params, example_options, en_sm_spacy_model, example_corpus
    ) -> None:
        term_extraction = TFIDFTermExtraction(
            token_sequence_preprocessing=lambda span: [
                token.lower_.strip() for token in span
            ],
            **example_params,
            **example_options,
        )

        pipeline = Pipeline(spacy_model=en_sm_spacy_model, corpus=example_corpus)

        term_extraction.run(pipeline=pipeline)

        assert len(pipeline.candidate_terms) == 12
