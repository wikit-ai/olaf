from typing import Any, Callable, Dict, List, Tuple

import pytest
import spacy.tokens

from olaf.pipeline.pipeline_component.term_extraction.pos_term_extraction import (
    POSTermExtraction,
)
from olaf.pipeline.pipeline_schema import Pipeline


@pytest.fixture(scope="session")
def default_parameters() -> Dict[str, Any]:
    parameters = {}
    return parameters


@pytest.fixture(scope="session")
def doc_attribute_parameters() -> Dict[str, Any]:
    parameters = {"token_sequences_doc_attribute": "selected_tokens"}
    return parameters


@pytest.fixture(scope="session")
def token_processing() -> Callable[[spacy.tokens.Token], str]:
    return lambda token: token.lemma_


@pytest.fixture(scope="session")
def corpus(en_sm_spacy_model) -> List[spacy.tokens.Span]:
    sentences = ["I bought a car and bikes.", "I eat vegetables and fruit."]
    corpus = list(en_sm_spacy_model.pipe(sentences))

    return corpus


@pytest.fixture(scope="session")
def doc_attribute_corpus(en_sm_spacy_model) -> List[spacy.tokens.Span]:
    sentences = ["I bought a car and bikes.", "I eat vegetables and fruit."]
    corpus = list(en_sm_spacy_model.pipe(sentences))
    if not spacy.tokens.Doc.has_extension("selected_tokens"):
        spacy.tokens.Doc.set_extension("selected_tokens", default=[], force=True)
    corpus[0]._.set(
        "selected_tokens", [en_sm_spacy_model("I bought a car and bikes.")[:]]
    )
    corpus[1]._.set("selected_tokens", [en_sm_spacy_model(" ")[:]])

    return corpus


@pytest.fixture(scope="session")
def sentences() -> List[spacy.tokens.Span]:
    sentences = ["I bought a car and bikes.", "I eat vegetables and fruit."]
    return sentences


@pytest.fixture(scope="session")
def doc_attribute_sentences() -> List[spacy.tokens.Span]:
    sentences = ["I bought a car and bikes.", " "]
    return sentences


@pytest.fixture(scope="session")
def token_sequences(en_sm_spacy_model) -> Tuple[spacy.tokens.Span]:
    sentences = ["I bought a car and bikes.", "I eat vegetables and fruit."]
    corpus = list(en_sm_spacy_model.pipe(sentences))

    token_seq = tuple([doc[:] for doc in corpus])
    return token_seq


@pytest.fixture(scope="session")
def candidate_tokens(en_sm_spacy_model) -> Dict[str, List[spacy.tokens.Token]]:
    sentences = ["I bought a car and bikes.", "I eat vegetables and fruit."]
    corpus = list(en_sm_spacy_model.pipe(sentences))

    cand_tokens = []
    for doc in corpus:
        for token in doc:
            if token.pos_ == "NOUN":
                cand_tokens.append(token.doc[token.i : token.i + 1])
    return cand_tokens


@pytest.fixture(scope="session")
def attribute_pipeline(en_sm_spacy_model) -> Pipeline:
    sentences = ["I bought a car and bikes.", "I eat vegetables and fruit."]
    corpus = list(en_sm_spacy_model.pipe(sentences))

    corpus[0]._.set(
        "selected_tokens", [en_sm_spacy_model("I bought a car and bikes.")[:]]
    )
    corpus[1]._.set("selected_tokens", [en_sm_spacy_model(" ")[:]])

    pipeline = Pipeline(spacy_model=en_sm_spacy_model, corpus=corpus)
    pipeline.candidate_terms = set()
    return pipeline


@pytest.fixture(scope="session")
def doc_pipeline(en_sm_spacy_model) -> Pipeline:
    sentences = ["I bought a car and bikes.", "I eat vegetables and fruit."]
    corpus = list(en_sm_spacy_model.pipe(sentences))

    pipeline = Pipeline(spacy_model=en_sm_spacy_model, corpus=corpus)
    pipeline.candidate_terms = set()
    return pipeline


@pytest.fixture(scope="session")
def candidate_terms() -> List[str]:
    candidate_terms = ["car", "bike", "vegetable", "fruit"]
    return candidate_terms


class TestPOSTermExtractionParameters:
    def test_default_parameters_value(self, default_parameters):
        pos_based_term_extraction = POSTermExtraction(**default_parameters)
        assert pos_based_term_extraction._token_sequences_doc_attribute is None
        assert pos_based_term_extraction._pos_selection == ["NOUN"]


class TestPOSTermExtractionFunctions:
    def test_extract_token_sequences_from_doc(
        self, token_processing, default_parameters, corpus, sentences
    ) -> Tuple[spacy.tokens.Span]:
        pos_based_term_extraction = POSTermExtraction(
            token_processing, **default_parameters
        )
        token_sequences = pos_based_term_extraction._extract_token_sequences(corpus)
        assert len(token_sequences) == 2
        conditions = [
            token_sequence.text in sentences for token_sequence in token_sequences
        ]
        assert all(conditions)

    def test_extract_token_sequences_from_attribute(
        self,
        token_processing,
        doc_attribute_parameters,
        doc_attribute_corpus,
        doc_attribute_sentences,
    ):
        pos_based_term_extraction = POSTermExtraction(
            token_processing, **doc_attribute_parameters
        )
        token_sequences = pos_based_term_extraction._extract_token_sequences(
            doc_attribute_corpus
        )
        assert len(token_sequences) == 2
        conditions = [
            token_sequence.text in doc_attribute_sentences
            for token_sequence in token_sequences
        ]
        assert all(conditions)

    def test_extract_candidate_tokens(
        self, token_processing, default_parameters, token_sequences, candidate_tokens
    ) -> List[spacy.tokens.Token]:
        pos_based_term_extraction = POSTermExtraction(
            token_processing, **default_parameters
        )
        cand_tokens = pos_based_term_extraction._extract_candidate_tokens(
            token_sequences
        )
        assert len(cand_tokens) == 4
        candidate_terms = [ctoken.text for ctoken in candidate_tokens]
        conditions = [cand_token.text in candidate_terms for cand_token in cand_tokens]
        assert all(conditions)

    def test_build_term_corpus_occ_map(
        self, token_processing, default_parameters, candidate_tokens, candidate_terms
    ) -> Dict[str, List[spacy.tokens.Token]]:
        pos_based_term_extraction = POSTermExtraction(
            token_processing, **default_parameters
        )
        term_corpus_occ_map = pos_based_term_extraction._build_term_corpus_occ_map(
            candidate_tokens
        )
        assert len(term_corpus_occ_map) == 4
        conditions = [
            term in candidate_terms for term in list(term_corpus_occ_map.keys())
        ]
        assert all(conditions)


class TestPOSTermExtractionProcess:
    def test_run_on_doc(
        self, token_processing, default_parameters, doc_pipeline, candidate_terms
    ):
        pos_based_term_extraction = POSTermExtraction(
            token_processing, **default_parameters
        )
        pos_based_term_extraction.run(doc_pipeline)
        assert len(doc_pipeline.candidate_terms) == 4
        conditions = [
            candidate.label in candidate_terms
            for candidate in doc_pipeline.candidate_terms
        ]
        assert all(conditions)

    def test_run_on_attribute(
        self,
        token_processing,
        doc_attribute_parameters,
        attribute_pipeline,
        candidate_terms,
    ):
        pos_based_term_extraction = POSTermExtraction(
            token_processing, **doc_attribute_parameters
        )
        pos_based_term_extraction.run(attribute_pipeline)
        assert len(attribute_pipeline.candidate_terms) == 2
        conditions = [
            candidate.label in candidate_terms
            for candidate in attribute_pipeline.candidate_terms
        ]
        assert all(conditions)
