from random import sample
from typing import Dict, List, Set, Tuple

import pytest
import spacy.tokens.doc

from olaf.commons.errors import OptionError
from olaf.commons.spacy_processing_tools import spacy_span_ngrams
from olaf.pipeline.pipeline_component.term_extraction.c_value_term_extraction import (
    CvalueTermExtraction,
)
from olaf.pipeline.pipeline_schema import Pipeline

custom_token_sequences_doc_attribute = "tokens_sequences_custom_attr"

if not spacy.tokens.doc.Doc.has_extension(custom_token_sequences_doc_attribute):
    spacy.tokens.doc.Doc.set_extension(custom_token_sequences_doc_attribute, default=[])


@pytest.fixture(scope="session")
def example_texts() -> List[str]:
    corpus_texts = [
        "Getter for Doc attributes. Since the getter is only called when we access the attribute, we can refer to the Span's 'is_country' attribute here, which is already set in the processing step.",
        "Knowledge Engineering (KE) refers to all technical, scientific and social aspects involved in building, maintaining and using knowledge-based systems.",
        "Regardless of your occupation or age, you’ve heard about OpenAI’s generative pre-trained transformer (GPT) technology on LinkedIn, YouTube, or in the news.",
        "connection module for 3p standard brake resistor",
    ]

    return corpus_texts


@pytest.fixture(scope="session")
def language_model():
    nlp = spacy.load("en_core_web_sm")
    return nlp


# @pytest.fixture(scope="session")
# def custom_token_sequences_doc_attribute():
#     token_sequences_doc_attribute = "tokens_sequences_custom_attr"
#     return token_sequences_doc_attribute


@pytest.fixture(scope="session")
def test_options() -> Dict[str, str]:
    options = {"threshold": 2.0, "max_term_token_length": 3}

    return options


@pytest.fixture(scope="session")
def test_parameters() -> Dict[str, str]:
    parameters = {"token_sequence_doc_attribute": custom_token_sequences_doc_attribute}

    return parameters


@pytest.fixture(scope="session")
def unknown_doc_attr_parameters() -> Dict[str, str]:
    parameters = {
        "token_sequence_doc_attribute": "unknown_custom_token_sequences_doc_attribute"
    }

    return parameters


@pytest.fixture(scope="session")
def corpus_raw(example_texts, language_model) -> List[spacy.tokens.doc.Doc]:
    docs = [language_model(text) for text in example_texts]

    return docs


@pytest.fixture(scope="class")
def corpus_custom_doc_attr(example_texts, language_model) -> List[spacy.tokens.doc.Doc]:
    docs = [language_model(text) for text in example_texts]

    for doc in docs[:2]:
        spans = [doc[2:7], doc[5:6]]
        doc._.set(custom_token_sequences_doc_attribute, spans)

    yield docs

    spacy.tokens.Doc.remove_extension(custom_token_sequences_doc_attribute)


class TestCvalueTermExtractionOptions:
    @pytest.mark.parametrize(
        "missing_val_options", [{"max_term_token_length": 5}, {"threshold": 150.3}]
    )
    def test_check_options_missing(self, missing_val_options):
        with pytest.raises(OptionError):
            c_val_term_extraction = CvalueTermExtraction(options=missing_val_options)

    @pytest.mark.parametrize(
        "wrong_val_options",
        [
            {"threshold": 150.3, "max_term_token_length": 5.0},
            {"threshold": 150, "max_term_token_length": 5},
        ],
    )
    def test_check_options_wrong_values(self, wrong_val_options):
        with pytest.raises(OptionError):
            c_val_term_extraction = CvalueTermExtraction(options=wrong_val_options)

    @pytest.mark.parametrize(
        "options, expected_cval_threshold_val",
        [
            ({"threshold": 150.3, "max_term_token_length": 5}, 150.3),
            (
                {
                    "threshold": 150.3,
                    "max_term_token_length": 5,
                    "c_value_threshold": 12.5,
                },
                12.5,
            ),
        ],
    )
    def test_check_options_c_val_threshold(self, options, expected_cval_threshold_val):
        c_val_term_extraction = CvalueTermExtraction(options=options)

        assert c_val_term_extraction._c_value_threshold == expected_cval_threshold_val


class TestCvalueTermExtractionParameters:
    def test_check_parameters_no_doc_attr_set(
        self, test_options, unknown_doc_attr_parameters, caplog
    ):
        caplog.clear()
        c_val_term_extraction = CvalueTermExtraction(
            parameters=unknown_doc_attr_parameters, options=test_options
        )

        log_messages = [rec.message for rec in caplog.records]
        expected_log_msg = f"""User defined c-value token sequence attribute unknown_custom_token_sequences_doc_attribute not set on spaCy Doc.
                    By default the system will use the entire content of the document."""

        assert expected_log_msg in log_messages
        assert c_val_term_extraction._token_sequences_doc_attribute is None

    def test_check_parameters_no_doc_attr_provided(self, test_options, caplog):
        caplog.clear()
        c_val_term_extraction = CvalueTermExtraction(options=test_options)

        log_messages = [rec.message for rec in caplog.records]
        expected_log_msg = """C-value token sequence attribute not set by the user.
                By default the system will use the entire content of the document."""

        assert expected_log_msg in log_messages
        assert c_val_term_extraction._token_sequences_doc_attribute is None

    def test_check_parameters_no_doc_attr_ok(self, test_parameters, test_options):
        c_val_term_extraction = CvalueTermExtraction(
            parameters=test_parameters, options=test_options
        )

        assert (
            c_val_term_extraction._token_sequences_doc_attribute
            == custom_token_sequences_doc_attribute
        )


class TestCvalueTermExtractionMethods:
    @pytest.fixture(scope="class")
    def example_pipeline(language_model, corpus_custom_doc_attr) -> Pipeline:
        pipeline = Pipeline(spacy_model=language_model, corpus=corpus_custom_doc_attr)

        return pipeline

    @pytest.fixture(scope="class")
    def expected_token_sequences(
        self, corpus_custom_doc_attr
    ) -> Tuple[spacy.tokens.span.Span]:
        token_sequences = []
        for doc in corpus_custom_doc_attr[:2]:
            token_sequences.extend([doc[2:5], doc[3:6], doc[4:7], doc[5:6]])

        return tuple(token_sequences)

    @pytest.fixture(scope="class")
    def c_value_term_extraction(
        self, test_parameters, test_options
    ) -> CvalueTermExtraction:
        c_value_term_extraction_instance = CvalueTermExtraction(
            parameters=test_parameters, options=test_options
        )
        return c_value_term_extraction_instance

    @pytest.fixture(scope="class")
    def spaced_expected_token_seqs(self, expected_token_sequences) -> List[str]:
        token_str = [
            " ".join([t.text for t in span]) for span in expected_token_sequences
        ]
        return token_str

    @pytest.fixture(scope="class")
    def expected_potential_terms(self, expected_token_sequences) -> Set[str]:
        terms = set()
        for token_seq in expected_token_sequences:
            term_spans = []
            for i in range(2, len(token_seq)):
                spans = spacy_span_ngrams(token_seq, i)
                term_spans.extend(spans)

            for span in term_spans:
                terms.add(" ".join([t.text for t in span]))

        return terms

    def test_extract_token_sequences_with_doc_attr(
        self, c_value_term_extraction, corpus_custom_doc_attr, expected_token_sequences
    ):
        token_sequences = c_value_term_extraction._extract_token_sequences(
            corpus_custom_doc_attr
        )
        assert expected_token_sequences == token_sequences

    def test_extract_token_sequences_no_doc_attr_provided(
        self, corpus_raw, test_options, expected_token_sequences
    ):
        c_value_term_extraction = CvalueTermExtraction(options=test_options)

        token_sequences = c_value_term_extraction._extract_token_sequences(corpus_raw)

        max_token_length = test_options["max_term_token_length"]

        conditions = [len(token_sequences) > len(expected_token_sequences)]
        conditions.extend(
            [len(token_seq) <= max_token_length for token_seq in token_sequences]
        )

        assert all(conditions)

    def test_spaced_term_corpus_occ_map_length(
        self,
        c_value_term_extraction,
        expected_token_sequences,
        expected_potential_terms,
    ):
        term_corpus_occ_map = c_value_term_extraction._spaced_term_corpus_occ_map(
            expected_token_sequences
        )
        assert len(expected_potential_terms) == len(term_corpus_occ_map)

    def test_spaced_term_corpus_occ_map_keys(
        self,
        c_value_term_extraction,
        expected_token_sequences,
        expected_potential_terms,
    ):
        term_corpus_occ_map = c_value_term_extraction._spaced_term_corpus_occ_map(
            expected_token_sequences
        )
        term_corpus_occ_map_keys = term_corpus_occ_map.keys()

        conditions = [
            term in term_corpus_occ_map_keys for term in expected_potential_terms
        ]

        assert all(conditions)

    def test_spaced_term_corpus_occ_map_non_empty_value(
        self, c_value_term_extraction, expected_token_sequences
    ):
        term_corpus_occ_map = c_value_term_extraction._spaced_term_corpus_occ_map(
            expected_token_sequences
        )
        term_corpus_occ_map_values = term_corpus_occ_map.values()

        conditions = [
            len(corpus_occs) > 0 for corpus_occs in term_corpus_occ_map_values
        ]

        assert all(conditions)

    def test_extract_terms_no_new_terms(
        self,
        c_value_term_extraction,
        spaced_expected_token_seqs,
        expected_potential_terms,
    ):
        terms = c_value_term_extraction._extract_terms(spaced_expected_token_seqs)

        conditions = [term in expected_potential_terms for term in terms]

        assert all(conditions)

    def test_get_corpus_occurrences_no_empty_values(
        self,
        c_value_term_extraction,
        spaced_expected_token_seqs,
        expected_token_sequences,
    ):
        terms = c_value_term_extraction._extract_terms(spaced_expected_token_seqs)

        term_corpus_occ_map = c_value_term_extraction._spaced_term_corpus_occ_map(
            expected_token_sequences
        )

        conditions = [len(term_corpus_occ_map.get(term, [])) > 0 for term in terms]

        assert all(conditions)

    def test_run_CTs_have_corpus_occ(self, example_pipeline, c_value_term_extraction):
        c_value_term_extraction.run(example_pipeline)

        conditions = [
            len(ct.corpus_occurrences) > 0 for ct in example_pipeline.candidate_terms
        ]

        assert all(conditions)
