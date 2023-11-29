from functools import partial
from typing import Dict, List, Set

import pytest
import spacy.tokens

from olaf.commons.candidate_term_tools import (
    filter_cts_on_first_token_in_term,
    filter_cts_on_last_token_in_term,
    filter_cts_on_token_in_term,
    split_cts_on_token,
)
from olaf.commons.errors import NotCallableError
from olaf.pipeline.pipeline_component.term_extraction.manual_candidate_terms import (
    ManualCandidateTermExtraction,
)
from olaf.pipeline.pipeline_schema import Pipeline


@pytest.fixture(scope="session")
def raw_corpus() -> List[str]:
    corpus = [
        "A Sentence with some spans that might be in common with other ones.",
        "Another sentence With some exiting spans!",
        "Also, and why not? A doc with a url http://super.truc.ad and some numbers 145 58.69!",
    ]
    return corpus


@pytest.fixture(scope="session")
def corpus(raw_corpus, en_sm_spacy_model) -> List[spacy.tokens.Doc]:
    docs = [doc for doc in en_sm_spacy_model.pipe(raw_corpus)]
    return docs


@pytest.fixture(scope="function")
def pipeline(en_sm_spacy_model, raw_corpus) -> Pipeline:
    custom_pipeline = Pipeline(
        spacy_model=en_sm_spacy_model,
        corpus=[doc for doc in en_sm_spacy_model.pipe(raw_corpus)],
    )
    return custom_pipeline


@pytest.fixture(scope="session")
def ct_string_map() -> Dict[str, Set[str]]:
    ct_strings_map = {
        "a Sentence with": {"a Sentence with"},
        "sentence": {"sentence"},
        "common with": {"common with"},
        "that might be in": {"that might be in"},
    }
    return ct_strings_map


class TestManualCandidateTermWithPostProcessing:
    @pytest.fixture(scope="class")
    def manual_ct_extract(self, ct_string_map) -> ManualCandidateTermExtraction:
        params = {"ct_label_strings_map": ct_string_map}
        ct_extract = ManualCandidateTermExtraction(parameters=params)
        return ct_extract

    def test_no_post_process(self, manual_ct_extract, pipeline, ct_string_map) -> None:
        manual_ct_extract.run(pipeline)

        assert len(pipeline.candidate_terms) == len(ct_string_map)

    def test_bad_post_process(self, manual_ct_extract, pipeline) -> None:
        post_processing_functions = [
            partial(filter_cts_on_first_token_in_term, filtering_tokens={"a", "that"}),
            "not a callable",
        ]
        params = {"ct_label_strings_map": ct_string_map}

        with pytest.raises(NotCallableError):
            ct_extract = ManualCandidateTermExtraction(
                cts_post_processing_functions=post_processing_functions,
                parameters=params,
            )

    def test_ct_post_process_on_first_token(self, manual_ct_extract, pipeline) -> None:
        manual_ct_extract.cts_post_processing_functions = [
            partial(filter_cts_on_first_token_in_term, filtering_tokens={"a", "that"})
        ]

        manual_ct_extract.run(pipeline)

        ct_labels = {ct.label for ct in pipeline.candidate_terms}
        assert ct_labels == {"sentence", "common with"}

    def test_ct_post_process_on_last_token(self, manual_ct_extract, pipeline) -> None:
        manual_ct_extract.cts_post_processing_functions = [
            partial(filter_cts_on_last_token_in_term, filtering_tokens={"in", "with"})
        ]

        manual_ct_extract.run(pipeline)

        ct_labels = {ct.label for ct in pipeline.candidate_terms}
        assert ct_labels == {"sentence"}

    def test_ct_post_process_on_token_in(self, manual_ct_extract, pipeline) -> None:
        manual_ct_extract.cts_post_processing_functions = [
            partial(filter_cts_on_token_in_term, filtering_tokens={"might", "a"})
        ]

        manual_ct_extract.run(pipeline)

        ct_labels = {ct.label for ct in pipeline.candidate_terms}
        assert ct_labels == {"sentence", "common with"}

    def test_ct_post_process_multiple(self, manual_ct_extract, pipeline) -> None:
        manual_ct_extract.cts_post_processing_functions = [
            partial(filter_cts_on_first_token_in_term, filtering_tokens={"a"}),
            partial(filter_cts_on_last_token_in_term, filtering_tokens={"with"}),
        ]

        manual_ct_extract.run(pipeline)

        ct_labels = {ct.label for ct in pipeline.candidate_terms}
        assert ct_labels == {"sentence", "that might be in"}

    def test_ct_post_process_split(
        self, manual_ct_extract, pipeline, en_sm_spacy_model, corpus
    ) -> None:
        manual_ct_extract.cts_post_processing_functions = [
            partial(
                split_cts_on_token,
                splitting_tokens={"might"},
                spacy_model=en_sm_spacy_model,
                docs=corpus,
            ),
        ]

        manual_ct_extract.run(pipeline)

        ct_labels = {ct.label for ct in pipeline.candidate_terms}
        assert ct_labels == {
            "a Sentence with",
            "sentence",
            "common with",
            "that",
            "be in",
        }

    def test_ct_post_process_split_filter(
        self, manual_ct_extract, pipeline, en_sm_spacy_model, corpus
    ) -> None:
        manual_ct_extract.cts_post_processing_functions = [
            partial(
                split_cts_on_token,
                splitting_tokens={"might"},
                spacy_model=en_sm_spacy_model,
                docs=corpus,
            ),
            partial(filter_cts_on_last_token_in_term, filtering_tokens={"with"}),
        ]

        manual_ct_extract.run(pipeline)

        ct_labels = {ct.label for ct in pipeline.candidate_terms}
        assert ct_labels == {
            "sentence",
            "that",
            "be in",
        }
