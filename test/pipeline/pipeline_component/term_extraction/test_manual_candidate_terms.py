from typing import Dict, List, Set

import pytest
from spacy.matcher import PhraseMatcher

from olaf.commons.errors import ParameterError
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
def pipeline(en_sm_spacy_model, raw_corpus) -> Pipeline:
    custom_pipeline = Pipeline(
        spacy_model=en_sm_spacy_model,
        corpus=[doc for doc in en_sm_spacy_model.pipe(raw_corpus)],
    )
    return custom_pipeline


@pytest.fixture(scope="session")
def ct_string_map() -> Dict[str, Set[str]]:
    ct_strings_map = {"with": {"with", "and"}, "sentence": {"sentence"}}
    return ct_strings_map


class TestManualCandidateTermExtractionDefault:
    @pytest.fixture(scope="class")
    def default_manual_ct_extract(self, ct_string_map) -> ManualCandidateTermExtraction:
        params = {"ct_label_strings_map": ct_string_map}
        ct_extract = ManualCandidateTermExtraction(parameters=params)
        return ct_extract

    def test_default_params(self, default_manual_ct_extract) -> None:
        assert default_manual_ct_extract.phrase_matcher is None
        assert default_manual_ct_extract.ct_label_strings_map is not None

    def test_build_matcher(self, default_manual_ct_extract, en_sm_spacy_model) -> None:
        matcher = default_manual_ct_extract._build_matcher(en_sm_spacy_model)

        assert isinstance(matcher, PhraseMatcher)

    def test_run(self, default_manual_ct_extract, pipeline) -> None:
        pipeline.candidate_terms = set()

        default_manual_ct_extract.run(pipeline)

        assert len(pipeline.candidate_terms) == 2

        ct_index = {ct.label: ct for ct in pipeline.candidate_terms}

        assert len(ct_index["with"].corpus_occurrences) == 6
        assert len(ct_index["sentence"].corpus_occurrences) == 2


class TestManualCandidateTermExtractionCustomMatcher:
    @pytest.fixture(scope="class")
    def custom_matcher(self, en_sm_spacy_model, ct_string_map) -> PhraseMatcher:
        matcher = PhraseMatcher(en_sm_spacy_model.vocab)

        for label, match_strings in ct_string_map.items():
            matcher.add(label, [en_sm_spacy_model(string) for string in match_strings])

        return matcher

    @pytest.fixture(scope="class")
    def manual_ct_extract(self, custom_matcher) -> ManualCandidateTermExtraction:
        params = {"custom_matcher": custom_matcher}
        ct_extract = ManualCandidateTermExtraction(parameters=params)
        return ct_extract

    def test_params(self, manual_ct_extract, custom_matcher) -> None:
        assert manual_ct_extract.phrase_matcher == custom_matcher
        assert manual_ct_extract.ct_label_strings_map is None

    def test_run(self, manual_ct_extract, pipeline) -> None:
        pipeline.candidate_terms = set()

        manual_ct_extract.run(pipeline)

        assert len(pipeline.candidate_terms) == 2

        ct_index = {ct.label: ct for ct in pipeline.candidate_terms}

        assert len(ct_index["with"].corpus_occurrences) == 5
        assert len(ct_index["sentence"].corpus_occurrences) == 1


def test_wrong_init() -> None:
    with pytest.raises(ParameterError):
        ct_extract = ManualCandidateTermExtraction()
