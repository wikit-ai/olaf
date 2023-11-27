from typing import List

import pytest
import spacy

from olaf.commons.spacy_processing_tools import (
    is_not_num,
    is_not_punct,
    is_not_stopword,
    is_not_url,
    select_on_pos,
    spacy_span_ngrams,
    spans_overlap,
)


@pytest.fixture(scope="session")
def raw_corpus() -> List[str]:
    corpus = [
        "A sentence with some spans that might be in common with other ones.",
        "Another sentence with some exiting spans!",
        "Also, and why not? A doc with a url http://super.truc.ad and some numbers 145 58.69!",
    ]
    return corpus


@pytest.fixture(scope="session")
def spacy_model() -> spacy.Language:
    spacy_nlp = spacy.load(
        "en_core_web_sm",
        exclude=["parser", "ner"],
    )
    return spacy_nlp


class TestTokenSelectors:
    def test_is_not_num(self, spacy_model, raw_corpus) -> None:
        doc = spacy_model(raw_corpus[2])

        assert is_not_num(doc[3])
        assert not is_not_num(doc[-2])
        assert not is_not_num(doc[-3])

    def test_is_not_punct(self, spacy_model, raw_corpus) -> None:
        doc = spacy_model(raw_corpus[2])

        assert is_not_punct(doc[0])
        assert not is_not_punct(doc[1])
        assert not is_not_punct(doc[-1])

    def test_is_not_stopword(self, spacy_model, raw_corpus) -> None:
        doc = spacy_model(raw_corpus[2])

        assert is_not_stopword(doc[1])
        assert not is_not_stopword(doc[0])
        assert not is_not_stopword(doc[3])

    def test_is_not_url(self, spacy_model, raw_corpus) -> None:
        doc = spacy_model(raw_corpus[2])

        assert is_not_url(doc[1])
        assert not is_not_url(doc[11])
        assert is_not_url(doc[3])

    def test_select_on_pos(self, spacy_model, raw_corpus) -> None:
        pos = ["NOUN", "DET"]
        doc = spacy_model(raw_corpus[2])

        assert not select_on_pos(doc[1], pos)
        assert select_on_pos(doc[6], pos)
        assert select_on_pos(doc[7], pos)


def test_spacy_span_ngrams(spacy_model, raw_corpus) -> None:
    doc = spacy_model(raw_corpus[1])

    trigrams = spacy_span_ngrams(doc[:5], 3)
    trigrams_texts = [span.text for span in trigrams]
    bigrams = spacy_span_ngrams(doc[:5], 2)
    bigrams_texts = [span.text for span in bigrams]
    too_big_gram = spacy_span_ngrams(doc[:5], 7)

    assert all([len(gram) == 3] for gram in trigrams)
    assert all(
        [
            span_text in trigrams_texts
            for span_text in [
                "Another sentence with",
                "sentence with some",
                "with some exiting",
            ]
        ]
    )
    assert all([len(gram) == 2] for gram in bigrams)
    assert all(
        [
            span_text in bigrams_texts
            for span_text in [
                "Another sentence",
                "sentence with",
                "with some",
                "some exiting",
            ]
        ]
    )
    assert len(too_big_gram[0]) == len(doc[:5])
    assert too_big_gram[0].text == doc[:5].text


def test_spans_overlap(spacy_model, raw_corpus) -> None:
    docs = [spacy_model(text) for text in raw_corpus]

    span1 = docs[0][:3]
    span2 = docs[0][2:4]
    span3 = docs[0][3:6]
    outer_span = docs[0][2:10]
    inner_span = docs[0][5:7]
    other_doc_span = docs[1][2:5]

    assert not spans_overlap(span1, other_doc_span)
    assert not spans_overlap(other_doc_span, span1)
    assert spans_overlap(span1, span2)
    assert spans_overlap(span2, span2)
    assert spans_overlap(span2, span3)
    assert spans_overlap(span3, span2)
    assert spans_overlap(outer_span, inner_span)
    assert spans_overlap(inner_span, outer_span)
    assert spans_overlap(span3, inner_span)
    assert spans_overlap(inner_span, span3)
    assert not spans_overlap(span1, inner_span)
    assert not spans_overlap(inner_span, span1)
