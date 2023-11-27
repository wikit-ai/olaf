from typing import List

import spacy.matcher
import spacy.tokens
from nltk.util import ngrams as nltk_ngrams

from .logging_config import logger


def spacy_span_ngrams(
    span: spacy.tokens.Span, gram_size: int
) -> List[spacy.tokens.Span]:
    """Adapt the NTLK ngrams function to work with spaCy Span objects.

    Parameters
    ----------
    span : spacy.tokens.span.Span
        The spaCy Span object to extract the ngrams from.
    gram_size : int
        The gram size.

    Returns
    -------
    List[spacy.tokens.span.Span]
        The list of ngrams as spaCy Span objects.
    """
    if len(span) >= gram_size:
        try:
            grams = nltk_ngrams(span, gram_size)
        except Exception as e:
            logger.error(
                "There has been an issue while computing %i-grams for span %s  using nltk.util.ngrams function. Trace : %s.",
                gram_size,
                span.text,
                e,
            )
        doc = span.doc
        gram_spans = [doc[gram[0].i : gram[-1].i + 1] for gram in grams]
    else:
        gram_spans = [span]

    logger.info("%i-grams extracted for span %s.", gram_size, span.text)

    return gram_spans


def is_not_stopword(token: spacy.tokens.Token) -> bool:
    """Return True if the Spacy Token is NOT a stopword.

    Parameters
    ----------
    token : spacy.tokens.Token
        The Spacy token to test.

    Returns
    -------
    bool
        Whether the Token Shape is NOT a stopword or it is.
    """
    keep_token = not token.is_stop
    return keep_token


def is_not_punct(token: spacy.tokens.Token) -> bool:
    """Return True if the Spacy Token is NOT a punctuation symbol.

    Parameters
    ----------
    token : spacy.tokens.Token
        The Spacy token to test.

    Returns
    -------
    bool
        Whether the Token Shape is NOT a punctuation symbol or it is.
    """
    keep_token = not token.is_punct
    return keep_token


def is_not_num(token: spacy.tokens.Token) -> bool:
    """Return True if the Spacy Token is NOT a numerical value.

    Parameters
    ----------
    token : spacy.tokens.Token
        The Spacy token to test.

    Returns
    -------
    bool
        Whether the Token Shape is NOT a numerical value or it is.
    """
    keep_token = not token.like_num
    return keep_token


def is_not_url(token: spacy.tokens.Token) -> bool:
    """Return True if the Spacy Token is NOT a url.

    Parameters
    ----------
    token : spacy.tokens.Token
        The Spacy token to test.

    Returns
    -------
    bool
        Whether the Token Shape is NOT a url or it is.
    """
    keep_token = not token.like_url
    return keep_token


def select_on_pos(token: spacy.tokens.Token, pos_to_select: List[str]) -> bool:
    """Return true if the Spacy Token POS string is in the pos_to_select list.

    Parameters
    ----------
    token : spacy.tokens.Token
        The Spacy token to test
    pos_to_select : List[str]
        The list of strings corresponding to the POS tags to keep.

    Returns
    -------
    bool
        Whether the Token POS tag is in pos_to_select or not
    """
    keep_token = token.pos_ in pos_to_select
    return keep_token


def spans_overlap(span1: spacy.tokens.Span, span2: spacy.tokens.Span) -> bool:
    """Return true is the spans are overlapping, else False.

    Parameters
    ----------
    span1 : spacy.tokens.Span
        The first spaCy span.
    span2 : spacy.tokens.Span
        The second spaCy span.

    Returns
    -------
    bool
        Whether or not the spans are overlapping.
    """
    spans_are_overlapping = False

    if span1.doc == span2.doc:
        overlap_conditions = [
            span1.end == span2.end,
            span1.start == span2.start,
            (span1.end > span2.start) and (span1.end < span2.end),
            (span1.start > span2.start) and (span1.start < span2.end),
            (span1.start < span2.start) and (span1.end > span2.end),
            (span1.start > span2.start) and (span1.end < span2.end),
        ]

        spans_are_overlapping = any(overlap_conditions)

    return spans_are_overlapping
