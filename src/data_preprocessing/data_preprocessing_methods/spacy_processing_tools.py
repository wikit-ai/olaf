from typing import List

from nltk.util import ngrams as nltk_ngrams
import spacy.tokens


def spacy_span_ngrams(span: spacy.tokens.span.Span, gram_size: int) -> List[spacy.tokens.span.Span]:
    """Adapt the NTLK ngrams function to work with Spacy Span objects.

    Parameters
    ----------
    span : spacy.tokens.span.Span
        The spacy Span object to extract the ngrams from.
    gram_size : int
        The gram size

    Returns
    -------
    List[spacy.tokens.span.Span]
        The list of ngrams as Spacy Span objects
    """
    grams = nltk_ngrams(span, gram_size)
    doc = span.doc
    gram_spans = [spacy.tokens.span.Span(
        doc, gram[0].i, gram[-1].i + 1) for gram in grams]

    return gram_spans


def build_spans_from_tokens(token_list: List[spacy.tokens.Token], doc: spacy.tokens.doc.Doc) -> List[spacy.tokens.span.Span]:
    """Go through a list of Spacy tokens and extract the Spans in it.
        Essentialy look for tokens following each other in the Doc and build Spans from them.

    Parameters
    ----------
    token_list : List[spacy.tokens.Token]
        The list of Tokens to extract the Spans from
    doc : spacy.tokens.doc.Doc
        The Spacy Doc the Tokens are from

    Returns
    -------
    List[spacy.tokens.span.Span]
        The list of Spans extracted
    """
    spans = []

    if len(token_list) > 0:  # we can not extract spans from an empty list of tokens
        start_span_token_idx = token_list[0].i
        previous_token_idx = token_list[0].i

        for token in token_list:
            if token.i > previous_token_idx + 1:
                spans.append(doc[start_span_token_idx:previous_token_idx+1])
                start_span_token_idx = token.i
                previous_token_idx = token.i
            previous_token_idx = token.i
        spans.append(doc[start_span_token_idx:previous_token_idx+1])

    return spans
