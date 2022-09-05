import re

from typing import Iterable, List

import spacy.tokens.doc
import spacy.tokenizer
import spacy.tokens.span
import spacy.language

from nltk.util import ngrams as nltk_ngrams


def extract_text_sequences_from_corpus(docs: Iterable[spacy.tokens.doc.Doc]) -> List[spacy.tokens.span.Span]:
    """Extract all the alphanumeric sequences of tokens from a corpus of texts.
        In the list returned, all occurences of grams are returned. There is no filtering or duplicates removal.
    Parameters
    ----------
    docs : Iterable[spacy.tokens.doc.Doc]
        An iterable over the Spacy documents.

    Returns
    -------
    List[spacy.tokens.span.Span]
        The list of token sequences (Span) contained in the corpus.
    """
    some_num_pattern = re.compile(r'''^[xX]+-?[xX]*$''')
    str_token_sequences = []

    for doc in docs:
        str_token_seq = []

        for token in doc:

            # we rely on the token.shape_ attribute to check that the token contains only letters and dashes
            if (some_num_pattern.match(token.shape_)):
                str_token_seq.append(token)

            elif len(str_token_seq) > 0:
                str_token_sequences.append(
                    spacy.tokens.span.Span(doc, str_token_seq[0].i, str_token_seq[-1].i + 1))
                str_token_seq = []

        if len(str_token_seq) > 0:
            str_token_sequences.append(spacy.tokens.span.Span(
                doc, str_token_seq[0].i, str_token_seq[-1].i + 1))

    return str_token_sequences


def spacy_span_ngrams(span: spacy.tokens.span.Span, gram_size: int) -> List[spacy.tokens.span.Span]:
    grams = nltk_ngrams(span, gram_size)
    doc = span.doc
    gram_spans = [spacy.tokens.span.Span(
        doc, gram[0].i, gram[-1].i + 1) for gram in grams]
    return gram_spans
