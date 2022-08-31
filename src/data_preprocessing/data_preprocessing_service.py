import re

from typing import Iterable, List
import spacy.tokens.doc
import spacy.tokenizer
import spacy.tokens.span
import spacy.language


def c_value_tokenizer(nlp_model: spacy.language.Language) -> spacy.tokenizer.Tokenizer:
    """Build a tokenizer based on Spacy Language model specifically to be able to extract ngrams for the C-value computation.
        In particular, the tokenizer does not split on dashes ('-') in words.

    Parameters
    ----------
    nlp : spacy.language.Language
        The Spacy Language model the tokenizer will be set on.

    Returns
    -------
    spacy.tokenizer.Tokenizer
        The tokenizer
    """

    special_cases = {}
    prefix_re = re.compile(r'''^[\[\("'!#$%&\\\*+,\-./:;<=>?@\^_`\{|~]''')
    suffix_re = re.compile(r'''[\]\)"'!#$%&\\\*+,\-./:;<=>?@\^_`\}|~]$''')
    infix_re = re.compile(r'''(?<=[0-9])[+\-\*^](?=[0-9-])''')
    simple_url_re = re.compile(r'''^https?://''')

    return spacy.tokenizer.Tokenizer(nlp_model.vocab, rules=special_cases,
                                     prefix_search=prefix_re.search,
                                     suffix_search=suffix_re.search,
                                     infix_finditer=infix_re.finditer,
                                     url_match=simple_url_re.match)


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
