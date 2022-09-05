import re
import spacy.language


def no_split_on_dash_in_words_sapcy_tokenizer(nlp_model: spacy.language.Language) -> spacy.tokenizer.Tokenizer:
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

    tokenizer = spacy.tokenizer.Tokenizer(nlp_model.vocab, rules=special_cases,
                                          prefix_search=prefix_re.search,
                                          suffix_search=suffix_re.search,
                                          infix_finditer=infix_re.finditer,
                                          url_match=simple_url_re.match)

    return tokenizer
