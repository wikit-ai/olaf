from typing import Any, Callable, Iterable, List, Tuple
from pprint import pprint as pp
import re

from spacy.tokens import span
from spacy.tokens import doc
import spacy.tokenizer
import spacy.language

ListElementFilter = Callable[[List[Any]], List[Any]]


def c_value_tokenizer(nlp: spacy.language.Language) -> spacy.tokenizer.Tokenizer:
    """Build a tokenizer based on Spacy Language model specifically to be able to extract ngrams for the C-value computation.
        In particular, the tokenizer does not split on dashes ('_').

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

    return spacy.tokenizer.Tokenizer(nlp.vocab, rules=special_cases,
                                     prefix_search=prefix_re.search,
                                     suffix_search=suffix_re.search,
                                     infix_finditer=infix_re.finditer,
                                     url_match=simple_url_re.match)


def extract_text_sequences_from_corpus(docs: Iterable[doc.Doc]) -> List[List[str]]:

    some_num_pattern = re.compile(r'''^[xX]+-?[xX]*$''')

    for doc in docs:
        str_token_sequences = []
        str_token_seq = []
        for token in doc:
            if (some_num_pattern.fullmatch(token.shape_)):
                str_token_seq.append(token.lower_)
            elif len(str_token_seq) > 0:
                str_token_sequences.append(str_token_seq)
                str_token_seq = []
        if len(str_token_seq) > 0:
            str_token_sequences.append(str_token_seq)

    return str_token_sequences


def build_ngrams_from_corpus(corpus: List[span.Span], ngram_max_size: int) -> Tuple[str]:
    """Extract all ngrams from the texts in a corpus.
        Notes: 
          - In the list returned, all occurences of grams are returned. There is no filtering or duplicates removal.
          - The grams are joined into a space separated string

    Parameters
    ----------
    corpus : List[spacy.Doc]
        List of spacy Documents forming the corpus
    ngram_max_size : int
        The maximum size of grams to build. For example, if 3 is given, unigrams, bigrams, and trigrams will be constructed.

    Returns
    -------
    List[str]
        The list of ngrams ordered by their size in terms of tokens from the biggest to th smallest.
    """

    pass


if __name__ == "__main__":

    test_texts = [
        'Toothed lock washers - Type V, countersunk',
        'Taper pin - conicity 1/50',
        'T-head bolts with double nib',
        'Handwheels, DIN 950, case-iron, d2 small, without keyway, without handle, form B-F/A',
        'Dog point hexagon socket set screw',
        'Butterfly valve SV04 DIN BF, actuator PAMS93-size 1/2 NC + TOP',
        'Splined Shafts acc. to DIN 5463 / ISO 14',
        'Grooved pins - Half-length reverse-taper grooved',
        'Rod ends DIN ISO 12240-4 (DIN 648) E series stainless version with female thread, maintenance-free',
        'Palm Grips, DIN 6335, light metal, with smooth blind hole, form C, DIN 6335-AL-63-20-C-PL',
        'Hexagon socket set screws with dog point, DIN EN ISO 4028-M5x12 - 45H',
        'Rivet DIN 661  - Type A - 1,6 x 6',
        'Welding neck flange - PN 400 - DIN 2627 - NPS 150',
        'Step Blocks, DIN 6326, adjustable, with spiral gearing, upper part, DIN 6326-K',
        'Loose Slot Tenons, DIN 6323, form C, DIN 6323-20x28-C',
        'Hexagon nut DIN EN 24036 - M3.5 - St'
    ]

    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = c_value_tokenizer(nlp)

    for text in test_texts:
        doc = nlp(text)
        print(doc.text)
        print([t.text for t in doc])
        print([t.shape_ for t in doc])
        print(extract_text_sequences_from_corpus([doc]))
        print()
