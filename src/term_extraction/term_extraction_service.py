from collections import Counter, defaultdict
import math
from typing import Any, Callable, Dict, Iterable, List, Tuple

import spacy.tokens
import spacy.tokenizer
import spacy.language
from nltk.util import ngrams

TokenSequenceFilter = Callable[[List[Any]], List[List[str]]]


class Cvalue:
    """A class to compute the C-value of each term (token sequence) in a corpus of texts.

       Notes:
         - Potential pitfall: when extracting terms we extract the span texts (required to get span frequences). 
           In the rest of the process we "retokenize" the spans by splitting the span text on spaces.
    """

    def __init__(self, tokenSequences: Iterable[spacy.tokens.span.Span], max_size_gram: int) -> None:
        self.tokenSequences = tokenSequences
        self.max_size_gram = max_size_gram
        self.candidateTerms = None
        self.candidateTermsCounter = None
        self.c_values = None

        self._extract_candidate_terms()
        self._compute_c_values()

    def __call__(self) -> List[Tuple[float, str]]:
        return self.c_values

    def _count_token_sequences(self) -> Counter:
        return Counter([span.text for span in self.tokenSequences])

    def _order_count_candidate_terms(self, candidate_terms_by_size: Dict[int, str]) -> Tuple[List[str], Counter]:

        all_candidate_terms = []

        for terms_size, terms in candidate_terms_by_size.items():
            all_candidate_terms.extend(terms)
            candidate_terms_by_size[terms_size] = list(set(terms))

        candidateTermsCounter = Counter(all_candidate_terms)

        # each group of ngram needs to be ordered by the frequence
        for terms_size in candidate_terms_by_size.keys():
            candidate_terms_by_size[terms_size].sort(
                key=lambda term: candidateTermsCounter[term], reverse=True)

        candidateTerms = []
        for terms_size in range(1, self.max_size_gram + 1).__reversed__():
            candidateTerms.extend(candidate_terms_by_size[terms_size])

        return candidateTerms, candidateTermsCounter

    def _extract_candidate_terms(self) -> None:

        tokenSeqCounter = self._count_token_sequences()
        tokenSeqStrings = tokenSeqCounter.keys()

        candidate_terms_by_size = defaultdict(list)

        for size in range(1, self.max_size_gram + 1):
            for tokenSeqStr in tokenSeqStrings:
                tokens = tokenSeqStr.split()
                candidate_terms_by_size[size].extend(
                    [" ".join(gram) for gram in ngrams(tokens, size)] * tokenSeqCounter[tokenSeqStr])

        self.candidateTerms, self.candidateTermsCounter = self._order_count_candidate_terms(
            candidate_terms_by_size)

    def _get_substrings(self, term: str) -> List[str]:
        tokens = term.split()
        token_len = len(tokens)

        substrings = set()
        for i in range(1, token_len):
            for gram in ngrams(tokens, i):
                substrings.add(" ".join(gram))

        return list(substrings)

    def _update_stat_triple(self, substring: str, stat_triples: Dict[str, int], parent_term: str, term_frequences: Counter) -> None:

        if substring in stat_triples.keys():

            if parent_term in stat_triples.keys():
                stat_triples[substring][1] = stat_triples[substring][1] + \
                    (term_frequences[parent_term] -
                     stat_triples[parent_term][1])
            else:
                stat_triples[substring][1] = stat_triples[substring][1] + \
                    term_frequences[parent_term]

            stat_triples[substring][2] += 1

        else:
            f_string = 0 if term_frequences.get(
                substring) is None else term_frequences[substring]
            stat_triples[substring] = [
                f_string, term_frequences[parent_term], 1]

    def _process_substrings(self, candidate_term: str, stat_triples: Dict[str, int], term_frequences: Counter) -> None:
        substrings = self._get_substrings(candidate_term)
        for substring in substrings:
            self._update_stat_triple(
                substring, stat_triples, candidate_term, term_frequences)

    def _compute_c_values(self) -> None:

        c_values = []
        stat_triples = dict()

        for candidate_term in self.candidateTerms:

            len_candidate_term = len(candidate_term.split())

            if len_candidate_term == self.max_size_gram:
                c_val = math.log2(len_candidate_term) * \
                    self.candidateTermsCounter[candidate_term]
                c_values.append((c_val, candidate_term))

                self._process_substrings(
                    candidate_term, stat_triples, self.candidateTermsCounter)

            else:
                if candidate_term not in stat_triples.keys():
                    c_val = math.log2(len_candidate_term) * \
                        self.candidateTermsCounter[candidate_term]
                    c_values.append((c_val, candidate_term))
                else:
                    c_val = math.log2(
                        len_candidate_term) * (self.candidateTermsCounter[candidate_term] - (stat_triples[candidate_term][1] / stat_triples[candidate_term][2]))
                    c_values.append((c_val, candidate_term))

                self._process_substrings(
                    candidate_term, stat_triples, self.candidateTermsCounter)

        c_values.sort(key=lambda c_val: c_val[0], reverse=True)
        self.c_values = c_values
