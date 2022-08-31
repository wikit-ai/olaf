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
       The C-values are computed based on <https://doi.org/10.1007/s007999900023>.

       Notes:
         - Potential pitfall: when extracting terms we extract the span texts (required to get span frequences). 
           In the rest of the process we "retokenize" the spans by splitting the span text on spaces.
    """

    def __init__(self, tokenSequences: Iterable[spacy.tokens.span.Span], max_size_gram: int) -> None:
        """The Cvalue class only requires a list of text sequences (Spacy Span objects) and a maximum size of ngrams.
            The C-value will be computed for all ngrams with size ranging from 1 to max_size_gram.

            Note:
              - The maximum length of a token sequence should be equal to max_size_gram. 
                If not, we will add ngrams of max_size_gram extracted from the longer sequence of tokens.

        Parameters
        ----------
        tokenSequences : Iterable[spacy.tokens.span.Span]
            The token sequences to compute the C-value from
        max_size_gram : int
            The maximum size for ngrams to consider.
        """
        self.tokenSequences = tokenSequences
        self.max_size_gram = max_size_gram
        self.candidateTerms = None
        self.candidateTermsCounter = None
        self.c_values = None

        self._extract_candidate_terms()
        self._compute_c_values()

    def __call__(self) -> List[Tuple[float, str]]:
        return self.c_values

    def _order_count_candidate_terms(self, candidate_terms_by_size: Dict[int, str]) -> Tuple[List[str], Counter]:
        """Order the candidate terms by their size (number of tokens) and their frequence in the corpus.
           The algorithm relies on this order.
           From <https://doi.org/10.1007/s007999900023>: 
                The lists are then filtered through the stop-list and are concatenated. The longest 
                strings appear at the top, and decrease in size as we move down, with the bigrams 
                being at the bottom. The strings of each length are ordered by their frequency of occurrence.

        Parameters
        ----------
        candidate_terms_by_size : Dict[int, str]
            A Dictionnary with the term size as key and the list of all terms of the latter size (with all duplicates)

        Returns
        -------
        Tuple[List[str], Counter]
            A list of candidate terms without duplicates
            The Counter containing the associated term frequencies 
        """
        all_candidate_terms = []  # need all the terms to compute the frequences

        for terms_size, terms in candidate_terms_by_size.items():
            all_candidate_terms.extend(terms)
            candidate_terms_by_size[terms_size] = list(
                set(terms))  # we do not need duplicates anymore

        candidateTermsCounter = Counter(all_candidate_terms)

        # each group of candidate terms needs to be ordered by the frequence
        for terms_size in candidate_terms_by_size.keys():
            candidate_terms_by_size[terms_size].sort(
                key=lambda term: candidateTermsCounter[term], reverse=True)

        candidateTerms = []
        # groups of candidate terms are concatenated from the the longest to the smallest
        for terms_size in range(1, self.max_size_gram + 1).__reversed__():
            candidateTerms.extend(candidate_terms_by_size[terms_size])

        return candidateTerms, candidateTermsCounter

    def _update_token_sequences(self) -> List[str]:
        """Check that no sequence of tokens is longer than the max_size_gram.
           If a sequence of tokens is longer than the max_size_gram we generate a grams of 
           size max_size_gram and add them to the list of token sequences.
           All the token sequences are space concatenated into a string.

        Returns
        -------
        List[str]
            token sequences are space concatenated into a string.
        """
        allTokenSeqStrings = []

        for span in self.tokenSequences:
            if len(span) <= self.max_size_gram:
                allTokenSeqStrings.append(span.text)
            else:
                # token sequence longer than max_size_gram, generate the ngrams
                tokens = span.text.split()
                allTokenSeqStrings.extend(
                    [" ".join(gram) for gram in ngrams(tokens, self.max_size_gram)])

        return allTokenSeqStrings

    def _extract_candidate_terms(self) -> None:
        """Extract the valid list of candidate terms and compute the corresponding frequences.
            This method sets the attributes:
              - self.candidateTerms
              - self.candidateTermsCounter
        """
        tokenSeqCounter = Counter(self._update_token_sequences())
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
        """Extract the list of substrings (string of token sequence) contained in a term.
           Substrings are extracted from unigram to term size-gram.

        Parameters
        ----------
        term : str
            The term to extract the substrings from.

        Returns
        -------
        List[str]
            The list of substrings
        """
        tokens = term.split()
        token_len = len(tokens)

        substrings = set()
        for i in range(1, token_len):
            # we need ngrams, i.e., all overlapping substrings
            for gram in ngrams(tokens, i):
                substrings.add(" ".join(gram))

        return list(substrings)

    def _update_stat_triple(self, substring: str, stat_triples: Dict[str, int], parent_term: str, term_frequences: Counter) -> None:
        """Update the stat triples table accroding to the algorithm in <https://doi.org/10.1007/s007999900023> (section 2.3, page 4).

        Parameters
        ----------
        substring : str
            The substring considered
        stat_triples : Dict[str, int]
            The set of stat triple. The keys are the substrings
        parent_term : str
            The term the substring has been extracted from.
        term_frequences : Counter
            The terms frequences
        """
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
        """Extract the substrings of candidate term and loop over them to update the stat triples

        Parameters
        ----------
        candidate_term : str
            The candidate term to process
        stat_triples : Dict[str, int]
            The set of stat triple. The keys are the substrings
        term_frequences : Counter
            The candidate terms frequences
        """
        substrings = self._get_substrings(candidate_term)
        for substring in substrings:
            self._update_stat_triple(
                substring, stat_triples, candidate_term, term_frequences)

    def _compute_c_values(self) -> None:
        """Compute the C-value following the algorithm in <https://doi.org/10.1007/s007999900023> (section 2.3, page 4).

           This method sets the attribute:
             - self.c_values
        """
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

        # reorder the c-values so we have the terms with the highest c-values at the top.
        c_values.sort(key=lambda c_val: c_val[0], reverse=True)

        self.c_values = c_values
