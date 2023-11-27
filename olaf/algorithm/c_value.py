from collections import Counter, defaultdict
from typing import List, Optional, Set, Tuple
import math

from nltk.util import ngrams as nltk_ngrams

from ..commons.logging_config import logger


class Cvalue:
    """A class to extract terms form a list of strings and compute the C-values of those.
       The C-values are computed based on <https://doi.org/10.1007/s007999900023>.

    Attributes
    ----------
    corpus_terms: List[str]
        The list of strings extracted from the corpus to extract the terms from.
        The strings should be space-tokenised.
    max_term_token_length: int
        The maximum number of tokens a term can have.
    stop_list: Set[str]
        A set of stop words that should not appear in a term.
    _terms_string_tokens: Tuple[Tuple[str]]
        Tuple of terms string tokens to compute the C-values with.
    c_value_threshold: float
        A threshold to decide wether or not a term should be added to the candidate terms.
    candidate_terms: Tuple[str]
        The tuple of selected candidate terms.
    _terms_counter: Counter
        A mapping of terms to their occurrences in the corpus.
    _term_stat_triples: Dict[str, Tuple[int, int, int]]
        Tuple of term occurrences values used for computing C-values.
    c_values: Tuple[Tuple[float, str]]
        An ordered tuple of candidate terms with their C-values.
    """

    def __init__(self,
                 corpus_terms: List[str],
                 max_term_token_length: Optional[int] = None,
                 stop_list: Optional[Set[str]] = set(),
                 # value to act as if there was no threshold
                 c_value_threshold: Optional[float] = 0.0
                 ) -> None:
        """Initialise Cvalue instance.

        Parameters
        ----------
        corpus_terms: List[str]
            The list of strings extracted from the corpus to extract the terms from.
            The strings should be space-tokenised.
        max_term_token_length : Optional[int]
            The maximum number of tokens a term can have, by default None.
            If not provided, the default value will be set to the maximum token length of candidate terms.
        stop_list : Optional[Set[str]]
            A set of stop words that should not appear in a term, by default set().
        c_value_threshold : Optional[float]
            A threshold to decide wether or not a term should be added to the candidate terms, by default 0.0.
            The default value is set to 0.0 if there is no threshold.
        """
        self.corpus_terms = corpus_terms

        if max_term_token_length is None:
            self._max_term_token_length = max_term_token_length
        else:
            self.max_term_token_length = max_term_token_length

        self.stop_list = stop_list
        self.c_value_threshold = c_value_threshold

        self._candidate_terms = None
        self._c_values = None

        self._term_stat_triples = dict()

        self._terms_string_tokens = None
        self._terms_counter = None
        self._extract_and_count_terms_string_tokens()
        self._order_terms_string_tokens()

    @property
    def max_term_token_length(self) -> int:
        """Getter for the max token length of terms.

        Returns
        -------
        int
            The terms max token length.
        """
        return self._max_term_token_length

    @max_term_token_length.setter
    def max_term_token_length(self, value: int) -> None:
        """Setter for the max token length of terms.
            The value should be an int greater than 1.

        Parameters
        ----------
        value : int
            The value for the terms max token length.

        Raises
        ------
        AttributeError
            An error raised in case the value to set is lower than 2.
        """
        if value > 1:
            self._max_term_token_length = value
        else:
            raise AttributeError(
                "Attribute max_term_token_length should be an int greater than 1.")

    @property
    def candidate_terms(self) -> Tuple[str]:
        """Getter for the candidate_terms attribute.
            Should log a warning in case the user forgot to run the C-value computation.

        Returns
        -------
        Tuple[str]
            The tuple of candidate terms.
        """
        if not self._candidate_terms:
            logger.warn(
                "No candidate terms found. Have you run the C-value computation?")
        else:
            return self._candidate_terms

    @property
    def c_values(self) -> Tuple[Tuple[float, str]]:
        """Getter for the c_values attribute.
            Should log a warning in case the user forgot to run the C-value computation.

        Returns
        -------
        Tuple[Tuple[float, str]]
            The tuple of terms and their C-values.
        """
        if not self._c_values:
            logger.warn(
                "No C-values found. Have you run the C-value computation?")
        else:
            return self._c_values

    def _extract_and_count_terms_string_tokens(self) -> None:
        """Extract terms string tokens from the strings provided as input and count their occurrences.

            The method sets the following attributes:
            - self._terms_counter
            - self._terms_string_tokens
        """
        all_terms_string_tokens = []

        # provide a maximum token length in case none have been given.
        # arbitrary value to act as if there were no maximum token length limit.
        max_term_token_length = self._max_term_token_length if self._max_term_token_length else 100

        for term in self.corpus_terms:
            term_tokens = term.split()
            if len(term_tokens) > 1:

                # avoid adding terms longer than the max token length
                if len(term_tokens) <= max_term_token_length:
                    all_terms_string_tokens.append(tuple(term_tokens))

                for i in range(2, min(max_term_token_length, len(term_tokens))):
                    i_length_token_seqs = nltk_ngrams(term_tokens, i)
                    terms_tokens = [tuple(tokens) for tokens in i_length_token_seqs if not set(
                        tokens).intersection(self.stop_list)]
                    all_terms_string_tokens.extend(terms_tokens)

        self._terms_counter = Counter(all_terms_string_tokens)
        self._terms_string_tokens = list(self._terms_counter.keys())

    def _order_terms_string_tokens(self) -> None:
        """Order the terms string tokens by token length and occurrence as it should 
            be for the C-value computation algorithm.

            The method updates the following attributes:
            - self._terms_string_tokens
            - self._max_term_token_length
        """
        terms_string_tokens_by_size = defaultdict(list)
        for term_tokens in self._terms_string_tokens:
            terms_string_tokens_by_size[len(term_tokens)].append(term_tokens)

        # set the max_term_token_length attribute if not provided.
        if not self._max_term_token_length:
            self._max_term_token_length = max(
                terms_string_tokens_by_size.keys())

        ordered_terms_string_tokens = []
        for _, terms_tokens in sorted(terms_string_tokens_by_size.items(), reverse=True):
            terms_tokens.sort(
                key=lambda term: self._terms_counter[term], reverse=True)
            ordered_terms_string_tokens.extend(terms_tokens)

        self._terms_string_tokens = tuple(ordered_terms_string_tokens)

    def _extract_term_substrings_tokens(self, term_string_tokens: Tuple[str]) -> Tuple[Tuple[str]]:
        """Extract term substrings tokens, i.e., the n-grams for n=[2:term token length - 1].

        Parameters
        ----------
        term_string : Tuple[str]
            The term string tokens to extract the substrings tokens from.

        Returns
        -------
        Tuple[Tuple[str]]
            The tuple of substrings tokens ordered by descending token length.
        """

        term_substrings_tokens = []

        for i in range(2, len(term_string_tokens)):
            i_length_token_seqs = nltk_ngrams(term_string_tokens, i)
            term_substrings_tokens.extend([tuple(tokens)
                                           for tokens in i_length_token_seqs])

        term_substrings_tokens.sort(key=lambda e: len(e), reverse=True)

        return tuple(term_substrings_tokens)

    def _update_term_stat_triples(self, term_string_tokens: Tuple[str]) -> None:
        """Update the triples used to compute C-values following the original paper algorithm.
            The triple is defined as "(f(b), t(b), c(b)), where f(b) is the total frequency of b in
            the corpus, t(b) is the frequency of b as a nested string of candidate terms, c(b) is 
            the number of these longer candidate terms" (citation from the original paper in which 
            "frequency" is meant as "occurrences"). Where a is the candidate term string and b is a 
            substring of a.

        The method updates the following attribute:
        - self._term_stat_triples

        Parameters
        ----------
        term_string : Tuple[str]
            The term string tokens based on which to update the triple.
        """
        substrings_tokens = self._extract_term_substrings_tokens(
            term_string_tokens)
        for substring_tokens in substrings_tokens:
            if not self._term_stat_triples.get(substring_tokens):
                term_stat_triple = [
                    self._terms_counter[substring_tokens],
                    self._terms_counter[term_string_tokens],
                    1
                ]
                self._term_stat_triples[substring_tokens] = term_stat_triple
            else:
                if self._term_stat_triples.get(term_string_tokens):
                    n_term_string_as_nested = self._term_stat_triples[term_string_tokens][2]
                else:
                    n_term_string_as_nested = 0

                self._term_stat_triples[substring_tokens][1] += (
                    self._terms_counter[term_string_tokens] - n_term_string_as_nested)
                self._term_stat_triples[substring_tokens][2] += 1

    def compute_c_values(self) -> None:
        """Compute the C-value scores.

            The method sets the following attributes:
            - self._c_values
            - self._candidate_terms
        """

        c_values = []

        for term_string_tokens in self._terms_string_tokens:
            candidate_term = " ".join(term_string_tokens)
            term_token_length = len(term_string_tokens)

            if term_token_length == self._max_term_token_length:
                c_val = math.log2(term_token_length) * \
                    self._terms_counter[term_string_tokens]

                if c_val >= self.c_value_threshold:
                    c_values.append((c_val, candidate_term))
                    self._update_term_stat_triples(term_string_tokens)

            else:
                if term_string_tokens not in self._term_stat_triples.keys():
                    c_val = math.log2(term_token_length) * \
                        self._terms_counter[term_string_tokens]
                else:
                    c_val = math.log2(term_token_length) * (self._terms_counter[term_string_tokens] -
                                                            (self._term_stat_triples[term_string_tokens][1] /
                                                             self._term_stat_triples[term_string_tokens][2])
                                                            )
                if c_val >= self.c_value_threshold:
                    c_values.append((c_val, candidate_term))
                    self._update_term_stat_triples(term_string_tokens)

        self._c_values = tuple(
            sorted(c_values, key=lambda e: e[0], reverse=True))
        self._candidate_terms = tuple([c_val_tuple[1]
                                       for c_val_tuple in self._c_values])
