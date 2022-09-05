from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Tuple

import spacy.tokens
import spacy.tokenizer
import spacy.language
from nltk.util import ngrams

from data_preprocessing.data_preprocessing_service import spacy_span_ngrams


@dataclass
class CValueResults:
    c_value: float
    candidate_term: str


@dataclass
class CandidateTermStatTriple:
    """From the algorithm in <https://doi.org/10.1007/s007999900023> (section 2.3, page 4):
    For every string a, that is extracted as a candidate term, we create for each of its substrings b, a triple (f(b), t(b), c(b)), 
    where f(b) is the total frequency of b in the corpus, t(b) is the frequency of b as a nested string of candidate terms, c(b) is the 
    number of these longer candidate terms.

    When an instance is created, the count of longer candidate term is initialized to 1 according to the algorithm.
    """
    candidate_term: str
    substring: str
    substring_corpus_frequency: int = 0
    substring_nested_frequency: int = 0
    count_longer_terms: int = 1


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
        self.candidateTermsSpans = None
        self.candidateTermsCounter = None
        self.CandidateTermStatTriples = dict()
        self.c_values = None

    def __call__(self) -> List[CValueResults]:
        if self.c_values is not None:
            return self.c_values
        else:
            return self.compute_c_values()

    def _extract_candidate_terms(self) -> None:
        """Extract the valid list of candidate terms and compute the corresponding frequences.
            This method sets the attributes:
              - self.candidateTerms
              - self.candidateTermsCounter
        """

        candidateTermsCounter = defaultdict(lambda: 0)
        candidateTermSpans = dict()
        candidateTerms = []
        candidate_terms_by_size = defaultdict(set)

        def update_term_containers(span) -> None:
            for size in range(1, self.max_size_gram + 1):  # for each gram size
                size_candidate_terms_spans = spacy_span_ngrams(
                    span, size)  # generate ngrams
                for size_candidate_terms_span in size_candidate_terms_spans:  # update variables for each ngram

                    candidate_terms_by_size[size].add(
                        size_candidate_terms_span.text)

                    candidateTermsCounter[size_candidate_terms_span.text] += 1

                    # select one spacy. Span for each text (in this case the last one)
                    candidateTermSpans[size_candidate_terms_span.text] = size_candidate_terms_span

        for span in self.tokenSequences:
            if len(span) <= self.max_size_gram:  # token sequence length ok
                update_term_containers(span)

            else:  # token sequence too long --> generate subsequences and process them
                tokenSubSequences = [
                    gram for gram in spacy_span_ngrams(span, self.max_size_gram)]

                for tokenSeq in tokenSubSequences:
                    update_term_containers(tokenSeq)

        # each group of candidate terms needs to be ordered by the frequence
        # groups of candidate terms are concatenated from the the longest to the smallest
        for terms_size in range(1, self.max_size_gram + 1).__reversed__():
            orderedByFreqTerms = list(candidate_terms_by_size[terms_size])
            orderedByFreqTerms.sort(
                key=lambda term: candidateTermsCounter[term], reverse=True)
            candidateTerms.extend(orderedByFreqTerms)

        # self.candidateTerms = candidateTerms
        self.candidateTermsSpans = [candidateTermSpans[term]
                                    for term in candidateTerms]
        self.candidateTermsCounter = candidateTermsCounter

    def _get_substrings_spans(self, term_span: str) -> List[str]:
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
        substrings_spans = set()
        for i in range(1, len(term_span)):
            # we need ngrams, i.e., all overlapping substrings
            for term_subspan in spacy_span_ngrams(term_span, i):
                substrings_spans.add(term_subspan)

        return list(substrings_spans)

    def _update_CandidateTermStatTriples(self, substring: str, parent_term: str) -> None:
        """Update the stat triples table according to the algorithm in <https://doi.org/10.1007/s007999900023> (section 2.3, page 4).

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

        if substring in self.CandidateTermStatTriples.keys():

            if parent_term in self.CandidateTermStatTriples.keys():
                self.CandidateTermStatTriples[substring].substring_nested_frequency += (
                    self.candidateTermsCounter[parent_term] - self.CandidateTermStatTriples[parent_term].substring_nested_frequency)
            else:
                self.CandidateTermStatTriples[substring].substring_nested_frequency += self.candidateTermsCounter[parent_term]

            self.CandidateTermStatTriples[substring].count_longer_terms += 1

        else:  # if substring never seen before, init a new CandidateTermStatTriple

            substr_corpus_frequency = 0
            # the substring might be an existing candidate term, if so its frenquency is the condidate term one
            if self.candidateTermsCounter.get(substring) is not None:
                self.candidateTermsCounter[substring]

            self.CandidateTermStatTriples[substring] = CandidateTermStatTriple(
                candidate_term=parent_term,
                substring=substring,
                substring_corpus_frequency=substr_corpus_frequency,
                substring_nested_frequency=self.candidateTermsCounter[parent_term],
                count_longer_terms=1  # init to one, it is the first time we encunter the substring
            )

    def _process_substrings_spans(self, candidate_term_span: spacy.tokens.span.Span) -> None:
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
        substrings_spans = self._get_substrings_spans(candidate_term_span)
        for substring_span in substrings_spans:
            self._update_CandidateTermStatTriples(
                substring_span.text, candidate_term_span.text)

    def compute_c_values(self) -> List[CValueResults]:
        """Compute the C-value following the algorithm in <https://doi.org/10.1007/s007999900023> (section 2.3, page 4).

           This method sets the attribute:
             - self.c_values
        """

        self._extract_candidate_terms()

        c_values = []

        for candidate_term_span in self.candidateTermsSpans:

            len_candidate_term = len(candidate_term_span)

            if len_candidate_term == self.max_size_gram:
                c_val = math.log2(len_candidate_term) * \
                    self.candidateTermsCounter[candidate_term_span.text]
                c_values.append(CValueResults(
                    c_value=c_val, candidate_term=candidate_term_span.text))

                self._process_substrings_spans(
                    candidate_term_span)

            else:
                if candidate_term_span.text not in self.CandidateTermStatTriples.keys():
                    c_val = math.log2(len_candidate_term) * \
                        self.candidateTermsCounter[candidate_term_span.text]
                    c_values.append(CValueResults(
                        c_value=c_val, candidate_term=candidate_term_span.text))
                else:
                    c_val = math.log2(
                        len_candidate_term) * (self.candidateTermsCounter[candidate_term_span.text] -
                                               (
                                                   self.CandidateTermStatTriples[candidate_term_span.text].substring_nested_frequency /
                            self.CandidateTermStatTriples[candidate_term_span.text].count_longer_terms)
                    )
                    c_values.append(CValueResults(
                        c_value=c_val, candidate_term=candidate_term_span.text))

                self._process_substrings_spans(
                    candidate_term_span)

        # reorder the c-values so we have the terms with the highest c-values at the top.
        c_values.sort(key=lambda c_val: c_val.c_value, reverse=True)

        self.c_values = c_values

        return self.c_values


class TermExtraction:
    def c_value(self, tokenSequences: Iterable[spacy.tokens.span.Span], max_size_gram: int) -> List[CValueResults]:
        pass


if __name__ == "__main__":
    test_terms = []
    test_terms.extend(["ADENOID CYSTIC BASAL CELL CARCINOMA"] * 5)
    test_terms.extend(["CYSTIC BASAL CELL CARCINOMA"] * 11)
    test_terms.extend(["ULCERATED BASAL CELL CARCINOMA"] * 7)
    test_terms.extend(["RECURRENT BASAL CELL CARCINOMA"] * 5)
    test_terms.extend(["CIRCUMSCRIBED BASAL CELL CARCINOMA"] * 3)
    test_terms.extend(["BASAL CELL CARCINOMA"] * 984)

    vocab_strings = []
    for term in test_terms:
        vocab_strings.extend(term.split())

    vocab = spacy.vocab.Vocab(strings=vocab_strings)

    test_terms_spans = []

    for term in test_terms:
        words = term.split()
        spaces = [True] * len(words)
        doc = spacy.tokens.Doc(vocab, words=words, spaces=spaces)
        span = spacy.tokens.Span(doc, doc[0].i, doc[-1].i + 1)
        test_terms_spans.append(span)

    my_c_val = Cvalue(tokenSequences=test_terms_spans, max_size_gram=5)

    test_candidate_terms_by_size = defaultdict(list)
    test_candidateTermsSpans = [span for span in test_terms_spans]
    for term_span in test_candidateTermsSpans:
        test_candidate_terms_by_size[len(term_span)].append(term_span)

    # we manually set the candidate terms and their frequences otherwise the process considers all
    # the ngrams extracted from the terms. This is not done like this in the paper.
    # my_c_val.candidateTerms, my_c_val.candidateTermsCounter = my_c_val._order_count_candidate_terms(
    #     test_candidate_terms_by_size)
    my_c_val.compute_c_values()

    c_values = my_c_val()

    print(c_values)
