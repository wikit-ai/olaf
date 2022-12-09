from collections import defaultdict
import math
from typing import Dict, List, Set

import spacy.tokens.doc

import config.logging_config as logging_config
from term_extraction.term_extraction_schema import CandidateTermStatTriple, DocAttributeNotFound, TermExtractionResults
from commons.spacy_processing_tools import spacy_span_ngrams


class Cvalue:
    """A class to compute the C-value of each term (token sequence) in a corpus of texts.
       The C-values are computed based on <https://doi.org/10.1007/s007999900023>.
    """

    def __init__(self, corpus: List[spacy.tokens.doc.Doc], tokenSequences_doc_attribute_name: str, max_size_gram: int) -> None:
        """The Cvalue class requires a list of text sequences (Spacy Span objects) and a maximum size of ngrams.
            The C-value will be computed for all ngrams with size ranging from 1 to max_size_gram.

            Note:
              - The maximum length of a token sequence should be equal to max_size_gram. 
                If not, we will add ngrams of max_size_gram extracted from the longer sequence of tokens.

            TODO:
              - Add an attribute to store the list of spacy Spans associated with each candidate term string.
                So we can keep track of the relation between candidate terms and the documents that contains them.
                add the method to update the corpus Docs with a custom attribute tagging the righ spans.

        Parameters
        ----------
        corpus: List[spacy.tokens.doc.Doc]
            The List of Spacy Doc composing the corpus
        tokenSequences_doc_attribute_name : str
            The name of the Spacy Doc custom attribute storing the token sequences to consider for the C-value
        max_size_gram : int
            The maximum size for ngrams to consider.

        Class instance attributes
        ----------
        candidateTermsSpans : Iterable[spacy.tokens.span.Span]
            The spacy Spans corresponding to the candidate terms extracted form the tokenSequences. 
            This attribute will be set by the private method _extract_candidate_terms.
        candidateTermsCounter : collection.Counter
            A counter of the candidate terms appearance in the corpus
            This attribute will be set by the private method _extract_candidate_terms. 
        CandidateTermStatTriples : Dict[str, CandidateTermStatTriple]
            An attribute updated and used during the C-value computation process
            This attribute will be set by the private method compute_c_values.
        c_values : List[TermExtractionResults]
            An attribute that will contained the candidate terms and their C-values ordered by descending C-values.
            This attribute will be set by the private method compute_c_values.
        """
        self.corpus = corpus
        self.tokenSequences_doc_attribute_name = tokenSequences_doc_attribute_name
        self.max_size_gram = max_size_gram
        self.candidateTermsSpans = None
        self.candidateTermsCounter = None
        self.CandidateTermStatTriples = dict()
        self.c_values = list()

    @property
    def c_values(self) -> List[TermExtractionResults]:
        """Getter for self.c_value property

        Returns
        -------
        List[TermExtractionResults]
            The list of C-values alongside the terms
        """
        if len(self._c_values) == 0:
            self.compute_c_values()

        return self._c_values

    @c_values.setter
    def c_values(self, value: List[TermExtractionResults]) -> None:
        """Setter for self.c_value attribute.

        Parameters
        ----------
        value : List[TermExtractionResults]
            The value to set.
        """
        if isinstance(value, list):
            self._c_values = value
        else:
            logging_config.logger.error(
                "Incompatible value type for self.c_value attribute. It should be List[TermExtractionResults]")

    def _extract_token_sequences(self) -> List[spacy.tokens.span.Span]:
        """Extract the list of Spacy spans contained in the Spacy Doc custum attribute

        Returns
        -------
        List[spacy.tokens.span.Span]
            The list of Spacy spans corresponding to the token sequences to use for the C-value computation.

        Raises
        ------
        DocAttributeNotFound
            An Exception to flag when a custom attribute on a Spacy Doc has not been found.
        """
        tokenSequences = []

        try:
            for doc in self.corpus:
                if not doc.has_extension(self.tokenSequences_doc_attribute_name):
                    raise DocAttributeNotFound(
                        f"Document custom attribute {self.tokenSequences_doc_attribute_name} not found for document object {doc}")
                else:
                    tokenSequences.extend([token_seq for token_seq in doc._.get(
                        self.tokenSequences_doc_attribute_name) if len(token_seq) > 1])
        except Exception as e:
            logging_config.logger.error(
                f"Trace : {e}")
        else:
            logging_config.logger.info(
                f"Token Sequences for C-value extracted")

        return tokenSequences

    def _update_term_containers(self,
                                span: spacy.tokens.span.Span,
                                candidate_terms_by_size: Dict[int, Set[str]],
                                candidateTermsCounter: Dict[str, int],
                                candidateTermSpans: Dict[str,
                                                         spacy.tokens.span.Span]
                                ) -> None:
        """Process a candidate term (Spacy span) and update data containers required for the C-value computation.

        Parameters
        ----------
        span : spacy.tokens.span.Span
            The candidate term to process
        candidate_terms_by_size : Dict[int, Set[str]]
            A data container with the candidate terms strings grouped by their size (i.e. number of words)
        candidateTermsCounter : Dict[str, int]
            A data container with the candidate terms occurrence counts in the corpus 
        candidateTermSpans : Dict[str, spacy.tokens.span.Span]
            A data container with the strings associated with each candidate term
        """

        for size in range(2, self.max_size_gram + 1):  # for each gram size

            size_candidate_terms_spans = spacy_span_ngrams(
                span, size)  # generate ngrams

            for size_candidate_terms_span in size_candidate_terms_spans:  # update variables for each ngram

                candidate_terms_by_size[size].add(
                    size_candidate_terms_span.text)

                candidateTermsCounter[size_candidate_terms_span.text] += 1

                # select one spacy. Span for each text (in this case the last one)
                candidateTermSpans[size_candidate_terms_span.text] = size_candidate_terms_span

    def _tokenSequence2CandidateTerm(self, tokenSequence: spacy.tokens.span.Span) -> List[spacy.tokens.span.Span]:
        """Extract the candidate terms spans from the selected token sequences. 
            Generate sub sequences when the token sequence is longer than the self.max_size_gram.

        Parameters
        ----------
        tokenSequence : spacy.tokens.span.Span
            The token sequence to extract the candidate term from.

        Returns
        -------
        List[spacy.tokens.span.Span]
            The list of candidate terms
        """
        if len(tokenSequence) <= self.max_size_gram:  # token sequence length ok
            if len(tokenSequence) > 1:
                return [tokenSequence]
            else:
                return []
        else:  # token sequence too long --> generate subsequences and process them
            return [gram for gram in spacy_span_ngrams(tokenSequence, self.max_size_gram)]

    def _order_candidate_terms(self, candidate_terms_by_size: Dict[int, Set[str]]) -> List[spacy.tokens.span.Span]:
        """Order the candidate terms as detailed in the algorithm defined in <https://doi.org/10.1007/s007999900023>.

        Parameters
        ----------
        candidate_terms_by_size : Dict[int, Set[str]]
            A data container with the candidate terms strings grouped by their size (i.e. number of words)

        Returns
        -------
        List[str]
            The ordered list of candidate terms
        """
        # each group of candidate terms needs to be ordered by occurrence
        # groups of candidate terms are concatenated from the the longest to the smallest
        ordered_candidate_terms = []
        for terms_size in range(1, self.max_size_gram + 1).__reversed__():
            orderedByOccurrenceTerms = list(
                candidate_terms_by_size[terms_size])
            orderedByOccurrenceTerms.sort(
                key=lambda term: self.candidateTermsCounter[term], reverse=True)
            ordered_candidate_terms.extend(orderedByOccurrenceTerms)

        return ordered_candidate_terms

    def _extract_candidate_terms(self) -> None:
        """Extract the valid list of candidate terms and compute the corresponding occurencies.
            This method sets the attributes:
              - self.candidateTermsSpans
              - self.candidateTermsCounter
        """

        candidateTermsCounter = defaultdict(lambda: 0)
        candidateTermSpans = dict()
        candidateTerms = []
        candidate_terms_by_size = defaultdict(set)

        tokenSequences = self._extract_token_sequences()

        for tokenSeq in tokenSequences:
            candidate_term_spans = self._tokenSequence2CandidateTerm(tokenSeq)

            for candidate_term_span in candidate_term_spans:
                self._update_term_containers(
                    candidate_term_span, candidate_terms_by_size, candidateTermsCounter, candidateTermSpans)

        self.candidateTermsCounter = candidateTermsCounter

        candidateTerms = self._order_candidate_terms(candidate_terms_by_size)
        self.candidateTermsSpans = [candidateTermSpans[term]
                                    for term in candidateTerms]

    def _get_substrings_spans(self, term_span: spacy.tokens.span.Span) -> List[spacy.tokens.span.Span]:
        """Extract the ngrams contained in the term. The result is returned as a list of spacy Spans.

        Parameters
        ----------
        term_span : spacy.tokens.span.Span
            The spacy Span of the term to extract the ngrams from

        Returns
        -------
        List[spacy.tokens.span.Span]
            The resulting list of ngrams as spacy Spans
        """
        substrings_spans = []
        for i in range(2, len(term_span)):
            # we need ngrams, i.e., all overlapping substrings
            for term_subspan in spacy_span_ngrams(term_span, i):
                substrings_spans.append(term_subspan)

        return substrings_spans

    def _update_CandidateTermStatTriples(self, substring: str, parent_term: str) -> None:
        """Update the self.CandidateTermStatTriples attribute according to the algorithm in <https://doi.org/10.1007/s007999900023> (section 2.3, page 4).

        Parameters
        ----------
        substring : str
            the ngram text extracted from the candidate term.
        parent_term : str
            the candidate term text
        """

        if substring in self.CandidateTermStatTriples.keys():

            if parent_term in self.CandidateTermStatTriples.keys():
                self.CandidateTermStatTriples[substring].substring_nested_occurrence += (
                    self.candidateTermsCounter[parent_term] - self.CandidateTermStatTriples[parent_term].substring_nested_occurrence)
            else:
                self.CandidateTermStatTriples[substring].substring_nested_occurrence += self.candidateTermsCounter[parent_term]

            self.CandidateTermStatTriples[substring].count_longer_terms += 1

        else:  # if substring never seen before, init a new CandidateTermStatTriple

            substr_corpus_occurrence = 0
            # the substring might be an existing candidate term, if so its frenquency is the condidate term one
            if self.candidateTermsCounter.get(substring) is not None:
                self.candidateTermsCounter[substring]

            self.CandidateTermStatTriples[substring] = CandidateTermStatTriple(
                candidate_term=parent_term,
                substring=substring,
                substring_corpus_occurrence=substr_corpus_occurrence,
                substring_nested_occurrence=self.candidateTermsCounter[parent_term],
                count_longer_terms=1  # init to one, it is the first time we encunter the substring
            )

    def _process_substrings_spans(self, candidate_term_span: spacy.tokens.span.Span) -> None:
        """Extract the ngrams contained in the candidate term and loop over them to update the CandidateTermStatTriples attribute according to 
        the algorithm in <https://doi.org/10.1007/s007999900023> (section 2.3, page 4).

        Parameters
        ----------
        candidate_term_span : spacy.tokens.span.Span
            The spacy Span object of the candidate term to process.
        """
        substrings_spans = self._get_substrings_spans(candidate_term_span)
        for substring_span in substrings_spans:
            self._update_CandidateTermStatTriples(
                substring_span.text, candidate_term_span.text)

    def compute_c_values(self) -> None:
        """Compute the C-value following the algorithm in <https://doi.org/10.1007/s007999900023> (section 2.3, page 4).

           This method sets the attribute:
             - self.c_values
        """

        if self.candidateTermsSpans is None:
            self._extract_candidate_terms()

        c_values = []

        for candidate_term_span in self.candidateTermsSpans:

            len_candidate_term = len(candidate_term_span)

            if len_candidate_term == 1:
                logging_config.logger.error(
                    f"Error with candidate term {candidate_term_span.text}. C-value does not make sens for term composed of only one token.")
                continue

            if len_candidate_term == self.max_size_gram:
                c_val = math.log2(len_candidate_term) * \
                    self.candidateTermsCounter[candidate_term_span.text]
                c_values.append(TermExtractionResults(
                    score=c_val, candidate_term=candidate_term_span.text))

                self._process_substrings_spans(
                    candidate_term_span)

            else:
                if candidate_term_span.text not in self.CandidateTermStatTriples.keys():
                    c_val = math.log2(len_candidate_term) * \
                        self.candidateTermsCounter[candidate_term_span.text]
                    c_values.append(TermExtractionResults(
                        score=c_val, candidate_term=candidate_term_span.text))
                else:
                    c_val = math.log2(
                        len_candidate_term) * (self.candidateTermsCounter[candidate_term_span.text] -
                                               (
                                                   self.CandidateTermStatTriples[candidate_term_span.text].substring_nested_occurrence /
                            self.CandidateTermStatTriples[candidate_term_span.text].count_longer_terms)
                    )
                    c_values.append(TermExtractionResults(
                        score=c_val, candidate_term=candidate_term_span.text))

                self._process_substrings_spans(
                    candidate_term_span)

        # reorder the c-values so we have the terms with the highest c-values at the top.
        c_values.sort(key=lambda c_val: c_val.score, reverse=True)

        self.c_values = c_values
