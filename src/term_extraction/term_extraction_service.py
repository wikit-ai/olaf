from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, List, Dict

import spacy
import spacy.tokens
import spacy.tokenizer
import spacy.language

from data_preprocessing.data_preprocessing_service import spacy_span_ngrams
from term_extraction.term_extraction_schema import CandidateTermStatTriple, CValueResults
from config import core
import logging_config


class Cvalue:
    """A class to compute the C-value of each term (token sequence) in a corpus of texts.
       The C-values are computed based on <https://doi.org/10.1007/s007999900023>.
    """

    def __init__(self, tokenSequences: Iterable[spacy.tokens.span.Span], max_size_gram: int) -> None:
        """The Cvalue class only requires a list of text sequences (Spacy Span objects) and a maximum size of ngrams.
            The C-value will be computed for all ngrams with size ranging from 1 to max_size_gram.

            Note:
              - The maximum length of a token sequence should be equal to max_size_gram. 
                If not, we will add ngrams of max_size_gram extracted from the longer sequence of tokens.

            TODO:
              - Add an attribute to store the list of spacy Spans associated with each candidate term string.
                So we can keep track of the relation between candidate terms and the documents that contains them.

        Parameters
        ----------
        tokenSequences : Iterable[spacy.tokens.span.Span]
            The token sequences extracted from the document in the corpus. 
            We will extract candidate terms from those sequences by computing ngrams.
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
        c_values : List[CValueResults]
            An attribute that will contained the candidate terms and their C-values ordered by descending C-values.
            This attribute will be set by the private method compute_c_values.
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

        # an inner function to not duplicate code
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
        for i in range(1, len(term_span)):
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

    def compute_c_values(self) -> List[CValueResults]:
        """Compute the C-value following the algorithm in <https://doi.org/10.1007/s007999900023> (section 2.3, page 4).

           This method sets the attribute:
             - self.c_values

        Returns
        -------
        List[CValueResults]
            the candidate terms and their C-values ordered by descending C-values. 
        """

        if self.candidateTermsSpans is None:
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


class Term_Extraction():
    """Second processing of the corpus.
    Finding of terms under interest.

    """

    def __init__(self) -> None:
        pass

    def c_value(self, tokenSequences: Iterable[spacy.tokens.span.Span], max_size_gram: int) -> Cvalue:
        self.c_value = Cvalue(tokenSequences, max_size_gram)
        return self.c_value

    def on_pos_token_filtering(self,corpus: List[List[spacy.tokens.token.Token]],token_pos_filter: List[str]) -> List[Dict[str,int]]:
        """Return unique candidate terms after filtering on pos-tagging labels.
        Candidate terms are lemmatized and put into lowercase.

        Parameters
        ----------
        corpus : List[List[spacy.tokens.token.Token]]
            Cleaned corpus.
        token_pos_filter : List[str]
            Pos-tagging filters to apply.
            
        Returns
        -------
        List[Dict[str,int]]
            List of unique candidate terms lemmatized and their occurences.
        """
        candidate_terms = []
        try :
            for document in corpus:
                for token in document :
                    if token.pos_ in token_pos_filter:
                        candidate_terms.append(token.lemma_.lower())
        except Exception as _e:
            logging_config.logger.error("Could not filter and lemmatize spacy tokens. Trace : %s", _e)
        else : 
            logging_config.logger.info("List of tokens filtered and lemmatized.")
        unique_candidate_terms = list(set(candidate_terms))
        count_candidate_terms = [{"term":term,"occurence":candidate_terms.count(term)} for term in unique_candidate_terms]
        return count_candidate_terms

    def frequency_filtering(self, count_candidate_terms: List[Dict[str,int]]) -> List[str]:
        """Return candidate terms with frequency higher than a configured threshold.

        Parameters
        ----------
        count_candidate_terms : List[Dict[str,int]]
            List of unique candidate terms and their occurences.

        Returns
        -------
        List[str]
            Candidate terms extracted.
        """
        nb_term_candidates = len(count_candidate_terms)
        validated_terms =[]
        if nb_term_candidates>0 :
            term_occurrence = []
            try :
                for candidate in count_candidate_terms:
                    term_occurrence.append({"term":candidate['term'],"occurence":candidate['occurence']})

                validated_terms = [term['term'] for term in term_occurrence if term['occurrence']>core.OCCURRENCE_THRESHOLD]
            except Exception as _e:
                logging_config.logger.error("Could not filter candidate terms by occurrence. Trace : %s", _e)
            else : 
                logging_config.logger.info("List of tokens filtered by occurrence.")
        else:
            logging_config.logger.error("No term candidate found.")
            validated_terms = None
        return validated_terms

term_extraction = Term_Extraction()