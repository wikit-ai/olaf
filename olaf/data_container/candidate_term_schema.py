from typing import Optional, Set, Tuple

import spacy.tokens

from .concept_schema import Concept
from .enrichment_schema import Enrichment


class CandidateTerm:
    """Candidate terms are created by the term extraction methods.
    They represent the words of interest in the corpus.
    They are used by the concept and relation extraction methods and transformed into
    linguistic realisations.
    """

    def __init__(
        self,
        label: str,
        corpus_occurrences: Set[spacy.tokens.Span],
        enrichment: Optional[Enrichment] = None,
    ) -> None:
        """Initialise candidate term instance.

        Parameters
        ----------
        label : str
            The candidate term human readable label.
        corpus_occurrences : Set[spacy.tokens.Span]
            Set of corpus occurrences for the candidate term.
        enrichment : Enrichment, optional
            Enrichment information for the candidate term, by default None.
        """
        self.label = label
        self.corpus_occurrences = corpus_occurrences
        self.enrichment = enrichment

    def add_corpus_occurrences(
        self, new_corpus_occurrences: Set[spacy.tokens.Span]
    ) -> None:
        """Add new corpus occurrences for the candidate terms.

        Parameters
        ----------
        new_corpus_occurrences : Set[spacy.tokensSpan]
            New corpus occurrences to add for the candidate term.
        """
        self.corpus_occurrences.update(new_corpus_occurrences)


class CandidateRelation(CandidateTerm):
    """Candidate relations are created from candidate terms by the ct_to_cr function.
    They represent the words of interest for relation in the corpus.
    They are used by the relation extraction methods and transformed into
    linguistic realisations.
    """

    def __init__(
        self,
        label: str,
        corpus_occurrences: Set[Tuple[spacy.tokens.Span]],
        source_concept: Optional[Concept] = None,
        destination_concept: Optional[Concept] = None,
        enrichment: Optional[Enrichment] = None,
    ) -> None:
        """Initialise candidate relation instance.

        Parameters
        ----------
        label : str
            The candidate term human readable label.
        corpus_occurrences : Set[Tuple[spacy.tokens.Span]]
            Set of corpus occurrences for the candidate relation.
        source_concept : Concept, optional
            Source concept of the relation if specified, by default None.
        destination_concept : Concept, optional
            Destination concept of the relation if specified, by default None.
        enrichment : Enrichment, optional
            Enrichment information for the candidate relation, by default None.
        """
        super().__init__(label, corpus_occurrences, enrichment)
        self.source_concept = source_concept
        self.destination_concept = destination_concept

    def add_corpus_occurrence(
        self, new_corpus_occurrence: Tuple[spacy.tokens.Span]
    ) -> None:
        """Add new corpus occurrence for the candidate relation.

        Parameters
        ----------
        new_corpus_occurrence : Tuple[spacy.tokens.Span]
            New corpus occurrence to add for the candidate relation.
        """
        self.corpus_occurrences.add(new_corpus_occurrence)

    def add_corpus_occurrences(
        self, new_corpus_occurrences: Set[Tuple[spacy.tokens.Span]]
    ) -> None:
        """Add new corpus occurrences for the candidate relations.

        Parameters
        ----------
        new_corpus_occurrences : Set[Tuple[spacy.tokensSpan]]
            New corpus occurrences to add for the candidate relation.
        """
        self.corpus_occurrences.update(new_corpus_occurrences)
