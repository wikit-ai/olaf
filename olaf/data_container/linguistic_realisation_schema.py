from abc import ABC
from typing import Any, Optional, Set, Tuple

import spacy.tokens


class LinguisticRealisation(ABC):
    """We distinguish between concept, relation and metarelation and their representations in text.
    The text denoting a concept, relation or metarelation is referred to as a linguistic
    realisation. The LinguisticRealisation class define a string (label) which is the text denoting
    the concept, relation or meta relation and keep an index of all occurrences in the corpus, i.e.
    corpus_occurrences.

    Parameters
    ----------
    label : str
        The linguistic realisation human readable label.
        The string should appear in one of the corpus occurrences or be a metarelation type.
    corpus_occurrences : Any
        The LinguisticRealisation occurrences in the corpus.
    """

    def __init__(
        self, label: str, corpus_occurrences: Optional[Set[Any]] = None
    ) -> None:
        """Initialise linguistic realisation instance.

        Parameters
        ----------
        label : str
            The linguistic realisation human readable label.
            The string should appear in one of the corpus occurrences or be a metarelation type.
        corpus_occurrences : Optional[Set[CORPUS_OCCURRENCE]]
            The linguistic realisation occurrences in the corpus.
        """
        self.label = label
        self.corpus_occurrences = corpus_occurrences if corpus_occurrences else set()

    def add_corpus_occurrences(self, new_corpus_occurrences: Set[Any]) -> None:
        """Add new corpus occurrences for the linguistic realisation.

        Parameters
        ----------
        new_corpus_occurrences : Set[Any]
            New corpus occurrences to add for the linguistic realisation.
        """
        
        self.corpus_occurrences.update(new_corpus_occurrences)

    def get_docs(self) -> Set[spacy.tokens.Doc]:
        """Fetch all the corpus documents contained in the corpus occurrences.

        Returns
        -------
        Set[spacy.tokens.doc.Doc]
            The set of corpus documents contained in the corpus occurrences.
        """
        corpus_occurrences_docs = set()

        for corpus_occ in self.corpus_occurrences:
            if isinstance(corpus_occ, tuple):
                doc = corpus_occ[0].doc
            else:
                doc = corpus_occ.doc

            corpus_occurrences_docs.add(doc)

        return corpus_occurrences_docs


class ConceptLR(LinguisticRealisation):
    """Linguistic Realisation specific to a concept. Corpus occurrences are represented
    as single span.

    Parameters
    ----------
    label : str
        The linguistic realisation human readable label.
        The string should appear in one of the corpus occurrences.
    corpus_occurrences : Set[spacy.tokens.Span]
        The concept linguistic realisation occurrences in the corpus which is a span.
    """

    def __init__(
        self,
        label: str,
        corpus_occurrences: Optional[Set[spacy.tokens.Span]] = None,
    ) -> None:
        """Initialise concept linguistic realisation instance.

        Parameters
        ----------
        label : str
            The linguistic realisation human readable label.
            The string should appear in one of the corpus occurrences.
        corpus_occurrences : Optional[Set[spacy.tokens.Span]]
            The concept linguistic realisation occurrences in the corpus.
        """
        super().__init__(label, corpus_occurrences)

    def add_corpus_occurrences(
        self, new_corpus_occurrences: Set[spacy.tokens.Span]
    ) -> None:
        """Add new corpus occurrences for the linguistic realisation.

        Parameters
        ----------
        new_corpus_occurrences : Set[spacy.tokens.Span]
            New corpus occurrences to add for the linguistic realisation.
        """
        super().add_corpus_occurrences(new_corpus_occurrences)


class RelationLR(LinguisticRealisation):
    """Linguistic Realisation specific to a relation. Corpus occurrences are represented as tuple
    of three spans.
    One for the source concept at position 0.
    One for the relation label at position 1.
    One for the destination concept at position 2.

    Parameters
    ----------
    label : str
        The linguistic realisation human readable label.
        The string should appear in one of the corpus occurrences.
    corpus_occurrences : Set[Tuple[spacy.tokens.Span, spacy.tokens.Span, spacy.tokens.Span]]
        The relation linguistic realisation occurrences in the corpus which is tuple of three spans.
    """

    def __init__(
        self,
        label: str,
        corpus_occurrences: Optional[
            Set[
                Tuple[
                    spacy.tokens.Span,
                    spacy.tokens.Span,
                    spacy.tokens.Span,
                ]
            ]
        ] = None,
    ) -> None:
        """Initialise relation linguistic realisation instance.

        Parameters
        ----------
        label : str
            The linguistic realisation human readable label.
            The string should appear in one of the corpus occurrences.
        corpus_occurrences : Optional[Set[Tuple[spacy.tokens.Span, spacy.tokens.Span, spacy.tokens.Span]]]
            The relation linguistic realisation occurrences in the corpus.
        """
        super().__init__(label, corpus_occurrences)

    def add_corpus_occurrences(
        self,
        new_corpus_occurrences: Set[
            Tuple[spacy.tokens.Span, spacy.tokens.Span, spacy.tokens.Span]
        ],
    ) -> None:
        """Add new corpus occurrences for the linguistic realisation.

        Parameters
        ----------
        new_corpus_occurrences : Set[Tuple[spacy.tokens.Span, spacy.tokens.Span, spacy.tokens.Span]]
            New corpus occurrences to add for the linguistic realisation.
        """
        super().add_corpus_occurrences(new_corpus_occurrences)


class MetaRelationLR(LinguisticRealisation):
    """Linguistic Realisation specific to a metarelation. Corpus occurrences are represented as tuple
    of two spans.
    One for the source concept at position 0.
    One for the destination concept at position 1.

    Parameters
    ----------
    label : str
        The linguistic realisation human readable label.
        The string should be a metarelation type.
    corpus_occurrences : Set[Tuple[spacy.tokens.Span, spacy.tokens.Span, spacy.tokens.Span]]
        The metarelation linguistic realisation occurrences in the corpus which is tuple of
        two spans.
    """

    def __init__(
        self,
        label: str,
        corpus_occurrences: Optional[
            Set[Tuple[spacy.tokens.Span, spacy.tokens.Span]]
        ] = None,
    ) -> None:
        """Initialise metarelation linguistic realisation instance.

        Parameters
        ----------
        label : str
            The linguistic realisation human readable label.
            The string should be a metarelation type.
        corpus_occurrences : Optional[Set[Tuple[spacy.tokens.Span, spacy.tokens.Span, spacy.tokens.Span]]]
            The relation linguistic realisation occurrences in the corpus.
        """
        super().__init__(label, corpus_occurrences)

    def add_corpus_occurrences(
        self,
        new_corpus_occurrences: Set[Tuple[spacy.tokens.Span, spacy.tokens.Span]],
    ) -> None:
        """Add new corpus occurrences for the linguistic realisation.

        Parameters
        ----------
        new_corpus_occurrences : Set[Tuple[spacy.tokens.Span, spacy.tokens.Span]]
            New corpus occurrences to add for the linguistic realisation.
        """
        super().add_corpus_occurrences(new_corpus_occurrences)
