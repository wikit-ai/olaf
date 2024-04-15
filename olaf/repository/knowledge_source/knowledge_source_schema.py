from abc import ABC, abstractmethod
from typing import Set


class KnowledgeSource(ABC):
    """Knowledge sources are any external sources of knowledge."""

    def __init__(self) -> None:
        """Initialise KnowledgeSource instance."""

    @abstractmethod
    def _check_resources(self) -> None:
        """Method to check that the component has access to all its required resources."""

    @abstractmethod
    def _check_parameters(self) -> None:
        """Check wether required parameters are given and correct. If this is not the case,
        suitable default ones are set.
        """

    @abstractmethod
    def match_external_concepts(self, matching_terms: Set[str]) -> Set[str]:
        """Method to fetch external concepts matching the set of terms.

        Parameters
        ----------
        matching_terms : Set[str]
            The term texts to use for matching concepts.

        Returns
        -------
        Set[str]
            The UIDs of the external concepts found matching the term texts.
        """

    @abstractmethod
    def fetch_terms_synonyms(self, terms: Set[str]) -> Set[str]:
        """Method to fetch synonyms of a set of terms according to the knowledge source.

        Parameters
        ----------
        terms : Set[str]
            The set of terms to find synonyms of.

        Returns
        -------
        Set[str]
            The set of terms synonyms.
        """

    @abstractmethod
    def fetch_terms_antonyms(self, terms: Set[str]) -> Set[str]:
        """Method to fetch antonyms of a set of terms according to the knowledge source.

        Parameters
        ----------
        terms : Set[str]
            The set of terms to find antonyms of.

        Returns
        -------
        Set[str]
            The set of terms antonyms.
        """

    @abstractmethod
    def fetch_terms_hypernyms(self, terms: Set[str]) -> Set[str]:
        """Method to fetch hypernyms of a set of terms according to the knowledge source.

        Parameters
        ----------
        terms : Set[str]
            The set of terms to find hypernyms of.

        Returns
        -------
        Set[str]
            The set of terms hypernyms.
        """

    @abstractmethod
    def fetch_terms_hyponyms(self, terms: Set[str]) -> Set[str]:
        """Method to fetch hyponyms of a set of terms according to the knowledge source.

        Parameters
        ----------
        terms : Set[str]
            The set of terms to find hyponyms of.

        Returns
        -------
        Set[str]
            The set of terms hyponyms.
        """
