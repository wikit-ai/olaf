from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set


class KnowledgeSource(ABC):
    """Knowledge sources are any external sources of knowledge.

    Attributes
    ----------
    parameters: Dict[str, Any]
        Parameters are fixed values to be defined when building the knowledge source.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        """Initialise KnowledgeSource instance.

        Parameters
        ----------
        parameters : Dict[str, Any], optional
            Parameters are fixed values to be defined when building the knowledge source,
            by default None.
        """
        self.parameters = parameters if parameters else dict()

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
