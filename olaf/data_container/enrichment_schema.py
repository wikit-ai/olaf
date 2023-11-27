from dataclasses import dataclass, field
from typing import Set


@dataclass
class Enrichment:
    """A dataclass to contain any information enriching a candidate term.
        Instances are typically created by the candidate term enrichment processes.
        By default we define at minimum synonyms but we can extend the class with any possible
        useful information, e.g., antonyms, or hypernyms.

        Note that the definition given to synonyms here largely depends on the knowledge
        representation application.It could be strict synonyms or closely related terms with
        regards to a specific context domain.

    Parameters
    ----------
    synonyms: Set[str]
        The set of synonyms.
        Empty set by default if it is initialised without terms.
    hypernyms: Set[str]
        The set of hypernyms.
        Empty set by default if it is initialised without terms.
    hyponyms: Set[str]
        The set of hyponyms.
        Empty set by default if it is initialised without terms.
    antonyms: Set[str]
        The set of antonyms.
        Empty set by default if it is initialised without terms.
    """

    synonyms: Set[str] = field(default_factory=set)
    hypernyms: Set[str] = field(default_factory=set)
    hyponyms: Set[str] = field(default_factory=set)
    antonyms: Set[str] = field(default_factory=set)

    def add_synonyms(self, synonyms: Set[str]) -> None:
        """Add new synonyms for the enrichment.

        Parameters
        ----------
        synonyms : Set[str]
            New synonyms to add on the enrichment.
        """
        self.synonyms.update(synonyms)

    def add_hypernyms(self, hypernyms: Set[str]) -> None:
        """Add new hypernyms for the enrichment.

        Parameters
        ----------
        hypernyms : Set[str]
            New hypernyms to add on the enrichment.
        """
        self.hypernyms.update(hypernyms)

    def add_hyponyms(self, hyponyms: Set[str]) -> None:
        """Add new hyponyms for the enrichment.

        Parameters
        ----------
        hyponyms : Set[str]
            New hyponyms to add on the enrichment.
        """
        self.hyponyms.update(hyponyms)

    def add_antonyms(self, antonyms: Set[str]) -> None:
        """Add new antonyms for the enrichment.

        Parameters
        ----------
        antonyms : Set[str]
            New antonyms to add on the enrichment.
        """
        self.antonyms.update(antonyms)

    def merge_with_enrichment(self, enrichment_to_integrate) -> None:
        """Merge the enrichment into another one.
        The enrichment is updated in place with the enrichment provided.

        Parameters
        ----------
        enrichment_to_integrate : Enrichment
            The enrichment to merge the current one with.
        """
        self.add_synonyms(enrichment_to_integrate.synonyms)
        self.add_antonyms(enrichment_to_integrate.antonyms)
        self.add_hypernyms(enrichment_to_integrate.antonyms)
        self.add_hyponyms(enrichment_to_integrate.hyponyms)
