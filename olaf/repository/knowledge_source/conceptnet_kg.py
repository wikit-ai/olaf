from typing import Any, Dict, List, Optional, Set

import requests

from ...commons.logging_config import logger
from ...commons.string_tools import space_to_underscore_str, underscore_to_space_str
from .knowledge_source_schema import KnowledgeSource


class ConceptNetKnowledgeResource(KnowledgeSource):
    """Adapter for the ConceptNet KG: https://conceptnet.io/.

    Attributes
    ----------
    lang: str, optional
        Language ISO code for the terms to find concepts for, by default 'en'.
    api_resp_batch_size: int, optional
        Batch size for the ConceptNet API when fetching data, by default 1000.
    check_sources: bool, optional
        Wether or not to filter the concepts based on provided sources, default False.
    validation_sources: Set[str], optional
        The sources to use to filter the concepts, default set().
    """

    def __init__(
        self,
        lang: Optional[str] = "en",
        api_resp_batch_size: Optional[int] = 1000,
        check_sources: Optional[bool] = False,
        validation_sources: Optional[Set[str]] = None,
    ) -> None:
        """Initialise ConceptNet knowledge resource instance.

        Parameters
        ----------
        lang: str, optional
            Language ISO code for the terms to find concepts for, by default 'en'.
        api_resp_batch_size: int, optional
            Batch size for the ConceptNet API when fetching data, by default 1000.
        check_sources: bool, optional
            Wether or not to filter the concepts based on provided sources, default False.
        validation_sources: Set[str], optional
            The sources to use to filter the concepts, default set().
        """

        self.lang = lang

        self.api_resp_batch_size = api_resp_batch_size
        self.check_sources = check_sources
        self.validation_sources = validation_sources
        self._check_parameters()

    def _check_parameters(self) -> None:
        """Check wether required parameters are given and correct. If this is not the case,
        suitable default ones are set.
        """
        if not self.lang:
            logger.warning(
                "No value given for lang parameter, default will be set to 'en'"
            )
            self.lang = "en"

        if not self.api_resp_batch_size:
            logger.warning(
                "No value given for api_resp_batch_size parameter, default will be set to 1000"
            )
            self.api_resp_batch_size = 1000

        if not self.validation_sources:
            logger.warning(
                "No value given for validation_sources parameter, default will be set to []"
            )
            self.validation_sources = []

        if self.check_sources and len(self.validation_sources) == 0:
            logger.warning(
                """Using sources checking (check_sources = True) but no source tags provided in parameter `validation_sources`.
                Defaulting to not checking sources.
                """
            )
            self.check_sources = False

    def _check_resources(self) -> None:
        # TODO
        """Method to check that the component has access to all its required resources."""

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

        term_conceptnet_uris = set()

        for term in matching_terms:
            term_conceptnet_uris.update(
                self._get_term_conceptnet_external_uris(space_to_underscore_str(term))
            )

        return term_conceptnet_uris

    def _get_term_conceptnet_external_uris(self, term_conceptnet_text: str) -> Set[str]:
        """Fetch term related data from ConceptNet api and extract the term related external URIs.

        Parameters
        ----------
        term_conceptnet_text : str
            The ConceptNet formatted term string.

        Returns
        -------
        Set[str]
            The ConceptNet term related external uris.
        """
        conceptnet_external_uris = {}
        conceptnet_term_edges = []

        conceptnet_term_res = self._conceptnet_api_fetch_term(
            term_conceptnet_text,
            self.lang,
            self.api_resp_batch_size,
        )

        if "error" in conceptnet_term_res.keys():  # The term is not in ConceptNet data.
            logger.info(
                "No match in ConceptNet for term %s",
                underscore_to_space_str(term_conceptnet_text),
            )

        else:
            conceptnet_term_edges = conceptnet_term_res.get("edges", [])

            if "view" in conceptnet_term_res.keys():  # Get rest of the edges.
                conceptnet_term_edges.extend(
                    self._get_paginated_conceptnet_edges(
                        conceptnet_term_res["view"], self.api_resp_batch_size
                    )
                )

            if self.check_sources:
                conceptnet_term_edges = self._filter_edges_on_sources(
                    conceptnet_term_edges
                )

            conceptnet_external_uris = self._get_concept_uris_from_edges(
                conceptnet_term_edges
            )

        return conceptnet_external_uris

    def _filter_edges_on_sources(
        self, conceptnet_edges_obj: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter ConceptNet edges to keep only the ones coming from provided trusted sources.

        Parameters
        ----------
        conceptnet_edges_obj : List[Dict[str, Any]].
            The list of ConceptNet edge objects to be updated

        Returns
        -------
        List[Dict[str, Any]]
            List of selected ConceptNet edges.
        """

        selected_edges = []

        for edge_object in conceptnet_edges_obj:
            if edge_object["rel"].get("@id") == "/r/ExternalURL":
                if edge_object["end"]["site"] in self.validation_sources:
                    selected_edges.append(edge_object)

        return selected_edges

    def _get_concept_uris_from_edges(self, edges: List[Dict[str, Any]]) -> Set[str]:
        """Extract ConceptNet Node external URIs from a list of ConceptNet edges.

        Parameters
        ----------
        edges : List[str, Any]
            The list of ConceptNet "/r/ExternalURL" edges.

        Returns
        -------
        Set[str]
            The ConceptNet Node external URIs.
        """
        concept_uris = set()
        for edge_object in edges:
            if edge_object["rel"].get("@id") == "/r/ExternalURL":
                concept_uri = edge_object["end"].get("@id")
                if concept_uri is not None:
                    concept_uris.add(edge_object["end"].get("@id"))

        return concept_uris

    def _get_paginated_conceptnet_edges(
        self, conceptnet_view_res: Dict[str, str], batch_size: int
    ) -> List[Dict[str, Any]]:
        """Fetch paginated edges from the ConceptNet api. The api return results by batch.
            This method iterate over the batches to fetch all of them.

        Parameters
        ----------
        conceptnet_view_res : Dict[str, str]
            The "view" section of the ConceptNet api first results response.
            It contains the information to iterate over the result pages.

        Returns
        -------
        List[Dict[str, Any]]
            The list of fetched ConceptNet edge objects.
        """
        last_page = False
        page_count = 0

        paginated_edges = []

        while not last_page:
            page_count += 1

            next_page_url = (
                "http://api.conceptnet.io"
                + conceptnet_view_res.get("nextPage").split("?")[0]
                + f"?offset={page_count*batch_size}&limit={batch_size}"
            )

            conceptnet_res = requests.get(next_page_url, timeout=1000).json()

            paginated_edges.extend(conceptnet_res.get("edges", []))

            last_page = conceptnet_res["view"].get("nextPage") is None

        return paginated_edges

    def _conceptnet_api_fetch_term(
        self, term_conceptnet_text: str, lang: str, batch_size: int
    ) -> Dict[str, Any]:
        """Wrapper to hit the ConceptNet API.

        Parameters
        ----------
        term_conceptnet_text : str
            Term to fetch the ConceptNet data from (spaces are replaced by underscores).
        lang : str
            The term language.
        batch_size : int
            The number of edges to fetch.

        Returns
        -------
        Dict[str, Any]
            The ConceptNet API result.
        """
        term_conceptnet_url = f"http://api.conceptnet.io/c/{lang}/{term_conceptnet_text}?limit={batch_size}"
        conceptnet_term_res = requests.get(term_conceptnet_url, timeout=1000).json()

        return conceptnet_term_res

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError
