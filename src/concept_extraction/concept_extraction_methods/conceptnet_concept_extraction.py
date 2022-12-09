from typing import Any, Dict, List, Set
from uuid import uuid4

from tqdm import tqdm

from commons.ontology_learning_repository import conceptnet_api_fetch_term, get_paginated_conceptnet_edges
from commons.ontology_learning_schema import CandidateTerm, Concept, KR
from commons.ontology_learning_utils import space2underscoreStr
import config.logging_config as logging_config


class ConceptNetConceptExtraction:

    def __init__(self, candidate_terms: List[CandidateTerm], kr: KR, options: Dict[str, Any]) -> None:
        """Initializer for a ConceptNet based concept extraction process.

        Parameters
        ----------
        candidate_terms: List[CandidateTerm]
            The list of candidate terms to extract the concepts from.
        options: Dict[str, Any]
            The specific parameters to setup the class.
        """
        self.candidate_terms = candidate_terms
        self.options = options
        self.kr = kr

        try:
            assert self.options["lang"] is not None
            assert self.options["api_resp_batch_size"] is not None
            assert self.options["validation_sources"] is not None
            assert isinstance(self.options["validation_sources"], list)
        except AssertionError as e:
            logging_config.logger.error(
                f"""Config information missing or wrong for ConceptNet extraction. Make sure you provided the right configuration fields:
                    - concept_extraction.conceptnet.lang: str
                    - concept_extraction.conceptnet.api_resp_batch_size: int
                    - concept_extraction.conceptnet.validation_sources: List[str]
                    Trace : {e}
                """)

    def __call__(self) -> None:
        """Extract the concepts from a list of candidate terms.
            Directly update the KR instance.
        """
        concepts = self.extract_concepts_from_terms(self.candidate_terms)
        self.kr.concepts.update(concepts)

    def extract_concepts_from_terms(self, terms: List[CandidateTerm]) -> List[Concept]:
        """Concept extraction from a list of candidate terms.

        Parameters
        ----------
        terms : List[CandidateTerm]
            The list of candidate terms to extract the concepts from.

        Returns
        -------
        List[Concept]
            The list of extracted concepts
        """
        concepts = []

        # "pre processing"
        if self.options.get("merge_candidate_terms_on_syns", False):
            terms = self._merge_terms_on_synonyms(terms)

        for term in tqdm(terms):
            term_external_ids = self._get_candidate_term_conceptnet_external_uris(
                term)
            if len(term_external_ids) > 0:
                new_concept = Concept(
                    uid=str(uuid4()),
                    terms=term.synonyms,
                    external_uris=term_external_ids
                )
                # don't forget to add the term value
                new_concept.terms.add(term.value)
                concepts.append(new_concept)

        # "post processing"
        if self.options.get("merge_concepts_on_external_ids", False):
            concepts = self._merge_concepts_on_external_ids(concepts)

        return concepts

    def _merge_terms_on_synonyms(self, terms: List[CandidateTerm]) -> List[CandidateTerm]:
        """Preprocessing candidate terms by merging them if they have synonyms in common.

        Parameters
        ----------
        terms : List[CandidateTerm]
            The list of candidate terms to process.

        Returns
        -------
        List[CandidateTerm]
            The list of new candidate terms resulting from the potential merges
        """
        merged_terms = terms
        merge_occured = True

        # we loop in undefinde number of times because merging two candidate terms means merging their synonyms
        # and therefore might result in missed merges
        while merge_occured:
            temp_merged_terms = []

            for term_idx, term_1 in enumerate(merged_terms):
                for term_2 in merged_terms[term_idx+1:]:
                    syns_and_value_term_1 = term_1.synonyms.union(
                        {term_1.value})
                    syns_and_value_term_2 = term_2.synonyms.union(
                        {term_2.value})
                    if not syns_and_value_term_1.isdisjoint(syns_and_value_term_2):
                        temp_merged_terms.append(
                            self._merge_candidate_terms(term_1, term_2))

            if len(temp_merged_terms) > 0:
                merged_terms = temp_merged_terms
            else:
                merge_occured = False

        return merged_terms

    def _merge_candidate_terms(self, term_1: CandidateTerm, term_2: CandidateTerm) -> CandidateTerm:
        """Merge two candidate terms to create one. 
            We arbitrarily chose the first candidate term value to become the new candidate term value.
            The second candidate term value is added to the new candidate terms synonyms.
            Synonmys, hypernyms, hyponyms and antonyms are merged.

        Parameters
        ----------
        term_1 : CandidateTerm
            The first candidate term
        term_2 : CandidateTerm
            The second candidate term

        Returns
        -------
        CandidateTerm
            The new candidate term resulting from the merge
        """
        new_candidate_term = CandidateTerm(
            value=term_1.value,  # arbitrary choice
            synonyms=term_1.synonyms.union(term_2.synonyms),
            antonyms=term_1.antonyms.union(term_2.antonyms),
            hypernyms=term_1.hypernyms.union(term_2.hypernyms),
            hyponyms=term_1.hyponyms.union(term_2.hyponyms)
        )
        # don't forget the other term
        new_candidate_term.synonyms.add(term_2.value)

        logging_config.logger.info(
            f"Candidate term {term_1.value} and {term_2.value} have been merged")

        return new_candidate_term

    def _merge_concepts_on_external_ids(self, candidate_concepts: List[Concept]) -> List[Concept]:
        """Post processing merging found concepts if they have external URIs in common.

        Parameters
        ----------
        candidate_concepts : List[Concept]
            The concepts to process

        Returns
        -------
        List[Concept]
            The list of new concepts resulting from the potential merges.
        """
        merged_concepts = candidate_concepts
        merge_occured = True

        while merge_occured:
            temp_merged_concepts = []

            for concept_idx, concept_1 in enumerate(merged_concepts):
                for concept_2 in merged_concepts[concept_idx+1:]:
                    if not concept_1.external_uris.isdisjoint(concept_2.external_uris):
                        temp_merged_concepts.append(
                            self._merge_concepts(concept_1, concept_2))

            if len(temp_merged_concepts) > 0:
                merged_concepts = temp_merged_concepts
            else:
                merge_occured = False

        return merged_concepts

    def _merge_concepts(self, concept_1: Concept, concept_2: Concept) -> Concept:
        """Merge two concepts.

        Parameters
        ----------
        concept_1 : Concept
            The first concept.
        concept_2 : Concept
            The second concept

        Returns
        -------
        Concept
            The concept resulting from the merge.
        """
        new_concept = Concept(
            uid=str(uuid4()),
            terms=concept_1.terms.union(concept_2.terms),
            external_uris=concept_1.external_uris.union(
                concept_2.external_uris)
        )
        return new_concept

    def _get_candidate_term_conceptnet_external_uris(self, candidate_term: CandidateTerm) -> Set[str]:
        """Fetch from ConceptNet the external URIs related to a candidate term.

        Parameters
        ----------
        candidate_term : CandidateTerm
            The candidate term to process.

        Returns
        -------
        Set[str]
            The set of external URIs related to the candidate term.
        """
        candidate_term_external_ids = self._get_term_conceptnet_external_uris(
            space2underscoreStr(candidate_term.value))

        if self.options.get("use_synonyms"):
            for synonym in candidate_term.synonyms:
                candidate_term_external_ids.update(self._get_term_conceptnet_external_uris(
                    space2underscoreStr(synonym)))

        return candidate_term_external_ids

    def _get_term_conceptnet_external_uris(self, term_conceptnet_text: str) -> Set[str]:
        """Fetch term related data from conceptnet api and extract the term related external URIs.

        Parameters
        ----------
        term_conceptnet_text : str
            The conceptnet formated term string

        Returns
        -------
        Set[str]
            The ConceptNet term related external uris.
        """
        conceptnet_external_ids = {}
        conceptnet_term_edges = []

        conceptnet_term_res = conceptnet_api_fetch_term(
            term_conceptnet_text, self.options['lang'], self.options['api_resp_batch_size'])

        if "error" in conceptnet_term_res.keys():  # the term is not in conceptnet data
            logging_config.logger.info(
                f"No match in ConceptNet for term {term_conceptnet_text.replace('_', ' ')}")

        else:

            conceptnet_term_edges = conceptnet_term_res.get('edges', [])

            if "view" in conceptnet_term_res.keys():  # get rest of the edges
                conceptnet_term_edges.extend(
                    get_paginated_conceptnet_edges(conceptnet_term_res["view"], self.options["api_resp_batch_size"]))

            conceptnet_term_edges = self._filter_edges(conceptnet_term_edges)

            conceptnet_external_ids = self._get_concept_ids_from_edges(
                conceptnet_term_edges)

        return conceptnet_external_ids

    def _filter_edges(self, conceptnet_edges_obj: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter conceptnet edges to keep only the useful ones.

        Parameters
        ----------
        conceptnet_edges_obj : List[Dict[str, Any]]
            The list of conceptnet edge objects to be updated

        Returns
        -------
        List[Dict[str, Any]]
            List of selected conceptnet edges
        """

        selected_edges = []

        for edge_object in conceptnet_edges_obj:

            if edge_object["rel"].get("@id") == "/r/ExternalURL":
                conditions = [
                    source == edge_object["end"]["site"] for source in self.options["validation_sources"]
                ]
                if any(conditions):
                    selected_edges.append(edge_object)

        return selected_edges

    def _get_concept_ids_from_edges(self, edges: List[Dict[str, Any]]) -> Set[str]:
        """Extraxt conceptnet Node external URLs from a list of "/r/ExternalURL" edges.

        Parameters
        ----------
        edges : List[str, Any]
            The list of conceptnet "/r/ExternalURL" edges

        Returns
        -------
        Set[str]
            conceptnet Node external URLs
        """
        concept_ids = {
            edge["end"].get("@id") for edge in edges
        }

        return concept_ids
