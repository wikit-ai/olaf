from typing import Any, Dict, List, Optional, Set

from commons.ontology_learning_repository import conceptnet_api_fetch_term, get_paginated_conceptnet_edges
from commons.ontology_learning_schema import CandidateTerm
from commons.ontology_learning_utils import space2underscoreStr
import config.logging_config as logging_config
from term_enrichment.term_enrichment_schema import ConceptNetEdgeData, ConceptNetTermData


class ConceptNetTermEnrichment:

    def __init__(self, options: Dict[str, Any]) -> None:
        """Initializer for a ConceptNet based term enrichment process.

        Parameters
        ----------
        options: Dict[str, Any]
            The specific parameters to setup the class.
        """
        self.options = options

        try:
            assert self.options["lang"] is not None
            assert self.options["api_resp_batch_size"] is not None
        except AssertionError as e:
            logging_config.logger.error(
                f"""Config information missing for WordNet term enrichment. Make sure you provided the configuration fields:
                    - term_enrichment.conceptnet.lang
                    - term_enrichment.conceptnet.api_resp_batch_size
                    Trace : {e}
                """)

    def enrich_candidate_term(self, candidate_term: CandidateTerm) -> None:
        """The method to enrich one candidate term.

        Parameters
        ----------
        candidate_term : CandidateTerm
            The candidate term to enrich.
        """
        term_conceptnet_text = space2underscoreStr(candidate_term.value)

        conceptnet_term_data = self._get_term_conceptnet_data(
            term_conceptnet_text)

        if conceptnet_term_data is not None:
            if (len(conceptnet_term_data.synonym_edges) > 0) or (len(conceptnet_term_data.formof_edges) > 0):
                candidate_term.synonyms.update(
                    self._get_term_synonyms_from_syn_edges(conceptnet_term_data.synonym_edges))
                candidate_term.synonyms.update(
                    self._get_term_synonyms_from_formof_edges(conceptnet_term_data.formof_edges))

            if len(conceptnet_term_data.isa_edges) > 0:
                candidate_term.hypernyms.update(
                    self._get_term_hypernyms_from_isa_edges(conceptnet_term_data.conceptnet_id, conceptnet_term_data.isa_edges))
                candidate_term.hyponyms.update(
                    self._get_term_hyponyms_from_isa_edges(conceptnet_term_data.conceptnet_id, conceptnet_term_data.isa_edges))

            if len(conceptnet_term_data.antonym_edges) > 0:
                candidate_term.antonyms.update(
                    self._get_term_antonyms_from_anto_edges(conceptnet_term_data.conceptnet_id, conceptnet_term_data.antonym_edges))

    def enrich_candidate_terms(self, candidate_terms: List[CandidateTerm]) -> None:
        """The main method of the class. It enriches a list of candidate terms.

        Parameters
        ----------
        candidate_terms : List[CandidateTerm]
            The list of candidate terms to enrich.
        """
        if self.options.get("term_max_tokens"):
            term_max_tokens = self.options.get("term_max_tokens")
            candidate_terms_2_process = [term for term in candidate_terms if len(
                term.value.split()) <= term_max_tokens]
        else:
            candidate_terms_2_process = candidate_terms

        for term in candidate_terms_2_process:
            self.enrich_candidate_term(term)

    def _get_term_conceptnet_data(self, term_conceptnet_text: str) -> Optional[ConceptNetTermData]:
        """Fetch term related data from conceptnet api.

        Parameters
        ----------
        term_conceptnet_text : str
            The conceptnet formated term string

        Returns
        -------
        Optional[ConceptNetTermData]
            The ConceptNet term related data embeded in a custom object.
        """
        conceptnet_term_edges = []

        conceptnet_term_res = conceptnet_api_fetch_term(
            term_conceptnet_text, self.options['lang'], self.options['api_resp_batch_size'])

        if "error" in conceptnet_term_res.keys():
            conceptnet_term_data = None  # the term is not in conceptnet data
            logging_config.logger.info(
                f"No match in ConceptNet for term {term_conceptnet_text.split('_')}")

        else:

            conceptnet_term_data = ConceptNetTermData(
                conceptnet_term_res["@id"])
            conceptnet_term_edges.extend(conceptnet_term_res.get('edges', []))

            if "view" in conceptnet_term_res.keys():  # get rest of the edges
                conceptnet_term_edges.extend(
                    get_paginated_conceptnet_edges(conceptnet_term_res["view"], self.options["api_resp_batch_size"]))

            conceptnet_term_edges = self._filter_edges(conceptnet_term_edges)

            for edge_data in conceptnet_term_edges:
                relation_key = edge_data.edge_rel_id

                if relation_key == "/r/Synonym":
                    conceptnet_term_data.synonym_edges.append(edge_data)

                elif relation_key == "/r/IsA":
                    conceptnet_term_data.isa_edges.append(edge_data)

                elif relation_key == "/r/FormOf":
                    conceptnet_term_data.formof_edges.append(edge_data)

                elif relation_key == "/r/Antonym":
                    conceptnet_term_data.antonym_edges.append(edge_data)

        return conceptnet_term_data

    def _filter_edges(self, conceptnet_edges_obj: List[Dict[str, Any]]) -> List[ConceptNetEdgeData]:
        """Filter conceptnet edges to keep only the useful ones.

        Parameters
        ----------
        conceptnet_edges_obj : List[Dict[str, Any]]
            The list of conceptnet edge objects to be updated

        Returns
        -------
        ConceptNetEdgeData
            List of selected conceptnet edges
        """

        lang = self.options['lang']
        relations_2_keep = {"/r/Synonym", "/r/IsA", "/r/FormOf", "/r/Antonym"}
        selected_edges = []

        for edge_object in conceptnet_edges_obj:

            conditions = [
                edge_object["end"].get("language") == lang,
                edge_object["start"].get("language") == lang,
                edge_object["rel"].get("@id") in relations_2_keep
            ]

            if all(conditions):
                selected_edges.append(ConceptNetEdgeData(
                    edge_rel_id=edge_object["rel"].get("@id"),
                    end_node_concept_id=edge_object["end"].get("term"),
                    start_node_concept_id=edge_object["start"].get("term"),
                    end_node_label=edge_object["end"].get("label"),
                    end_node_lang=edge_object["end"].get("language"),
                    end_node_sense_label=edge_object["end"].get("sense_label"),
                    start_node_label=edge_object["start"].get("label"),
                    start_node_lang=edge_object["start"].get("language"),
                    start_node_sense_label=edge_object["start"].get(
                        "sense_label")
                ))

        return selected_edges

    def _get_term_synonyms_from_syn_edges(self, conceptnet_syn_edges: List[ConceptNetEdgeData]) -> Set[str]:
        """Private method to extract the synonyms (strings) from conceptnet term related data.
            This method focuses on "/r/Synonym" conceptnet relation.

        Parameters
        ----------
        conceptnet_syn_edges: List[ConceptNetEdgeData]
            The term data extracted from conceptnet.

        Returns
        -------
        Set[str]
            The set of synonyms strings.
        """
        term_synonyms = set()

        for edge in conceptnet_syn_edges:
            conditions = [
                edge.end_node_sense_label == "n, artifact",
                edge.start_node_sense_label == "n, artifact"
            ]

            if all(conditions):

                term_synonyms.add(edge.start_node_label)
                term_synonyms.add(edge.end_node_label)

        return term_synonyms

    def _get_term_synonyms_from_formof_edges(self, conceptnet_formof_edges: List[ConceptNetEdgeData]) -> Set[str]:
        """Private method to extract the synonyms (strings) from conceptnet term related data.
            This method focuses on "/r/FormOf" conceptnet relation.

        Parameters
        ----------
        conceptnet_formof_edges: List[ConceptNetEdgeData]
            The term data extracted from conceptnet.

        Returns
        -------
        Set[str]
            The set of synonyms strings.
        """
        term_synonyms = set()

        for edge in conceptnet_formof_edges:

            if edge.end_node_sense_label == "n":
                term_synonyms.add(edge.start_node_label)
                term_synonyms.add(edge.end_node_label)

        return term_synonyms

    def _get_term_hypernyms_from_isa_edges(self, term_conceptnet_id: str, conceptnet_isa_edges: List[ConceptNetEdgeData]) -> Set[str]:
        """Private method to extract the hypernyms (strings) from conceptnet term related data.
            This method focuses on "/r/IsA" conceptnet relation.

        Parameters
        ----------
        term_conceptnet_id: str
            The term ConceptNet ID
        conceptnet_isa_edges: List[ConceptNetEdgeData]
            The term data extracted from conceptnet.

        Returns
        -------
        Set[str]
            The set of hypernyms strings.
        """

        term_hypernyms = set()

        for edge in conceptnet_isa_edges:
            if edge.start_node_concept_id == term_conceptnet_id:
                term_hypernyms.add(edge.end_node_label)

        return term_hypernyms

    def _get_term_hyponyms_from_isa_edges(self, term_conceptnet_id: str, conceptnet_isa_edges: List[ConceptNetEdgeData]) -> Set[str]:
        """Private method to extract the hyponyms (strings) from conceptnet term related data.
            This method focuses on "/r/IsA" conceptnet relation.

        Parameters
        ----------
        term_conceptnet_id: str
            The term ConceptNet ID
        conceptnet_isa_edges: List[ConceptNetEdgeData]
            The term data extracted from conceptnet.

        Returns
        -------
        Set[str]
            The set of hyponyms strings.
        """

        term_hyponyms = set()

        for edge in conceptnet_isa_edges:
            if edge.end_node_concept_id == term_conceptnet_id:
                term_hyponyms.add(edge.start_node_label)

        return term_hyponyms

    def _get_term_antonyms_from_anto_edges(self, term_conceptnet_id: str, conceptnet_anto_edges: List[ConceptNetEdgeData]) -> Set[str]:
        """Private method to extract the antonyms (strings) from conceptnet term related data.
            This method focuses on "/r/Antonym" conceptnet relation.

        Parameters
        ----------
        term_conceptnet_id: str
            The term ConceptNet ID
        conceptnet_anto_edges: List[ConceptNetEdgeData]
            The term data extracted from conceptnet.

        Returns
        -------
        Set[str]
            The set of antonyms strings.
        """
        term_antonyms = set()

        for edge in conceptnet_anto_edges:
            if edge.start_node_concept_id == term_conceptnet_id:
                term_antonyms.add(edge.end_node_label)

        return term_antonyms
