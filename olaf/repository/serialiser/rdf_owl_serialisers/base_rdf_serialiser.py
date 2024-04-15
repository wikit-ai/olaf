import urllib.parse
from typing import Set

from rdflib import Graph, URIRef

from ....commons.logging_config import logger
from ....data_container.concept_schema import Concept
from ....data_container.knowledge_representation_schema import KnowledgeRepresentation
from ....data_container.metarelation_schema import Metarelation
from ....data_container.relation_schema import Relation


class BaseRDFserialiser:
    """
    Serialize KnowledgeRepresentation data into RDF graph.

    This class provides methods to build an RDF graph from a KnowledgeRepresentation (KR) instance
    and export it in various RDF formats. It allows users to represent concepts, relations, and
    metarelations as RDF triples based on a specified base URI.

    Attributes
    ----------
    base_uri : URIRef
        The base URI to be used for constructing RDF URIs.
    graph : Graph
        An RDFlib Graph for storing the RDF triples.
    """

    def __init__(self, base_uri: str) -> None:
        """
        Initialize an BaseRDFserialiser instance.

        Parameters
        ----------
        base_uri : str
            The base URI to be used for constructing RDF URIs.
        """
        self.base_uri: URIRef = URIRef(base_uri)
        self._graph: Graph = None

    @property
    def graph(self) -> Graph:
        """
        Get the RDF graph. If not built yet, it will issue a warning.

        Returns
        -------
        Graph
            An RDFlib Graph containing RDF triples.
        """
        if not self._graph:
            logger.warning(
                """The graph has not been built yet. Did you forget to build it?
                >>> my_serialiser.build_graph(kr)
                """
            )
            self._graph = Graph()

        return self._graph

    def build_concept_uri(self, concept: Concept) -> URIRef:
        """
        Build a URI for a concept based on its label.

        Parameters
        ----------
        concept : Concept
            The concept instance for which to build a URI.

        Returns
        -------
        URIRef
            The constructed URI for the concept.
        """
        concept_label = "_".join(concept.label.lower().split())
        concept_uri = self.base_uri + URIRef(urllib.parse.quote(concept_label))
        return concept_uri

    def build_relation_uri(self, relation: Relation) -> URIRef:
        """
        Build a URI for a relation based on its label.

        Parameters
        ----------
        relation : Relation
            The relation instance for which to build a URI.

        Returns
        -------
        URIRef
            The constructed URI for the relation.
        """
        relation_label = "_".join(relation.label.lower().split())
        relation_uri = self.base_uri + URIRef(urllib.parse.quote(relation_label))
        return relation_uri

    def build_metarelation_uri(self, metarelation: Metarelation) -> URIRef:
        """
        Build a URI for a metarelation based on its label.

        Parameters
        ----------
        metarelation : Metarelation
            The metarelation instance for which to build a URI.

        Returns
        -------
        URIRef
            The constructed URI for the metarelation.
        """
        metarelation_label = "_".join(metarelation.label.lower().split())
        metarelation_uri = self.base_uri + URIRef(
            urllib.parse.quote(metarelation_label)
        )
        return metarelation_uri

    def _add_relation_triples(self, relations: Set[Relation]) -> None:
        """
        Add RDF triples for relations to the RDF graph.

        Parameters
        ----------
        relations : Set[Relation]
            The set of relation instances to add as RDF triples.
        """
        for rel in relations:
            rel_uri = self.build_relation_uri(rel)
            src_concept_uri = self.build_concept_uri(rel.source_concept)
            dest_concept_uri = self.build_concept_uri(rel.destination_concept)

            self._graph.add((src_concept_uri, rel_uri, dest_concept_uri))

    def _add_metarelation_triples(self, metarelations: Set[Metarelation]) -> None:
        """
        Add RDF triples for metarelations to the RDF graph.

        Parameters
        ----------
        metarelations : Set[Metarelation]
            The set of metarelation instances to add as RDF triples.
        """
        for rel in metarelations:
            rel_uri = self.build_metarelation_uri(rel)
            src_concept_uri = self.build_concept_uri(rel.source_concept)
            dest_concept_uri = self.build_concept_uri(rel.destination_concept)

            self._graph.add((src_concept_uri, rel_uri, dest_concept_uri))

    def build_graph(self, kr: KnowledgeRepresentation) -> None:
        """
        Build the RDF graph from a KnowledgeRepresentation instance.

        Parameters
        ----------
        kr : KnowledgeRepresentation
            The KnowledgeRepresentation instance containing concepts, relations, and metarelations.
        """
        self._graph = Graph()

        self._add_relation_triples(kr.relations)
        self._add_metarelation_triples(kr.metarelations)

    def export_graph(self, file_path: str, rdf_format: str = "turtle") -> None:
        """
        Export the RDF graph to a specified file path and RDF format.
        Available formats are the ones defined in RDFlib,
        see: <https://rdflib.readthedocs.io/en/stable/plugin_serializers.html>

        Parameters
        ----------
        file_path : str
            The file path where the RDF graph will be exported.
        rdf_format : str, optional
            The RDF serialization format, e.g., 'turtle', 'xml', 'json-ld', by default 'turtle'.
        """
        if not self._graph:
            logger.warning(
                """The graph has not been built yet. The exported graph will be empty. 
                Did you forget to build it? >>> my_serialiser.build_graph(kr)
                """
            )
            self._graph = Graph()

        self._graph.serialize(destination=file_path, format=rdf_format)
