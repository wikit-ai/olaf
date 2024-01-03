import urllib.parse
from typing import Dict, Optional, Set

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD, Namespace

from ....commons.logging_config import logger
from ....data_container.concept_schema import Concept
from ....data_container.knowledge_representation_schema import \
    KnowledgeRepresentation
from ....data_container.metarelation_schema import Metarelation
from ....data_container.relation_schema import Relation


class BaseOWLSerialiser:
    """
    Serialize KnowledgeRepresentation data into an RDF graph following the OWL language.

    This class provides methods to build an RDF graph following the OWL language from a
    KnowledgeRepresentation (KR) instance and export it in various RDF formats. It allows users to
    represent concepts, relations, and metarelations as RDF triples based on a specified base URI.

    Attributes
    ----------
    base_uri : URIRef
        The base URI to be used for constructing RDF URIs.
    graph : Graph
        An RDFlib Graph for storing the RDF triples.
    metarelation_map : Dict[str, Namespace], optional
        A dictionary mapping metarelation labels to RDF namespaces, by default None.
        If not provided, it uses a default mapping for "is_generalised_by" metarelations.
    keep_all_labels : bool, optional
        A boolean indicating whether to to include all labels for concepts and their different
        linguistic realisations as SKOS altLabels, by default True.

    Notes
    -----
    This class is designed to serialize KR data into RDF following the OWL language as follow:
    - All concepts will be considered OWL classes.
    - All relations will be considered OWL object properties.
    - All metarelations will be considered OWL object properties unless they have a mapping
        specified in the metarelation_map attribute.
    """

    def __init__(
        self,
        base_uri: str,
        metarelation_map: Optional[Dict[str, Namespace]] = None,
        keep_all_labels: Optional[bool] = True,
    ) -> None:
        """
        Initialize a BaseOWLSerialiser instance.

        Parameters
        ----------
        base_uri : str
            The base URI to be used for constructing RDF URIs.
        metarelation_map : Dict[str, Namespace], optional
            A dictionary mapping metarelation labels to RDF namespaces, by default None.
            If not provided, it uses a default mapping for "is_generalised_by" metarelations.
        keep_all_labels : bool, optional
            A boolean indicating whether to include all labels as SKOS altLabels, by default True.
        """
        self.base_uri: URIRef = URIRef(base_uri)
        self.metarelation_map = (
            metarelation_map
            if metarelation_map is not None
            else {"is_generalised_by": RDFS.subClassOf}
        )
        self.keep_all_labels = keep_all_labels
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
        concept_label = "".join(
            [token.capitalize() for token in concept.label.lower().split()]
        )
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
        relation_label = "".join(
            [token.capitalize() for token in relation.label.lower().split()]
        )
        relation_label = relation_label[0].lower() + relation_label[1:]

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

        metarelation_uri = self.metarelation_map.get(metarelation.label)

        if metarelation_uri is None:
            metarelation_label = "".join(
                [token.capitalize() for token in metarelation.label.lower().split()]
            )
            metarelation_label = metarelation_label[0].lower() + metarelation_label[1:]
            metarelation_uri = self.base_uri + URIRef(
                urllib.parse.quote(metarelation_label)
            )

        return metarelation_uri

    def _add_concept_triples(self, concepts: Set[Concept]) -> None:
        """
        Add RDF triples for concepts to the RDF graph.
        This method consider each concept an OWL class and will create the triples accordingly.
        It will also add the concept label as OWL class label.

        Parameters
        ----------
        concepts : Set[Concept]
            The set of concept instances to add as RDF triples.
        """
        for concept in concepts:
            concept_uri = self.build_concept_uri(concept)

            self._graph.add((concept_uri, RDF.type, OWL.Class))
            self._graph.add(
                (concept_uri, RDFS.label, Literal(concept.label, datatype=XSD.string))
            )

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

            self._graph.add((rel_uri, RDF.type, OWL.ObjectProperty))
            self._graph.add(
                (rel_uri, RDFS.label, Literal(rel.label, datatype=XSD.string))
            )

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

            if rel.label in self.metarelation_map:
                src_concept_uri = self.build_concept_uri(rel.source_concept)
                dest_concept_uri = self.build_concept_uri(rel.destination_concept)

                self._graph.add((src_concept_uri, rel_uri, dest_concept_uri))

            else:
                self._graph.add((rel_uri, RDF.type, OWL.ObjectProperty))
                self._graph.add(
                    (rel_uri, RDFS.label, Literal(rel.label, datatype=XSD.string))
                )

    def _add_concepts_labels(self, concepts: Set[Concept]) -> None:
        """
        Add SKOS altLabels for concepts to the RDF graph.

        Parameters
        ----------
        concepts : Set[Concept]
            The set of concept instances to add as SKOS altLabels.
        """
        for concept in concepts:
            concept_uri = self.build_concept_uri(concept)
            for lr in concept.linguistic_realisations:
                self._graph.add(
                    (
                        concept_uri,
                        SKOS.altLabel,
                        Literal(lr.label, datatype=XSD.string),
                    )
                )

    def _add_relations_labels(self, relations: Set[Relation]) -> None:
        """
        Add SKOS altLabels for relations to the RDF graph.

        Parameters
        ----------
        relations : Set[Relation]
            The set of relation instances to add as SKOS altLabels.
        """
        for rel in relations:
            rel_uri = self.build_relation_uri(rel)
            for lr in rel.linguistic_realisations:
                self._graph.add(
                    (
                        rel_uri,
                        SKOS.altLabel,
                        Literal(lr.label, datatype=XSD.string),
                    )
                )

    def build_graph(self, kr: KnowledgeRepresentation) -> None:
        """
        Build the RDF graph from a KnowledgeRepresentation instance.

        Parameters
        ----------
        kr : KnowledgeRepresentation
            The KnowledgeRepresentation instance containing concepts, relations, and metarelations.
        """
        self._graph = Graph()

        self._add_concept_triples(kr.concepts)
        self._add_relation_triples(kr.relations)
        self._add_metarelation_triples(kr.metarelations)

        if self.keep_all_labels:
            self._add_concepts_labels(kr.concepts)
            self._add_relations_labels(kr.relations)

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
            The RDF serialization format, e.g., 'turtle', 'xml', 'json-ld'. Defaults to 'turtle'.
        """
        if not self._graph:
            logger.warning(
                """The graph has not been built yet. The exported graph will be empty. 
                Did you forget to build it? >>> my_serialiser.build_graph(kr)
                """
            )
            self._graph = Graph()

        self._graph.serialize(destination=file_path, format=rdf_format)