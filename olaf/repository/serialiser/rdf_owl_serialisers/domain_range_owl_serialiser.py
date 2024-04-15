from typing import Dict, Set

from rdflib import Literal
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace

from ....data_container.metarelation_schema import Metarelation
from ....data_container.relation_schema import Relation

from .base_owl_serialiser import BaseOWLSerialiser


class DomainRangeOWLSerialiser(BaseOWLSerialiser):
    """
    Serialize KnowledgeRepresentation data into an RDF graph following the OWL language.

    This class provides methods to build an RDF graph following the OWL language from a
    KnowledgeRepresentation (KR) instance and export it in various RDF formats. It allows users to
    represent concepts, relations, and metarelations as RDF triples based on a specified base URI.
    This serialiser makes sure relations have source and destination concepts as domain and range.

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
    - All relations will have the source concept class as RDFS domain
    - All relations will have the destination concept class as RDFS range
    - All metarelations will be considered OWL object properties unless they have a mapping
        specified in the metarelation_map attribute.
    """

    def __init__(
        self,
        base_uri: str,
        metarelation_map: Dict[str, Namespace] = None,
        keep_all_labels: bool = True,
    ) -> None:
        super().__init__(base_uri, metarelation_map, keep_all_labels)

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

            self._graph.add((rel_uri, RDF.type, OWL.ObjectProperty))
            self._graph.add(
                (rel_uri, RDFS.label, Literal(rel.label, datatype=XSD.string))
            )
            self._graph.add((rel_uri, RDFS.domain, src_concept_uri))
            self._graph.add((rel_uri, RDFS.range, dest_concept_uri))

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
                self._graph.add((rel_uri, RDFS.domain, src_concept_uri))
                self._graph.add((rel_uri, RDFS.range, dest_concept_uri))
