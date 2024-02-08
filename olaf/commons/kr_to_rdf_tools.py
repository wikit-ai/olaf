import urllib.parse
from collections import defaultdict
from itertools import product
from typing import Tuple

from rdflib import OWL, RDF, RDFS, BNode, Graph, URIRef
from rdflib.collection import Collection

from ..data_container import KnowledgeRepresentation
from ..data_container.metarelation_schema import METARELATION_RDFS_OWL_MAP


def owl_class_uri(label: str, base_uri: URIRef) -> URIRef:
    """Build an OWL class URI.

    Parameters
    ----------
    label : str
        The label to use in the URI.
    base_uri : URIRef
        The base URI to use.

    Returns
    -------
    URIRef
        The OWL class URI.
    """
    class_label = "".join([token.capitalize() for token in label.lower().split()])
    concept_uri = base_uri + URIRef(urllib.parse.quote(class_label))

    return concept_uri


def owl_obj_prop_uri(label: str, base_uri: URIRef) -> URIRef:
    """Build an OWL object property URI.

    Parameters
    ----------
    label : str
        The label to use in the URI.
    base_uri : URIRef
        The base URI to use.

    Returns
    -------
    URIRef
        The OWL object property URI.
    """
    relation_label = "".join([token.capitalize() for token in label.lower().split()])
    relation_label = relation_label[0].lower() + relation_label[1:]
    relation_uri = base_uri + URIRef(urllib.parse.quote(relation_label))

    return relation_uri


def owl_instance_uri(label: str, base_uri: URIRef) -> URIRef:
    """Build an OWL named instance URI.

    Parameters
    ----------
    label : str
        The label to use in the URI.
    base_uri : URIRef
        The base URI to use.

    Returns
    -------
    URIRef
        The OWL named instance URI.
    """
    instance_label = "".join([token.capitalize() for token in label.lower().split()])
    instance_label = "_" + instance_label[0].lower() + instance_label[1:]
    instance_uri = base_uri + URIRef(urllib.parse.quote(instance_label))

    return instance_uri


def kr_concepts_to_owl_classes(kr: KnowledgeRepresentation, base_uri: URIRef) -> Graph:
    """Create the RDF triples corresponding to making each KR concepts an OWL class.

    Parameters
    ----------
    kr : KnowledgeRepresentation
        The Knowledge Representation containing the concepts.
    base_uri : URIRef
        The base URI to use when creating the class URIs.

    Returns
    -------
    Graph
        The constructed RDF triples.
    """
    rdf_graph = Graph()

    for concept in kr.concepts:
        concept_uri = owl_class_uri(label=concept.label, base_uri=base_uri)
        rdf_graph.add((concept_uri, RDF.type, OWL.Class))

    return rdf_graph


def kr_relations_to_owl_obj_props(
    kr: KnowledgeRepresentation, base_uri: URIRef
) -> Graph:
    """Create the RDF triples corresponding to making each KR relations an OWL object property.

    Parameters
    ----------
    kr : KnowledgeRepresentation
        The Knowledge Representation containing the concepts.
    base_uri : URIRef
        The base URI to use when creating the class URIs.

    Returns
    -------
    Graph
        The constructed RDF triples.
    """
    rdf_graph = Graph()

    for relation in kr.relations:
        rel_uri = owl_obj_prop_uri(label=relation.label, base_uri=base_uri)
        rdf_graph.add((rel_uri, RDF.type, OWL.ObjectProperty))

    return rdf_graph


def kr_metarelations_to_owl(
    kr: KnowledgeRepresentation, base_uri: URIRef
) -> Graph:
    """Create the RDF triples corresponding to mapping the KR metarelations with OWL vocabulary.

    The mapping depends on the providing dictionary.
    The KR metarelations not matching any keys in the mapping dictionary is created as an OWL object property.

    Parameters
    ----------
    kr : KnowledgeRepresentation
        The Knowledge Representation containing the concepts.
    base_uri : URIRef
        The base URI to use when creating the class URIs.

    Returns
    -------
    Graph
        The constructed RDF triples.
    """

    rdf_graph = Graph()

    for relation in kr.metarelations:

        rel_uri = METARELATION_RDFS_OWL_MAP.get(relation.label)

        if rel_uri is not None:
            src_concept_uri = owl_class_uri(
                label=relation.source_concept.label, base_uri=base_uri
            )
            dest_concept_uri = owl_class_uri(
                label=relation.destination_concept.label, base_uri=base_uri
            )

            rdf_graph.add((src_concept_uri, rel_uri, dest_concept_uri))

        else:
            rel_uri = owl_obj_prop_uri(label=relation.label, base_uri=base_uri)
            rdf_graph.add((rel_uri, RDF.type, OWL.ObjectProperty))

    return rdf_graph


def kr_relations_to_domain_range_obj_props(
    kr: KnowledgeRepresentation, base_uri: URIRef
) -> Graph:
    """Create the RDF triples corresponding to making each KR relations OWL object properties with domain and range
    their source and destination concepts.

    Source and destination concepts will be created as OWL classes.

    Parameters
    ----------
    kr : KnowledgeRepresentation
        The Knowledge Representation containing the concepts.
    base_uri : URIRef
        The base URI to use when creating the class URIs.

    Returns
    -------
    Graph
        The constructed RDF triples.
    """
    rdf_graph = Graph()

    for relation in kr.relations:
        rel_uri = owl_obj_prop_uri(label=relation.label, base_uri=base_uri)
        rdf_graph.add((rel_uri, RDF.type, OWL.ObjectProperty))

        if relation.source_concept:
            src_concept_uri = owl_class_uri(
                label=relation.source_concept.label, base_uri=base_uri
            )
            rdf_graph.add((rel_uri, RDFS.domain, src_concept_uri))
            rdf_graph.add((src_concept_uri, RDF.type, OWL.Class))

        if relation.destination_concept:
            dest_concept_uri = owl_class_uri(
                label=relation.destination_concept.label, base_uri=base_uri
            )
            rdf_graph.add((rel_uri, RDFS.range, dest_concept_uri))
            rdf_graph.add((dest_concept_uri, RDF.type, OWL.Class))

    return rdf_graph


def kr_concepts_to_disjoint_classes(
    kr: KnowledgeRepresentation, base_uri: URIRef
) -> Graph:
    """Create the RDF triples corresponding to making each KR concepts an OWL class and making each classes disjoint.

    Parameters
    ----------
    kr : KnowledgeRepresentation
        The Knowledge Representation containing the concepts.
    base_uri : URIRef
        The base URI to use when creating the class URIs.

    Returns
    -------
    Graph
        The constructed RDF triples.
    """
    rdf_graph = Graph()

    concept_uris = set()

    for concept in kr.concepts:
        concept_uri = owl_class_uri(label=concept.label, base_uri=base_uri)
        concept_uris.add(concept_uri)
        rdf_graph.add((concept_uri, RDF.type, OWL.Class))

    rdf_collection = Collection(graph=rdf_graph, uri=BNode(), seq=list(concept_uris))
    b_node = BNode()

    rdf_graph.add((b_node, RDF.type, OWL.AllDisjointClasses))
    rdf_graph.add((b_node, OWL.members, rdf_collection.uri))

    return rdf_graph

def create_obj_prop_some_restriction_triples(
        rel_uri: URIRef, dest_concept_uri: URIRef
    ) -> Tuple[URIRef, Graph]:
    """Create the triples corresponding to an existential OWL property restriction part
    of the graph.

    Parameters
    ----------
    rel_uri : URIRef
        The URI or the relation the OWL property restriction is focusing on.
    dest_concept_uri : URIRef
        The URI of the concept (i.e., OWL class) involved in the OWL property restriction.

    Returns
    -------
    Tuple[URIRef, Graph]
        The blank node ID origin of the OWL property restriction and the corresponding graph.
    """
    obj_prop_restriction_g = Graph()

    b_node = BNode()
    obj_prop_restriction_g.add((b_node, RDF.type, OWL.Restriction))
    obj_prop_restriction_g.add((b_node, OWL.onProperty, rel_uri))
    obj_prop_restriction_g.add((b_node, OWL.someValuesFrom, dest_concept_uri))

    return b_node, obj_prop_restriction_g

def create_obj_prop_all_restriction_triples(
        rel_uri: URIRef, dest_concept_uri: URIRef
    ) -> Tuple[URIRef, Graph]:
    """Create the triples corresponding to an universal OWL property restriction part
    of the graph.

    Parameters
    ----------
    rel_uri : URIRef
        The URI or the relation the OWL property restriction is focusing on.
    dest_concept_uri : URIRef
        The URI of the concept (i.e., OWL class) involved in the OWL property restriction.

    Returns
    -------
    Tuple[URIRef, Graph]
        The blank node ID origin of the OWL property restriction and the corresponding graph.
    """
    obj_prop_restriction_g = Graph()

    b_node = BNode()
    obj_prop_restriction_g.add((b_node, RDF.type, OWL.Restriction))
    obj_prop_restriction_g.add((b_node, OWL.onProperty, rel_uri))
    obj_prop_restriction_g.add((b_node, OWL.allValuesFrom, dest_concept_uri))

    return b_node, obj_prop_restriction_g

def kr_relations_to_anonymous_some_parent(
    kr: KnowledgeRepresentation, base_uri: URIRef
) -> Graph:
    """Create RDF triples corresponding to saying in plain english:
        'Each source concept is A SUBSET OF the set of all the things that are related to SOME instances
        of the destination concept by the relation.'

    Parameters
    ----------
    kr : KnowledgeRepresentation
        The Knowledge Representation containing the concepts and relations.
    base_uri : URIRef
        The base URI to use when creating the URIs.

    Returns
    -------
    Graph
        The constructed RDF triples.
    """
    rdf_graph = Graph()

    for relation in kr.relations:
        rel_uri = owl_obj_prop_uri(label=relation.label, base_uri=base_uri)
        rdf_graph.add((rel_uri, RDF.type, OWL.ObjectProperty))

        if relation.source_concept:
            src_concept_uri = owl_class_uri(
                label=relation.source_concept.label, base_uri=base_uri
            )
            rdf_graph.add((src_concept_uri, RDF.type, OWL.Class))

        if relation.destination_concept:
            dest_concept_uri = owl_class_uri(
                label=relation.destination_concept.label, base_uri=base_uri
            )
            rdf_graph.add((dest_concept_uri, RDF.type, OWL.Class))

        if relation.source_concept and relation.destination_concept:
            restriction_b_node, restriction_g = create_obj_prop_some_restriction_triples(
                rel_uri=rel_uri, dest_concept_uri=dest_concept_uri
            )
            rdf_graph += restriction_g
            rdf_graph.add((src_concept_uri, RDFS.subClassOf, restriction_b_node))

    return rdf_graph

def kr_relations_to_anonymous_only_parent(
    kr: KnowledgeRepresentation, base_uri: URIRef
) -> Graph:
    """Create RDF triples corresponding to said in plain english:
        'Each source concept is A SUBSET OF the set of all the things that are related to ONLY
        instances of the destination concept by the relation.'

    Parameters
    ----------
    kr : KnowledgeRepresentation
        The Knowledge Representation containing the concepts and relations.
    base_uri : URIRef
        The base URI to use when creating the URIs.

    Returns
    -------
    Graph
        The constructed RDF triples.
    """
    rdf_graph = Graph()

    for relation in kr.relations:
        
        rel_uri = owl_obj_prop_uri(label=relation.label, base_uri=base_uri)
        rdf_graph.add((rel_uri, RDF.type, OWL.ObjectProperty))

        if relation.source_concept:
            src_concept_uri = owl_class_uri(
                label=relation.source_concept.label, base_uri=base_uri
            )
            rdf_graph.add((src_concept_uri, RDF.type, OWL.Class))

        if relation.destination_concept:
            dest_concept_uri = owl_class_uri(
                label=relation.destination_concept.label, base_uri=base_uri
            )
            rdf_graph.add((dest_concept_uri, RDF.type, OWL.Class))
        
        if relation.source_concept and relation.destination_concept:
            restriction_b_node, restriction_g = create_obj_prop_all_restriction_triples(
                rel_uri=rel_uri, dest_concept_uri=dest_concept_uri
            )
            rdf_graph += restriction_g
            rdf_graph.add((src_concept_uri, RDFS.subClassOf, restriction_b_node))

    return rdf_graph


def kr_relations_to_anonymous_some_equivalent(
    kr: KnowledgeRepresentation, base_uri: URIRef
) -> Graph:
    """Create RDF triples corresponding to said in plain english:
        'Each source concept is EQUIVALENT TO the set of all the things that are related to SOME
        instances of the destination concept by the relation.'

    Parameters
    ----------
    kr : KnowledgeRepresentation
        The Knowledge Representation containing the concepts and relations.
    base_uri : URIRef
        The base URI to use when creating the URIs.

    Returns
    -------
    Graph
        The constructed RDF triples.
    """
    rdf_graph = Graph()

    for relation in kr.relations:

        rel_uri = owl_obj_prop_uri(label=relation.label, base_uri=base_uri)
        rdf_graph.add((rel_uri, RDF.type, OWL.ObjectProperty))

        if relation.source_concept:
            src_concept_uri = owl_class_uri(
                label=relation.source_concept.label, base_uri=base_uri
            )
            rdf_graph.add((src_concept_uri, RDF.type, OWL.Class))

        if relation.destination_concept:
            dest_concept_uri = owl_class_uri(
                label=relation.destination_concept.label, base_uri=base_uri
            )
            rdf_graph.add((dest_concept_uri, RDF.type, OWL.Class))

        if relation.source_concept and relation.destination_concept:
            restriction_b_node, restriction_g = create_obj_prop_some_restriction_triples(
                rel_uri=rel_uri, dest_concept_uri=dest_concept_uri
            )
            rdf_graph += restriction_g
            rdf_graph.add((src_concept_uri, OWL.equivalentClass, restriction_b_node))

    return rdf_graph


def concept_lrs_to_owl_individuals(
    kr: KnowledgeRepresentation, base_uri: URIRef
) -> Graph:
    """Create the RDF triples corresponding to making each KR concepts an OWL class with each
    concept linguistic representations instances of the concept class.

    Parameters
    ----------
    kr : KnowledgeRepresentation
        The Knowledge Representation containing the concepts.
    base_uri : URIRef
        The base URI to use when creating the class URIs.

    Returns
    -------
    Graph
        The constructed RDF triples.
    """
    rdf_graph = Graph()

    for concept in kr.concepts:

        concept_uri = owl_class_uri(label=concept.label, base_uri=base_uri)
        rdf_graph.add((concept_uri, RDF.type, OWL.Class))

        for c_lr in concept.linguistic_realisations:
            instance_uri = owl_instance_uri(label=c_lr.label, base_uri=base_uri)
            rdf_graph.add((instance_uri, RDF.type, concept_uri))

        concepts_lrs_map = defaultdict(set)
        for relation in kr.relations:
            rel_uri = owl_obj_prop_uri(label=relation.label, base_uri=base_uri)
            rdf_graph.add((rel_uri, RDF.type, OWL.ObjectProperty))

            if relation.source_concept not in concepts_lrs_map:
                for c_lr in relation.source_concept.linguistic_realisations:
                    concepts_lrs_map[relation.source_concept].add(
                        owl_instance_uri(label=c_lr.label, base_uri=base_uri)
                    )

            if relation.destination_concept not in concepts_lrs_map:
                for c_lr in relation.destination_concept.linguistic_realisations:
                    concepts_lrs_map[relation.destination_concept].add(
                        owl_instance_uri(label=c_lr.label, base_uri=base_uri)
                    )

            concepts_product = product(
                concepts_lrs_map[relation.source_concept],
                concepts_lrs_map[relation.destination_concept],
            )

            for source_uri, dest_uri in concepts_product:
                rdf_graph.add((source_uri, rel_uri, dest_uri))

    return rdf_graph


def all_individuals_different(kr: KnowledgeRepresentation, base_uri: URIRef) -> Graph:
    """Create the RDF triples corresponding to making each KR concepts linguistic representation an
    OWL named instance and making each instance different.

    Parameters
    ----------
    kr : KnowledgeRepresentation
        The Knowledge Representation containing the concepts.
    base_uri : URIRef
        The base URI to use when creating the class URIs.

    Returns
    -------
    Graph
        The constructed RDF triples.
    """
    rdf_graph = Graph()

    instance_uris = set()

    for concept in kr.concepts:
        concept_uri = owl_class_uri(label=concept.label, base_uri=base_uri)
        rdf_graph.add((concept_uri, RDF.type, OWL.Class))

        for c_lr in concept.linguistic_realisations:
            instance_uri = owl_instance_uri(label=c_lr.label, base_uri=base_uri)
            rdf_graph.add((instance_uri, RDF.type, concept_uri))
            instance_uris.add(instance_uri)

    rdf_collection = Collection(graph=rdf_graph, uri=BNode(), seq=list(instance_uris))
    b_node = BNode()

    rdf_graph.add((b_node, RDF.type, OWL.AllDifferent))
    rdf_graph.add((b_node, OWL.distinctMembers, rdf_collection.uri))

    return rdf_graph
