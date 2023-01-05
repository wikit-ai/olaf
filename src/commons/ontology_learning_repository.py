import kglab
import os.path
from rdflib import BNode, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD
import requests
from typing import Any, Dict, List, Optional, Set, Tuple
import urllib.parse

from commons.ontology_learning_schema import Concept, KR, MetaRelation, Relation
from config.core import config, DATA_PATH


def get_paginated_conceptnet_edges(conceptnet_view_res: Dict[str, str], batch_size: int) -> List[Dict[str, Any]]:
    """Fetch paginated edges from the conceptnet api. The api return results by batch. 
        This method ietrate over the batches to fecth all of them. 

    Parameters
    ----------
    conceptnet_view_res : Dict[str, str]
        The "view" section of the conceptnet api first results response.
        (it contains the information to iterate over the result pages)

    Returns
    -------
    List[Dict[str, Any]]
        The list of fetched conceptnet edge objects
    """
    last_page = False
    page_count = 0

    paginated_edges = []

    while not last_page:
        page_count += 1

        next_page_url = 'http://api.conceptnet.io' + \
            conceptnet_view_res.get("nextPage").split(
                "?")[0] + f"?offset={page_count*batch_size}&limit={batch_size}"

        conceptnet_res = requests.get(next_page_url).json()

        paginated_edges.extend(conceptnet_res.get("edges", []))

        last_page = conceptnet_res["view"].get("nextPage") is None

    return paginated_edges


def conceptnet_api_fetch_term(term_conceptnet_text: str, lang: str, batch_size: int) -> Dict[str, Any]:
    """Wrapper to hit the conceptnet API.

    Parameters
    ----------
    term_conceptnet_text : str
        Term to fetch the ConceptNetdata from (spaces are replaced underscores)
    lang : str
        The term language
    batch_size : int
        The number of edges to fetch

    Returns
    -------
    Dict[str, Any]
        ConceptNet API result.
    """
    term_conceptnet_url = f"http://api.conceptnet.io/c/{lang}/{term_conceptnet_text}?limit={batch_size}"
    conceptnet_term_res = requests.get(term_conceptnet_url).json()

    return conceptnet_term_res


RDFS_SUBCLASS_OF = RDFS.subClassOf
RDFS_SUBPROPERTY_OF = RDFS.subPropertyOf
RDFS_LABEL = RDFS.label

RDF_TYPE = RDF.type

OWL_CLASS = OWL.Class
OWL_OBJECT_PROPERTY = OWL.ObjectProperty
OWL_RESTRICTION = OWL.Restriction
OWL_ON_PROPERTY = OWL.onProperty
OWL_SOME_VALUES_FROM = OWL.someValuesFrom
OWL_DISJOINT_WITH = OWL.disjointWith

XSD_STRING = XSD.string

SKOS_RELATED = SKOS.related

META_REL_2_RDFS_OWL = {
    "generalisation": RDFS_SUBCLASS_OF,
    "related_to": SKOS_RELATED,
    "hasPart": URIRef("http://ontologies.ms2.com/o/kr#hasPart"),
    "specificTo": URIRef("http://ontologies.ms2.com/o/kr#specificTo"),
    "hasType": URIRef("http://ontologies.ms2.com/o/kr#hasType")
}


def KR2RDF(kr: KR, format: str = "ttl", saving_file: Optional[str] = None) -> None:
    """Export a KR instance to RDF in the Turtle format.
        The created triples are partially based on OWL and RDFS.
        They do not respect a specific semantic. The aim is to be able du visualize 
        the graph in Protégé and other graph visualization coming with a triple store.

    Parameters
    ----------
    kr : KR
        The KR instance to export as triples.
    format: str, default "ttl"
        The format to save use for saving the KR instance. Should be "ttl", "xml", or one 
        of the RDFlib serializers (https://rdflib.readthedocs.io/en/stable/plugin_serializers.html)
    saving_file : Optional[str], optional
        The location to save the generated turtle file. If None the value will be taken from the config file.
    """
    namespaces = {
        "ms2": "http://ontologies.ms2.com/o/kr#",
    }

    kg = kglab.KnowledgeGraph(
        name="MS2 KR",
        namespaces=namespaces
    )

    kg.add(URIRef("http://ontologies.ms2.com/o/kr#hasPart"),
           RDF_TYPE, OWL_OBJECT_PROPERTY)
    kg.add(URIRef("http://ontologies.ms2.com/o/kr#specificTo"),
           RDF_TYPE, OWL_OBJECT_PROPERTY)
    kg.add(URIRef("http://ontologies.ms2.com/o/kr#hasType"),
           RDF_TYPE, OWL_OBJECT_PROPERTY)
    kg.add(SKOS_RELATED, RDF_TYPE, OWL_OBJECT_PROPERTY)

    kg.add(URIRef("http://ontologies.ms2.com/o/kr#hasPart"),
           RDFS_SUBPROPERTY_OF, SKOS_RELATED)
    kg.add(URIRef("http://ontologies.ms2.com/o/kr#specificTo"),
           RDFS_SUBPROPERTY_OF, SKOS_RELATED)
    kg.add(URIRef("http://ontologies.ms2.com/o/kr#hasType"),
           RDFS_SUBPROPERTY_OF, SKOS_RELATED)

    for concept in kr.concepts:
        concept_uri = URIRef(
            kg.get_ns('ms2') + urllib.parse.quote(concept.uid))

        kg.add(concept_uri, RDF_TYPE, OWL_CLASS)

        for label in concept.terms:
            kg.add(concept_uri, RDFS_LABEL, Literal(
                label, datatype=XSD.string))

    for relation in kr.relations:
        relation_uri = URIRef(
            kg.get_ns('ms2') + urllib.parse.quote(relation.uid))

        kg.add(relation_uri, RDF_TYPE, OWL_OBJECT_PROPERTY)

        for label in relation.terms:
            kg.add(relation_uri, RDFS_LABEL, Literal(
                label, datatype=XSD.string))

        source_concept_uri = URIRef(
            kg.get_ns('ms2') + urllib.parse.quote(relation.source_concept_id))
        dest_concept_uri = URIRef(
            kg.get_ns('ms2') + urllib.parse.quote(relation.destination_concept_id))

        kg.add(source_concept_uri, relation_uri, dest_concept_uri)

    for meta_relation in kr.meta_relations:

        meta_relation_uri = META_REL_2_RDFS_OWL[meta_relation.relation_type]

        source_concept_uri = URIRef(
            kg.get_ns('ms2') + urllib.parse.quote(meta_relation.source_concept_id))
        dest_concept_uri = URIRef(
            kg.get_ns('ms2') + urllib.parse.quote(meta_relation.destination_concept_id))

        kg.add(dest_concept_uri, meta_relation_uri, source_concept_uri)

    if saving_file is None:
        saving_file = config['kr_export']['saving_file']

    if not os.path.isabs(saving_file):
        saving_file = os.path.join(DATA_PATH, saving_file)

    kg.save_rdf(saving_file, format=format)


def create_anonymous_subclass_triples(blank_node: BNode, restricted_prop: URIRef, target_class: URIRef) -> Set[Tuple[URIRef, URIRef, URIRef]]:
    """Create the triples to represent a anonymous class of an OWL Some Values From restriction 
    on a property with a specific target class.

    Parameters
    ----------
    blank_node : BNode
        The blank node denoting the anonymous class
    restricted_prop : URIRef
        The OWL object property the restriction targets
    target_class : URIRef
        The class the restriction targets

    Returns
    -------
    Set[Tuple[URIRef, URIRef, URIRef]]
        _description_
    """
    triples = {
        (blank_node, RDF_TYPE, OWL_RESTRICTION),
        (blank_node, OWL_ON_PROPERTY, restricted_prop),
        (blank_node, OWL_SOME_VALUES_FROM, target_class)
    }

    return triples


def KR2OWL_restriction_on_concepts(kr: KR, format: str = "ttl", saving_file: Optional[str] = None) -> None:
    """Export a KR instance to RDF in the Turtle format.
        The created triples form a valid OWL ontology.
        This function adds restrictions on concepts in the ontology.

    Parameters
    ----------
    kr : KR
        The KR instance to export as triples.
    format: str, default "ttl"
        The format to save use for saving the KR instance. Should be "ttl", "xml", or one 
        of the RDFlib serializers (https://rdflib.readthedocs.io/en/stable/plugin_serializers.html)
    saving_file : Optional[str], optional
        The location to save the generated turtle file. If None the value will be taken from the config file.
    """
    namespaces = {
        "ms2": "http://ontologies.ms2.com/o/kr#",
    }

    kg = kglab.KnowledgeGraph(
        name="MS2 KR",
        namespaces=namespaces
    )

    kg.add(URIRef("http://ontologies.ms2.com/o/kr#hasPart"),
           RDF_TYPE, OWL_OBJECT_PROPERTY)
    kg.add(URIRef("http://ontologies.ms2.com/o/kr#specificTo"),
           RDF_TYPE, OWL_OBJECT_PROPERTY)
    kg.add(URIRef("http://ontologies.ms2.com/o/kr#hasType"),
           RDF_TYPE, OWL_OBJECT_PROPERTY)
    kg.add(SKOS_RELATED, RDF_TYPE, OWL_OBJECT_PROPERTY)

    kg.add(URIRef("http://ontologies.ms2.com/o/kr#hasPart"),
           RDFS_SUBPROPERTY_OF, SKOS_RELATED)
    kg.add(URIRef("http://ontologies.ms2.com/o/kr#specificTo"),
           RDFS_SUBPROPERTY_OF, SKOS_RELATED)
    kg.add(URIRef("http://ontologies.ms2.com/o/kr#hasType"),
           RDFS_SUBPROPERTY_OF, SKOS_RELATED)

    for concept in kr.concepts:
        concept_uri = URIRef(
            kg.get_ns('ms2') + urllib.parse.quote(concept.uid))

        kg.add(concept_uri, RDF_TYPE, OWL_CLASS)

        for label in concept.terms:
            kg.add(concept_uri, RDFS_LABEL, Literal(
                label, datatype=XSD.string))

    for relation in kr.relations:
        relation_uri = URIRef(
            kg.get_ns('ms2') + urllib.parse.quote(relation.uid))

        kg.add(relation_uri, RDF_TYPE, OWL_OBJECT_PROPERTY)
        kg.add(relation_uri, RDFS_SUBPROPERTY_OF, SKOS_RELATED)

        for label in relation.terms:
            kg.add(relation_uri, RDFS_LABEL, Literal(
                label, datatype=XSD.string))

        source_concept_uri = URIRef(
            kg.get_ns('ms2') + urllib.parse.quote(relation.source_concept_id))
        dest_concept_uri = URIRef(
            kg.get_ns('ms2') + urllib.parse.quote(relation.destination_concept_id))

        blank_node = BNode()
        subclassof_restriction_triples = create_anonymous_subclass_triples(
            blank_node=blank_node,
            restricted_prop=relation_uri,
            target_class=dest_concept_uri
        )

        kg.add(blank_node, RDFS_SUBCLASS_OF, source_concept_uri)
        # kg.add(source_concept_uri, RDFS_SUBCLASS_OF, blank_node)

        for s, p, o in subclassof_restriction_triples:
            kg.add(s, p, o)

    for meta_relation in kr.meta_relations:

        meta_relation_uri = META_REL_2_RDFS_OWL[meta_relation.relation_type]

        source_concept_uri = URIRef(
            kg.get_ns('ms2') + urllib.parse.quote(meta_relation.source_concept_id))
        dest_concept_uri = URIRef(
            kg.get_ns('ms2') + urllib.parse.quote(meta_relation.destination_concept_id))

        if meta_relation_uri != RDFS_SUBCLASS_OF:
            blank_node = BNode()
            subclassof_restriction_triples = create_anonymous_subclass_triples(
                blank_node=blank_node,
                restricted_prop=meta_relation_uri,
                target_class=dest_concept_uri
            )

            kg.add(blank_node, RDFS_SUBCLASS_OF, source_concept_uri)
            # kg.add(source_concept_uri, RDFS_SUBCLASS_OF, blank_node)

            if meta_relation_uri != SKOS_RELATED:
                kg.add(source_concept_uri, OWL_DISJOINT_WITH, dest_concept_uri)

            for s, p, o in subclassof_restriction_triples:
                kg.add(s, p, o)
        else:
            kg.add(source_concept_uri, RDFS_SUBCLASS_OF, dest_concept_uri)

    if saving_file is None:
        saving_file = config['kr_export']['saving_file']

    if not os.path.isabs(saving_file):
        saving_file = os.path.join(DATA_PATH, saving_file)

    kg.save_rdf(saving_file, format=format)


def KR2TXT(kr: KR, saving_file: Optional[str] = None) -> None:
    """Export a KR instance to a text file.
        The created text file is mostly intended to ease later loading of a KR instance.

    Parameters
    ----------
    kr : KR
        The KR instance to export as triples.
    saving_file : Optional[str], optional
        The location to save the generated text file. If None the value will be taken from the config file.
    """

    if saving_file is None:
        saving_file = config['kr_export']['saving_file']

    if not os.path.isabs(saving_file):
        saving_file = os.path.join(DATA_PATH, saving_file)

    with open(saving_file, "w", encoding='utf8') as file:

        file.write(
            "==============================================================================\n")
        file.write(
            "=======                  CONCEPTS                        =====================\n")
        file.write(
            "==============================================================================\n")

        for concept in kr.concepts:
            file.write(str(concept) + "\n")

        file.write("\n")
        file.write(
            "==============================================================================\n")
        file.write(
            "=======                  RELATIONS                       =====================\n")
        file.write(
            "==============================================================================\n")

        for relation in kr.relations:
            file.write(str(relation) + "\n")

        file.write("\n")
        file.write(
            "==============================================================================\n")
        file.write(
            "=======                  META RELATIONS                  =====================\n")
        file.write(
            "==============================================================================\n")

        for meta_relation in kr.meta_relations:
            file.write(str(meta_relation) + "\n")


def load_KR_from_text(kr_file: Optional[str] = None) -> KR:
    """Load a KR instance from a text file.
        Make sure the text file contains a KR instance saved using the KR2TXT function.

    Parameters
    ----------
    kr_file : Optional[str], optional
        The text file containing the KR instance., by default None

    Returns
    -------
    KR
        The loaded KR instance
    """
    if kr_file is None:
        kr_file = config['kr_export']['saving_file']

    if not os.path.isabs(kr_file):
        kr_file = os.path.join(DATA_PATH, kr_file)

    kr = KR()

    with open(kr_file, "r", encoding='utf8') as file:

        for line in file.readlines():

            if line.startswith("Concept"):
                kr.concepts.add(eval(line))
            elif line.startswith("Relation"):
                kr.relations.add(eval(line))
            elif line.startswith("MetaRelation"):
                kr.meta_relations.add(eval(line))

    return kr
