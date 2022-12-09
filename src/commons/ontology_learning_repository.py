import kglab
import os.path
from rdflib import URIRef, Literal
from rdflib.namespace import OWL, RDF, RDFS, XSD, SKOS
import requests
from tqdm import tqdm
from typing import Any, Dict, List, Optional
import urllib.parse

from commons.ontology_learning_schema import KR
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
OWL_CLASS = OWL.Class
OWL_OBJECT_PROPERTY = OWL.ObjectProperty
RDFS_LABEL = RDFS.label
RDF_TYPE = RDF.type
XSD_STRING = XSD.string
SKOS_RELATED = SKOS.related

META_REL_2_RDFS_OWL = {
    "generalisation": RDFS_SUBCLASS_OF,
    "related_to": SKOS_RELATED
}


def KR2RDF(kr: KR, saving_file: Optional[str] = None) -> None:
    """Export a KR instance to RDF in the Turtle format.
        The created triples are partially based on OWL and RDFS.
        They do not respect a specific semantic. The aim is to be able du visualize 
        the graph in Protégé and other graph visualization coming with a triple store.

    Parameters
    ----------
    kr : KR
        The KR instance to export as triples.
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

    kg.save_rdf(saving_file, format="ttl")
