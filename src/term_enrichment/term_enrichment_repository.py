import os.path
import requests
from typing import Any, Dict, List, Set

from commons.ontology_learning_schema import CandidateTerm
from config.core import config, DATA_PATH
import config.logging_config as logging_config


def load_candidate_terms_from_file() -> List[CandidateTerm]:
    """Load a list of candidate term from a file containing one term string per line.

    Returns
    -------
    List[CandidateTerm]
        The list of candidate terms
    """
    candidate_terms_file = config['term_enrichment'].get(
        'candidate_terms_path')

    if not os.path.isabs(candidate_terms_file):
        candidate_terms_file = os.path.join(
            DATA_PATH, candidate_terms_file)

    try:
        with open(candidate_terms_file, "r", encoding="utf8") as file:
            candidate_terms_texts = [line.strip() for line in file.readlines()]
    except Exception as e:
        logging_config.logger.error(
            f"Could not load candidate terms from file {candidate_terms_file}. Trace : {e}")
    else:
        logging_config.logger.info(
            f"Candidate terms loaded from file {candidate_terms_file}.")

    candidate_terms = [CandidateTerm(term) for term in candidate_terms_texts]

    return candidate_terms


def load_wordnet_domains() -> Dict[str, List[str]]:
    """Load the mapping of WordNet Synsets to domains from a file.
        The file should have the structure: `synset_code\tdomain1 domain2`
        Function inspired from project <https://github.com/argilla-io/spacy-wordnet>

    Returns
    -------
    Dict[str, List[str]]
        The mapping of WordNet Synsets to domains.
    """
    domain_file_path = config['term_enrichment']['wordnet'].get(
        'wordnet_domain_path')

    if not os.path.isabs(domain_file_path):
        domain_file_path = os.path.join(
            DATA_PATH, domain_file_path)

    domains_map = dict()

    try:
        for line in open(domain_file_path, "r", encoding="utf8"):
            ssid, domains = line.strip().split("\t")
            domains_map[ssid] = domains.split(" ")
    except Exception as e:
        logging_config.logger.error(
            f"Could not load wordnet domains from file {domain_file_path}. Trace : {e}")
    else:
        logging_config.logger.info(f"Wordnet domains loaded.")

    return domains_map


def load_enrichment_wordnet_domains_from_file() -> Set[str]:
    """Load a set of domains (strings) from a file.
        The file is expected to contain one domain sring per lin.

    Returns
    -------
    Set[str]
        The set of domains.
    """
    domain_file_path = config['term_enrichment']['wordnet'].get(
        'enrichment_domains_file')

    if not os.path.isabs(domain_file_path):
        domain_file_path = os.path.join(
            DATA_PATH, domain_file_path)

    enrichment_domains = set()

    try:
        for line in open(domain_file_path, "r", encoding="utf8"):
            enrichment_domains.add(line.strip())
    except Exception as e:
        logging_config.logger.error(
            f"Could not load enrichment wordnet domains from file {domain_file_path}. Trace : {e}")
    else:
        logging_config.logger.info(f"Enrichment Wordnet domains loaded.")

    return enrichment_domains


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
