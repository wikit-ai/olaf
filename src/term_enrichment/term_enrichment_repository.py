from typing import List, Dict
import os.path

from term_enrichment.term_enrichment_schema import CandidateTerm
from config.core import config, DATA_PATH


def load_candidate_terms_from_file() -> List():

    candidate_terms_file = config['term_enrichment'].get(
        'candidate_terms_path')

    if not os.path.isabs(candidate_terms_file):
        candidate_terms_file = os.path.join(
            DATA_PATH, candidate_terms_file)

    with open(candidate_terms_file, "r", encoding="utf8") as file:
        candidate_terms_texts = [line.strip() for line in file.readlines()]

    candidate_terms = [CandidateTerm(term) for term in candidate_terms_texts]

    return candidate_terms


def load_wordnet_domains() -> Dict[str, List[str]]:

    domain_file_path = config['term_enrichment']['wordnet'].get(
        'wordnet_domain_path')

    if not os.path.isabs(domain_file_path):
        domain_file_path = os.path.join(
            DATA_PATH, domain_file_path)

    domains_map = dict()

    for line in open(domain_file_path, "r", encoding="utf8"):
        ssid, domains = line.strip().split("\t")
        domains_map[ssid] = domains.split(" ")

    return domains_map
