import requests
from typing import Any, Dict, List


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
