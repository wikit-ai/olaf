from typing import List, Set

from commons.ontology_learning_schema import CandidateTerm


def filter_candidate_terms_on_first_token(candidate_terms: List[CandidateTerm], filtering_tokens: Set[str]) -> List[CandidateTerm]:
    """Filter candidate terms based on the appearance of specific tokens at the beginning of the candidate term value.

    Parameters
    ----------
    candidate_terms : List[CandidateTerm]
        List of tokens to filter
    filtering_tokens : Set[str]
        The set of tokens string to filter on.

    Returns
    -------
    List[CandidateTerm]
        The list of filtered candidate terms.
    """

    selected_candidate_terms = list()

    for term in candidate_terms:
        term_value = term.value
        tokenized_term = term_value.strip().split()

        if (tokenized_term[0] in filtering_tokens):
            continue
        else:
            selected_candidate_terms.append(term)

    return selected_candidate_terms


def filter_candidate_terms_on_last_token(candidate_terms: List[CandidateTerm], filtering_tokens: Set[str]) -> List[CandidateTerm]:
    """Filter candidate terms based on the appearance of specific tokens at the end of the candidate term value.

    Parameters
    ----------
    candidate_terms : List[CandidateTerm]
        List of tokens to filter
    filtering_tokens : Set[str]
        The set of tokens string to filter on.

    Returns
    -------
    List[CandidateTerm]
        The list of filtered candidate terms.
    """

    selected_candidate_terms = list()

    for term in candidate_terms:
        term_value = term.value
        tokenized_term = term_value.strip().split()

        if (tokenized_term[-1] in filtering_tokens):
            continue
        else:
            selected_candidate_terms.append(term)

    return selected_candidate_terms


def filter_candidate_terms_if_token_in_term(candidate_terms: List[CandidateTerm], filtering_tokens: Set[str]) -> List[CandidateTerm]:
    """Filter candidate terms based on the appearance of specific tokens at in the candidate term value.

    Parameters
    ----------
    candidate_terms : List[CandidateTerm]
        List of tokens to filter
    filtering_tokens : Set[str]
        the set of tokens string to filter on.

    Returns
    -------
    List[CandidateTerm]
        The list of filtered candidate terms.
    """

    selected_candidate_terms = list()

    for term in candidate_terms:
        term_value = term.value
        tokenized_term = term_value.strip().split()

        conditions = [token in filtering_tokens for token in tokenized_term]

        if any(conditions):
            continue
        else:
            selected_candidate_terms.append(term)

    return selected_candidate_terms


str2candidateTermFilter = {
    "on_first_token": filter_candidate_terms_on_first_token,
    "on_last_token": filter_candidate_terms_on_last_token,
    "if_token_in_term": filter_candidate_terms_if_token_in_term
}
