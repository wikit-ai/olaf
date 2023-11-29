from typing import List, Set

import spacy
from spacy.matcher import PhraseMatcher

from ..commons.logging_config import logger
from ..data_container.candidate_term_schema import CandidateRelation, CandidateTerm
from ..data_container.concept_schema import Concept
from ..data_container.linguistic_realisation_schema import ConceptLR


def group_cts_on_synonyms(
    candidate_terms: Set[CandidateTerm],
) -> List[Set[CandidateTerm]]:
    """Group candidate terms based on common synonyms.

    Parameters
    ----------
    candidate_terms : Set[CandidateTerm]
        Set of candidate terms to group.

    Returns
    -------
    List[Set[CandidateTerm]]
        List of candidate terms grouped.
        Each set contains candidate terms with common synonyms.
    """
    common_groups = []
    common_candidates = set()
    candidate_terms = list(candidate_terms)
    while candidate_terms:
        reference_term = candidate_terms.pop(0)
        common_candidates.add(reference_term)
        find_synonym_candidates(reference_term, candidate_terms, common_candidates)
        common_groups.append(common_candidates)
        common_candidates = set()
    return common_groups


def find_synonym_candidates(
    reference_ct: CandidateTerm,
    candidate_terms: List[CandidateTerm],
    common_candidates: Set[CandidateTerm],
) -> None:
    """Find candidate terms with common synonyms based on
    the reference term and group them in the a set.
    If the candidate terms are candidate relations,
    they should have the same source and destination concepts.

    Parameters
    ----------
    reference_ct : CandidateTerm
        Candidate term used as reference to start the synonym search with.
    candidate_terms : List[CandidateTerm]
        List of candidate terms to compare with the reference term.
    common_candidates : Set[CandidateTerm]
        Set of candidate terms with common synonyms.
    """
    i = 0
    while i < len(candidate_terms):
        check_source_destination = True
        if isinstance(reference_ct, CandidateRelation):
            check_source_destination = (
                reference_ct.source_concept == candidate_terms[i].source_concept
                and reference_ct.destination_concept
                == candidate_terms[i].destination_concept
            )
        check_common_synonyms = cts_have_common_synonyms(
            reference_ct, candidate_terms[i]
        )
        if check_source_destination and check_common_synonyms:
            new_reference_ct = candidate_terms.pop(i)
            common_candidates.add(new_reference_ct)
            remaining_candidates = len(candidate_terms) - i
            find_synonym_candidates(
                new_reference_ct, candidate_terms, common_candidates
            )
            i = len(candidate_terms) - remaining_candidates
        else:
            i += 1


def cts_have_common_synonyms(c_term_1: CandidateTerm, c_term_2: CandidateTerm) -> bool:
    """Check if two terms have common synonyms.

    Parameters
    ----------
    c_term_1 : CandidateTerm
        First candidate term to compare.
    c_term_2 : CandidateTerm
        Second candidate term to compare.

    Returns
    -------
    bool
        True if the two candidate terms have common synonyms, False otherwise.
    """
    conditions = [c_term_1.label == c_term_2.label]
    if (c_term_1.enrichment is not None) and (c_term_2.enrichment is not None):
        conditions.extend(
            [
                c_term_1.label in c_term_2.enrichment.synonyms,
                c_term_2.label in c_term_1.enrichment.synonyms,
                c_term_1.enrichment.synonyms & c_term_2.enrichment.synonyms,
            ]
        )
    elif (c_term_1.enrichment is None) and (c_term_2.enrichment is not None):
        conditions.append(c_term_1.label in c_term_2.enrichment.synonyms)
    elif (c_term_2.enrichment is None) and (c_term_1.enrichment is not None):
        conditions.append(c_term_2.label in c_term_1.enrichment.synonyms)

    return any(conditions)


def cts_to_concept(concept_candidates: Set[CandidateTerm]) -> Concept:
    """Create a concept out of a set of candidate terms.

    Parameters
    ----------
    concept_candidates : Set[CandidateTerm]
        Set of candidate terms to be merged in a same concept.

    Returns
    -------
    Concept
        The created concept.
    """
    candidates = list(concept_candidates)
    new_concept = Concept(candidates[0].label)
    for candidate in candidates:
        candidate_lr = ConceptLR(
            label=candidate.label, corpus_occurrences=candidate.corpus_occurrences
        )
        new_concept.add_linguistic_realisation(candidate_lr)
        if candidate.enrichment:
            for synonym in candidate.enrichment.synonyms:
                syn_lr = ConceptLR(label=synonym)
                new_concept.add_linguistic_realisation(syn_lr)

    return new_concept


def filter_cts_on_token_in_term(
    candidate_terms: Set[CandidateTerm], filtering_tokens: Set[str]
) -> Set[CandidateTerm]:
    """Filter a set of candidate terms based on tokens appearing in them.

    Note: this function acts only at the candidate term label level.

    Parameters
    ----------
    candidate_terms: Set[CandidateTerm]
        Set of candidate terms to filter.
    filtering_tokens: Set[str]
        The set of token strings to use for filtering the candidate terms.

    Returns
    -------
    Set[CandidateTerm]
        The set of filtered candidate terms.
    """

    if len(filtering_tokens) == 0:
        logger.warning(
            """The set of tokens to use for filtering out
            candidate terms is empty. This function have no effect."""
        )

    selected_candidate_terms = set()

    for ct in candidate_terms:
        ct_tokens_to_check = set(ct.label.strip().split())

        if not (ct_tokens_to_check & filtering_tokens):
            selected_candidate_terms.add(ct)

    return selected_candidate_terms


def filter_cts_on_last_token_in_term(
    candidate_terms: Set[CandidateTerm], filtering_tokens: Set[str]
) -> Set[CandidateTerm]:
    """Filter a set of candidate terms based on their last token.

    Note: this function acts only at the candidate term label level.

    Parameters
    ----------
    candidate_terms: Set[CandidateTerm]
        Set of candidate terms to filter.
    filtering_tokens: Set[str]
        The set of token strings to use for filtering the candidate terms.

    Returns
    -------
    Set[CandidateTerm]
        The set of filtered candidate terms.
    """
    if len(filtering_tokens) == 0:
        logger.warning(
            """The set of tokens to use for filtering out
            candidate terms is empty. This function have no effect."""
        )

    selected_candidate_terms = set()

    for ct in candidate_terms:
        ct_token_to_check = ct.label.strip().split()[-1]

        if ct_token_to_check not in filtering_tokens:
            selected_candidate_terms.add(ct)

    return selected_candidate_terms


def filter_cts_on_first_token_in_term(
    candidate_terms: Set[CandidateTerm], filtering_tokens: Set[str]
) -> Set[CandidateTerm]:
    """Filter a set of candidate terms based on their first token.

    Note: this function acts only at the candidate term label level.

    Parameters
    ----------
    candidate_terms: Set[CandidateTerm]
        Set of candidate terms to filter.
    filtering_tokens: Set[str]
        The set of token strings to use for filtering the candidate terms.

    Returns
    -------
    Set[CandidateTerm]
        The set of filtered candidate terms.
    """
    if len(filtering_tokens) == 0:
        logger.warning(
            """The set of tokens to use for filtering out
             candidate terms is empty. This function have no effect."""
        )

    selected_candidate_terms = set()

    for ct in candidate_terms:
        ct_token_to_check = ct.label.strip().split()[0]

        if ct_token_to_check not in filtering_tokens:
            selected_candidate_terms.add(ct)

    return selected_candidate_terms


def build_cts_from_strings(
    ct_label_strings: Set[str],
    spacy_model: spacy.language.Language,
    docs: List[spacy.tokens.Doc],
) -> Set[CandidateTerm]:
    """Create candidate terms from a set of strings label.

    Parameters
    ----------
    ct_label_strings : Set[str]
        The set of strings to use for candidate terms labels.
    spacy_model : spacy.language.Language
        The spaCy model to retrieve the corpus occurrences.
    docs : List[spacy.tokens.Doc]
        The corpus in which to find the corpus occurrences.

    Returns
    -------
    Set[CandidateTerm]
        The set of created candidate terms.
    """
    phrase_matcher = PhraseMatcher(spacy_model.vocab, attr="LOWER")

    for label in ct_label_strings:
        phrase_matcher.add(label, [spacy_model(label)])

    candidate_terms_index = {}

    for doc in docs:
        matches = phrase_matcher(doc, as_spans=True)

        for match in matches:
            if match.label not in candidate_terms_index:
                candidate_terms_index[match.label] = CandidateTerm(
                    label=spacy_model.vocab.strings[match.label],
                    corpus_occurrences={match},
                )
            else:
                candidate_terms_index[match.label].add_corpus_occurrences({match})

    return set(candidate_terms_index.values())


def split_cts_on_token(
    candidate_terms: Set[CandidateTerm],
    splitting_tokens: Set[str],
    spacy_model: spacy.language.Language,
    docs: List[spacy.tokens.Doc],
) -> Set[CandidateTerm]:
    """Split candidate terms based on a set of token strings.

    Note: this function acts only at the candidate term label level.

    Parameters
    ----------
    candidate_terms: Set[CandidateTerm]
        The set of candidate terms to split.
    splitting_tokens: Set[str]
        The token strings to split candidate terms on.
    spacy_model : spacy.language.Language
        The spaCy model to retrieve the candidate terms' corpus occurrences.
    docs : List[spacy.tokens.Doc]
        The corpus in which to find the candidate terms' corpus occurrences.

    Returns
    -------
    Set[CandidateTerm]
        The new set of candidate terms.
    """
    if len(splitting_tokens) == 0:
        logger.warning(
            """The set of tokens to use for candidate terms splitting is empty. 
            This function have not effect."""
        )

    new_candidate_terms = set()
    new_ct_to_construct_strings = set()

    for ct in candidate_terms:
        tokenized_ct_label = ct.label.strip().split()

        splitting_token_found = set(tokenized_ct_label) & splitting_tokens

        if splitting_token_found:
            token_accumulator = []
            for token in tokenized_ct_label:
                if token not in splitting_tokens:
                    token_accumulator.append(token)
                elif token_accumulator:  # to avoid empty string
                    new_ct_to_construct_strings.add(" ".join(token_accumulator))
                    token_accumulator = []
            if token_accumulator:
                # flush the accumulator before next candidate term
                new_ct_to_construct_strings.add(" ".join(token_accumulator))
        else:
            new_candidate_terms.add(ct)

    if len(new_ct_to_construct_strings) > 0:
        new_candidate_terms.update(
            build_cts_from_strings(
                ct_label_strings=new_ct_to_construct_strings,
                spacy_model=spacy_model,
                docs=docs,
            )
        )

    return new_candidate_terms
