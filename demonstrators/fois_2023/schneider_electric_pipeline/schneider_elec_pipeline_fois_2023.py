from typing import List

from axiom_extraction.axiom_extraction_service import AxiomExtraction
from commons.ontology_learning_schema import CandidateTerm
from concept_extraction.concept_extraction_service import ConceptExtraction
from concept_hierarchy.concept_hierarchy_service import ConceptHierarchy
from data_preprocessing.data_preprocessing_service import DataPreprocessing
from relation_extraction.relation_extraction_service import RelationExtraction
from term_enrichment.term_enrichment_service import TermEnrichment
from term_extraction.term_extraction_service import TermExtraction


def filter_duplicated_terms(candidate_terms: List[CandidateTerm]) -> List[CandidateTerm]:
    """Filter the list of candidate terms to remove duplicates.

    Parameters
    ----------
    candidate_terms : List[CandidateTerm]
        The list of candidate terms

    Returns
    -------
    List[CandidateTerm]
        The list of candidate terms duplicates filtered out.
    """
    new_candidate_terms = list()
    term_values = set()

    for term in candidate_terms:
        if not (term.value in term_values):
            new_candidate_terms.append(term)
            term_values.add(term.value)

    return new_candidate_terms


def split_candidate_terms(candidate_terms: List[CandidateTerm], splitting_token: str) -> List[CandidateTerm]:
    """Function to manually split some candidate terms based on predifined tokens.

    Parameters
    ----------
    candidate_terms : List[CandidateTerm]
        The list of candidate terms to process;
    splitting_token : str
        The token to split the candidate terms on.

    Returns
    -------
    List[CandidateTerm]
        The resulting list of candidate terms
    """

    new_candidate_terms = list()

    for term in candidate_terms:
        term_value = term.value
        tokenized_term = term_value.strip().split()

        if splitting_token in tokenized_term:
            splitting_index = tokenized_term.index(splitting_token)
            new_candidate_terms.append(CandidateTerm(
                " ".join(tokenized_term[:splitting_index])))
            new_candidate_terms.append(CandidateTerm(
                " ".join(tokenized_term[splitting_index + 1:])))
        else:
            new_candidate_terms.append(term)

    return new_candidate_terms


def main() -> None:

    ##########################################
    #           DATA PREPROCESSING           #
    ##########################################

    data_prep = DataPreprocessing()
    data_prep._set_corpus()

    STOP_WORDS = set(data_prep.spacy_model.Defaults.stop_words)
    STOP_WORDS.update({
        "ø", "kw", "ac", "dc"
    })

    ##########################################
    #            TERM EXTRACTION             #
    ##########################################

    term_extraction = TermExtraction(data_prep.corpus)

    print("C-value Term extraction")
    candidate_terms = term_extraction.c_value_term_extraction()
    print()

    print("Post filtering candidate terms")
    candidate_terms = term_extraction.post_filter_candidate_terms_on_tokens_presence(
        candidate_terms=candidate_terms,
        filter_type="on_first_token",
        filtering_tokens=STOP_WORDS
    )

    candidate_terms = term_extraction.post_filter_candidate_terms_on_tokens_presence(
        candidate_terms=candidate_terms,
        filter_type="on_last_token",
        filtering_tokens=STOP_WORDS
    )

    candidate_terms = split_candidate_terms(candidate_terms, "with")
    candidate_terms = split_candidate_terms(candidate_terms, "and")
    candidate_terms = split_candidate_terms(candidate_terms, "by")
    candidate_terms = split_candidate_terms(candidate_terms, "or")
    candidate_terms = split_candidate_terms(candidate_terms, "for")

    # We manually added some known Candidate Terms.
    # Here we add the terms corresponding to the Schneider Electric product ranges.
    candidate_terms.append(CandidateTerm("acti9"))
    candidate_terms.append(CandidateTerm("tesys"))
    candidate_terms.append(CandidateTerm("lexium"))
    candidate_terms.append(CandidateTerm("easy9"))
    candidate_terms.append(CandidateTerm("resi9"))
    candidate_terms.append(CandidateTerm("zelio"))
    candidate_terms.append(CandidateTerm("harmony"))
    candidate_terms.append(CandidateTerm("modicon"))

    candidate_terms = filter_duplicated_terms(candidate_terms)

    print(f"{len(candidate_terms)} candidate terms have been found")

    ##########################################
    #            TERM ENRICHMENT             #
    ##########################################

    print("Enriching candidate terms using WordNet ...")
    term_enrichment = TermEnrichment(candidate_terms)
    term_enrichment.wordnet_term_enrichment()

    ##########################################
    #           CONCEPT EXTRACTION           #
    ##########################################

    print("Extracting concepts ...")
    concept_extraction = ConceptExtraction(candidate_terms)
    concept_extraction.group_by_synonyms()

    kr = concept_extraction.kr

    print(f"{len(kr.concepts)} concepts have been found.")

    ##########################################
    #            CONCEPT HIERARCHY           #
    ##########################################

    print("Extracting hierarchical relations...")
    concept_hierarchy = ConceptHierarchy(
        data_prep.corpus, kr)
    concept_hierarchy.term_subsumption()

    ##########################################
    #           RELATION EXTRACTION          #
    ##########################################

    print("Relation Extraction based on occurrence ...")

    relation_extraction = RelationExtraction(
        data_prep.corpus, kr)
    relation_extraction.on_occurence_relation_extraction()

    print("Relation Extraction based on occurrence with sep term...")
    relation_extraction.on_cooc_with_sep_term_map_meta_rel_extraction(
        spacy_nlp=data_prep.spacy_model)

    ##########################################
    #           AXIOM EXTRACTION             #
    ##########################################

    print("Extracting axioms and saving OWL ontology ...")
    axiom_extraction = AxiomExtraction(kr)
    axiom_extraction.owl_concept_restriction()


if __name__ == "__main__":
    main()
