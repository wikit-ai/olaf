import os
from nltk.corpus import stopwords
from random import sample
import re
from typing import List, Set
import spacy.tokens.doc
from collections import defaultdict

from axiom_extraction.axiom_extraction_service import AxiomExtraction
from config.core import DATA_PATH
from term_extraction.term_extraction_service import TermExtraction
from data_preprocessing.data_preprocessing_service import DataPreprocessing
from term_extraction.term_extraction_methods.c_value import Cvalue
from term_extraction.term_extraction_methods.tf_idf_term_extraction import TFIDF_TermExtraction
from commons.ontology_learning_schema import CandidateTerm
from commons.ontology_learning_repository import KR2RDF, KR2OWL_restriction_on_concepts, KR2TXT
from term_enrichment.term_enrichment_service import TermEnrichment

from concept_extraction.concept_extraction_service import ConceptExtraction
from concept_hierarchy.concept_hierarchy_service import ConceptHierarchy
from relation_extraction.relation_extraction_service import RelationExtraction

STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.update({
    "ø", "kw", "ac", "dc"
})


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

    data_prep = DataPreprocessing()
    data_prep._set_corpus()

    term_extraction = TermExtraction(data_prep.corpus)

    ###############################################################
    # Only to visualize c - value scores and determine a treshold
    ###############################################################
    # doc_attribute_name = "selected_tokens"
    # max_size_gram = 5

    # c_value = Cvalue(data_prep.corpus, doc_attribute_name, max_size_gram)
    # schneider_c_values = c_value.c_values

    # with open(os.path.join(DATA_PATH, "data_files", "schneider_texts_cvalues.txt"), "w", encoding='utf8') as file:
    #     for c_val in schneider_c_values:
    #         file.write(f"{c_val.score:.4} -- {c_val.candidate_term}\n")
    ###############################################################

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
    print()
    with open(os.path.join(DATA_PATH, "data_files", "schneider_texts_candidate_terms.txt"), "w", encoding='utf8') as file:
        for term in candidate_terms:
            file.write(f"{term.value}\n")

    print("Sample candidate terms:")
    for term in sample(candidate_terms, 10):
        print(term.value)

    print()
    print("Enriching candidate terms using WordNet ...")
    print()
    term_enrichment = TermEnrichment(candidate_terms)
    print("Wordnet term enrichment ...")
    term_enrichment.wordnet_term_enrichment()

    # Visualize some examples of some candidate terms after enrichment
    print("Examples of some candidate terms after enrichment : ")
    i = 0
    count_printed = 0
    while (count_printed < 10) and (i < len(candidate_terms)):
        if len(candidate_terms[i].synonyms) > 1:
            print(candidate_terms[i])
            count_printed += 1
        i += 1

    print("Extracting concepts ...")
    concept_extraction = ConceptExtraction(candidate_terms)
    concept_extraction.group_by_synonyms()

    kr = concept_extraction.kr

    print(f"{len(kr.concepts)} concepts have been found.")
    print()

    print("Extracting hierarchical relations...")
    concept_hierarhy_options = {
        "use_span": True,
        "term_subsumption": {
            "algo_type": "UNIQUE",
            "subsumption_threshold": 0.8,
            "use_lemma": False
        }
    }
    concept_hierarchy = ConceptHierarchy(
        data_prep.corpus, kr, concept_hierarhy_options)
    concept_hierarchy.term_subsumption()

    print("Relation Extraction based on occurrence ...")

    rel_extraction_config = {
        "on_occurrence": {
            "use_lemma": False,
            "threshold": 3
        },
        "on_occurrence_with_sep_term": {
            "term_relation_map": {
                "with": "hasPart",
                "for": "specificTo",
                "by": "hasType"
            },
            "use_lemma": False,
            "cooc_treshold": 0,
            "cooc_scope": "doc",
            "concept_distance_limit": 3
        }
    }

    relation_extraction = RelationExtraction(
        data_prep.corpus, kr, configuration=rel_extraction_config)
    relation_extraction.on_occurence_relation_extraction()

    print("Relation Extraction based on occurrence with sep term...")
    relation_extraction.on_cooc_with_sep_term_map_meta_rel_extraction(
        spacy_nlp=data_prep.spacy_model)

    axiom_extraction_configs = {
        "owl_restriction_on_concepts": {
            "owl_onto_saving_file": os.path.join(DATA_PATH, "data_files", "tp_schneider_kr.owl"),
            "reasoner": "ELK",
            # Update the paths based on your configuration
            "java_exe": "C:\\Program Files\\Common Files\\Oracle\\Java\\javapath\\java.exe",
            "robot_jar": "C:\\Users\\msesboue\\Documents\\\software_tools\\robot\\robot.jar"
        }
    }

    print("Saving KR to RDF...")
    KR2RDF(kr, saving_file=os.path.join(DATA_PATH, "data_files",
           "tp_schneider_kr.ttl"))

    print("Saving KR to OWL...")
    KR2OWL_restriction_on_concepts(kr, format="xml", saving_file=os.path.join(DATA_PATH, "data_files",
                                                                              "tp_schneider_kr_no_axiom_validation.owl"))

    axiom_extraction = AxiomExtraction(kr, axiom_extraction_configs)
    axiom_extraction.owl_concept_restriction()

    print("Saving KR to text...")
    KR2TXT(kr, saving_file=os.path.join(DATA_PATH, "data_files",
           "schneider_texts_kr.txt"))


if __name__ == "__main__":
    main()
