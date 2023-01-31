from data_preprocessing.data_preprocessing_service import DataPreprocessing
from term_extraction.term_extraction_service import TermExtraction
from term_enrichment.term_enrichment_service import TermEnrichment
from concept_extraction.concept_extraction_service import ConceptExtraction
from concept_hierarchy.concept_hierarchy_service import ConceptHierarchy
from relation_extraction.relation_extraction_service import RelationExtraction
from commons.ontology_learning_schema import CandidateTerm
from commons.ontology_learning_repository import KR2RDF

def main() -> None:

    ##########################################
    #           DATA PREPROCESSING           #
    ##########################################

    print()
    print("Data preprocessing using Spacy ...")
    print()

    data_prep = DataPreprocessing()
    data_prep._set_corpus()

    ##########################################
    #            TERM EXTRACTION             #
    ##########################################

    print()
    print("Term extraction based on pos tagging and occurrence filtering ...")
    print()

    term_extraction = TermExtraction(data_prep.corpus)
    pos_candidate_terms = term_extraction.on_pos_term_extraction()
    occurence_candidate_terms = term_extraction.on_occurrence_term_extraction()

    # Merge candidate terms
    candidates_terms = [candidate_term for candidate_term in pos_candidate_terms if candidate_term in occurence_candidate_terms]

    # Remove wrong candidate terms
    candidates_terms.remove(CandidateTerm("cc"))
    candidates_terms.remove(CandidateTerm("pouvoir"))
    candidates_terms.remove(CandidateTerm("bit"))
    candidates_terms.remove(CandidateTerm(">"))
    candidates_terms.remove(CandidateTerm("<"))
    candidates_terms.remove(CandidateTerm("oui"))

    ##########################################
    #            TERM ENRICHMENT             #
    ##########################################

    print()
    print("Term enrichment based on embeddings similarity ...")
    print()

    term_enrichment = TermEnrichment(candidates_terms)
    term_enrichment.embedding_term_enrichment(data_prep.spacy_model)


    ##########################################
    #           CONCEPT EXTRACTION           #
    ##########################################

    print()
    print("Concept extraction with synonyms grouping ...")
    print()


    concept_extraction = ConceptExtraction(candidates_terms)
    concept_extraction.group_by_synonyms()
    kr = concept_extraction.kr


    ##########################################
    #            CONCEPT HIERARCHY           #
    ##########################################

    print()
    print("Concept hierarchisation based on term subsumption ...")
    print()

    concept_hierarchy = ConceptHierarchy(data_prep.corpus, kr)
    concept_hierarchy.term_subsumption()
    kr = concept_hierarchy.kr

    ##########################################
    #           RELATION EXTRACTION          #
    ##########################################

    print()
    print("Relation extraction based on pos tagging ...")
    print()

    relation_extraction = RelationExtraction(data_prep.corpus, kr)
    relation_extraction.on_pos_relation_extraction()
    kr = relation_extraction.kr

    KR2RDF(kr, "ttl", "teams_kr.ttl")

if __name__ == "__main__":
    main()