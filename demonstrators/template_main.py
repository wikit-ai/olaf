from data_preprocessing.data_preprocessing_service import DataPreprocessing
from term_extraction.term_extraction_service import TermExtraction
from term_enrichment.term_enrichment_service import TermEnrichment
from concept_extraction.concept_extraction_service import ConceptExtraction
from concept_hierarchy.concept_hierarchy_service import ConceptHierarchy
from commons.ontology_learning_schema import KR


def main() -> None:

    # initialisation
    corpus = list()
    candidate_terms = list()
    knowledge_representation = KR()

    # ontology learning process

    # data prep
    data_prep = DataPreprocessing()
    data_prep._set_corpus()

    corpus = data_prep.corpus

    # term extraction --> set global variable candidate_terms

    term_extraction = TermExtraction(data_prep.corpus)
    candidate_terms = term_extraction.on_pos_term_extraction()
    

    print(candidate_terms)

    # term enrichment --> update a list of Candidate Terms

    term_enrich = TermEnrichment(candidate_terms)
    term_enrich.wordnet_term_enrichment()
    enrich_candidate_terms = term_enrich.candidate_terms

    # Concept extraction

    concept_extraction = ConceptExtraction(enrich_candidate_terms)
    concept_extraction.group_by_synonyms()

    kr = concept_extraction.kr

    # Concept hierarchy

    concept_hierarchy = ConceptHierarchy(corpus, kr)
    concept_hierarchy.term_subsumption()

    kr = concept_hierarchy.kr

if __name__ == "__main__":
    main()
