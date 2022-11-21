import uuid

from data_preprocessing.data_preprocessing_service import Data_Preprocessing
from term_extraction.term_extraction_service import Term_Extraction
from concept_hierarchy.concept_hierarchy_service import Concept_Hierarchy
from commons.ontology_learning_schema import Concept,KR

def main() -> None:

    data_prep = Data_Preprocessing()
    data_prep._set_corpus()

    term_extraction = Term_Extraction(data_prep.corpus)
    pos_candidate_terms = term_extraction.on_pos_term_extraction()
    occurence_candidate_terms = term_extraction.on_occurence_term_extraction()
    candidates_terms = list(set(pos_candidate_terms) & set(occurence_candidate_terms))
    print(candidates_terms) 

    kr = KR()

    for candidate_term in candidates_terms:
        concept = Concept(uuid.uuid4())
        concept.terms.add(candidate_term)
        kr.concepts.add(concept)

    concept_hierarchy = Concept_Hierarchy(data_prep.corpus,kr)
    concept_hierarchy.term_subsumption()

    print(kr)

if __name__ == "__main__":
    main()

