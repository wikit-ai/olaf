from commons.ontology_learning_schema import CandidateTerm
from concept_extraction.concept_extraction_service import ConceptExtraction

def main() -> None:

    candidate_terms = [
        CandidateTerm("téléphone", synonyms={"mobile", "portable"}),
        CandidateTerm("portable", synonyms={"smartphone"}),
        CandidateTerm("iphone", synonyms={"smartphone"}),
        CandidateTerm("ordinateur", synonyms={"pc", "ordi", "ordinateur portable"})
    ]

    print("\nList of candidate tersm : ")
    print(candidate_terms)

    concept_extraction = ConceptExtraction(candidate_terms)
    concept_extraction.group_by_synonyms()

    print("\n\nKnowledge representation built : ")
    print(concept_extraction.kr)

if __name__ == "__main__":
    main()
