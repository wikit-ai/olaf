from term_enrichment.term_enrichment_service import TermEnrichment
from commons.ontology_learning_schema import CandidateTerm


def main() -> None:
    terms = [
        "screw",
        "bolt",
        "screwdriver",
        "circuit breaker",
        "pump",
        "notexistingword"
    ]

    candidate_terms = [CandidateTerm(term) for term in terms]

    print("Candidate terms before ConceptNet enrichment : ")
    for candidate_term in candidate_terms:
        print(candidate_term)

    print()
    print("Enriching candidate terms using ConceptNet ...")
    print()
    term_enrichment = TermEnrichment(candidate_terms)
    term_enrichment.conceptnet_term_enrichment()

    print("Candidate terms after ConceptNet enrichment : ")
    for candidate_term in candidate_terms:
        print(candidate_term)


if __name__ == "__main__":
    main()
