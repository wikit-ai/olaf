from term_enrichment.term_enrichment_service import TermEnrichment
from term_enrichment.term_enrichment_schema import CandidateTerm


def main() -> None:
    terms = [
        "screw",
        "bolts",
        "nuts",
        "screwdriver",
        "circuit breaker"
    ]

    candidate_terms = [CandidateTerm(term) for term in terms]

    print("Candidate terms before WordNet enrichment : ")
    for candidate_term in candidate_terms:
        print(candidate_term)

    print()
    print("Enriching candidate terms using WordNet ...")
    print()
    term_enrichment = TermEnrichment(candidate_terms)
    term_enrichment.wordnet_term_enrichment()

    print("Candidate terms after WordNet enrichment : ")
    for candidate_term in candidate_terms:
        print(candidate_term)


if __name__ == "__main__":
    main()
