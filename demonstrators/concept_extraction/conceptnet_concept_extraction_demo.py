from commons.ontology_learning_schema import CandidateTerm
from concept_extraction.concept_extraction_service import ConceptExtraction
from term_enrichment.term_enrichment_service import TermEnrichment

config = {
    "conceptnet": {
        "api_resp_batch_size": 1000,
        "lang": "en",
        "term_max_tokens": 1,
        "validation_sources": ["dbpedia.org", "wikidata.dbpedia.org"],
        "use_synonyms": True,
        "merge_concepts_on_external_ids": True,
        "merge_candidate_terms_on_syns": True
    }
}

terms = [
    "control unit",
    "selector switch",
    "communication module",
    "pilot light",
    "control circuits",
    "push button",
    "circuit boards",
    "protective cover",
    "power base",
    "supply line",
    "circuit breaker",
    "miniature circuit",
    "residual current",
    "motor control",
    "motor",
    "unit",
    "circuit",
    "screw",
    "current"
]


def main() -> None:

    candidate_terms = [CandidateTerm(term) for term in terms]

    print("Candidate terms:")
    for t in candidate_terms:
        print(t.value, ", synonyms: ", t.synonyms)

    print()
    print("Wordnet term enrichment ...")
    term_enrichment = TermEnrichment(candidate_terms)
    term_enrichment.wordnet_term_enrichment()
    print("Candidate terms:")
    for t in candidate_terms:
        print(t.value, ", synonyms: ", t.synonyms)

    print()
    print("ConceptNet concept extraction")
    concept_extraction = ConceptExtraction(
        candidate_terms, config=config)
    concept_extraction.conceptnet_concept_extraction()

    print("Knowledge Representation:")
    for concept in concept_extraction.kr.concepts:
        print(concept.uid, concept.terms)
        print()


if __name__ == "__main__":
    main()
