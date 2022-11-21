from data_preprocessing.data_preprocessing_service import Data_Preprocessing
from term_extraction.term_extraction_service import Term_Extraction


def main() -> None:

    # initialisation
    corpus = list()
    candidate_terms = list()
    knowledge_representation = KR()

    # ontology learning process

    # data prep
    data_prep = Data_Preprocessing()
    data_prep._set_corpus()

    corpus = data_prep.corpus

    # term extraction --> set global variable candidate_terms

    term_extraction = Term_Extraction(data_prep.corpus)
    pos_candidate_terms = term_extraction.on_pos_term_extraction()
    occurence_candidate_terms = term_extraction.on_occurence_term_extraction()
    candidates_terms = list(set(pos_candidate_terms) &
                            set(occurence_candidate_terms))

    print(candidates_terms)

    # term enrichment --> update a list of Candidate Terms

    term_enrich = TermEnrichment(candidate_terms)
    term_enrich.wordnet_enrichment()

    # Concept extraction


if __name__ == "__main__":
    main()
