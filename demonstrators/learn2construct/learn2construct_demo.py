from data_preprocessing.data_preprocessing_service import DataPreprocessing
from term_extraction.term_extraction_service import TermExtraction

def main() -> None:

    data_prep = DataPreprocessing()
    data_prep._set_corpus()

    term_extraction = TermExtraction(data_prep.corpus)
    pos_candidate_terms = term_extraction.on_pos_term_extraction()
    occurence_candidate_terms = term_extraction.on_occurrence_term_extraction()
    candidates_terms = [candidate_term for candidate_term in pos_candidate_terms if candidate_term in occurence_candidate_terms]
    print(candidates_terms) 

if __name__ == "__main__":
    main()

