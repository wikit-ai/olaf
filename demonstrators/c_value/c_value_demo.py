from term_extraction.term_extraction_service import Term_Extraction
from data_preprocessing.data_preprocessing_service import Data_Preprocessing


def main() -> None:

    data_prep = Data_Preprocessing()
    data_prep._set_corpus()

    term_extraction = Term_Extraction(data_prep.corpus)

    candidate_terms = term_extraction.c_value_term_extraction()

    for term in candidate_terms:
        print(term)


if __name__ == "__main__":
    main()