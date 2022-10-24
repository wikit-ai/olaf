import os

from config.core import DATA_PATH
from term_extraction.term_extraction_service import Term_Extraction
import tqdm
from data_preprocessing.data_preprocessing_service import Data_Preprocessing


def main() -> None:

    data_prep = Data_Preprocessing()
    data_prep._set_corpus()

    term_extraction = Term_Extraction(data_prep.corpus)

    term_extraction.c_value_term_extraction("selected_tokens", 5)

    c_values = term_extraction.c_value()

    with open(os.path.join(DATA_PATH, "data_files", "cvalues_demo.txt"), "w", encoding='utf8') as file:
        for c_val in tqdm.tqdm(c_values):
            file.write(f"{c_val.c_value:.5f} -- {c_val.candidate_term}\n")


if __name__ == "__main__":
    main()
