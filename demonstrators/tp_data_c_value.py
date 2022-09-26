

import os

from config.core import DATA_PATH
from term_extraction.term_extraction_service import Term_Extraction
# to make sure the methods are registered inthe spacy registry
import data_preprocessing.data_preprocessing_methods.spacy_pipeline_components
import tqdm
from data_preprocessing.data_preprocessing_repository import load_corpus
from data_preprocessing.data_preprocessing_service import Data_Preprocessing


def main() -> None:

    data_prep = Data_Preprocessing()
    data_prep._set_corpus()
    docs = data_prep.document_representation()

    term_extraction = Term_Extraction()

    term_extraction.c_value_term_extraction(docs, "c_value_token_sequences", 5)

    c_values = term_extraction.c_value()

    with open(os.path.join(DATA_PATH, "schneider_texts_cvalues.txt"), "w", encoding='utf8') as file:
        for c_val in tqdm.tqdm(c_values):
            file.write(f"{c_val.c_value:.2f} -- {c_val.candidate_term}\n")


if __name__ == "__main__":
    main()
