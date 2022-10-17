import re
from data_preprocessing.data_preprocessing_service import Data_Preprocessing
from term_extraction.term_extraction_service import Term_Extraction
# from relation_extraction.relation_extraction_service import relation_extraction

# ----------Corpus preprocessing--------------------
data_preprocessing = Data_Preprocessing()
data_preprocessing._set_corpus()
print(data_preprocessing.corpus[0])
filtered_corpus = data_preprocessing.corpus_cleaning()
print(filtered_corpus[0])

# -----------Term extraction------------------------

term_extraction = Term_Extraction(data_preprocessing.corpus)
pos_candidate_terms = term_extraction.on_pos_token_filtering(on_lemma=True)
occurence_candidate_terms = term_extraction.occurence_filtering(on_lemma=True)
candidates_terms = list(set(pos_candidate_terms) & set(occurence_candidate_terms))
print(candidates_terms)