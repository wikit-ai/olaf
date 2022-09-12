import spacy
from typing import List
import logging_config
from data_preprocessing.data_preprocessing_repository import load_corpus, load_spacy_model

class Data_Preprocessing():
    """First basic processing of the corpus.

    Attributes: 
        corpus: List(str)
            Documents to process
    """

    def __init__(self) -> None:
        self.corpus = load_corpus()
    
    def document_representation(self) -> List[spacy.tokens.doc.Doc]:
        """Convert text to spacy document representation.

        Returns
        -------
        List(spacy.tokens.doc)
            Corpus of spacy document representation.
        """
        spacy_model = load_spacy_model()
        corpus_preprocessed = []

        try : 
            for spacy_document in spacy_model.pipe(self.corpus):
                corpus_preprocessed.append(spacy_document)
        except Exception as _e: 
            logging_config.logger.error("Could not load content as spacy document. Trace : %s", _e)
        else : 
            logging_config.logger.info("File content converted to spacy document.")
        return corpus_preprocessed

    def standard_cleaning(self,spacy_corpus: List[spacy.tokens.doc.Doc])-> List[List[spacy.tokens.token.Token]]:
        """Cleaning of the corpus from spacy pipeline.
        Removing stop-words, punctuation, number and url.

        Parameters
        ----------
        spacy_corpus : List[spacy.tokens.doc.Doc]
            Corpus with spacy representation.

        Returns
        -------
        List[List[spacy.tokens.token.Token]]
            Corpus filtered.
        """
        clean_corpus = []
        try : 
            for document in spacy_corpus:
                clean_doc = []
                for token in document:
                    if (not(token.is_stop) and
                        not(token.is_punct) and
                        not(token.like_num) and
                        not(token.like_url)):
                        clean_doc.append(token)
                clean_corpus.append(clean_doc)
        except Exception as _e:
            logging_config.logger.error("Could not filter spacy tokens. Trace : %s", _e)
        else : 
            logging_config.logger.info("File content cleaned with spacy filters.")
        return clean_corpus


data_preprocessing = Data_Preprocessing()