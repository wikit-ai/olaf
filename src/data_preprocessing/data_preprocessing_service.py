import re
from typing import Iterable, List

import spacy
import spacy.tokens.doc
import spacy.tokenizer
import spacy.tokens.span
import spacy.language

from data_preprocessing.data_preprocessing_repository import load_corpus, load_spacy_model
import config.logging_config as logging_config


def extract_text_sequences_from_corpus(docs: Iterable[spacy.tokens.doc.Doc]) -> List[spacy.tokens.span.Span]:
    """Extract all the alphanumeric sequences of tokens from a corpus of texts.
        In the list returned, all occurences of grams are returned. There is no filtering or duplicates removal.
    Parameters
    ----------
    docs : Iterable[spacy.tokens.doc.Doc]
        An iterable over the Spacy documents.

    Returns
    -------
    List[spacy.tokens.span.Span]
        The list of token sequences (Span) contained in the corpus.
    """
    some_num_pattern = re.compile(r'''^[xX]+-?[xX]*$''')
    str_token_sequences = []

    for doc in docs:
        str_token_seq = []

        for token in doc:

            # we rely on the token.shape_ attribute to check that the token contains only letters and dashes
            if (some_num_pattern.match(token.shape_)):
                str_token_seq.append(token)

            elif len(str_token_seq) > 0:
                str_token_sequences.append(
                    spacy.tokens.span.Span(doc, str_token_seq[0].i, str_token_seq[-1].i + 1))
                str_token_seq = []

        if len(str_token_seq) > 0:
            str_token_sequences.append(spacy.tokens.span.Span(
                doc, str_token_seq[0].i, str_token_seq[-1].i + 1))

    return str_token_sequences


class Data_Preprocessing():
    """First basic processing of the corpus.

    Attributes: 
        corpus: List(str)
            Documents to process
    """

    def __init__(self) -> None:
        pass

    def _set_corpus(self) -> None:
        self.corpus = load_corpus()

    def _set_tokenizer(self) -> None:
        pass

    def get_token_filters(self):
        pass

    def document_representation(self) -> List[spacy.tokens.doc.Doc]:
        """Convert text to spacy document representation.

        Returns
        -------
        List(spacy.tokens.doc)
            Corpus of spacy document representation.
        """
        spacy_model = load_spacy_model()
        corpus_preprocessed = []

        try:
            for spacy_document in spacy_model.pipe(self.corpus):
                corpus_preprocessed.append(spacy_document)
        except Exception as _e:
            logging_config.logger.error(
                "Could not load content as spacy document. Trace : %s", _e)
        else:
            logging_config.logger.info(
                "File content converted to spacy document.")
        return corpus_preprocessed

    def standard_cleaning(self, spacy_corpus: List[spacy.tokens.doc.Doc]) -> List[List[spacy.tokens.token.Token]]:
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
        try:
            for document in spacy_corpus:
                clean_doc = []
                for token in document:
                    if (not (token.is_stop) and
                        not (token.is_punct) and
                        not (token.like_num) and
                            not (token.like_url)):
                        clean_doc.append(token)
                clean_corpus.append(clean_doc)
        except Exception as _e:
            logging_config.logger.error(
                "Could not filter spacy tokens. Trace : %s", _e)
        else:
            logging_config.logger.info(
                "File content cleaned with spacy filters.")
        return clean_corpus


data_preprocessing = Data_Preprocessing()
