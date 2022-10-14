import unittest

import spacy

from term_extraction.term_extraction_methods.c_value import Cvalue


class TestCvalue(unittest.TestCase):
    """Test the C-value computation according to the examples in <https://doi.org/10.1007/s007999900023> (section 2.3.1, page 5).
    """

    @classmethod
    def setUpClass(self):
        self.test_terms = [
            "ADENOID CYSTIC BASAL CELL CARCINOMA",
            "CYSTIC BASAL CELL CARCINOMA",
            "ULCERATED BASAL CELL CARCINOMA",
            "RECURRENT BASAL CELL CARCINOMA",
            "CIRCUMSCRIBED BASAL CELL CARCINOMA",
            "BASAL CELL CARCINOMA"
        ]

        test_terms_counter = {
            "ADENOID CYSTIC BASAL CELL CARCINOMA": 5,
            "CYSTIC BASAL CELL CARCINOMA": 11,
            "ULCERATED BASAL CELL CARCINOMA": 7,
            "RECURRENT BASAL CELL CARCINOMA": 5,
            "CIRCUMSCRIBED BASAL CELL CARCINOMA": 3,
            "BASAL CELL CARCINOMA": 984
        }

        vocab_strings = []
        for term in self.test_terms:
            vocab_strings.extend(term.split())

        vocab = spacy.vocab.Vocab(strings=vocab_strings)

        self.doc_attribute_name = "cvalue_token_sequences"

        if not spacy.tokens.doc.Doc.has_extension(self.doc_attribute_name):
            spacy.tokens.doc.Doc.set_extension(
                self.doc_attribute_name, default=[])

        test_terms_spans = []
        corpus = []
        for term in self.test_terms:
            words = term.split()
            spaces = [True] * len(words)
            doc = spacy.tokens.Doc(vocab, words=words, spaces=spaces)
            doc._.set(self.doc_attribute_name, [doc[:]])
            test_terms_spans.append(doc[:])
            corpus.append(doc)

        my_c_val = Cvalue(
            corpus=corpus, tokenSequences_doc_attribute_name=self.doc_attribute_name, max_size_gram=5)

        # we manually set the candidate terms and their frequences otherwise the process considers all
        # the ngrams extracted from the terms. This is not done like this in the paper.
        my_c_val.candidateTermsSpans = test_terms_spans
        my_c_val.candidateTermsCounter = test_terms_counter

        self.c_values = my_c_val()

    def test_Cvalue_results(self):
        self.assertEqual(len(self.c_values), len(set(self.test_terms)))

        self.assertEqual(round(self.c_values[0].c_value, 2), 1551.36)
        self.assertEqual(
            self.c_values[0].candidate_term, "BASAL CELL CARCINOMA")

        self.assertEqual(round(self.c_values[1].c_value), 14.0)
        self.assertEqual(
            self.c_values[1].candidate_term, "ULCERATED BASAL CELL CARCINOMA")

        self.assertEqual(round(self.c_values[2].c_value), 12.0)
        self.assertEqual(
            self.c_values[2].candidate_term, "CYSTIC BASAL CELL CARCINOMA")

        self.assertEqual(round(self.c_values[3].c_value, 4), 11.6096)
        self.assertEqual(self.c_values[3].candidate_term,
                         "ADENOID CYSTIC BASAL CELL CARCINOMA")

        self.assertEqual(round(self.c_values[4].c_value), 10.0)
        self.assertEqual(
            self.c_values[4].candidate_term, "RECURRENT BASAL CELL CARCINOMA")

        self.assertEqual(round(self.c_values[5].c_value), 6.0)
        self.assertEqual(self.c_values[5].candidate_term,
                         "CIRCUMSCRIBED BASAL CELL CARCINOMA")


if __name__ == '__main__':
    unittest.main()
