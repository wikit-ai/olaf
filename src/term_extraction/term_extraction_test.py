from collections import defaultdict
import unittest

import spacy

from term_extraction.term_extraction_service import Cvalue

test_terms = []
test_terms.extend(["ADENOID CYSTIC BASAL CELL CARCINOMA"] * 5)
test_terms.extend(["CYSTIC BASAL CELL CARCINOMA"] * 11)
test_terms.extend(["ULCERATED BASAL CELL CARCINOMA"] * 7)
test_terms.extend(["RECURRENT BASAL CELL CARCINOMA"] * 5)
test_terms.extend(["CIRCUMSCRIBED BASAL CELL CARCINOMA"] * 3)
test_terms.extend(["BASAL CELL CARCINOMA"] * 984)

vocab_strings = []
for term in test_terms:
    vocab_strings.extend(term.split())

vocab = spacy.vocab.Vocab(strings=vocab_strings)

test_terms_spans = []

for term in test_terms:
    words = term.split()
    spaces = [True] * len(words)
    doc = spacy.tokens.Doc(vocab, words=words, spaces=spaces)
    span = spacy.tokens.Span(doc, doc[0].i, doc[-1].i + 1)
    test_terms_spans.append(span)

my_c_val = Cvalue(tokenSequences=test_terms_spans, max_size_gram=5)

test_candidate_terms_by_size = defaultdict(list)
test_candidateTerms = [span.text for span in test_terms_spans]
for term in test_candidateTerms:
    test_candidate_terms_by_size[len(term.split())].append(term)

my_c_val.candidateTerms, my_c_val.candidateTermsCounter = my_c_val._order_count_candidate_terms(
    test_candidate_terms_by_size)
my_c_val._compute_c_values()

c_values = my_c_val()


class TestCvalue(unittest.TestCase):
    def test_Cvalue_results(self):
        self.assertEqual(len(c_values), len(set(test_terms)))

        self.assertEqual(round(c_values[0][0], 2), 1551.36)
        self.assertEqual(c_values[0][1], "BASAL CELL CARCINOMA")

        self.assertEqual(round(c_values[1][0]), 14.0)
        self.assertEqual(c_values[1][1], "ULCERATED BASAL CELL CARCINOMA")

        self.assertEqual(round(c_values[2][0]), 12.0)
        self.assertEqual(c_values[2][1], "CYSTIC BASAL CELL CARCINOMA")

        self.assertEqual(round(c_values[3][0], 4), 11.6096)
        self.assertEqual(c_values[3][1], "ADENOID CYSTIC BASAL CELL CARCINOMA")

        self.assertEqual(round(c_values[4][0]), 10.0)
        self.assertEqual(c_values[4][1], "RECURRENT BASAL CELL CARCINOMA")

        self.assertEqual(round(c_values[5][0]), 6.0)
        self.assertEqual(c_values[5][1], "CIRCUMSCRIBED BASAL CELL CARCINOMA")


if __name__ == '__main__':
    unittest.main()
