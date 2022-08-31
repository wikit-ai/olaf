import unittest

import spacy

from data_preprocessing.data_preprocessing_service import c_value_tokenizer, extract_text_sequences_from_corpus

test_texts_tokens = [
    ('Toothed lock washers - Type V, countersunk',
     ['Toothed', 'lock', 'washers', '-', 'Type', 'V', ',', 'countersunk']),
    ('Taper pin - conicity 1/50', ['Taper', 'pin', '-', 'conicity', '1/50']),
    ('T-head bolts with double nib',
     ['T-head', 'bolts', 'with', 'double', 'nib']),
    ('Handwheels, DIN 950, case-iron, d2 small, without keyway, without handle, form B-F/A',
     ['Handwheels', ',', 'DIN', '950', ',', 'case-iron', ',', 'd2', 'small', ',', 'without', 'keyway', ',', 'without', 'handle', ',', 'form', 'B-F/A']),
    ('Dog point hexagon socket set screw', [
     'Dog', 'point', 'hexagon', 'socket', 'set', 'screw']),
    ('Butterfly valve SV04 DIN BF, actuator PAMS93-size 1/2 NC + TOP',
     ['Butterfly', 'valve', 'SV04', 'DIN', 'BF', ',', 'actuator', 'PAMS93-size', '1/2', 'NC', '+', 'TOP']),
    ('Splined Shafts acc. to DIN 5463 / ISO 14',
     ['Splined', 'Shafts', 'acc', '.', 'to', 'DIN', '5463', '/', 'ISO', '14']),
    ('Grooved pins - Half-length reverse-taper grooved',
     ['Grooved', 'pins', '-', 'Half-length', 'reverse-taper', 'grooved']),
    ('Rod ends DIN ISO 12240-4 (DIN 648) E series stainless version with female thread, maintenance-free',
     ['Rod', 'ends', 'DIN', 'ISO', '12240', '-', '4', '(', 'DIN', '648', ')', 'E', 'series', 'stainless', 'version', 'with', 'female', 'thread', ',', 'maintenance-free']),
    ('Palm Grips, DIN 6335, light metal, with smooth blind hole, form C, DIN 6335-AL-63-20-C-PL',
     ['Palm', 'Grips', ',', 'DIN', '6335', ',', 'light', 'metal', ',', 'with', 'smooth', 'blind', 'hole', ',', 'form', 'C', ',', 'DIN', '6335-AL-63', '-', '20-C-PL']),
    ('Hexagon socket set screws with dog point, DIN EN ISO 4028-M5x12 - 45H',
     ['Hexagon', 'socket', 'set', 'screws', 'with', 'dog', 'point', ',', 'DIN', 'EN', 'ISO', '4028-M5x12', '-', '45H']),
    ('Rivet DIN 661  - Type A - 1,6 x 6',
     ['Rivet', 'DIN', '661', ' ', '-', 'Type', 'A', '-', '1,6', 'x', '6']),
    ('Welding neck flange - PN 400 - DIN 2627 - NPS 150',
     ['Welding', 'neck', 'flange', '-', 'PN', '400', '-', 'DIN', '2627', '-', 'NPS', '150']),
    ('Step Blocks, DIN 6326, adjustable, with spiral gearing, upper part, DIN 6326-K',
     ['Step', 'Blocks', ',', 'DIN', '6326', ',', 'adjustable', ',', 'with', 'spiral', 'gearing', ',', 'upper', 'part', ',', 'DIN', '6326-K']),
    ('Loose Slot Tenons, DIN 6323, form C, DIN 6323-20x28-C',
     ['Loose', 'Slot', 'Tenons', ',', 'DIN', '6323', ',', 'form', 'C', ',', 'DIN', '6323', '-', '20x28-C']),
    ('Hexagon nut DIN EN 24036 - M3.5 - St',
     ['Hexagon', 'nut', 'DIN', 'EN', '24036', '-', 'M3.5', '-', 'St'])
]

test_texts_text_sequences = [
    ('Toothed lock washers - Type V, countersunk',
     ['Toothed lock washers', 'Type V', 'countersunk']),
    ('Taper pin - conicity 1/50', ['Taper pin', 'conicity']),
    ('T-head bolts with double nib', ['T-head bolts with double nib']),
    ('Handwheels, DIN 950, case-iron, d2 small, without keyway, without handle, form B-F/A',
     ['Handwheels', 'DIN', 'case-iron', 'small', 'without keyway', 'without handle', 'form']),
    ('Dog point hexagon socket set screw', [
     'Dog point hexagon socket set screw']),
    ('Butterfly valve SV04 DIN BF, actuator PAMS93-size 1/2 NC + TOP',
     ['Butterfly valve', 'DIN BF', 'actuator', 'NC', 'TOP']),
    ('Splined Shafts acc. to DIN 5463 / ISO 14',
     ['Splined Shafts acc', 'to DIN', 'ISO']),
    ('Grooved pins - Half-length reverse-taper grooved',
     ['Grooved pins', 'Half-length reverse-taper grooved']),
    ('Rod ends DIN ISO 12240-4 (DIN 648) E series stainless version with female thread, maintenance-free',
     ['Rod ends DIN ISO', 'DIN', 'E series stainless version with female thread', 'maintenance-free']),
    ('Palm Grips, DIN 6335, light metal, with smooth blind hole, form C, DIN 6335-AL-63-20-C-PL',
     ['Palm Grips', 'DIN', 'light metal', 'with smooth blind hole', 'form C', 'DIN']),
    ('Hexagon socket set screws with dog point, DIN EN ISO 4028-M5x12 - 45H',
     ['Hexagon socket set screws with dog point', 'DIN EN ISO']),
    ('Rivet DIN 661  - Type A - 1,6 x 6', ['Rivet DIN', 'Type A', 'x']),
    ('Welding neck flange - PN 400 - DIN 2627 - NPS 150',
     ['Welding neck flange', 'PN', 'DIN', 'NPS']),
    ('Step Blocks, DIN 6326, adjustable, with spiral gearing, upper part, DIN 6326-K',
     ['Step Blocks', 'DIN', 'adjustable', 'with spiral gearing', 'upper part', 'DIN']),
    ('Loose Slot Tenons, DIN 6323, form C, DIN 6323-20x28-C',
     ['Loose Slot Tenons', 'DIN', 'form C', 'DIN']),
    ('Hexagon nut DIN EN 24036 - M3.5 - St', ['Hexagon nut DIN EN', 'St'])
]

nlp = spacy.load("en_core_web_sm", disable=[
                 'tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])
nlp.tokenizer = c_value_tokenizer(nlp)


class TestCvalueTokenize(unittest.TestCase):
    def test_c_value_tokenizer(self):
        for idx, doc in enumerate([nlp(e[0]) for e in test_texts_tokens]):
            self.assertEqual([token.text for token in doc],
                             test_texts_tokens[idx][1])


class TestExtractTextSequencesFromCorpus(unittest.TestCase):
    def test_extract_text_sequences_from_corpus(self):
        for idx, doc in enumerate([nlp(e[0]) for e in test_texts_text_sequences]):
            selected_sequences = [
                span.text for span in extract_text_sequences_from_corpus([doc])]
            self.assertEqual(selected_sequences,
                             test_texts_text_sequences[idx][1])


if __name__ == '__main__':
    unittest.main()
