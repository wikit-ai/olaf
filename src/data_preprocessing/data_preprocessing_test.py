import re
from collections import Counter
import unittest

import spacy
from commons.spacy_processing_tools import build_spans_from_tokens
from data_preprocessing.data_preprocessing_methods.token_selectors import TokenSelectionPipeline, select_on_pos, select_on_occurrence_count

from data_preprocessing.data_preprocessing_service import DataPreprocessing
# from data_preprocessing.data_preprocessing_methods.tokenizers import create_no_split_on_dash_in_words_tokenizer
from config.logging_config import logger


class TestDataPreprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        self.data_preprocessing = DataPreprocessing()
        self.texts_and_tokens = [
            ('Toothed lock washers - Type V, countersunk',
             ['Toothed', 'lock', 'washers', '-', 'Type', 'V', ',', 'countersunk']),
            ('Taper pin - conicity 1/50',
             ['Taper', 'pin', '-', 'conicity', '1/50']),
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

        self.texts_and_sequences2extract = [
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
            ('Rivet DIN 661  - Type A - 1,6 x 6',
             ['Rivet DIN', 'Type A', 'x']),
            ('Welding neck flange - PN 400 - DIN 2627 - NPS 150',
             ['Welding neck flange', 'PN', 'DIN', 'NPS']),
            ('Step Blocks, DIN 6326, adjustable, with spiral gearing, upper part, DIN 6326-K',
             ['Step Blocks', 'DIN', 'adjustable', 'with spiral gearing', 'upper part', 'DIN']),
            ('Loose Slot Tenons, DIN 6323, form C, DIN 6323-20x28-C',
             ['Loose Slot Tenons', 'DIN', 'form C', 'DIN']),
            ('Hexagon nut DIN EN 24036 - M3.5 - St',
             ['Hexagon nut DIN EN', 'St'])
        ]

        self.spacy_model = spacy.load("en_core_web_sm")

        create_tokenizer = spacy.util.registry.get(
            "tokenizers", "no_split_on_dash_in_words_tokenizer")
        custom_tokenizer = create_tokenizer()(self.spacy_model)
        self.spacy_model.tokenizer = custom_tokenizer
        # self.spacy_model.tokenizer = create_no_split_on_dash_in_words_tokenizer()(self.spacy_model)

        self.doc_attribute_name = "selected_tokens_4_test"
        self.spacy_model.add_pipe("token_selector", last=True, config={
            "token_selector_config": {
                "pipeline_name": "test_pipeline",
                "token_selector_names": ["filter_punct", "filter_num", "filter_url"],
                "doc_attribute_name": self.doc_attribute_name,
                'make_spans': False,
                "select_on_pos": {
                    "pos_to_select": ["NOUN"]
                },
                "select_on_occurrence": {
                    "occurrence_threshold" : 3
                }
            }
        })

    def test_extension_set(self) -> None:
        txt = "hello, my name is Matthias, I am 26, and I love pasta. By the way my website is http://matthias.com"
        doc = self.spacy_model(txt)
        self.assertTrue(doc.has_extension(self.doc_attribute_name))

    def test_no_split_on_dash_in_words_tokenizer(self) -> None:
        for idx, doc in enumerate([self.spacy_model(e[0]) for e in self.texts_and_tokens]):
            self.assertEqual([token.text for token in doc],
                             self.texts_and_tokens[idx][1])

    def test_token_selector_pipeline_component(self) -> None:
        txt = "hello, my name is Matthias, I am 26, and I love pasta. By the way my website is http://matthias.com"
        tokens_text_to_be_selected = [
            "hello", "my", "name", "is", "Matthias", "I", "am",
            "and", "I", "love", "pasta", "By", "the", "way", "my", "website", "is"
        ]
        doc = self.spacy_model(txt)
        selected_tokens_text = [
            token.text for token in doc._.get(self.doc_attribute_name)]
        self.assertListEqual(tokens_text_to_be_selected, selected_tokens_text)

    def test_select_on_pos(self):
        txt = "hello, my name is Matthias, I am 26, and I love pasta. By the way my website is http://matthias.com"
        doc = self.spacy_model(txt)
        text_noun = ["name", "pasta", "way","website"]
        results = [token.text for token in doc if select_on_pos(token, ["NOUN"])]
        self.assertEqual(text_noun,results)

    def test_select_on_occurrence(self):
        txt = "hello, my name is Matthias, I am 26, and I love pasta and website. By the way my website is http://matthias.com . My cats names are Jack and Hector but they do not have website."
        doc = self.spacy_model(txt)
        terms = [token.text for token in doc._.get(self.doc_attribute_name)]
        term_lemmas = [token.lemma_ for token in doc._.get(self.doc_attribute_name)]

        counter_terms = Counter(terms)
        counter_term_lemmas = Counter(term_lemmas)

        self.assertFalse(select_on_occurrence_count(doc._.get(self.doc_attribute_name)[2],1,counter_terms,False)) 
        self.assertTrue(select_on_occurrence_count(doc._.get(self.doc_attribute_name)[12],2,counter_terms,False))

        self.assertTrue(select_on_occurrence_count(doc._.get(self.doc_attribute_name)[3],1,counter_term_lemmas,True))
        self.assertTrue(select_on_occurrence_count(doc._.get(self.doc_attribute_name)[12],2,counter_terms,True))

    def test_build_spans_from_tokens(self) -> None:
        txt = "hello, my name is Matthias, I am 26, and I love pasta. By the way my website is http://matthias.com"
        spans_text_to_be_extracted = [
            "hello", "my name is Matthias", "I am", "and I love pasta", "By the way my website is"
        ]
        doc = self.spacy_model(txt)
        selected_spans = build_spans_from_tokens(
            doc._.get(self.doc_attribute_name), doc)
        selected_span_texts = [span.text for span in selected_spans]
        self.assertListEqual(spans_text_to_be_extracted, selected_span_texts)

    def test_token_selector_pipeline_component_with_spans(self) -> None:
        txt = "hello, my name is Matthias, I am 26, and I love pasta. By the way my website is http://matthias.com"
        spans_text_to_be_extracted = [
            "hello", "my name is Matthias", "I am", "and I love pasta", "By the way my website is"
        ]

        span_attribute_name = self.doc_attribute_name + "_span"
        self.spacy_model.replace_pipe("token_selector", "token_selector", config={
            "token_selector_config": {
                "make_spans": True,
                "pipeline_name": "test_pipeline",
                "token_selector_names": ["filter_punct", "filter_num", "filter_url"],
                "doc_attribute_name": span_attribute_name
            }
        })
        doc = self.spacy_model(txt)
        selected_spans = doc._.get(span_attribute_name)
        selected_span_texts = [span.text for span in selected_spans]
        self.assertListEqual(spans_text_to_be_extracted, selected_span_texts)

    def test_load_selectors_from_config(self) -> None:
        token_select_pipeline_config = {"pipeline_name": "test_pipeline",
                                        "token_selector_names": ["not_exist_token_selector", "select_on_pos", "filter_punct", "filter_num", "filter_url"],
                                        "select_on_pos": {}}

        test_token_select_pipeline = TokenSelectionPipeline(
            token_select_pipeline_config)

        with self.assertLogs(logger, level='ERROR') as cm:
            test_token_select_pipeline = TokenSelectionPipeline(
                token_select_pipeline_config)
            self.assertRegex(
                " ".join(cm.output), re.compile("not_exist_token_selector token selector not found"))

        with self.assertLogs(logger, level='ERROR') as cm:
            token_select_pipeline_config[
                "token_selector_names"] = ["select_on_pos", "filter_punct", "filter_num", "filter_url"]
            test_token_select_pipeline = TokenSelectionPipeline(
                token_select_pipeline_config)
            self.assertRegex(
                " ".join(cm.output), re.compile("Parameter pos_to_select for token selector select_on_pos not found in pipeline config"))

        with self.assertLogs(logger, level='INFO') as cm:
            token_select_pipeline_config["select_on_pos"] = {
                "pos_to_select": ["NOUN", "VERB"]}
            test_token_select_pipeline = TokenSelectionPipeline(
                token_select_pipeline_config)
            self.assertRegex(
                " ".join(cm.output), re.compile("Token selectors loaded for pipeline test_pipeline"))

        self.assertEqual(len(test_token_select_pipeline.token_selectors), 4)

if __name__ == '__main__':
    unittest.main()
