import os.path
import re
import unittest

import spacy
from data_preprocessing.data_preprocessing_methods.spacy_processing_tools import build_spans_from_tokens
from data_preprocessing.data_preprocessing_schema import TokenSelectionPipeline

from data_preprocessing.data_preprocessing_service import Data_Preprocessing, extract_text_sequences_from_corpus
from data_preprocessing.data_preprocessing_methods.spacy_pipeline_components import create_no_split_on_dash_in_words_tokenizer
from config.core import CONFIG_PATH
from config.logging_config import logger


class TestDataPreprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        self.data_preprocessing = Data_Preprocessing()
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

        self.spacy_model = spacy.load("en_core_web_sm", exclude=[
            'tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'])
        self.spacy_model.tokenizer = create_no_split_on_dash_in_words_tokenizer()(self.spacy_model)

        self.doc_attribute_name = "selected_tokens_4_test"
        self.spacy_model.add_pipe("token_selector", last=True, config={
            "token_selection_config_path": os.path.join(CONFIG_PATH, "token_selector_config4test.ini"),
            "doc_attribute_name": self.doc_attribute_name
        })

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
            "make_spans": True,
            "token_selection_config_path": os.path.join(CONFIG_PATH, "token_selector_config4test.ini"),
            "doc_attribute_name": span_attribute_name
        })

        doc = self.spacy_model(txt)
        selected_spans = doc._.get(span_attribute_name)
        selected_span_texts = [span.text for span in selected_spans]
        self.assertListEqual(spans_text_to_be_extracted, selected_span_texts)

    def test_load_selectors_from_config(self) -> None:
        token_select_pipeline_config = {
            "TOKEN_SELECTION_PIPELINE_CONFIG": {
                "PIPELINE_NAME": "Test TokenSelectionPipeline",
                "TOKEN_SELECTOR_NAMES": "not_exist_token_selector select_on_pos filter_punct filter_num filter_url"
            }
        }

        test_token_select_pipeline = TokenSelectionPipeline(
            token_select_pipeline_config)

        with self.assertLogs(logger, level='ERROR') as cm:
            test_token_select_pipeline = TokenSelectionPipeline(
                token_select_pipeline_config)
            self.assertRegex(
                " ".join(cm.output), re.compile("not_exist_token_selector token selector not found"))

        with self.assertLogs(logger, level='ERROR') as cm:
            token_select_pipeline_config["TOKEN_SELECTION_PIPELINE_CONFIG"][
                "TOKEN_SELECTOR_NAMES"] = "select_on_pos filter_punct filter_num filter_url"
            test_token_select_pipeline = TokenSelectionPipeline(
                token_select_pipeline_config)
            self.assertRegex(
                " ".join(cm.output), re.compile("Parameter pos_to_select for token selector select_on_pos not found in pipeline config"))

        with self.assertLogs(logger, level='INFO') as cm:
            token_select_pipeline_config["TOKEN_SELECTION_PIPELINE_CONFIG"]["pos_to_select"] = "NOUN VERB"
            test_token_select_pipeline = TokenSelectionPipeline(
                token_select_pipeline_config)
            self.assertRegex(
                " ".join(cm.output), re.compile("Token selectors loaded for pipeline Test TokenSelectionPipeline"))

        self.assertEqual(len(test_token_select_pipeline.token_selectors), 4)

    def test_extract_text_sequences_from_corpus(self) -> None:
        for idx, doc in enumerate([self.spacy_model(e[0]) for e in self.texts_and_sequences2extract]):
            selected_sequences = [
                span.text for span in extract_text_sequences_from_corpus([doc])]
            self.assertEqual(selected_sequences,
                             self.texts_and_sequences2extract[idx][1])


if __name__ == '__main__':
    unittest.main()
