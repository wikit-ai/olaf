import pytest
import spacy


@pytest.fixture(scope="session")
def en_sm_spacy_model():
    spacy_model = spacy.load("en_core_web_sm", exclude=["ner"])
    return spacy_model


@pytest.fixture(scope="session")
def en_md_spacy_model():
    spacy_model = spacy.load("en_core_web_md", exclude=["ner"])
    return spacy_model
