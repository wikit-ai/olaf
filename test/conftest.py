import pytest
import spacy


@pytest.fixture(scope="session")
def en_sm_spacy_model():
    spacy_model = spacy.load("en_core_web_sm")
    return spacy_model
