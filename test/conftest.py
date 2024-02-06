import os
import tempfile

import pytest
import spacy
from pytest import MonkeyPatch


@pytest.fixture(scope="session")
def test_data_path():
    with tempfile.TemporaryDirectory() as newpath:
        old_cwd = os.getcwd()
        os.chdir(newpath)
        mp = MonkeyPatch()
        mp.setenv("DATA_PATH", newpath)
        yield
        os.chdir(old_cwd)


@pytest.fixture(scope="session")
def en_sm_spacy_model():
    spacy_model = spacy.load("en_core_web_sm", exclude=["ner"])
    return spacy_model


@pytest.fixture(scope="session")
def en_md_spacy_model():
    spacy_model = spacy.load("en_core_web_md", exclude=["ner"])
    return spacy_model
