import pandas as pd
import pytest
import spacy.tokens

from olaf.commons.errors import EmptyCorpusError, FileOrDirectoryNotFoundError
from olaf.repository.corpus_loader.csv_corpus_loader import CsvCorpusLoader


@pytest.fixture(scope="session")
def csv_data_folder(tmp_path_factory):
    path = tmp_path_factory.mktemp("test_data")
    for i in range(3):
        filename = path / f"doc{i}.csv"
        df = pd.DataFrame([f"content for doc {i}"], columns=["content"])
        df.to_csv(filename, index=False)

    return path


@pytest.fixture(scope="session")
def schneider_csv_sample(tmp_path_factory):
    path = tmp_path_factory.mktemp("test_data") / "schneider_texts_sample.csv"

    sentences = [
        "prismaset_lvs08879_plain metal gl.pl.for pack",
        "connection module for 3p standard brake resistor",
        "tesys gv4l - magnetic motor breaker 2a to 115a - gv4l03b",
        "lexium servo motor bsh with straight connector, flange 70, peak ip50 untapped, with holding break and compatible with lxm05ad10m2",
    ]

    df = pd.DataFrame(sentences, columns=["content"])
    df.to_csv(path, index=False)

    return path


def test_read_corpus_invalid_path():
    corpus_path = "invalid_path"
    column_name = "content"
    corpus_loader = CsvCorpusLoader(corpus_path, column_name)

    with pytest.raises(FileOrDirectoryNotFoundError):
        corpus = corpus_loader._read_corpus()


def test_read_corpus_folder(csv_data_folder):
    column_name = "content"
    corpus_loader = CsvCorpusLoader(csv_data_folder, column_name)
    corpus = corpus_loader._read_corpus()

    assert len(corpus) == 3


def test_read_corpus_file(schneider_csv_sample):
    column_name = "content"
    corpus_loader = CsvCorpusLoader(schneider_csv_sample, column_name)
    corpus = corpus_loader._read_corpus()

    assert len(corpus) == 4


def test_corpus_loader_empty_error(schneider_csv_sample, en_sm_spacy_model):
    column_name = "description"
    corpus_loader = CsvCorpusLoader(schneider_csv_sample, column_name)

    with pytest.raises(EmptyCorpusError):
        corpus = corpus_loader(en_sm_spacy_model)


def test_corpus_loader(schneider_csv_sample, en_sm_spacy_model):
    column_name = "content"
    corpus_loader = CsvCorpusLoader(schneider_csv_sample, column_name)
    corpus = corpus_loader(en_sm_spacy_model)

    assert len(corpus) == 4
    assert isinstance(corpus[0], spacy.tokens.Doc)
