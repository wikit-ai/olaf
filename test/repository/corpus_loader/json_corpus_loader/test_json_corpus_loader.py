import json
import os.path

from olaf.commons.errors import FileOrDirectoryNotFoundError
from olaf.commons.paths import PROJECT_ROOT_PATH
from olaf.repository.corpus_loader.json_corpus_loader import JsonCorpusLoader

import pytest
import spacy

@pytest.fixture(scope="session")
def labour_code_json_sample(tmp_path_factory):
    path = tmp_path_factory.mktemp("test_data") 

    docs = [
        {"content": " Les dispositions du présent livre sont applicables aux employeurs de droit privé ainsi qu'à leurs salariés. Elles sont également applicables aux établissements publics à caractère industriel et commercial."},
        {"content": "La durée du travail effectif est le temps pendant lequel le salarié est à la disposition de l'employeur et se conforme à ses directives sans pouvoir vaquer librement à des occupations personnelles. "},
        {"content": "Si le temps de trajet entre le domicile et le lieu habituel de travail est majoré du fait d'un handicap, il peut faire l'objet d'une contrepartie sous forme de repos. "},
        {"content": "A défaut d'accord prévu à l'article L. 3121-14, le régime d'équivalence peut être institué par décret en Conseil d'Etat."}
    ]

    for i,doc in enumerate (docs) :
        filename = path / f"doc{i}.json"
        with open(filename, 'w') as outfile:
            json.dump([doc], outfile)

    return path


@pytest.fixture(scope="session")
def json_sample(tmp_path_factory):
    path = tmp_path_factory.mktemp("test_data") / "json_sample.json"

    file_content = [
        {"content": "doc1"},
        {"content": "doc2"},
        {"content": "doc3"}
    ]

    with open(path, 'w') as outfile:
        json.dump(file_content, outfile)

    return path


def test_read_corpus_invalid_path():
    corpus_path = os.path.join(PROJECT_ROOT_PATH, "invalid_path")
    json_field = "content"
    corpus_loader = JsonCorpusLoader(corpus_path, json_field)
    with pytest.raises(FileOrDirectoryNotFoundError):
        corpus = corpus_loader._read_corpus()


def test_read_corpus_folder_invalid_field(labour_code_json_sample):
    json_field = "description"
    corpus_loader = JsonCorpusLoader(labour_code_json_sample, json_field)
    with pytest.raises(Exception):
        corpus = corpus_loader._read_corpus()


def test_read_corpus_file_invalid_field(json_sample):
    json_field = "description"
    corpus_loader = JsonCorpusLoader(json_sample, json_field)
    with pytest.raises(Exception):
        corpus = corpus_loader._read_corpus()


def test_read_corpus_folder(labour_code_json_sample):
    json_field = "content"
    corpus_loader = JsonCorpusLoader(labour_code_json_sample, json_field)
    corpus = corpus_loader._read_corpus()

    assert len(corpus) == 4


def test_read_corpus_file(json_sample):
    json_field = "content"
    corpus_loader = JsonCorpusLoader(json_sample, json_field)
    corpus = corpus_loader._read_corpus()

    assert len(corpus) == 3


def test_corpus_loader(labour_code_json_sample):
    json_field = "content"
    corpus_loader = JsonCorpusLoader(labour_code_json_sample, json_field)
    spacy_model = spacy.load("fr_core_news_sm")
    corpus = corpus_loader(spacy_model)

    assert len(corpus) == 4
    assert type(corpus[0]) == spacy.tokens.Doc
