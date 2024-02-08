import os
from os import PathLike

import pytest
import spacy.tokens

from olaf.commons.errors import FileOrDirectoryNotFoundError
from olaf.repository.corpus_loader import TextCorpusLoader


@pytest.fixture(scope="session")
def labour_code_sample() -> list[str]:
    docs = [
        "Les dispositions du présent livre sont applicables aux employeurs de droit privé ainsi qu'à leurs salariés. Elles sont également applicables aux établissements publics à caractère industriel et commercial.",
        "La durée du travail effectif est le temps pendant lequel le salarié est à la disposition de l'employeur et se conforme à ses directives sans pouvoir vaquer librement à des occupations personnelles. ",
        "Si le temps de trajet entre le domicile et le lieu habituel de travail est majoré du fait d'un handicap, il peut faire l'objet d'une contrepartie sous forme de repos. ",
        "A défaut d'accord prévu à l'article L. 3121-14, le régime d'équivalence peut être institué par décret en Conseil d'Etat."
    ]
    return docs

@pytest.fixture(scope="session")
def corpus_one_doc_per_file_path(tmp_path_factory, labour_code_sample) -> PathLike:
    path = tmp_path_factory.mktemp("test_data")
    
    for i, doc in enumerate(labour_code_sample):
        filename = path / f"doc{i}.txt"
        with open(filename, "w", encoding="utf8") as outfile:
            outfile.write(doc)

    return str(path)

@pytest.fixture(scope="session")
def corpus_one_doc_per_line_path(tmp_path_factory, labour_code_sample) -> PathLike:
    path = tmp_path_factory.mktemp("test_data")
    
    filename = path / "test_doc.txt"
    doc_content = "\n\n".join(labour_code_sample)

    with open(filename, "w", encoding="utf8") as outfile:
        outfile.write(doc_content)

    return str(filename)

def test_read_corpus_invalid_path() -> None:
    corpus_path = os.path.join(os.getenv("DATA_PATH"), "invalid_path")

    corpus_loader = TextCorpusLoader(corpus_path)
    with pytest.raises(FileOrDirectoryNotFoundError):
        corpus = corpus_loader._read_corpus()

def test_read_corpus_one_doc_per_file(labour_code_sample, corpus_one_doc_per_file_path) -> None:
    corpus_loader = TextCorpusLoader(corpus_one_doc_per_file_path)
    corpus = corpus_loader._read_corpus()
    assert len(corpus) == len(labour_code_sample)

def test_read_corpus_one_doc_per_line(labour_code_sample, corpus_one_doc_per_line_path) -> None:
    corpus_loader = TextCorpusLoader(corpus_one_doc_per_line_path)
    corpus = corpus_loader._read_corpus()
    assert len(corpus) == len(labour_code_sample)