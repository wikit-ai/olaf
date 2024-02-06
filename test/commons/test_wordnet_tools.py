import os

import pytest
from nltk.corpus.reader.wordnet import ADJ as WN_ADJ
from nltk.corpus.reader.wordnet import ADV as WN_ADV
from nltk.corpus.reader.wordnet import NOUN as WN_NOUN
from nltk.corpus.reader.wordnet import VERB as WN_VERB

from olaf.commons.wordnet_tools import (
    fetch_wordnet_lang, load_enrichment_wordnet_domains_from_file,
    load_wordnet_domains, spacy2wordnet_pos)


@pytest.fixture(scope="module")
def sample_wordnet_domains_path(test_data_path):
    wordnet_domains_fn = (
        os.path.join(os.getenv("DATA_PATH"), "sample_wordnet_domains.txt")
    )

    lines = [
        "04154152-n\tgas mechanics nautical\n",
        "01352806-v\thydraulics\n",
        "01995549-v\tvehicles transport skiing town_planning\n",
        "05087173-n\tvehicles\n",
        "10695192-n\tvehicles transport\n",
        "04021798-n\tgas applied_science hydraulics railway mechanics surgery vehicles cycling transport nautical\n",
        "00624738-n\tsport athletics health\n",
        "14049711-n\tphysiology medicine health psychiatry\n",
        "01017738-a\tmedicine health\n",
        "10834337-n\tlaw\n",
        "00779599-n\tlaw\n",
        "10149867-n\tadministration law\n",
        "00707322-v\tadministration law politics\n",
        "06734702-n\tlaw\n",
        "13339844-n\texchange administration law\n",
        "10514643-n\tlaw\n",
    ]

    with open(wordnet_domains_fn, "w", encoding="utf8") as file:
        file.writelines(lines)

    return wordnet_domains_fn


@pytest.fixture(scope="session")
def sample_domains_path(test_data_path):
    wordnet_domains_fn = os.path.join(os.getenv("DATA_PATH"), "sample_domains.txt")

    lines = ["administration\n", "hydraulics"]

    with open(wordnet_domains_fn, "w", encoding="utf8") as file:
        file.writelines(lines)

    return wordnet_domains_fn


def test_load_wordnet_domains(sample_wordnet_domains_path) -> None:
    
    wordnet_domains = load_wordnet_domains(
        wordnet_domains_path=sample_wordnet_domains_path
    )

    assert isinstance(wordnet_domains, dict)
    assert len(wordnet_domains) == 16
    assert wordnet_domains.get("10695192-n") == {"vehicles", "transport"}


def test_load_enrichment_wordnet_domains_from_file(sample_domains_path) -> None:
    enrichment_domains = load_enrichment_wordnet_domains_from_file(
        enrichment_domains_path=sample_domains_path
    )

    assert isinstance(enrichment_domains, set)
    assert "administration" in enrichment_domains
    assert "hydraulics" in enrichment_domains


def test_spacy2wordnet_pos() -> None:
    assert spacy2wordnet_pos("ADJ") == WN_ADJ
    assert spacy2wordnet_pos("NOUN") == WN_NOUN
    assert spacy2wordnet_pos("ADV") == WN_ADV
    assert spacy2wordnet_pos("VERB") == WN_VERB
    assert spacy2wordnet_pos("AUX") == WN_VERB


def test_fetch_wordnet_lang() -> None:
    assert fetch_wordnet_lang(lang="en") == "eng"
    assert fetch_wordnet_lang(lang="fr") == "fra"
