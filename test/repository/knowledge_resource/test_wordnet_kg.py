from typing import Any, Dict, Set

import pytest
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import NOUN as WN_NOUN
from nltk.corpus.reader.wordnet import VERB as WN_VERB
from nltk.corpus.reader.wordnet import Synset

from olaf.repository.knowledge_source.wordnet_kg import WordNetKnowledgeResource


@pytest.fixture(scope="session")
def sample_wordnet_domains_path(tmp_path_factory):
    wordnet_domains_fn = (
        tmp_path_factory.mktemp("test_data") / "sample_wordnet_domains.txt"
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
def sample_enrichment_domains_path(tmp_path_factory):
    wordnet_domains_fn = tmp_path_factory.mktemp("test_data") / "sample_domains.txt"

    lines = ["administration\n", "hydraulics"]

    with open(wordnet_domains_fn, "w", encoding="utf8") as file:
        file.writelines(lines)

    return wordnet_domains_fn


@pytest.fixture(scope="session")
def nut_expected_wn_synsets() -> Set[Synset]:
    synsets = {
        wn.synset("nut.n.01"),
        wn.synset("nut.n.02"),
        wn.synset("nut.n.03"),
        wn.synset("en.n.01"),
        wn.synset("crackpot.n.01"),
        wn.synset("addict.n.01"),
        wn.synset("testis.n.01"),
        wn.synset("nut.v.01"),
    }

    return synsets


@pytest.fixture(scope="session")
def nut_expected_verb_wn_synsets() -> Set[Synset]:
    synsets = {wn.synset("nut.v.01")}

    return synsets


class TestDefaultWordNetKG:
    @pytest.fixture(scope="class")
    def default_wordnet_kg(self) -> WordNetKnowledgeResource:
        params = {}

        kg = WordNetKnowledgeResource(**params)

        return kg

    @pytest.fixture(scope="class")
    def prison_guard_lemmas_texts(self) -> Set[str]:
        texts = {
            "prison guard",
            "jailer",
            "jail",
            "jailor",
            "gaoler",
            "gaol",
            "screw",
            "turnkey",
        }

        return texts

    def test_default_attributes(self, default_wordnet_kg) -> None:
        assert default_wordnet_kg.lang == "en"
        assert not default_wordnet_kg.use_domains
        assert not default_wordnet_kg.use_pos
        assert default_wordnet_kg.wordnet_lang == "eng"
        assert default_wordnet_kg.wordnet_domains_map is None
        assert default_wordnet_kg.enrichment_domains is None
        assert default_wordnet_kg.wordnet_pos is None

    def test_get_term_wordnet_synsets(
        self, default_wordnet_kg, nut_expected_wn_synsets, nut_expected_verb_wn_synsets
    ) -> None:
        nut_synsets = default_wordnet_kg._get_term_wordnet_synsets("nut")

        assert nut_synsets == nut_expected_wn_synsets

        default_wordnet_kg.use_pos = True
        default_wordnet_kg.wordnet_pos = {WN_VERB}
        nut_verb_synsets = default_wordnet_kg._get_term_wordnet_synsets("nut")
        assert nut_verb_synsets == nut_expected_verb_wn_synsets

        default_wordnet_kg.wordnet_lang = "fra"
        nut_fra_verb_synsets = default_wordnet_kg._get_term_wordnet_synsets("nut")
        assert len(nut_fra_verb_synsets) == 0

        default_wordnet_kg.use_pos = False
        default_wordnet_kg.wordnet_pos = None
        default_wordnet_kg.wordnet_lang = "eng"

        air_pump_synsets = default_wordnet_kg._get_term_wordnet_synsets("air_pump")
        assert air_pump_synsets == {wn.synset("air_pump.n.01")}

    def test_fetch_terms_synsets(self, default_wordnet_kg) -> None:
        synsets = default_wordnet_kg._fetch_terms_synsets({"air pump", "nut"})
        assert wn.synset("air_pump.n.01") in synsets
        assert wn.synset("nut.n.03") in synsets

    def test_get_lemmas_texts(
        self, default_wordnet_kg, prison_guard_lemmas_texts
    ) -> None:
        prison_guard_lemmas = wn.synset("prison_guard.n.01").lemmas(lang="eng")
        lemma_texts = default_wordnet_kg._get_lemmas_texts(prison_guard_lemmas)
        assert lemma_texts == prison_guard_lemmas_texts

    def test_get_synset_hypernyms(self, default_wordnet_kg) -> None:
        screw_hypernyms = default_wordnet_kg._get_synset_hypernyms(
            wn.synset("screw.n.02")
        )
        assert screw_hypernyms == {wn.synset("inclined_plane.n.01")}

    def test_get_synset_hyponyms(self, default_wordnet_kg) -> None:
        inclined_plane_hypernyms = default_wordnet_kg._get_synset_hypernyms(
            wn.synset("inclined_plane.n.01")
        )
        assert inclined_plane_hypernyms == {wn.synset("machine.n.04")}

    def test_fetch_terms_synonyms(
        self, default_wordnet_kg, prison_guard_lemmas_texts
    ) -> None:
        synonyms = default_wordnet_kg.fetch_terms_synonyms({"screw", "pump"})

        conditions = [text in synonyms for text in prison_guard_lemmas_texts]
        conditions.append("pump" in synonyms)
        assert all(conditions)

    def test_fetch_terms_antonyms(self, default_wordnet_kg) -> None:
        antonyms = default_wordnet_kg.fetch_terms_antonyms({"screw", "pump"})
        assert antonyms == {"unscrew"}

    def test_fetch_terms_hyponyms(self, default_wordnet_kg) -> None:
        hyponyms = default_wordnet_kg.fetch_terms_hyponyms({"screw", "pump"})

        assert "gas pump" in hyponyms
        assert "oil pump" in hyponyms
        assert "machine screw" in hyponyms
        assert "thumbscrew" in hyponyms

    def test_fetch_terms_hypernyms(self, default_wordnet_kg) -> None:
        hypernyms = default_wordnet_kg.fetch_terms_hypernyms({"screw", "pump"})

        assert "mechanical device" in hypernyms
        assert "fastener" in hypernyms
        assert "bring up" in hypernyms
        assert "internal organ" in hypernyms


class TestWordNetKGWithDomains:
    @pytest.fixture(scope="class")
    def wordnet_kg_with_domains(
        self, sample_wordnet_domains_path, sample_enrichment_domains_path
    ) -> WordNetKnowledgeResource:
        params = {
            "use_domains": True,
            "wordnet_domains_path": sample_wordnet_domains_path,
            "enrichment_domains_path": sample_enrichment_domains_path,
        }

        kg = WordNetKnowledgeResource(**params)

        return kg

    @pytest.fixture(scope="class")
    def screw_filtered_synsets(self) -> Set[Synset]:
        synsets = {wn.synset("prison_guard.n.01"), wn.synset("screw.v.04")}
        return synsets

    def test_attributes(self, wordnet_kg_with_domains) -> None:
        assert wordnet_kg_with_domains.use_domains
        assert len(wordnet_kg_with_domains.wordnet_domains_map) == 16
        assert "administration" in wordnet_kg_with_domains.enrichment_domains
        assert "hydraulics" in wordnet_kg_with_domains.enrichment_domains

    def test_get_domains_for_synset(self, wordnet_kg_with_domains) -> None:
        synset_1 = wn.synset("screw.n.03")
        synset_1_wn_domains = wordnet_kg_with_domains._get_domains_for_synset(synset_1)

        synset_2 = wn.synset("screw.v.04")
        synset_2_wn_domains = wordnet_kg_with_domains._get_domains_for_synset(synset_2)

        assert synset_1_wn_domains == {"gas", "nautical", "mechanics"}
        assert synset_2_wn_domains == {"hydraulics"}

    def test_check_enrichment_domains_exist(self, wordnet_kg_with_domains) -> None:
        assert wordnet_kg_with_domains._check_enrichment_domains_exist()

        wordnet_kg_with_domains.enrichment_domains.add("unknown domain")
        assert not wordnet_kg_with_domains._check_enrichment_domains_exist()

        wordnet_kg_with_domains.enrichment_domains.remove("unknown domain")

    def test_filter_synsets_on_domains(
        self, wordnet_kg_with_domains, screw_filtered_synsets
    ) -> None:
        screw_synsets = wn.synsets("screw")
        filtered_synsets = wordnet_kg_with_domains._filter_synsets_on_domains(
            synsets=screw_synsets
        )

        assert filtered_synsets == screw_filtered_synsets

    def test_get_term_wordnet_synsets(
        self, wordnet_kg_with_domains, screw_filtered_synsets
    ) -> None:
        screw_synsets = wordnet_kg_with_domains._get_term_wordnet_synsets("screw")

        assert screw_synsets == screw_filtered_synsets


class TestWordNetKGWithPOS:
    @pytest.fixture(scope="class")
    def wordnet_kg_with_pos(self) -> WordNetKnowledgeResource:
        params = {
            "use_pos": True,
            "wordnet_pos": {"NOUN", "VERB"},
        }

        kg = WordNetKnowledgeResource(**params)

        return kg

    def test_attributes(self, wordnet_kg_with_pos) -> None:
        assert wordnet_kg_with_pos.use_pos
        assert WN_NOUN in wordnet_kg_with_pos.wordnet_pos
        assert WN_VERB in wordnet_kg_with_pos.wordnet_pos
        assert len(wordnet_kg_with_pos.wordnet_pos) == 2
