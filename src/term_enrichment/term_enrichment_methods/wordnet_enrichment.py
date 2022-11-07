from collections import Dict, List, Set, Optional, Any
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import (
    ADJ as WN_ADJ,
    ADV as WN_ADV,
    NOUN as WN_NOUN,
    VERB as WN_VERB,
    Synset, Lemma
)

from term_enrichment.term_enrichment_schema import CandidateTerm
from term_enrichment.term_enrichment_repository import load_wordnet_domains, load_enrichment_wordnet_domains_from_file
from config.core import config

# Adapted from <https://github.com/argilla-io/spacy-wordnet/tree/b9efd800e02d55e848d56ce7acfacafb2089f587>
# The Open Multi Wordnet corpus contains the following languages:
#   als arb bul cat cmn dan ell eus fas fin fra glg heb hrv ind ita jpn nld nno nob pol por qcn slv spa swe tha zsm
#   ('deu' can be found in Extended Open Multi Wordnet)
# the available spacy languages are:
# af am ar bg bn ca cs da de el en es et eu fa fi fr ga gu he hi hr hu hy id is it ja kn ko ky lb lij
# lt lv mk ml mr nb ne nl pl pt ro ru sa si sk sl sq sr sv ta te th ti tl tn tr tt uk ur vi xx yo zh
# then the full mapping is
_WN_LANGUAGES_MAPPING = dict(
    es="spa",
    en="eng",
    fr="fra",
    it="ita",
    pt="por",
    de="deu",
    # other languages from omw
    sq="als",  # Albanian
    ar="arb",  # Arabic
    bg="bul",  # Bulgarian
    ca="cat",  # Catalan
    zh="cmn",  # Chinese Open Wordnet
    da="dan",  # Danish
    el="ell",  # Greek
    eu="eus",  # Basque
    fa="fas",  # Persian
    fi="fin",  # Finnish
    # ?? ='glg',  # Galician
    he="heb",  # Hebrew
    hr="hrv",  # Croatian
    id="ind",  # Indonesian
    ja="jpn",  # Japanese
    nl="nld",  # Dutch
    # no ='nno', # Norwegian
    # nb ='nob', # Norwegian Bokmal
    pl="pol",  # Polish
    # ?? ='qcn', # Chinese (Taiwan)
    sl="slv",  # Slovenian
    sv="swe",  # Swedish
    th="tha",  # Thai
    ml="zsm",  # Malayalam
)

_WN_POS_MAPPING = {
    "ADJ": WN_ADJ,
    "NOUN": WN_NOUN,
    "ADV": WN_ADV,
    "VERB": WN_VERB,
    "AUX": WN_VERB,
}


def spacy2wordnet_pos(spacy_pos: int) -> Optional[str]:
    return _WN_POS_MAPPING.get(spacy_pos)


def fetch_wordnet_lang(lang: Optional[str] = None) -> str:
    language = _WN_LANGUAGES_MAPPING.get(lang, None)

    if not language:
        raise Exception(f"Language {lang} not supported")

    return language


def termStr2wordnetStr(term_text: str) -> str:
    return "_".join(term_text.split())


def wordnetStr2termStr(wordnet_text: str) -> str:
    return " ".join(wordnet_text.split("_"))


class WordNetTermEnrichment:

    def __init__(self, lang: str = 'en', use_domains: bool = False, use_pos: bool = False) -> None:
        self.wordnet_lang = fetch_wordnet_lang(lang)
        self.use_domains = use_domains
        self.use_pos = use_pos

        self.wordnet_domains_map: Dict[str, List[str]] = None
        self.enrichment_domains: Set[str] = None
        self.wordnet_pos: Set[Any] = None

        if self.use_domains:
            self.wordnet_domains_map = load_wordnet_domains()
            self._get_enrichment_domains()

        if self.use_pos:
            self._get_wordnet_pos()

    def _get_enrichment_domains(self) -> None:
        domains = config["term_enrichment"]["wordnet"].get(
            "enrichment_domains")

        if domains is not None:
            if isinstance(domains, list):
                self.enrichment_domains = set(domains)
        else:
            self.enrichment_domains = load_enrichment_wordnet_domains_from_file()

    def _check_domains_exists(self) -> bool:
        all_wordnet_domains = set()
        for synset_domains in self.wordnet_domains_map.values():
            all_wordnet_domains.update(set(synset_domains))

        return len(all_wordnet_domains.intersection(self.enrichment_domains)) > 0

    def _get_wordnet_pos(self) -> None:
        config_pos = config["term_enrichment"]["wordnet"].get(
            "synset_pos")
        if isinstance(config_pos, list):
            self.wordnet_pos = {spacy2wordnet_pos(pos) for pos in config_pos}

    def _get_domains_for_synset(self, synset: Synset) -> Set[str]:
        ssid = "{}-{}".format(str(synset.offset()).zfill(8), synset.pos())
        return self.wordnet_domains_map.get(ssid, set())

    def _find_wordnet_domains(self, synsets: List[Synset]) -> Set[str]:

        wordnet_domains = {
            domain
            for synset in synsets
            for domain in self._get_domains_for_synset(synset)
        }

        return wordnet_domains

    def _filter_synsets_on_domains(self, synsets: Set[Synset]) -> Set[Synset]:

        kept_synsets = set()

        for synset in synsets:
            synset_domains = self._get_domains_for_synset(synset)
            if len(self.enrichment_domains.intersection(synset_domains)):
                kept_synsets.add(synset)

        return synset

    def enrich_candidate_term(self, candidate_term: CandidateTerm) -> None:
        term_wordnet_text = termStr2wordnetStr(candidate_term.value)

        term_synsets = self._get_term_wordnet_synsets(term_wordnet_text)

        candidate_term.synonyms.update(self._get_term_synonyms(term_synsets))

        candidate_term.hypernyms.update(self._get_term_hypernyms(term_synsets))

        candidate_term.hyponyms.update(self._get_term_hyponyms(term_synsets))

        candidate_term.antonyms.update(self._get_term_antonyms(term_synsets))

    def enrich_candidate_terms(self, candidate_terms: List[CandidateTerm]) -> None:
        for term in candidate_terms:
            self.enrich_candidate_term(term)

    def _get_lemmas_texts(lemmas: Set[Lemma]) -> Set[str]:
        lemmas_names = set()

        for lemma in lemmas:
            lemmas_names.add(lemma.name())

            for derived_lemma in lemma.derivationally_related_forms():
                lemmas_names.add(derived_lemma.name())

        lemmas_texts = {wordnetStr2termStr(name) for name in lemmas_names}

        return lemmas_texts

    def _get_term_synonyms(self, term_synsets: Set[Synset]) -> Set[str]:
        term_synonyms = set()
        for synset in term_synsets:
            synset_lemmas = synset.lemmas(lang=self.wordnet_lang)
            synset_lemmas_texts = self._get_lemmas_texts(synset_lemmas)
            term_synonyms.update(synset_lemmas_texts)
        return term_synonyms

    def _get_term_hypernyms(self, term_synsets: Set[Synset]) -> Set[str]:
        term_hypernyms_synsets = set()
        for synset in term_synsets:
            synset_hypernyms = self._get_synset_hypernyms(synset)
            term_hypernyms_synsets.update(synset_hypernyms)

        term_hypernyms = set()
        for synset in term_hypernyms_synsets:
            synset_lemmas = synset.lemmas(lang=self.wordnet_lang)
            synset_lemmas_texts = self._get_lemmas_texts(synset_lemmas)
            term_hypernyms.update(synset_lemmas_texts)

    def _get_term_hyponyms(self, term_synsets: Set[Synset]) -> Set[str]:
        term_hyponyms_synsets = set()
        for synset in term_synsets:
            synset_hyponyms = self._get_synset_hyponyms(synset)
            term_hyponyms_synsets.update(synset_hyponyms)

        term_hyponyms = set()
        for synset in term_hyponyms_synsets:
            synset_lemmas = synset.lemmas(lang=self.wordnet_lang)
            synset_lemmas_texts = self._get_lemmas_texts(synset_lemmas)
            term_hyponyms.update(synset_lemmas_texts)

    def _get_term_antonyms(self, term_synsets: Set[Synset]) -> Set[str]:
        term_antonyms_lemmas = set()
        for synset in term_synsets:
            synset_lemmas = synset.lemmas(lang=self.wordnet_lang)
            synset_antonyms = set()
            for lemma in synset_lemmas:
                synset_antonyms.update(set(lemma.antonyms()))

            term_antonyms_lemmas.update(synset_antonyms)

        term_antonyms_texts = self._get_lemmas_texts(term_antonyms_lemmas)

        return term_antonyms_texts

    def _get_term_wordnet_synsets(self, term_text: str) -> Set[Synset]:
        term_synsets = set()

        if self.use_pos:
            for pos in self.wordnet_pos:
                term_synsets.update(
                    set(wn.synsets(term_text, pos=pos, lang=self.wordnet_lang)))

        if self.use_domains:
            term_synsets = self._filter_synsets_on_domains(
                synsets=term_synsets)

        return term_synsets

    def _get_synset_hypernyms(self, synset: Synset) -> Set[Synset]:
        synset_hypernyms = set(synset.hypernyms())

        if self.use_domains:
            synset_hypernyms = self._filter_synsets_on_domains(
                synsets=synset_hypernyms)

        return synset_hypernyms

    def _get_synset_hyponyms(self, synset: Synset) -> Set[Synset]:
        synset_hyponyms = set(synset.hyponyms())

        if self.use_domains:
            synset_hyponyms = self._filter_synsets_on_domains(
                synsets=synset_hyponyms)

        return synset_hyponyms
