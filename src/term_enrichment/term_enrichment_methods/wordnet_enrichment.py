from collections import Dict, List, Set, Optional, Any
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import (
    ADJ as WN_ADJ,
    ADV as WN_ADV,
    NOUN as WN_NOUN,
    VERB as WN_VERB,
    Synset,
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

    def __call__(self, terms_to_enrich: List[CandidateTerm]) -> None:
        for candidate_term in terms_to_enrich:
            term_string = candidate_term.value
            term_wordnet_string = term_string.replace(" ", "_")
            synsets = wn.synsets(term_wordnet_string, pos=wn.NOUN)

            candidate_term_synsets = self._filter_synsets_on_domains(synsets)

            enriching_terms = set()
            enriching_synsets = set()

            for selected_synset in candidate_term_synsets:
                enriching_terms.update(
                    {" ".join(lemma.name().split("_")) for lemma in selected_synset.lemmas()})
                enriching_synsets.add(selected_synset.name())

            candidate_term.enriching_terms.update(enriching_terms)
            candidate_term.source_ids['wordnet'].update(enriching_synsets)

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

    def _filter_synsets_on_domains(self, synsets: List[Synset]) -> List[Synset]:

        kept_synsets = []

        for synset in synsets:
            synset_domains = self._get_domains_for_synset(synset)
            if len(self.enrichment_domains.intersection(synset_domains)):
                kept_synsets.append(synset)

        return synset
