from collections import Dict, List, Set
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
import os.path

from term_enrichment.term_enrichment_schema import CandidateTerm
from term_enrichment.term_enrichment_repository import load_wordnet_domains
from config.core import config, DATA_PATH


class WordNetTermEnrichment:

    def __init__(self, lang: str = 'en', use_domains: bool = False) -> None:
        self.lang = lang
        self.use_domains = use_domains

        self.wordnet_domains_map: Dict[str, List[str]] = None
        self.enrichment_domains: Set[str] = None

        if self.use_domains:
            self.wordnet_domains_map = load_wordnet_domains()
            self._get_enrichment_domains()

    def __call__(self) -> List[CandidateTerm]:
        pass

    def _get_enrichment_domains(self) -> None:
        domains = config["term_enrichment"]["wordnet"].get(
            "enrichment_domains")
        if isinstance(domains, list):
            self.enrichment_domains = set(domains)

    def _check_domains_exists(self) -> bool:
        all_wordnet_domains = set()
        for synset_domains in self.wordnet_domains_map.values():
            all_wordnet_domains.update(set(synset_domains))

        return len(all_wordnet_domains.intersection(self.enrichment_domains)) > 0

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
