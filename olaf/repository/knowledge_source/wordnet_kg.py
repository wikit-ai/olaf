from typing import Dict, Optional, Set

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Lemma, Synset

from ...commons.logging_config import logger
from ...commons.string_tools import space_to_underscore_str, underscore_to_space_str
from ...commons.wordnet_tools import (
    WORDNET_DOMAINS_SSID_NUM_SIZE,
    fetch_wordnet_lang,
    load_enrichment_wordnet_domains_from_file,
    load_wordnet_domains,
    spacy2wordnet_pos,
)
from .knowledge_source_schema import KnowledgeSource


class WordNetKnowledgeResource(KnowledgeSource):
    """Adapter for the WordNet linguistic knowledge base: .

    Attributes
    ----------
    lang: str, optional
        Language ISO code for the terms to find concepts and terms for, by default 'en'.
    use_domains: bool, optional
        Wether or not to filter the matchings on provided domains, by default False.
    use_pos: bool, optional
        Wether or not to filter the matchings on provided part of speech tags, by default False.
    wordnet_domains_map: Dict[str, Set[str]], optional
        The mapping between WordNet synsets ids and domains ids, by default None.
        The expected file can be found at
        <https://github.com/argilla-io/spacy-wordnet/blob/master/spacy_wordnet/data/wordnet_domains.txt>
    enrichment_domains: Set[str], optional
        The set of enrichment domains strings to use for matching.
        Mandatory when use_domains is True, by default to None.
    wordnet_pos: Set[str], optional
        The set of part of speech tags to use for matching.
        Mandatory when use_pos is True, by default to None.
    """

    def __init__(
        self,
        lang: Optional[str] = "en",
        use_domains: Optional[bool] = False,
        use_pos: Optional[bool] = False,
        wordnet_domains_map: Optional[Dict[str, Set[str]]] = None,
        wordnet_domains_path: Optional[str] = None,
        enrichment_domains: Optional[Set[str]] = None,
        enrichment_domains_path: Optional[str] = None,
        wordnet_pos: Optional[Set[str]] = None,
    ) -> None:
        """Initialise WordNet knowledge resource instance.

        Parameters
        ----------
        lang: str, optional
            Language ISO code for the terms to find concepts and terms for, by default 'en'.
        use_domains: bool, optional
            Wether or not to filter the matchings on provided domains, by default False.
        use_pos: bool, optional
            Wether or not to filter the matchings on provided part of speech tags, by default False.
        wordnet_domains_path : str, optional
            The full or relative path to wordnet domains synsets mapping file, by default None.
        wordnet_domains_map: Dict[str, Set[str]], optional
            The mapping between WordNet synsets ids and domains ids, by default None.
            The expected file can be found at
            <https://github.com/argilla-io/spacy-wordnet/blob/master/spacy_wordnet/data/wordnet_domains.txt>
        wordnet_domains_path : str, optional
            The full or relative path to wordnet domains synsets mapping file, by default None.
        enrichment_domains: Set[str], optional
            The set of enrichment domains strings to use for matching.
            Mandatory when use_domains is True, by default to None.
        enrichment_domains_path : str, optional
            The full or relative path to the file containing wordnet domains to use for enrichment,
            by default None.
        wordnet_pos: Set[str], optional
            The set of part of speech tags to use for matching.
            Mandatory when use_pos is True, by default to None.
        """

        self.lang = lang
        self.use_domains = use_domains
        self.use_pos = use_pos

        self.wordnet_lang = fetch_wordnet_lang(self.lang)
        self.wordnet_domains_map = wordnet_domains_map
        self.wordnet_domains_path = wordnet_domains_path
        self.enrichment_domains = enrichment_domains
        self.enrichment_domains_path = enrichment_domains_path
        self.wordnet_pos = wordnet_pos

        self._check_parameters()

    def _check_parameters(self) -> None:
        """Check wether required parameters are given and correct. If this is not the case,
        suitable default ones are set.
        """

        if self.use_domains:
            if self.enrichment_domains is None:
                if self.enrichment_domains_path is None:
                    logger.warning(
                        """Using enrichment wordnet domains but file path not provided in parameters (key `enrichment_domains`). 
                                Defaulting to NOT using wordnet domains.
                            """
                    )
                    self.use_domains = False
                else:
                    self.enrichment_domains = load_enrichment_wordnet_domains_from_file(
                        self.enrichment_domains_path
                    )

            if not self.enrichment_domains:
                self.use_domains = False
                self.enrichment_domains = None
                logger.warning(
                    """Using enrichment wordnet domains (use_domains = True) but could not find in the 
                        parameters nor load from file any enrichment domain.
                        See parameters 'enrichment_domains_path' or 'enrichment_domains'.
                        Defaulting to NOT using wordnet domains.
                        """
                )
            elif self.wordnet_domains_path is None:
                self._extracted_from__check_parameters_35(
                    """Using enrichment wordnet domains (use_domains = True) but config parameter
                        `wordnet_domains_path` is not provided.
                        Defaulting to not using wordnet domains.
                        """
                )
                self.use_domains = False
            else:
                self.wordnet_domains_map = load_wordnet_domains(
                    self.wordnet_domains_path
                )
            if not self.wordnet_domains_map:
                self._extracted_from__check_parameters_35(
                    """Using enrichment wordnet domains (use_domains = True) but could not find in the 
                        parameters nor load from file any wordNet domains mapping.
                        See parameter 'wordnet_domains_path'.
                        Defaulting to not using wordnet domains.
                        """
                )
                self.use_domains = False
        if self.use_domains and not self._check_enrichment_domains_exist():
            logger.warning(
                """Some Wordnet domains have not been found in the mappings."""
            )

        if self.use_pos:
            spacy_pos = self.wordnet_pos
            if spacy_pos:
                self.wordnet_pos = {spacy2wordnet_pos(pos) for pos in spacy_pos}
            else:
                logger.warning(
                    """Using POS tags (use_pos = True) but parameter `wordnet_pos` is not provided.
                        Defaulting to not using pos.
                    """
                )
                self.use_pos = False

            if len(self.wordnet_pos) == 0:
                logger.warning(
                    """Using specified POS but parameter `wordnet_pos` is an empty set or list.  
                        Defaulting to not using pos.
                    """
                )
                self.use_pos = False

    # TODO Rename this here and in `_check_parameters`
    def _extracted_from__check_parameters_35(self, arg0):
        logger.warning(arg0)
        self.use_domains = False
        self.enrichment_domains = None

    def _check_resources(self) -> None:
        # TODO
        """Method to check that the component has access to all its required resources."""

    def _check_enrichment_domains_exist(self) -> bool:
        """Private method to test wether all the WordNet domains provided for enrichment
            exist in the mapping of WordNet Synsets to domains.

        Returns
        -------
        bool
            Wether or not all the WordNet domains provided for enrichment exist.
        """
        all_wordnet_domains = set()
        for synset_domains in self.wordnet_domains_map.values():
            all_wordnet_domains.update(synset_domains)

        domains_exist = self.enrichment_domains.issubset(all_wordnet_domains)

        return domains_exist

    def match_external_concepts(self, matching_terms: Set[str]) -> Set[str]:
        """Method to fetch external concepts matching the set of terms.

        Parameters
        ----------
        matching_terms : Set[str]
            The term texts to use for matching concepts.

        Returns
        -------
        Set[str]
            The UIDs of the external concepts found matching the term texts.
        """
        raise NotImplementedError

    def _fetch_synsets_lemmas_texts(self, synsets: Set[Synset]) -> Set[str]:
        """Fetch the texts of synsets' lemmas.

        Parameters
        ----------
        synsets : Set[Synset]
            The synsets to extract the text lemmas from.

        Returns
        -------
        Set[str]
            The synsets' lemmas texts.
        """
        texts = set()
        for synset in synsets:
            synset_lemmas = synset.lemmas(lang=self.wordnet_lang)
            synset_lemmas_texts = self._get_lemmas_texts(synset_lemmas)
            texts.update(synset_lemmas_texts)

        return texts

    def fetch_terms_synonyms(self, terms: Set[str]) -> Set[str]:
        """Method to fetch synonyms of a set of terms from WordNet.

        Parameters
        ----------
        terms : Set[str]
            The set of terms to find synonyms of.

        Returns
        -------
        Set[str]
            The set of terms synonyms.
        """
        terms_synsets = self._fetch_terms_synsets(terms)

        terms_synonyms = self._fetch_synsets_lemmas_texts(terms_synsets)

        return terms_synonyms

    def fetch_terms_antonyms(self, terms: Set[str]) -> Set[str]:
        """Method to fetch antonyms of a set of terms from WordNet.

        Parameters
        ----------
        terms : Set[str]
            The set of terms to find antonyms of.

        Returns
        -------
        Set[str]
            The set of terms antonyms.
        """
        terms_synsets = self._fetch_terms_synsets(terms)

        term_antonyms_lemmas = set()
        for synset in terms_synsets:
            # antonyms are linked with Lemmas in WordNet
            # extract synsets lemmas
            synset_lemmas = synset.lemmas(lang=self.wordnet_lang)
            synset_antonyms = set()

            # Extract lemmas antonyms
            for lemma in synset_lemmas:
                synset_antonyms.update(set(lemma.antonyms()))

            term_antonyms_lemmas.update(synset_antonyms)

        terms_antonyms = self._get_lemmas_texts(term_antonyms_lemmas)

        return terms_antonyms

    def fetch_terms_hypernyms(self, terms: Set[str]) -> Set[str]:
        """Method to fetch hypernyms of a set of terms from WordNet.

        Parameters
        ----------
        terms : Set[str]
            The set of terms to find hypernyms of.

        Returns
        -------
        Set[str]
            The set of terms hypernyms.
        """
        terms_synsets = self._fetch_terms_synsets(terms)

        terms_hypernyms_synsets = set()
        for synset in terms_synsets:
            synset_hypernyms = self._get_synset_hypernyms(synset)
            terms_hypernyms_synsets.update(synset_hypernyms)

        # extract lemmas texts from hypernyms Synsets
        terms_hypernyms = self._fetch_synsets_lemmas_texts(terms_hypernyms_synsets)

        return terms_hypernyms

    def fetch_terms_hyponyms(self, terms: Set[str]) -> Set[str]:
        """Method to fetch hyponyms of a set of terms according to the knowledge source.

        Parameters
        ----------
        terms : Set[str]
            The set of terms to find hyponyms of.

        Returns
        -------
        Set[str]
            The set of terms hyponyms.
        """
        terms_synsets = self._fetch_terms_synsets(terms)

        # extract hyponyms Synsets
        terms_hyponyms_synsets = set()
        for synset in terms_synsets:
            synset_hyponyms = self._get_synset_hyponyms(synset)
            terms_hyponyms_synsets.update(synset_hyponyms)

        # extract lemmas texts from hyponyms Synsets
        terms_hyponyms = self._fetch_synsets_lemmas_texts(terms_hyponyms_synsets)

        return terms_hyponyms

    def _fetch_terms_synsets(self, terms: Set[str]) -> Set[Synset]:
        """Fetch terms corresponding WordNet synsets.

        Parameters
        ----------
        terms : Set[str]
            The set of terms to fetch the WordNet synsets for.

        Returns
        -------
        Set[Synset]
            The set of corresponding WordNet synsets.
        """
        terms_wordnet_texts = {space_to_underscore_str(term) for term in terms}

        terms_synsets = set()
        for term_wordnet_text in terms_wordnet_texts:
            terms_synsets.update(self._get_term_wordnet_synsets(term_wordnet_text))

        return terms_synsets

    def _get_domains_for_synset(self, synset: Synset) -> Set[str]:
        """Private method to extract the domains associated with a WordNet Sysnset.
            Adapted from project <https://github.com/argilla-io/spacy-wordnet>.

        Parameters
        ----------
        synset : Synset
            The WordNet Synset to extract the domains from.

        Returns
        -------
        Set[str]
            The set of domains associated with the synset.
        """
        ssid = f"{str(synset.offset()).zfill(WORDNET_DOMAINS_SSID_NUM_SIZE)}-{synset.pos()}"
        synset_domains = self.wordnet_domains_map.get(ssid, set())

        return synset_domains

    def _filter_synsets_on_domains(self, synsets: Set[Synset]) -> Set[Synset]:
        """Private method to filter out synsets not associated with the WordNet domains
            to use for term enrichment.

        Parameters
        ----------
        synsets : Set[Synset]
            The set of WordNet synsets to filter.

        Returns
        -------
        Set[Synset]
            The filtered set of synsets.
        """
        kept_synsets = set()

        for synset in synsets:
            synset_domains = self._get_domains_for_synset(synset)
            if len(self.enrichment_domains & synset_domains) > 0:
                kept_synsets.add(synset)

        return kept_synsets

    def _get_term_wordnet_synsets(self, term_text: str) -> Set[Synset]:
        """Private method to get WordNet Synsets associated with a term string.

        Parameters
        ----------
        term_text : str
            The term string.

        Returns
        -------
        Set[Synset]
            The corresponding WordNet Synsets.
        """
        term_synsets = set()

        if self.use_pos:
            for pos in self.wordnet_pos:
                term_synsets.update(
                    set(wn.synsets(term_text, pos=pos, lang=self.wordnet_lang))
                )
        else:
            term_synsets.update(set(wn.synsets(term_text, lang=self.wordnet_lang)))

        if self.use_domains:
            term_synsets = self._filter_synsets_on_domains(synsets=term_synsets)

        return term_synsets

    def _get_lemmas_texts(self, lemmas: Set[Lemma]) -> Set[str]:
        """Private method to extract the strings associated with a set of WordNet Lemmas.

        Parameters
        ----------
        lemmas : Set[Lemma]
            The set of Lemmas to extract the strings from.

        Returns
        -------
        Set[str]
            The set of extracted strings.
        """
        lemmas_names = set()

        for lemma in lemmas:
            lemmas_names.add(lemma.name())

            for derived_lemma in lemma.derivationally_related_forms():
                lemmas_names.add(derived_lemma.name())

        lemmas_texts = {underscore_to_space_str(name) for name in lemmas_names}

        return lemmas_texts

    def _get_synset_hypernyms(self, synset: Synset) -> Set[Synset]:
        """Private method to get WordNet hypernyms Synsets associated with a Synset.

        Parameters
        ----------
        synset : Synset
            The Synset to extract the hypernyms from.

        Returns
        -------
        Set[Synset]
            The hypernyms Synsets.
        """
        synset_hypernyms = set(synset.hypernyms())

        if self.use_domains:
            synset_hypernyms = self._filter_synsets_on_domains(synsets=synset_hypernyms)

        return synset_hypernyms

    def _get_synset_hyponyms(self, synset: Synset) -> Set[Synset]:
        """Private method to get WordNet hyponyms Synsets associated with a Synset.

        Parameters
        ----------
        synset : Synset
            The Synset to extract the hyponyms from.

        Returns
        -------
        Set[Synset]
            The hyponyms Synsets.
        """
        synset_hyponyms = set(synset.hyponyms())

        if self.use_domains:
            synset_hyponyms = self._filter_synsets_on_domains(synsets=synset_hyponyms)

        return synset_hyponyms
