import os
from typing import Dict, Optional, Set

from nltk.corpus.reader.wordnet import ADJ as WN_ADJ
from nltk.corpus.reader.wordnet import ADV as WN_ADV
from nltk.corpus.reader.wordnet import NOUN as WN_NOUN
from nltk.corpus.reader.wordnet import VERB as WN_VERB

from .logging_config import logger


def load_wordnet_domains(wordnet_domains_path: str) -> Dict[str, Set[str]]:
    """Load the mapping of WordNet Synsets to domains from a file.
    The file should have the structure: `synset_code\tdomain1 domain2`.
    Function inspired from project <https://github.com/argilla-io/spacy-wordnet>

    Parameters
    ----------
    wordnet_domains_path : str
        The full or relative path to wordnet domains synsets mapping file.

    Returns
    -------
    Dict[str, List[str]]
        The mapping of WordNet Synsets to domains.
    """
    domain_file_path = wordnet_domains_path

    if not os.path.isabs(wordnet_domains_path):
        domain_file_path = os.path.join(os.getenv("DATA_PATH"), domain_file_path)

    domains_map = dict()

    try:
        for line in open(domain_file_path, "r", encoding="utf8"):
            ssid, domains = line.strip().split("\t")
            domains_map[ssid] = set(domains.split())
    except Exception as e:
        logger.error(
            "Could not load wordnet domains from file %s. Trace : %s",
            domain_file_path,
            e,
        )
    else:
        logger.info("Wordnet domains loaded.")

    return domains_map


def load_enrichment_wordnet_domains_from_file(enrichment_domains_path: str) -> Set[str]:
    """Load a set of domains (strings) from a file.
        The file is expected to contain one domain string per line.

    Parameters
    ----------
    enrichment_domains_path : str
        The full or relative path to the file containing wordnet domains to use for enrichment.

    Returns
    -------
    Set[str]
        The set of domains.
    """

    if not os.path.isabs(enrichment_domains_path):
        enrichment_domains_path = os.path.join(os.getenv("DATA_PATH"), enrichment_domains_path)

    enrichment_domains = set()

    try:
        for line in open(enrichment_domains_path, "r", encoding="utf8"):
            enrichment_domains.add(line.strip())
    except Exception as e:
        logger.error(
            "Could not load enrichment wordnet domains from file %s. Trace : %s",
            enrichment_domains_path,
            e,
        )
    else:
        logger.info("Enrichment Wordnet domains loaded.")

    return enrichment_domains


# Adapted from <https://github.com/argilla-io/spacy-wordnet/tree/b9efd800e02d55e848d56ce7acfacafb2089f587>
# The Open Multi Wordnet corpus contains the following languages:
#   als arb bul cat cmn dan ell eus fas fin fra glg heb hrv ind ita jpn nld nno nob pol por qcn slv spa swe tha zsm
#   ('deu' can be found in Extended Open Multi Wordnet)
# the available spaCy languages are:
# af am ar bg bn ca cs da de el en es et eu fa fi fr ga gu he hi hr hu hy id is it ja kn ko ky lb lij
# lt lv mk ml mr nb ne nl pl pt ro ru sa si sk sl sq sr sv ta te th ti tl tn tr tt uk ur vi xx yo zh
# then the full mapping is

WORDNET_DOMAINS_SSID_NUM_SIZE = 8

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


def spacy2wordnet_pos(spacy_pos: str) -> Optional[str]:
    """Tool function to map a spaCy POS tag to the corresponding WordNet one.
        Return None if no mapping is found.
        Adapted from project <https://github.com/argilla-io/spacy-wordnet>.


    Parameters
    ----------
    spacy_pos : str
        The spaCy POS tag.

    Returns
    -------
    Optional[str]
        The WordNet POS tag.
    """
    return _WN_POS_MAPPING.get(spacy_pos)


def fetch_wordnet_lang(lang: str) -> str:
    """Tool function to map a Spacy language tag to the corresponding WordNet one.
        Return None if no mapping is found.
        Adapted from project <https://github.com/argilla-io/spacy-wordnet>.

    Parameters
    ----------
    lang : str
        The spaCy language tag.

    Returns
    -------
    str
        The WordNet language tag.

    Raises
    ------
    Exception
        An exception to spot a language not existing.
    """
    language = _WN_LANGUAGES_MAPPING.get(lang, None)

    if not language:
        raise TypeError(f"Language {lang} not supported")

    return language
