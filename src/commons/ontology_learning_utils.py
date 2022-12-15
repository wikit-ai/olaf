from typing import List

def space2underscoreStr(text_with_space: str) -> str:
    """Tool function to replace spaces by underscores in a text.

    Parameters
    ----------
    text_with_space : str
        The text containing spaces

    Returns
    -------
    str
        the text with underscores
    """

    text_with_underscore = "_".join(text_with_space.split())

    return text_with_underscore


def underscore2spaceStr(text_with_underscore: str) -> str:
    """Tool function to replace underscores by spaces in a text.

    Parameters
    ----------
    text_with_underscore : str
        The text containing underscores

    Returns
    -------
    str
        the text with spaces
    """

    text_with_space = " ".join(text_with_underscore.split("_"))

    return text_with_space

def check_term_in_content(term:str, words: List[str]) -> bool:
    """Check if a term is in a list of words.
    For term with multiple words, all words must be in the list and indexes must follow each other.
    Parameters
    ----------
    term : str
        Term to find.
    words : List[str]
        List of words to analyze.
    Returns
    -------
    bool
        True if the term is in the list of words, false otherwise.
    """
    term_words = term.strip().split()
    term_presence = True
    if term_words[0] in words:
        term_index = words.index(term_words[0])
        for term in term_words[1:]:
            if (term in words) and (words.index(term) == term_index+1):
                term_index += 1
            else:
                term_presence = False  
                break           
    else:
        term_presence = False
    return term_presence
