from typing import Dict, List

import spacy.tokens
import re

"""
    The functions should return True if the token should be kept, False otherwise.
    Specifying the parameter types as annotations is critical since the token selector loading from config
    process relies on these typing annotations.
    If you introduce a new typing annotation make sure you provide the way to precess it from a config file.
"""


def select_on_pos(token: spacy.tokens.Token, pos_to_select: List[str]) -> bool:
    """Return true if the Spacy Token POS string is in the pos_to_select list.  

    Parameters
    ----------
    token : spacy.tokens.Token
        The Spacy token to test
    pos_to_select : List[str]
        The list of strings corresponding to the POS tags to keep.

    Returns
    -------
    bool
        Wether the Token POS tag is in pos_to_select or not
    """
    if token.pos_ in pos_to_select:
        return True
    else:
        return False


def select_on_shape_match_pattern(token: spacy.tokens.Token, shape_pattern_to_select: re.Pattern) -> bool:
    """Return true if the Spacy Token Shape string matches the pattern shape_pattern_to_select. 

    Parameters
    ----------
    token : spacy.tokens.Token
        The Spacy token to test
    shape_pattern_to_select : re.Pattern
        A re.Pattern object to test the Token shape against.

    Returns
    -------
    bool
        Wether the Token Shape string matches the pattern shape_pattern_to_select or not
    """
    if (shape_pattern_to_select.match(token.shape_)):
        return True
    else:
        return False


def filter_stopwords(token: spacy.tokens.Token) -> bool:
    return not (token.is_stop)


def filter_punct(token: spacy.tokens.Token) -> bool:
    return not (token.is_punct)


def filter_num(token: spacy.tokens.Token) -> bool:
    return not (token.like_num)


def filter_url(token: spacy.tokens.Token) -> bool:
    return not (token.like_url)


def select_on_occurence_count(token: spacy.tokens.Token, treshold: int, occurence_counts: Dict[str, int]) -> bool:
    """Return true if the Spacy Token Text has an occurence above a defined treshold.

    Parameters
    ----------
    token : spacy.tokens.Token
        The Spacy token to test
    treshold : int
        The treshold below which the Token is not selected 
    occurence_counts : Dict[str, int]
        A Dictionnary with token texts as keys and their occurence as value.

    Returns
    -------
    bool
        Wether the token text occurence is above the defined treshold or not.
    """
    token_occurrence = occurence_counts.get(token.text)
    selected = False
    if token_occurrence is not None:
        if token_occurrence > treshold:
            selected = True
    return selected
