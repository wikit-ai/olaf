
import json
import re
from typing import Callable, Dict, List

import spacy.tokens.doc

"""A Type to define token filters.
    A TokenFilter is a function taking an iterable on spacy documents as input.
    A Token filter sets the selected attribute of the spacy Doc instance to False based on some conditions.
"""
TokenSelector = Callable[[spacy.tokens.Token], bool]

str2type_processes = {
    re.Pattern: lambda pattern_str: re.compile("^" + pattern_str + "$"),
    List[str]: lambda l_str: l_str,
    Dict[str, int]: lambda json_str: json.loads(json_str),
    int: lambda int_str: int(int_str)
}


class FileTypeDetailsNotFound(Exception):
    """An Exception to flag when the details specific to a corpus file type is not found.
    """
    pass


class TokenSelectorNotFound(Exception):
    """An Exception to flag when the Token selector has not been found.
    """
    pass


class TokenSelectorParamNotFound(Exception):
    """An Exception to flag when the Token selector parameters details have not been found.
    """
    pass


class TokenSelectorParamTypingProcessNotFound(Exception):
    """An Exception to flag when the Token selector parameter type has no string process defined.
        Or the process has not been found.
    """
    pass
