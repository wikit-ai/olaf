
from typing import Callable, Iterable

import spacy.tokens.doc

"""A Type to define token filters.
    A TokenFilter is a function taking an iterable on spacy documents as input.
    A Token filter sets the selected attribute of the spacy Doc instance to False based on some conditions.
"""
TokenFilter = Callable[[spacy.tokens.doc.Doc], spacy.tokens.doc.Doc]
