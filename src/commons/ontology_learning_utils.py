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
