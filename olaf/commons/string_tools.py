def space_to_underscore_str(text_with_space: str) -> str:
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

    text_with_underscore = text_with_space.replace(" ", "_")

    return text_with_underscore


def underscore_to_space_str(text_with_underscore: str) -> str:
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

    text_with_space = text_with_underscore.replace("_", " ")

    return text_with_space
