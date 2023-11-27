import json
import os
from typing import List

from ...commons.errors import FileOrDirectoryNotFoundError
from ...commons.logging_config import logger
from .corpus_loader_schema import CorpusLoader


class JsonCorpusLoader(CorpusLoader):
    """Corpus loader for json files in a same folder.

    Parameters
    ----------
    corpus_path : str
        Path of the text corpus to use.
    json_field : str
        Name of the field to use in json files.
    """

    def __init__(self, corpus_path: str, json_field: str) -> None:
        """Initialise json corpus loader.

        Parameters
        ----------
        corpus_path : str
            Path of the text corpus to use.
        json_field : str
            Name of the field to use in json files.
        """
        super().__init__(corpus_path)
        self.json_field = json_field

    def _read_corpus(self) -> List[str]:
        """Load json contents and convert them as a list of texts.

        Returns
        -------
        List[str]
            Corpus represented as a list of texts.
        """
        text_corpus = []

        if os.path.isdir(self.corpus_path):
            for filename in os.listdir(self.corpus_path):
                file_path = os.path.join(self.corpus_path, filename)
                if os.path.isfile(file_path):
                    with open(file_path, "r", encoding="utf-8") as file:
                        file_content = json.load(file)
                    try:
                        text_corpus += [
                            content[self.json_field] for content in file_content
                        ]
                    except Exception as _e:
                        logger.error(
                            f"Invalid json field {self.json_field} for file {filename}."
                        )
                        raise _e

        elif os.path.isfile(self.corpus_path):
            with open(self.corpus_path, "r", encoding="utf-8") as file:
                file_content = json.load(file)
                try:
                    text_corpus += [
                        content[self.json_field] for content in file_content
                    ]
                except Exception as _e:
                    logger.error(
                        f"Invalid json field {self.json_field} for file {self.corpus_path}."
                    )
                    raise _e
        else:
            logger.error(f"Corpus path {self.corpus_path} is invalid.")
            raise FileOrDirectoryNotFoundError(self.corpus_path)

        return text_corpus
