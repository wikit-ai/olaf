import os

from ...commons.errors import FileOrDirectoryNotFoundError
from ...commons.logging_config import logger
from .corpus_loader_schema import CorpusLoader


class TextCorpusLoader(CorpusLoader):
    """Corpus loader for text files in a same folder.

    If the corpus path is a folder, each text file in the folder is considered one document.
    If the corpus path is a text file, each line in the text file is considered one document.

    Parameters
    ----------
    corpus_path : str
        Path of the text corpus to use.
        It can be a folder or a file.
    """

    def __init__(self, corpus_path: str) -> None:
        """Initialise text corpus loader.

        Parameters
        ----------
        corpus_path : str
            Path of the text corpus to use.
        """
        super().__init__(corpus_path)

    def _read_corpus(self) -> list[str]:
        """Load text contents and convert them as a list of texts.

        Returns
        -------
        List[str]
            Corpus represented as a list of texts.
        """
        text_corpus = []

        if os.path.isdir(self.corpus_path):
            for filename in os.listdir(self.corpus_path):
                file_path = os.path.join(self.corpus_path, filename)
                file_extension = filename.split(".")[-1]
                if file_extension == "txt":
                    with open(file_path, "r", encoding="utf-8") as file:
                        text_corpus.append(file.read())

        elif os.path.isfile(self.corpus_path) and (
            self.corpus_path.split(".")[-1] == "txt"
        ):
            with open(self.corpus_path, "r", encoding="utf-8") as file:
                for line in file.readlines():
                    if len(line.strip()):
                        text_corpus.append(line)
        else:
            logger.error(
                "Corpus path %s is invalid, or the file extension is not '.txt'.",
                self.corpus_path,
            )
            raise FileOrDirectoryNotFoundError(self.corpus_path)

        return text_corpus
