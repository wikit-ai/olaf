import os
from typing import List

import pandas as pd

from ...commons.errors import FileOrDirectoryNotFoundError
from ...commons.logging_config import logger
from .corpus_loader_schema import CorpusLoader


class CsvCorpusLoader(CorpusLoader):
    """Corpus loader for csv file.

    Parameters
    ----------
    corpus_path : str
        Path of the text corpus to use.
    column_name : str
        Name of the column to use in the csv file.
    """

    def __init__(self, corpus_path: str, column_name: str) -> None:
        """Initialise csv corpus loader.

        Parameters
        ----------
        corpus_path : str
            Path of the text corpus to use.
        column_name : str
            Name of the column to use in the csv file.
        """
        super().__init__(corpus_path)
        self.column_name = column_name

    def _extract_column_from_dataframe(self, dataframe: pd.DataFrame) -> List[str]:
        """Extract content from a specific column and convert it into list.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe representing the corpus.

        Returns
        -------
        List[str]
            List of data to add in the corpus.
        """
        file_content = []
        if self.column_name in dataframe:
            file_content += dataframe[self.column_name].to_list()
        else:
            logger.warning(
                f"File {self.corpus_path} do not have column {self.column_name}."
            )
        return file_content

    def _read_corpus(self) -> List[str]:
        """Load csv file(s) contained in a file or a folder and convert it/them as a list of texts.

        Returns
        -------
        List[str]
            Corpus represented as a list of texts.
        """
        corpus = []

        if os.path.isdir(self.corpus_path):
            for filename in os.listdir(self.corpus_path):
                file_path = os.path.join(self.corpus_path, filename)
                if os.path.isfile(file_path):
                    df = pd.read_csv(file_path)
                    corpus += self._extract_column_from_dataframe(df)

        elif os.path.isfile(self.corpus_path):
            df = pd.read_csv(self.corpus_path)
            corpus += self._extract_column_from_dataframe(df)

        else:
            logger.error(f"File path {self.corpus_path} is invalid.")
            raise FileOrDirectoryNotFoundError(self.corpus_path)
        return corpus
