import os
import spacy
from abc import ABC, abstractmethod
from olaf import Pipeline
from olaf.repository.corpus_loader import TextCorpusLoader


class Runner(ABC):
    def __init__(self, model_name: str, corpus_path: str):
        """Initialise a pipeline Runner.

        Attributes
        ----------
        pipeline: Pipeline
            The pipeline to execute.
        """
        spacy_model = spacy.load(model_name)
        print(os.getcwd())
        corpus_loader = TextCorpusLoader(corpus_path=corpus_path)
        self.pipeline = Pipeline(spacy_model=spacy_model, corpus_loader=corpus_loader)

    @abstractmethod
    def run(self) -> None:
        """pipeline execution."""

    @abstractmethod
    def add_pipeline_components(self) -> None:
        """Add all neccesary components in the pipeline"""

    def describe(self) -> None:
        self.add_pipeline_components()
        print("Pipeline components: ")
        for component in self.pipeline.pipeline_components:
            print("\t", component.__class__.__name__)
