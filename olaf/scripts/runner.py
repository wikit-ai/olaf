import os
import spacy
from abc import ABC, abstractmethod
from olaf import Pipeline
from olaf.repository.corpus_loader import TextCorpusLoader
from olaf.repository.serialiser.kr_serialisers import KRJSONSerialiser


class Runner(ABC):
    def __init__(self, model_name: str, corpus_path: str):
        """Initialise a pipeline Runner.

        Attributes
        ----------
        pipeline: Pipeline
            The pipeline to execute.
        """
        spacy_model = spacy.load(model_name)
        corpus_loader = TextCorpusLoader(corpus_path=corpus_path)
        self.pipeline = Pipeline(spacy_model=spacy_model, corpus_loader=corpus_loader)

    @abstractmethod
    def add_pipeline_components(self) -> None:
        """Add all neccesary components in the pipeline"""

    def run(self, name: str) -> None:
        """LLM pipeline execution."""

        self.add_pipeline_components()
        self.pipeline.run()

        kr_serialiser = KRJSONSerialiser()
        kr_serialisation_path = os.path.join("data/", f"{name}_kr.json")
        kr_serialiser.serialise(kr=self.pipeline.kr, file_path=kr_serialisation_path)

        kr_rdf_graph_path = os.path.join(
            "data/",
            f"{name}_kr_rdf_graph.ttl",
        )
        self.pipeline.kr.rdf_graph.serialize(kr_rdf_graph_path, format="ttl")

        print(f"Nb concepts: {len(self.pipeline.kr.concepts)}")
        print(f"Nb relations: {len(self.pipeline.kr.relations)}")
        print(f"Nb metarelations: {len(self.pipeline.kr.metarelations)}")
        print(f"The KR object has been JSON serialised in : {kr_serialisation_path}")
        print(f"The KR RDF graph has been serialised in : {kr_rdf_graph_path}")

    def describe(self) -> None:
        self.add_pipeline_components()
        print("Pipeline components: ")
        for component in self.pipeline.pipeline_components:
            print("\t", component.__class__.__name__)
