import os

import spacy
from dotenv import load_dotenv
from .runner import Runner

from olaf import Pipeline
from olaf.pipeline.pipeline_component.concept_relation_extraction import (
    AgglomerativeClusteringConceptExtraction,
    AgglomerativeClusteringRelationExtraction,
)
from olaf.pipeline.pipeline_component.term_extraction import (
    POSTermExtraction,
)
from olaf.commons.kr_to_rdf_tools import (
    kr_concepts_to_owl_classes,
    kr_relations_to_owl_obj_props,
    kr_metarelations_to_owl,
    kr_relations_to_anonymous_some_parent,
    concept_lrs_to_owl_individuals,
)
from olaf.pipeline.pipeline_component.axiom_extraction import OWLAxiomExtraction
from olaf.repository.corpus_loader import TextCorpusLoader
from olaf.repository.serialiser import KRJSONSerialiser

load_dotenv()


class PipelineRunner(Runner):
    def __init__(self, model_name="en_core_web_md"):
        """Initialise a pipeline Runner."""
        super.__init__(model_name)


    def add_pipeline_components(self) -> None:
        """Create pipeline without LLM components."""

        pos_term_extraction = POSTermExtraction(pos_selection=["NOUN"])
        self.pipeline.add_pipeline_component(pos_term_extraction)

        ac_param = {
            "embedding_model": "sentence-transformers/sentence-t5-base",
            "distance_threshold": 0.1,
        }
        ac_concept_extraction = AgglomerativeClusteringConceptExtraction(**ac_param)
        self.pipeline.add_pipeline_component(ac_concept_extraction)

        pos_term_extraction = POSTermExtraction(pos_selection=["VERB"])

        self.pipeline.add_pipeline_component(pos_term_extraction)

        ac_relation_extraction = AgglomerativeClusteringRelationExtraction(**ac_param)
        self.pipeline.add_pipeline_component(ac_relation_extraction)

        axiom_generators = {
            kr_concepts_to_owl_classes,
            kr_relations_to_owl_obj_props,
            kr_metarelations_to_owl,
            kr_relations_to_anonymous_some_parent,
            concept_lrs_to_owl_individuals,
        }
        owl_axiom_extraction = OWLAxiomExtraction(
            owl_axiom_generators=axiom_generators,
            base_uri="https://github.com/wikit-ai/olaf/o/example#",
        )
        self.pipeline.add_pipeline_component(owl_axiom_extraction)

    def run(self) -> None:
        """LLM pipeline execution."""

        self.add_pipeline_components()
        self.pipeline.run()

        kr_serialiser = KRJSONSerialiser()
        kr_serialisation_path = os.path.join(os.getcwd(), "no_llm_pipeline_kr.json")
        kr_serialiser.serialise(kr=self.pipeline.kr, file_path=kr_serialisation_path)

        kr_rdf_graph_path = os.path.join(
            os.getcwd(),
            "no_llm_pipeline_kr_rdf_graph.ttl",
        )
        self.pipeline.kr.rdf_graph.serialize(kr_rdf_graph_path, format="ttl")

        print(f"Nb concepts: {len(self.pipeline.kr.concepts)}")
        print(f"Nb relations: {len(self.pipeline.kr.relations)}")
        print(f"Nb metarelations: {len(self.pipeline.kr.metarelations)}")
        print(f"The KR object has been JSON serialised in : {kr_serialisation_path}")
        print(f"The KR RDF graph has been serialised in : {kr_rdf_graph_path}")
