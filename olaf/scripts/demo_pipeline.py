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
    def __init__(
        self, model_name: str = "en_core_web_md", corpus_path: str = "data/demo.txt"
    ):
        """Initialise a pipeline Runner."""
        super().__init__(model_name, corpus_path)

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
        return super().run(name=__name__)
