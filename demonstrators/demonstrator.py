import os

from dotenv import load_dotenv

load_dotenv()

import spacy

from olaf import Pipeline
from olaf.data_container import CandidateTerm
from olaf.commons.spacy_processing_tools import is_not_punct, is_not_stopword
from olaf.pipeline.pipeline_component.candidate_term_enrichment import (
    KnowledgeBasedCTermEnrichment,
)
from olaf.pipeline.pipeline_component.concept_relation_extraction import (
    AgglomerativeClusteringConceptExtraction,
    AgglomerativeClusteringRelationExtraction,
)
from olaf.pipeline.pipeline_component.concept_relation_hierarchy import (
    SubsumptionHierarchisation,
)
from olaf.pipeline.pipeline_component.term_extraction import (
    POSTermExtraction,
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
from olaf.repository.knowledge_source import WordNetKnowledgeResource
from olaf.repository.serialiser import KRJSONSerialiser


def create_pipeline(model_name="en_core_web_md", corpus_file="") -> Pipeline:
    """Initialise a pipeline.

    Returns
    -------
    Pipeline
        The new pipeline created.
    """
    spacy_model = spacy.load(model_name)
    if os.path.isfile(corpus_file):
        corpus_loader = TextCorpusLoader(corpus_path=corpus_file)
    elif corpus_path := os.path.isfile(
        os.path.join(os.getenv("DATA_PATH"), corpus_file)
    ):
        corpus_loader = TextCorpusLoader(corpus_path=corpus_path)
    else:
        corpus_path = os.path.isfile(
            os.path.join(os.getenv("DATA_PATH"), "pizza_description.txt")
        )
    return Pipeline(spacy_model=spacy_model, corpus_loader=corpus_loader)


def add_pipeline_components(pipeline: Pipeline) -> Pipeline:
    """Create pipeline without LLM components.

    Parameters
    ----------
    pipeline: Pipeline
        The pipeline into which the components are to be added.

    Returns
    -------
    Pipeline
        The pipeline updated with new components.
    """

    pos_term_extraction = POSTermExtraction(pos_selection=["NOUN"])
    pipeline.add_pipeline_component(pos_term_extraction)

    ac_param = {
        "embedding_model": "sentence-transformers/sentence-t5-base",
        "distance_threshold": 0.1,
    }
    ac_concept_extraction = AgglomerativeClusteringConceptExtraction(**ac_param)
    pipeline.add_pipeline_component(ac_concept_extraction)

    pos_term_extraction = POSTermExtraction(pos_selection=["VERB"])

    pipeline.add_pipeline_component(pos_term_extraction)

    ac_relation_extraction = AgglomerativeClusteringRelationExtraction(**ac_param)
    pipeline.add_pipeline_component(ac_relation_extraction)

    return pipeline


def main() -> None:
    """LLM pipeline execution."""
    pipeline = create_pipeline()
    pipeline = add_pipeline_components(pipeline)
    pipeline.run()

    kr_serialiser = KRJSONSerialiser()
    kr_serialisation_path = os.path.join(
        os.getenv("RESULTS_PATH"), "no_llm_pipeline", "no_llm_pipeline_kr.json"
    )
    kr_serialiser.serialise(kr=pipeline.kr, file_path=kr_serialisation_path)

    kr_rdf_graph_path = os.path.join(
        os.getenv("RESULTS_PATH"), "no_llm_pipeline", "no_llm_pipeline_kr_rdf_graph.ttl"
    )
    pipeline.kr.rdf_graph.serialize(kr_rdf_graph_path, format="ttl")

    print(f"Nb concepts: {len(pipeline.kr.concepts)}")
    print(f"Nb relations: {len(pipeline.kr.relations)}")
    print(f"Nb metarelations: {len(pipeline.kr.metarelations)}")
    print(f"The KR object has been JSON serialised in : {kr_serialisation_path}")
    print(f"The KR RDF graph has been serialised in : {kr_rdf_graph_path}")
