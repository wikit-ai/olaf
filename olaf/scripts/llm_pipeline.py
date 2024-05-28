import os

import spacy
import openai
from .runner import Runner
from dotenv import load_dotenv

from olaf import Pipeline
from olaf.commons.errors import MissingEnvironmentVariable
from olaf.commons.llm_tools import LLMGenerator
from olaf.commons.logging_config import logger
from olaf.commons.prompts import (
    openai_prompt_concept_term_extraction,
    openai_prompt_concept_extraction,
    openai_prompt_hierarchisation,
    openai_prompt_owl_axiom_extraction,
    openai_prompt_relation_extraction,
    openai_prompt_relation_term_extraction,
    openai_prompt_term_enrichment,
)
from olaf.pipeline.pipeline_component.axiom_extraction import LLMBasedOWLAxiomExtraction
from olaf.pipeline.pipeline_component.candidate_term_enrichment import (
    LLMBasedTermEnrichment,
)
from olaf.pipeline.pipeline_component.concept_relation_extraction import (
    LLMBasedConceptExtraction,
    LLMBasedRelationExtraction,
)
from olaf.pipeline.pipeline_component.concept_relation_hierarchy import (
    LLMBasedHierarchisation,
)
from olaf.pipeline.pipeline_component.term_extraction import LLMTermExtraction
from olaf.repository.corpus_loader import TextCorpusLoader
from olaf.repository.serialiser import KRJSONSerialiser

load_dotenv()


class CustomLLMGenerator(LLMGenerator):
    """Text generator based on the new version of OpenAI gpt-3.5-turbo model."""

    def check_resources(self) -> None:
        """Check that the resources needed to use the custom generator are available."""
        if "OPENAI_API_KEY" not in os.environ:
            raise MissingEnvironmentVariable(self.__class__, "OPENAI_API_KEY")

    def generate_text(self, prompt: list[dict[str, str]]) -> str:
        """Generate text based on a chat completion prompt for the OpenAI gtp-3.5-turbo-0125 model.

        Parameters
        ----------
        prompt: list[dict[str, str]]
            The prompt to use for text generation.pipeline

        Returns
        -------
        str
            The output generated by the LLM.
        """
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        llm_output = ""
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                temperature=0,
                max_tokens=10000,
                messages=prompt,
            )
            llm_output = response.choices[0].message.content
        except Exception as e:
            logger.error(
                """Exception %s still occurred after retries on OpenAI API.
                         Skipping document %s...""",
                e,
                prompt[-1]["content"][5:100],
            )
        return llm_output


class PipelineRunner(Runner):
    """
    Attributes
    ----------
    pipeline : Pipeline
        The pipeline to build and run.
    """

    def __init__(self, model_name="en_core_web_md", corpus_path=""):
        """Initialise a pipeline Runner.
        Parameters
        ----------
        spacy_model: spacy.language.Language
            The spacy model used to represent text corpus.
        corpus_path : str
            Path of the text corpus to use.
            It can be a folder or a file."""
        spacy_model = spacy.load(model_name)
        if os.path.isfile(corpus_path) or os.path.isdir(corpus_path):
            corpus_loader = TextCorpusLoader(corpus_path=corpus_path)
        else:
            corpus_loader = TextCorpusLoader(
                corpus_path=os.path.join(os.getenv("DATA_PATH"), "demo.txt")
            )
        self.pipeline = Pipeline(spacy_model=spacy_model, corpus_loader=corpus_loader)

    def add_pipeline_components(self) -> None:
        """Create pipeline with only LLM components."""

        openai_generator = CustomLLMGenerator()
        llm_cterm_extraction = LLMTermExtraction(
            prompt_template=openai_prompt_concept_term_extraction,
            llm_generator=openai_generator,
        )
        self.pipeline.add_pipeline_component(llm_cterm_extraction)

        llm_cterm_enrichment = LLMBasedTermEnrichment(
            openai_prompt_term_enrichment, openai_generator
        )
        self.pipeline.add_pipeline_component(llm_cterm_enrichment)

        llm_concept_extraction = LLMBasedConceptExtraction(
            openai_prompt_concept_extraction, openai_generator
        )
        self.pipeline.add_pipeline_component(llm_concept_extraction)

        llm_hierarchisation = LLMBasedHierarchisation(
            openai_prompt_hierarchisation, openai_generator
        )
        self.pipeline.add_pipeline_component(llm_hierarchisation)

        llm_rterm_extraction = LLMTermExtraction(
            prompt_template=openai_prompt_relation_term_extraction,
            llm_generator=openai_generator,
        )
        self.pipeline.add_pipeline_component(llm_rterm_extraction)

        llm_rterm_enrichment = LLMBasedTermEnrichment(
            openai_prompt_term_enrichment, openai_generator
        )
        self.pipeline.add_pipeline_component(llm_rterm_enrichment)

        llm_relation_extraction = LLMBasedRelationExtraction(
            openai_prompt_relation_extraction, openai_generator
        )
        self.pipeline.add_pipeline_component(llm_relation_extraction)

        llm_axiom_extraction = LLMBasedOWLAxiomExtraction(
            openai_prompt_owl_axiom_extraction,
            openai_generator,
            namespace="https://github.com/wikit-ai/olaf/o/example#",
        )
        self.pipeline.add_pipeline_component(llm_axiom_extraction)

    def run(self) -> None:
        """Pipeline execution."""

        self.add_pipeline_components()
        self.pipeline.run()

        kr_serialiser = KRJSONSerialiser()
        kr_serialisation_path = os.path.join(os.getcwd(), "llm_pipeline_kr.json")
        kr_serialiser.serialise(kr=self.pipeline.kr, file_path=kr_serialisation_path)

        kr_rdf_graph_path = os.path.join(os.getcwd(), "llm_pipeline_kr_rdf_graph.ttl")
        self.pipeline.kr.rdf_graph.serialize(kr_rdf_graph_path, format="ttl")

        print(f"Nb concepts: {len(self.pipeline.kr.concepts)}")
        print(f"Nb relations: {len(self.pipeline.kr.relations)}")
        print(f"Nb metarelations: {len(self.pipeline.kr.metarelations)}")
        print(f"The KR object has been JSON serialised in : {kr_serialisation_path}")
        print(f"The KR RDF graph has been serialised in : {kr_rdf_graph_path}")
