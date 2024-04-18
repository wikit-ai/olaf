from typing import Any, Callable, Dict, List, Optional, Set

from rdflib import Graph
from spacy.tokens import Doc

from ....commons.llm_tools import HuggingFaceGenerator, LLMGenerator
from ....commons.logging_config import logger
from ....commons.prompts import hf_prompt_owl_axiom_extraction
from ....data_container import Concept, Metarelation, Relation
from ....data_container.metarelation_schema import METARELATION_RDFS_OWL_MAP
from ..pipeline_component_schema import PipelineComponent


class LLMBasedOWLAxiomExtraction(PipelineComponent):
    """LLM based OWL axiom extraction.

    Attributes
    ----------
    prompt_template: Callable[[str], List[Dict[str, str]]]
        Prompt template used to give instructions and context to the LLM.
    llm_generator: LLMGenerator
        The LLM model used to generate axioms.
    namespace: str
        The name space used for axiom generation, by default "http://www.ms2.org/o/example#".
    """

    def __init__(
        self,
        prompt_template: Optional[Callable[[str], List[Dict[str, str]]]] = None,
        llm_generator: Optional[LLMGenerator] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """Initialise LLM-based OWL axiom extraction pipeline component instance.

        Parameters
        ----------
        prompt_template: Callable[[str], List[Dict[str, str]]], optional
            Prompt template used to give instructions and context to the LLM.
            By default the axiom extraction prompt is used.
        llm_generator: LLMGenerator, optional
            The LLM model used to generate the axioms.
            By default, the zephyr-7b-beta HuggingFace model is used.
        namespace: str, optional
            The namespace used for axiom generation, by default "http://www.ms2.org/o/example#".
        """
        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else hf_prompt_owl_axiom_extraction
        )
        self.llm_generator = (
            llm_generator if llm_generator is not None else HuggingFaceGenerator()
        )
        self.namespace = (
            namespace if namespace is not None else "http://www.ms2.org/o/example#"
        )
        self.check_resources()

    def optimise(
        self, validation_terms: Set[str], option_values_map: Set[float]
    ) -> None:
        """A method to optimise the pipeline component by tuning the configuration."""
        raise NotImplementedError

    def check_resources(self) -> None:
        """Method to check that the component has access to all its required resources."""
        self.llm_generator.check_resources()

    def _compute_metrics(self) -> None:
        """A method to compute component performance metrics.
        It is used by the optimise method to update the options.
        """
        raise NotImplementedError

    def get_performance_report(self) -> Dict[str, Any]:
        """A getter for the pipeline component performance report.
        If the component has been optimised, it only returns the best performance.
        Otherwise, it returns the results obtained with the set parameters.

        Returns
        -------
        Dict[str, Any]
            The pipeline component performance report.
        """
        raise NotImplementedError

    def _concepts_to_text(self, concepts: Set[Concept]) -> str:
        """Create textual description of concepts.

        Parameters
        ----------
        concepts: Set[Concept]
            Set of concepts.

        Returns
        -------
        str
            The textual description of the concepts.
        """
        kr_description = ""
        kr_description += "Concepts:\n"
        for concept in concepts:
            lrs = [
                lr.label
                for lr in concept.linguistic_realisations
                if not (lr.label == concept.label)
            ]
            if len(lrs):
                kr_description += f"{concept.label} ({', '.join(lrs)})\n"
            else:
                kr_description += f"{concept.label}\n"

        return kr_description

    def _relations_to_text(self, owl_graph: Graph, relations: Set[Relation]) -> str:
        """Create textual description of relations.

        Parameters
        ----------
        owl_graph: Graph
            The graph with possible existing classes and individuals.
        relations: Set[Relation]
            Set of relations.

        Returns
        -------
        str
            The textual description of the relations with classes and individuals description.
        """
        kr_description = ""
        kr_description += "Classes:\n"
        q_res = owl_graph.query("SELECT ?class WHERE {?class rdf:type owl:Class .}")
        kr_description += ", ".join([item.fragment for res in q_res for item in res])
        kr_description += "Individuals:\n"
        q_res = owl_graph.query(
            "SELECT ?instance WHERE {?instance rdf:type owl:NamedIndividual .}"
        )
        kr_description += ", ".join([item.fragment for res in q_res for item in res])
        kr_description += "\nRelations:\n"
        rel = set()
        for relation in relations:
            if relation.label not in rel:
                lrs = [
                    lr.label
                    for lr in relation.linguistic_realisations
                    if not (lr.label == relation.label)
                ]
                if len(lrs):
                    kr_description += f"({relation.source_concept.label if relation.source_concept is not None else ''}, {relation.label} ({', '.join(lrs)}), {relation.destination_concept.label if relation.destination_concept is not None else ''})\n"
                else:
                    kr_description += f"({relation.source_concept.label if relation.source_concept is not None else ''}, {relation.label}, {relation.destination_concept.label if relation.destination_concept is not None else ''})\n"
                rel.add(relation.label)

        return kr_description

    def _metarelations_to_text(
        self, owl_graph: Graph, metarelations: Set[Metarelation]
    ) -> str:
        """Create textual description of metarelations.

        Parameters
        ----------
        owl_graph: Graph
            The graph with possible existing classes and individuals.
        metarelations: Set[Metarelation]
            Set of metarelations.

        Returns
        -------
        str
            The textual description of the metarelations with classes and individuals description.
        """
        kr_description = ""
        kr_description += "Classes:\n"
        q_res = owl_graph.query("SELECT ?class WHERE {?class rdf:type owl:Class .}")
        kr_description += ", ".join([item.fragment for res in q_res for item in res])
        kr_description += "Individuals:\n"
        q_res = owl_graph.query(
            "SELECT ?instance WHERE {?instance rdf:type owl:NamedIndividual .}"
        )
        kr_description += ", ".join([item.fragment for res in q_res for item in res])
        kr_description += "\nRelations:\n"
        for meta in metarelations:
            kr_description += f"({meta.source_concept.label}, {METARELATION_RDFS_OWL_MAP[meta.label].replace('http://www.w3.org/2000/01/rdf-schema#','rdfs:')}, {meta.destination_concept.label})\n"

        return kr_description

    def _llm_output_to_owl_graph(self, llm_output: str) -> Graph:
        """Convert llm output in turtle format as an RDF graph.

        Parameters
        ----------
        llm_output: str
            The llm output in turtle format.

        Returns
        -------
        Graph
            The RDF graph created.
        """
        if not (llm_output[-1] == "."):
            llm_output = llm_output[: llm_output.rfind(".") + 1]
        owl_graph = Graph()
        try:
            owl_graph.parse(data=llm_output, format="ttl")
        except SyntaxError as e:
            logger.error(
                "LLM generator output is not in the expected format. Axioms can not be extracted. Trace : %s",
                e,
            )
        return owl_graph

    def run(self, pipeline: Any) -> None:
        """Method that is responsible for the execution of the component.
        Axioms are created based on the knowledge representation.
        The RDF graph is set on the knowledge representation.

        Parameters
        ----------
        pipeline: Pipeline
            The pipeline to run the component with."""

        kr_owl_graph = Graph()

        if len(pipeline.kr.concepts):
            kr_description = self._concepts_to_text(pipeline.kr.concepts)
            prompt = self.prompt_template(kr_description, self.namespace)
            llm_output = self.llm_generator.generate_text(prompt)
            kr_owl_graph += self._llm_output_to_owl_graph(llm_output)

        if len(pipeline.kr.relations):
            kr_description = self._relations_to_text(
                kr_owl_graph, pipeline.kr.relations
            )
            prompt = self.prompt_template(kr_description, self.namespace)
            llm_output = self.llm_generator.generate_text(prompt)
            kr_owl_graph += self._llm_output_to_owl_graph(llm_output)

        if len(pipeline.kr.metarelations):
            kr_description = self._metarelations_to_text(
                kr_owl_graph, pipeline.kr.metarelations
            )
            prompt = self.prompt_template(kr_description, self.namespace)
            llm_output = self.llm_generator.generate_text(prompt)
            kr_owl_graph += self._llm_output_to_owl_graph(llm_output)

        pipeline.kr.rdf_graph = kr_owl_graph
