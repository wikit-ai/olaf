import ast
from typing import Any, Callable, Dict, List, Optional, Set

from spacy.tokens import Doc

from ....commons.llm_tools import HuggingFaceGenerator, LLMGenerator
from ....commons.logging_config import logger
from ....commons.prompts import hf_prompt_hierarchisation
from ....data_container.concept_schema import Concept
from ....data_container.metarelation_schema import Metarelation
from ..pipeline_component_schema import PipelineComponent


class LLMBasedHierarchisation(PipelineComponent):
    """LLM based concept hierarchisation.

    Attributes
    ----------
    prompt_template: Callable[[str], List[Dict[str, str]]]
        Prompt template used to give instructions and context to the LLM.
    llm_generator: LLMGenerator
        The LLM model used to generate the concept hierarchy.
    doc_context_max_len: int
        Maximum number of characters for the document context in the prompt.
    """

    def __init__(
        self,
        prompt_template: Optional[Callable[[str], List[Dict[str, str]]]] = None,
        llm_generator: Optional[LLMGenerator] = None,
        doc_context_max_len: Optional[int] = 4000,
    ) -> None:
        """Initialise LLM concept hierarchisation pipeline component instance.

        Parameters
        ----------
        prompt_template: Callable[[str], List[Dict[str, str]]], optional
            Prompt template used to give instructions and context to the LLM.
            By default the concept hierarchisation prompt is used.
        llm_generator: LLMGenerator, optional
            The LLM model used to generate the hierarchy.
            By default, the zephyr-7b-beta HuggingFace model is used.
        doc_context_max_len: int, optional
            Maximum number of characters for the document context in the prompt, by default 4000.
        """
        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else hf_prompt_hierarchisation
        )
        self.llm_generator = (
            llm_generator if llm_generator is not None else HuggingFaceGenerator()
        )
        self.doc_context_max_len = doc_context_max_len
        self._check_resources()

    def optimise(
        self, validation_terms: Set[str], option_values_map: Set[float]
    ) -> None:
        """A method to optimise the pipeline component by tuning the configuration."""
        raise NotImplementedError

    def _check_resources(self) -> None:
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

    def _extract_popular_docs(self, docs: List[Doc]) -> Set[Doc]:
        # TODO
        return set(docs)

    def _generate_doc_context(self, popular_docs: Set[Doc]) -> str:
        """Create context from documents with a fix size.

        Parameters
        ----------
        popular_docs: Set[Doc]
            spaCy docs to fill the context with.

        Returns
        -------
        str
            Concatenation of document contents up to a fixed size.
        """
        context = ""
        for doc in popular_docs:
            if len(doc.text) < self.doc_context_max_len - len(context):
                context += doc.text
                context += " "
            else:
                context += doc.text[: self.doc_context_max_len - len(context)]
                break
        return context

    def _create_concepts_description(self, concepts: Set[Concept]) -> str:
        """Create concepts textual description.

        Parameters
        ----------
        concepts: Set[Concept]
            Concepts to describe.

        Returns
        -------
        str
            Textual description of the concepts.
        """
        concepts_description = "Concepts:\n"
        for concept in concepts:
            lrs = [
                lr.label
                for lr in concept.linguistic_realisations
                if not (lr.label == concept.label)
            ]
            if len(lrs):
                concepts_description += f"{concept.label} ({', '.join(lrs)})\n"
            else:
                concepts_description += f"{concept.label}\n"
        return concepts_description

    def _find_concept_by_label(self, label: str, concepts: Set[Concept]) -> Concept:
        """Find a concept based on its label.

        Parameters
        ----------
        label: str
            The label of the wanted concept.
        concepts: Set[Concept]
            The set of concepts to be searched.

        Returns
        -------
        Concept
            The concept with the wanted label.
        """
        selected_concept = None
        for concept in concepts:
            if concept.label == label:
                selected_concept = concept
                break
        return selected_concept

    def _create_metarelations(
        self, llm_output: str, concepts: Set[Concept]
    ) -> Set[Metarelation] | None:
        """Create metarelations based on the LLM output.

        Parameters
        ----------
        llm_output: str
            Answer of the LLM for the hierarchy.
        concepts: Set[Concept]
            The set of existing concepts.

        Returns
        -------
        Set[Metarelation]
            The metarelations created.
        """
        metarelations = set()
        try:
            list_metarelations = ast.literal_eval(llm_output)
            for meta_tuple in list_metarelations:
                source_concept = self._find_concept_by_label(meta_tuple[0], concepts)
                destination_concept = self._find_concept_by_label(
                    meta_tuple[2], concepts
                )
                if source_concept is not None and destination_concept is not None:
                    new_metarelation = Metarelation(
                        source_concept, destination_concept, "is_generalised_by"
                    )
                    metarelations.add(new_metarelation)
        except (SyntaxError, ValueError):
            logger.error(
                """LLM generator output is not in the expected format. 
                The metarelations can not be extracted."""
            )
        return metarelations

    def run(self, pipeline: Any) -> None:
        """Method that is responsible for the execution of the component.
        Metarelations are created based on the concepts.

        Parameters
        ----------
        pipeline: Pipeline
            The pipeline to run the component with."""

        popular_docs = self._extract_popular_docs(pipeline.corpus)
        context = self._generate_doc_context(popular_docs)
        concepts_description = self._create_concepts_description(pipeline.kr.concepts)
        prompt = self.prompt_template(context, concepts_description)
        llm_output = self.llm_generator.generate_text(prompt)
        metarelations = self._create_metarelations(llm_output, pipeline.kr.concepts)

        pipeline.kr.metarelations.update(metarelations)
