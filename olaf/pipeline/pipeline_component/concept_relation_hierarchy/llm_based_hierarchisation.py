from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Set

from spacy.tokens import Doc
from tqdm import tqdm

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
        prompt_template: Callable[[str], List[Dict[str, str]]]
            Prompt template used to give instructions and context to the LLM.
            By default the concept hierarchisation prompt is used.
        llm_generator: LLMGenerator
            The LLM model used to generate the hierarchy.
            By default, the zephyr-7b-beta HuggingFace model is used.
        doc_context_max_len: int
            Maximum number of characters for the document context in the prompt.
            By default, it is set to 4000.
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

    def _find_concept_cooc(self, concept_1: Concept, concept_2: Concept) -> Set[Doc]:
        """Extract documents where both concepts appear.

        Parameters
        ----------
        concept_1: Concept
            First concept to find the documents.
        concept_2: Concept
            Second concept to find the documents.

        Returns
        -------
        Set[Doc]
            Set of spaCy docs where the both concepts appear.
        """
        c1_docs = set()
        c2_docs = set()
        for lr in concept_1.linguistic_realisations:
            c1_docs.update(lr.get_docs())
        for lr in concept_2.linguistic_realisations:
            c2_docs.update(lr.get_docs())
        return c1_docs & c2_docs

    def _generate_doc_context(self, concepts_docs: Set[Doc]) -> str:
        """Create context from documents with a fix size.

        Parameters
        ----------
        concepts_docs: Set[Doc]
            spaCy docs to fill the context with.

        Returns
        -------
        str
            Concatenation of document contents up to a fixed size.
        """
        context = ""
        for doc in concepts_docs:
            if len(doc.text) < self.doc_context_max_len - len(context):
                context += doc.text
                context += " "
            else:
                context += doc.text[: self.doc_context_max_len - len(context)]
                break
        return context

    def _create_metarelation(
        self, llm_output: str, c1: Concept, c2: Concept
    ) -> Metarelation | None:
        """Create a metarelation based on the LLM output.
        If no metarelation is created, the output is None.

        Parameters
        ----------
        llm_output: str
            Answer of the LLM for the hierarchy.
        c1: Concept
            First concept implied in the metarelation.
        C2: Concept
            Second concept implied in the metarelation.

        Returns
        -------
        Metarelation | None
            The metarelation created or None if no relation is created.
        """
        new_metarelation = None
        if llm_output == "1":
            new_metarelation = Metarelation(
                source_concept=c2, destination_concept=c1, label="is_generalised_by"
            )
        elif llm_output == "2":
            new_metarelation = Metarelation(
                source_concept=c1, destination_concept=c2, label="is_generalised_by"
            )
        elif not (llm_output == "3"):
            logger.warning(
                "LLM generator output is not in the expected format. Hierarchical relations not extracted between concepts %s and %s.",
                c1.label,
                c2.label,
            )
        return new_metarelation

    def run(self, pipeline: Any) -> None:
        """Method that is responsible for the execution of the component.
        Metarelations are created based on the concepts.

        Parameters
        ----------
        pipeline: Pipeline
            The pipeline to run the component with."""
        concept_pairs = list(combinations(pipeline.kr.concepts, 2))
        for concept_1, concept_2 in tqdm(concept_pairs):
            concepts_docs = self._find_concept_cooc(concept_1, concept_2)
            if len(concepts_docs) > 0:
                context = self._generate_doc_context(concepts_docs)
                prompt = self.prompt_template(concept_1.label, concept_2.label, context)
                llm_output = self.llm_generator.generate_text(prompt)
                new_metarelation = self._create_metarelation(
                    llm_output, concept_1, concept_2
                )
                if new_metarelation is not None:
                    pipeline.kr.metarelations.add(new_metarelation)
