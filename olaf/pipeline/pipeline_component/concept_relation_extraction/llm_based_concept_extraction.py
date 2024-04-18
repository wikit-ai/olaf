import ast
from typing import Any, Callable, Dict, List, Optional, Set

from spacy.tokens import Doc

from ....commons.candidate_term_tools import cts_to_concept
from ....commons.llm_tools import HuggingFaceGenerator, LLMGenerator
from ....commons.logging_config import logger
from ....commons.prompts import hf_prompt_concept_extraction
from ....data_container.candidate_term_schema import CandidateTerm
from ..pipeline_component_schema import PipelineComponent


class LLMBasedConceptExtraction(PipelineComponent):
    """LLM based concept extraction.

    Attributes
    ----------
    prompt_template: Callable[[str], List[Dict[str, str]]]
        Prompt template used to give instructions and context to the LLM.
    llm_generator: LLMGenerator
        The LLM model used to generate the concepts.
    doc_context_max_len: int
        Maximum number of characters for the document context in the prompt.
    """

    def __init__(
        self,
        prompt_template: Optional[Callable[[str], List[Dict[str, str]]]] = None,
        llm_generator: Optional[LLMGenerator] = None,
        doc_context_max_len: Optional[int] = 4000,
    ) -> None:
        """Initialise LLM concept extraction pipeline component instance.

        Parameters
        ----------
        prompt_template: Callable[[str], List[Dict[str, str]]], optional
            Prompt template used to give instructions and context to the LLM.
            By default the concept extraction prompt is used.
        llm_generator: LLMGenerator, optional
            The LLM model used to generate the concepts.
            By default, the zephyr-7b-beta HuggingFace model is used.
        doc_context_max_len: int, optional
            Maximum number of characters for the document context in the prompt, by defaut 4000.
        """
        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else hf_prompt_concept_extraction
        )
        self.llm_generator = (
            llm_generator if llm_generator is not None else HuggingFaceGenerator()
        )
        self.doc_context_max_len = doc_context_max_len
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

    def _create_doc_count(self, cterms: Set[CandidateTerm]) -> Dict[Doc, int]:
        """Count the number of candidate terms found in each documents of the corpus.

        Parameters
        ----------
        cterms: Set[CandidateTerm]
            The set of candidate terms found in the documents.

        Returns
        -------
        Dict[Doc, int]
            Document count with document as key and count of candidate terms in the document as value.
        """
        doc_count = {}
        for cterm in cterms:
            for co in cterm.corpus_occurrences:
                doc_count[co.doc] = doc_count.get(co.doc, 0) + 1
        return doc_count

    def _generate_doc_context(self, doc_count: Dict[Doc, int]) -> str:
        """Create context from documents with a fix size.
        Most relevant documents are the documents with the largest number of candidate terms found.

        Parameters
        ----------
        doc_count: Dict[Doc, int]
            Dictionary with documents as keys and the number of candidate terms found inside as values.

        Returns
        -------
        str
            Concatenation of document contents up to a fixed size.
        """
        context = ""
        sorted_doc_count = dict(
            sorted(doc_count.items(), key=lambda x: x[1], reverse=True)
        )
        for doc in sorted_doc_count:
            if len(doc.text) < self.doc_context_max_len - len(context):
                context += doc.text
                context += " "
            else:
                context += doc.text[: self.doc_context_max_len - len(context)]
                break
        return context

    def _convert_llm_output_to_cc(
        self, llm_output: str, cterm_index: Dict[str, CandidateTerm]
    ) -> List[Set[CandidateTerm]]:
        """Convert the output of the LLM to groups of candidate terms to merge as concepts.

        Parameters
        ----------
        llm_output: str
            The text generated by the LLM.
        cterm_index: Dict[str, CandidateTerm]
            Index of candidate terms with labels as keys and candidate term objects as values.

        Returns
        -------
        List[Set[CandidateTerm]]
            Groups of candidate terms to merge as concepts.
        """
        concept_candidates = []
        try:
            cc_labels = ast.literal_eval(llm_output)
            for cc_group in cc_labels:
                cc_set = {
                    cterm_index[cc_label]
                    for cc_label in cc_group
                    if cc_label in cterm_index
                }
                concept_candidates.append(cc_set)
        except (SyntaxError, ValueError):
            logger.error(
                """LLM generator output is not in the expected format. 
                The concepts can not be extracted."""
            )
        return concept_candidates

    def run(self, pipeline: Any) -> None:
        """Method that is responsible for the execution of the component.
        Concepts are created and candidate terms are purged.

        Parameters
        ----------
        pipeline: Pipeline
            The pipeline to run the component with."""
        cterm_index = {cterm.label: cterm for cterm in pipeline.candidate_terms}
        doc_count = self._create_doc_count(pipeline.candidate_terms)
        doc_context = self._generate_doc_context(doc_count)
        ct_str_list = "\n".join(cterm_index.keys())
        prompt = self.prompt_template(doc_context, ct_str_list)
        llm_output = self.llm_generator.generate_text(prompt)
        concept_candidates = self._convert_llm_output_to_cc(llm_output, cterm_index)
        for concept_candidate in concept_candidates:
            new_concept = cts_to_concept(concept_candidate)
            pipeline.kr.concepts.add(new_concept)

        pipeline.candidate_terms = set()
