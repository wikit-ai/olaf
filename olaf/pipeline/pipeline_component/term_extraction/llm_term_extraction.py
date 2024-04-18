import ast
import re
from typing import Any, Callable, Dict, List, Optional, Set

from spacy.tokens import Doc

from ....commons.llm_tools import HuggingFaceGenerator, LLMGenerator
from ....commons.logging_config import logger
from ....commons.prompts import hf_prompt_concept_term_extraction
from ....data_container.candidate_term_schema import CandidateTerm
from .term_extraction_schema import TermExtractionPipelineComponent


class LLMTermExtraction(TermExtractionPipelineComponent):
    """Extract candidate terms using LLM based on the corpus.

    Attributes
    ----------
    prompt_template: Callable[[str], List[Dict[str, str]]]
        Prompt template used to give instructions and context to the LLM.
    llm_generator: LLMGenerator
        The LLM model used to generate the candidate terms.
    cts_post_processing_functions: List[Callable[[Set[CandidateTerm]], Set[CandidateTerm]]], optional
        A list of candidate term post processing functions to run after candidate term extraction
        and before assigning the extracted candidate terms to the pipeline. Default to None.
    """

    def __init__(
        self,
        prompt_template: Optional[Callable[[str], List[Dict[str, str]]]] = None,
        llm_generator: Optional[LLMGenerator] = None,
        cts_post_processing_functions: Optional[
            List[Callable[[Set[CandidateTerm]], Set[CandidateTerm]]]
        ] = None,
    ) -> None:
        """Initialise LLM term extraction pipeline component instance.

        Parameters
        ----------
        cts_post_processing_functions: List[Callable[[Set[CandidateTerm]], Set[CandidateTerm]]], optional
            A list of candidate term post processing functions to run after candidate term extraction
            and before assigning the extracted candidate terms to the pipeline. Default to None.
        prompt_template: Callable[[str], List[Dict[str, str]]]
            Prompt template used to give instructions and context to the LLM.
            By default the concept term extraction prompt is used.
        llm_generator: LLMGenerator
            The LLM model used to generate the candidate terms.
            By default, the zephyr-7b-beta HuggingFace model is used.
        """
        super().__init__(cts_post_processing_functions)
        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else hf_prompt_concept_term_extraction
        )
        self.llm_generator = (
            llm_generator if llm_generator is not None else HuggingFaceGenerator()
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

    def _generate_candidate_terms(self, doc: Doc) -> Set[str]:
        """Generate candidate term labels from a document content with a LLM.

        Parameters
        ----------
        doc: Doc
            The spaCy doc used to generate candidate terms from.

        Returns
        -------
        Set[str]
            The set of candidate term labels generated.
        """
        doc_prompt = self.prompt_template(doc.text)
        llm_output = self.llm_generator.generate_text(doc_prompt)
        try:
            ct_labels = ast.literal_eval(llm_output)
            if isinstance(ct_labels, List):
                ct_labels = set(ct_labels)
            else:
                logger.error(
                    """LLM generator output is not in the expected format. The candidate terms can not be processed.\nDoc concerned : %s...""",
                    doc.text[:100],
                )
                ct_labels = set()
        except Exception:
            logger.error(
                """LLM generator output is not in the expected format.
                The candidate terms can not be processed.
                \nDoc concerned : %s...""",
                doc.text[:100],
            )
            ct_labels = set()
        return ct_labels

    def _update_candidate_terms(
        self, doc: Doc, ct_labels: Set[str], ct_index: Dict[str, CandidateTerm]
    ) -> Set[CandidateTerm]:
        """Update the candidate terms by adding new candidates if their labels appear in the corpus.
        Corpus occurrences are also found to create candidate term instance.

        Parameters
        ----------
        doc: Doc
            The spaCy doc used to validate candidate terms from.
        ct_labels: Set[str]
            The candidate term labels to validate.
        ct_index: Dict[str, CandidateTerm]
            The index of candidate terms with label as key and the candidate term object as value.
        """
        for label in ct_labels:
            if label in doc.text:
                occurrences = set()
                for string_match in re.finditer(label, doc.text):
                    span = doc.char_span(string_match.start(), string_match.end())
                    if span is not None:
                        occurrences.add(span)
                if label in ct_index.keys():
                    ct_index[label].add_corpus_occurrences(occurrences)
                else:
                    ct_index[label] = CandidateTerm(label, occurrences)

    def run(self, pipeline: Any) -> None:
        """Method that is responsible for the execution of the component.

        Parameters
        ----------
        pipeline: Pipeline
            The pipeline to run the component with.
        """
        ct_index = {}
        for ct in pipeline.candidate_terms:
            ct_index[ct.label] = ct

        for doc in pipeline.corpus:
            ct_labels = self._generate_candidate_terms(doc)
            self._update_candidate_terms(doc, ct_labels, ct_index)

        new_cts = set(ct_index.values())
        new_cts = self.apply_post_processing(new_cts)
        pipeline.candidate_terms.update(new_cts)
