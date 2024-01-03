import ast
import os
import re
from typing import Any, Callable, Dict, List, Optional, Set

from spacy.tokens import Doc

from ....commons.errors import MissingEnvironmentVariable
from ....commons.llm_tools import LLMGenerator, OpenAIGenerator
from ....commons.prompts import prompt_concept_term_extraction
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
            By default, the OpenAI gpt-3.5-turbo model is used.
        """
        super().__init__(cts_post_processing_functions)
        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else prompt_concept_term_extraction
        )
        self.llm_generator = (
            self.llm_generator if llm_generator is not None else OpenAIGenerator()
        )

    def optimise(
        self, validation_terms: Set[str], option_values_map: Set[float]
    ) -> None:
        """A method to optimise the pipeline component by tuning the configuration."""
        raise NotImplementedError

    def _check_resources(self) -> None:
        """Method to check that the component has access to all its required resources."""
        if isinstance(self.llm_generator, OpenAIGenerator):
            if "OPENAI_API_KEY" not in os.environ:
                raise MissingEnvironmentVariable(self.__class__, "OPENAI_API_KEY")

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
        ct_labels = set(ast.literal_eval(llm_output))
        return ct_labels

    def _update_candidate_terms(
        self, doc: Doc, ct_labels: Set[str], candidate_terms: Set[CandidateTerm]
    ) -> None:
        """Update the candidate terms by adding new candidates if their labels appear in the corpus.
        Corpus occurrences are also found to create candidate term instance.

        Parameters
        ----------
        doc: Doc
            The spaCy doc used to validate candidate terms from.
        ct_labels: Set[str]
            The candidate term labels to validate.
        candidate_terms: Set[CandidateTerm]
            The set of already existing candidate terms to update.
        """
        for label in ct_labels:
            if label in doc.text:
                occurrences = set()
                for string_match in re.finditer(label, doc.text):
                    occurrences.add(
                        doc.char_span(string_match.start(), string_match.end())
                    )
                label_ct = next(
                    (ct for ct in candidate_terms if ct.label == label),
                    None,
                )
                if label_ct is None:
                    candidate_terms.add(CandidateTerm(label, occurrences))
                else:
                    label_ct.add_corpus_occurrences(occurrences)

    def run(self, pipeline: Any) -> None:
        """Method that is responsible for the execution of the component.

        Parameters
        ----------
        pipeline: Pipeline
            The pipeline to run the component with.
        """
        candidate_terms = set()
        for doc in pipeline.corpus:
            ct_labels = self._generate_candidate_terms(doc)
            self._update_candidate_terms(doc, ct_labels, candidate_terms)

        candidate_terms = self.apply_post_processing(candidate_terms)
        pipeline.candidate_terms.update(candidate_terms)
