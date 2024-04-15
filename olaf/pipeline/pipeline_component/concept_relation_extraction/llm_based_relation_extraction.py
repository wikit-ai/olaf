import ast
from typing import Any, Callable, Dict, List, Optional, Set

from spacy.tokens import Doc

from ....commons.llm_tools import HuggingFaceGenerator, LLMGenerator
from ....commons.logging_config import logger
from ....commons.prompts import hf_prompt_relation_extraction
from ....commons.relation_tools import crs_to_relation, cts_to_crs, group_cr_by_concepts
from ....data_container.candidate_term_schema import CandidateTerm
from ..pipeline_component_schema import PipelineComponent


class LLMBasedRelationExtraction(PipelineComponent):
    """LLM based relation extraction.

    Attributes
    ----------
    prompt_template: Callable[[str], List[Dict[str, str]]], optional
        Prompt template used to give instructions and context to the LLM, by default None.
    llm_generator: LLMGenerator, optional
        The LLM model used to generate the relation, by default None.
    doc_context_max_len: int, optional
        Maximum number of characters for the document context in the prompt, by default 4000.
    concept_max_distance: int, optional
        The maximum distance between the candidate term and the concept sought.
        Set to 5 by default if not specified.
    scope: str, optional
        Scope used to search concepts. Can be "doc" for the entire document or "sent" for
        the candidate term "sentence". Set to "doc" by default if not specified.
    """

    def __init__(
        self,
        prompt_template: Optional[Callable[[str], List[Dict[str, str]]]] = None,
        llm_generator: Optional[LLMGenerator] = None,
        doc_context_max_len: Optional[int] = 4000,
        concept_max_distance: Optional[int] = None,
        scope: Optional[str] = "doc",
    ) -> None:
        """Initialise LLM relation extraction pipeline component instance.

        Parameters
        ----------
        prompt_template: Callable[[str], List[Dict[str, str]]], optional
            Prompt template used to give instructions and context to the LLM.
            By default the relation extraction prompt is used.
        llm_generator: LLMGenerator, optional
            The LLM model used to generate the relations.
            By default, the zephyr-7b-beta HuggingFace model is used.
        doc_context_max_len: int, optional
            Maximum number of characters for the document context in the prompt.
            By default, it is set to 4000.
        concept_max_distance: int, optional
            The maximum distance between the candidate term and the concept sought.
            Set to 5 by default if not specified.
        scope: str, optional
            Scope used to search concepts. Can be "doc" for the entire document or "sent" for
            the candidate term "sentence". Set to "doc" by default if not specified.
        """
        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else hf_prompt_relation_extraction
        )
        self.llm_generator = (
            llm_generator if llm_generator is not None else HuggingFaceGenerator()
        )
        self.doc_context_max_len = doc_context_max_len
        self.concept_max_distance = concept_max_distance
        self.scope = scope
        self.check_parameters()
        self._check_resources()

    def check_parameters(self) -> None:
        """Check whether required parameters are given and correct.
        If this is not the case, suitable default ones are set or errors are raised.

        This method affects the self.scope attribute.
        """
        if self.concept_max_distance is None:
            self.concept_max_distance = 5
            logger.warning(
                "No value given for concept_max_distance parameter, default will be set to 5."
            )
        elif not isinstance(self.concept_max_distance, int):
            self.concept_max_distance = 5
            logger.warning(
                "Incorrect type given for concept_max_distance parameter, default will be set to 5."
            )

        if self.scope not in {"sent", "doc"}:
            self.scope = "doc"
            logger.warning(
                """Wrong scope value. Possible values are 'sent' or 'doc'. Default to scope = 'doc'."""
            )

    def optimise(
        self, validation_terms: Set[str], option_values_map: Set[float]
    ) -> None:
        """A method to optimise the pipeline component by tuning the configuration."""
        raise NotImplementedError

    def _check_resources(self) -> None:
        """Method to check that the component has access to all its required resources."""
        self.llm_generator.check_resources()

        if self.scope not in {"sent", "doc"}:
            self.scope = "doc"
            logger.warning(
                """Wrong scope value. Possible values are 'sent' or 'doc'. Default to scope = 'doc'."""
            )

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
        for doc, _ in sorted_doc_count.items():
            if len(doc.text) < self.doc_context_max_len - len(context):
                context += doc.text
                context += " "
            else:
                context += doc.text[: self.doc_context_max_len - len(context)]
                break
        return context

    def _convert_llm_output_to_rc(
        self, llm_output: str, cterm_index: Dict[str, CandidateTerm]
    ) -> List[Set[CandidateTerm]]:
        """Convert the output of the LLM to groups of candidate terms to merge as relations.

        Parameters
        ----------
        llm_output: str
            The text generated by the LLM.
        cterm_index: Dict[str, CandidateTerm]
            Index of candidate terms with labels as keys and candidate term objects as values.

        Returns
        -------
        List[Set[CandidateTerm]]
            Groups of candidate terms to merge as relations.
        """
        relation_candidates = []
        try:
            rc_labels = ast.literal_eval(llm_output)
            for rc_group in rc_labels:
                rc_set = set()
                for rc_label in rc_group:
                    if rc_label in cterm_index.keys():
                        rc_set.add(cterm_index[rc_label])
                relation_candidates.append(rc_set)
        except (SyntaxError, ValueError):
            logger.error(
                """LLM generator output is not in the expected format. 
                The relations can not be extracted."""
            )
        return relation_candidates

    def run(self, pipeline: Any) -> None:
        """Method that is responsible for the execution of the component.
        Relations are created and candidate terms are purged.

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
        relation_candidates = self._convert_llm_output_to_rc(llm_output, cterm_index)
        concept_map = {concept.label: concept for concept in pipeline.kr.concepts}
        for rc_group in relation_candidates:
            crs = cts_to_crs(
                rc_group,
                concept_map,
                pipeline.spacy_model,
                self.concept_max_distance,
                self.scope,
            )
            new_relations = group_cr_by_concepts(crs)
            for new_relation in new_relations:
                new_relation = crs_to_relation(new_relation)
                pipeline.kr.relations.add(new_relation)

        pipeline.candidate_terms = set()
