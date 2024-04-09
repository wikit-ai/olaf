import ast
from typing import Any, Callable, Dict, List, Optional, Set

from ....commons.llm_tools import HuggingFaceGenerator, LLMGenerator
from ....commons.logging_config import logger
from ....commons.prompts import hf_prompt_term_enrichment
from ....data_container import CandidateTerm, Enrichment
from ..pipeline_component_schema import PipelineComponent


class LLMBasedTermEnrichment(PipelineComponent):
    """Enrich candidate terms using LLM knowledge.

    Attributes
    ----------
    prompt_template: Callable[[str], List[Dict[str, str]]]
        Prompt template used to give instructions and context to the LLM.
    llm_generator: LLMGenerator
        The LLM model used to enrich the candidate terms.
        By default, the zephyr-7b-beta HuggingFace model is used.
    """

    def __init__(
        self,
        prompt_template: Optional[Callable[[str], List[Dict[str, str]]]] = None,
        llm_generator: Optional[LLMGenerator] = None,
    ) -> None:
        """Initialise LLM term enrichment pipeline component instance.

        Parameters
        ----------
        prompt_template: Callable[[str], List[Dict[str, str]]], optional
            Prompt template used to give instructions and context to the LLM.
            By default the term enrichment prompt is used.
        llm_generator: LLMGenerator, optional
            The LLM model used to generate the enrichment.
            By default, the zephyr-7b-beta HuggingFace model is used.
        """

        self.prompt_template = (
            prompt_template
            if prompt_template is not None
            else hf_prompt_term_enrichment
        )
        self.llm_generator = (
            llm_generator if llm_generator is not None else HuggingFaceGenerator()
        )
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

    def _enrich_cterm(self, cterm: CandidateTerm) -> None:
        """Enrich a candidate term based on the LLM knowledge.

        Parameters
        ----------
        cterm: CandidateTerm
            The candidate term to enrich.
        """
        cterm_prompt = self.prompt_template(cterm.label)
        llm_output = self.llm_generator.generate_text(cterm_prompt)
        try:
            enrichment = ast.literal_eval(llm_output)
            if isinstance(enrichment, Dict):
                if cterm.enrichment is None:
                    cterm.enrichment = Enrichment()
                if "synonyms" in enrichment.keys():
                    cterm.enrichment.add_synonyms(set(enrichment["synonyms"]))
                if "hypernyms" in enrichment.keys():
                    cterm.enrichment.add_hypernyms(set(enrichment["hypernyms"]))
                if "hyponyms" in enrichment.keys():
                    cterm.enrichment.add_hyponyms(set(enrichment["hyponyms"]))
                if "antonyms" in enrichment.keys():
                    cterm.enrichment.add_antonyms(set(enrichment["antonyms"]))
            else:
                logger.error(
                    """LLM generator output is not in the expected format. The candidate term %s can not be enriched.""",
                    cterm.label,
                )
                enrichment = None
        except Exception:
            logger.error(
                """LLM generator output is not in the expected format. The candidate term %s can not be enriched.""",
                cterm.label,
            )
            enrichment = None

    def run(self, pipeline: Any) -> None:
        """Method that is responsible for the execution of the component.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline running.
        """

        for cterm in pipeline.candidate_terms:
            self._enrich_cterm(cterm)
