from typing import Any, Dict, Set

from ....commons.logging_config import logger
from ....data_container.enrichment_schema import Enrichment
from ....repository.knowledge_source.knowledge_source_schema import KnowledgeSource
from ..pipeline_component_schema import PipelineComponent


class KnowledgeBasedCTermEnrichment(PipelineComponent):
    """Pipeline component to enrich candidate terms based on an external source of knowledge,
    e.g., a KG.

    Attributes
    ----------
    knowledge_source : KnowledgeSource
        The source of knowledge to use for enrichment.
    parameters : Dict[str, Any] | None, optional
        Parameters are fixed values to be defined when building the knowledge source,
        by default None.
    options : Dict[str, Any] | None, optional
        Options are tunable parameters which will be updated to optimise the
        component performance, by default None.
    use_synonyms: bool, optional
        Wether to use the existing candidate terms synonyms, by default True.
    enrichment_kinds: Set[str], optional
        The kinds of enrichments to perform. Accepted values are: 'synonyms' (default), 'antonyms',
        'hypernyms', and 'hyponyms'. Other values will be ignored.
    """

    def __init__(
        self,
        knowledge_source: KnowledgeSource,
        parameters: Dict[str, Any] | None = None,
        options: Dict[str, Any] | None = None,
    ) -> None:
        """Initialise knowledge based concept extraction instance.

        Parameters
        ----------
        knowledge_source : KnowledgeSource
            The source of knowledge to use for concept matching.
        parameters : Dict[str, Any] | None, optional
            Parameters are fixed values to be defined when building the knowledge source,
            by default None.
        options : Dict[str, Any] | None, optional
            Options are tunable parameters which will be updated to optimise the
            component performance, by default None.
        """
        super().__init__(parameters, options)
        self.knowledge_source = knowledge_source

        self.use_synonyms: bool = self.parameters.get("use_synonyms", True)
        self.enrichment_kinds: Set[str] = self.parameters.get(
            "enrichment_kinds", {"synonyms"}
        )

        self._check_resources()

    def _check_resources(self) -> None:
        """Method to check that the component has access to all its required resources."""

        self.knowledge_source._check_resources()

    def optimise(self) -> None:
        # TODO
        """A method to optimise the pipeline component by tuning the options."""
        raise NotImplementedError

    def _compute_metrics(self) -> None:
        """A method to compute component performance metrics. It is used by the optimise
        method to update the options.
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

    def run(self, pipeline: Any) -> None:
        """Method that is responsible for the execution of the component.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline running.
        """

        unknown_enrichment_kinds = self.enrichment_kinds.difference(
            {"synonyms", "hypernyms", "hyponyms", "antonyms"}
        )

        if len(unknown_enrichment_kinds) > 0:
            logger.warning(
                """Some unknown enrichment kinds have been provided in parameter enrichment_kinds. 
                The following unknown enrichment kinds will be ignored:
                """
            )
            for enrichment_kind in unknown_enrichment_kinds:
                logger.warning("%s", enrichment_kind)

        for c_term in pipeline.candidate_terms:
            if not c_term.enrichment:
                c_term.enrichment = Enrichment()

            terms_to_use = {c_term.label}
            if self.use_synonyms:
                terms_to_use.update(c_term.enrichment.synonyms)

            if "synonyms" in self.enrichment_kinds:
                c_term.enrichment.add_synonyms(
                    self.knowledge_source.fetch_terms_synonyms(terms=terms_to_use)
                )

            if "hypernyms" in self.enrichment_kinds:
                c_term.enrichment.add_hypernyms(
                    self.knowledge_source.fetch_terms_hypernyms(terms=terms_to_use)
                )

            if "hyponyms" in self.enrichment_kinds:
                c_term.enrichment.add_hyponyms(
                    self.knowledge_source.fetch_terms_hyponyms(terms=terms_to_use)
                )

            if "antonyms" in self.enrichment_kinds:
                c_term.enrichment.add_antonyms(
                    self.knowledge_source.fetch_terms_antonyms(terms=terms_to_use)
                )
