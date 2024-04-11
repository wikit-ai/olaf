from typing import Any, Dict, Set

from ....commons.candidate_term_tools import cts_to_concept, group_cts_on_synonyms
from ....data_container.candidate_term_schema import CandidateTerm
from ..pipeline_component_schema import PipelineComponent
from ....repository.knowledge_source.knowledge_source_schema import KnowledgeSource


class KnowledgeBasedConceptExtraction(PipelineComponent):
    """Pipeline component to extract concepts based on an external source of knowledge,
    e.g., a KG.

    Attributes
    ----------
    knowledge_source : KnowledgeSource
        The source of knowledge to use for concept matching.
    parameters : Dict[str, Any], optional
        Parameters are fixed values to be defined when building the knowledge source,
        by default None.
    options : Dict[str, Any], optional
        Options are tunable parameters which will be updated to optimise the
        component performance, by default None.
    group_ct_on_synonyms: bool, optional
        Wether or not to group the candidate terms on synonyms before proceeding to the
        concept matching with the external source of knowledge, by default True.
    """

    def __init__(
        self,
        knowledge_source: KnowledgeSource,
        parameters: Dict[str, Any] = None,
        options: Dict[str, Any] = None,
    ) -> None:
        """Initialise knowledge based concept extraction instance.

        Parameters
        ----------
        knowledge_source : KnowledgeSource
            The source of knowledge to use for concept matching.
        parameters : Dict[str, Any], optional
            Parameters are fixed values to be defined when building the knowledge source,
            by default None.
        options : Dict[str, Any], optional
            Options are tunable parameters which will be updated to optimise the
            component performance, by default None.
        """
        super().__init__(parameters, options)
        self.knowledge_source = knowledge_source
        self.group_ct_on_synonyms = (
            parameters.get("group_ct_on_synonyms", True)
            if parameters is not None
            else True
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

    def c_terms_texts_to_match(self, ct_group: Set[CandidateTerm]) -> Set[str]:
        """Extract from a set of candidate terms the strings to use for concept matching.

        Parameters
        ----------
        ct_group : Set[CandidateTerm]
            The set of candidate terms.

        Returns
        -------
        Set[str]
            The set of strings to use for concept matching.
        """
        c_term_texts = set()

        for ct in ct_group:
            c_term_texts.add(ct.label)
            if ct.enrichment:
                c_term_texts.update(ct.enrichment.synonyms)

        return c_term_texts

    def run(self, pipeline: Any) -> None:
        """Method that is responsible for the execution of the component.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline running.
        """

        if self.group_ct_on_synonyms:
            cts_groups = group_cts_on_synonyms(pipeline.candidate_terms)
        else:
            cts_groups = [{ct} for ct in pipeline.candidate_terms]

        for ct_group in cts_groups:
            ct_group_texts = self.c_terms_texts_to_match(ct_group)

            concept_uids = self.knowledge_source.match_external_concepts(
                matching_terms=ct_group_texts
            )

            if len(concept_uids) > 0:
                c_term_concept = cts_to_concept(ct_group)
                c_term_concept.external_uids.update(concept_uids)
                pipeline.kr.concepts.add(c_term_concept)

        pipeline.candidate_terms = set()
