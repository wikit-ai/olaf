from typing import Any, Dict, Set, Optional

from ....commons.candidate_term_tools import group_cts_on_synonyms
from ....commons.logging_config import logger
from ....commons.relation_tools import crs_to_relation, cts_to_crs
from ....data_container.candidate_term_schema import CandidateRelation
from ..pipeline_component_schema import PipelineComponent
from ....repository.knowledge_source.knowledge_source_schema import KnowledgeSource


class KnowledgeBasedRelationExtraction(PipelineComponent):
    """Pipeline component to extract relations based on an external source of knowledge,
    e.g., a KG.
    Candidate terms are converted into candidate relations.
    Then, candidate relations are validated as relations if their labels match the external
    source of knowledge.

    Attributes
    ----------
    knowledge_source : KnowledgeSource
        The source of knowledge to use for relation matching.
    group_ct_on_synonyms: bool, optional
        Whether or not to group the candidate terms on synonyms before proceeding to the
        relation matching with the external source of knowledge, by default True.
    concept_max_distance: int, optional
        The maximum distance between the candidate term and the concept sought,
        by default 5.
    scope: str
        Scope used to search concepts. Can be "doc" for the entire document or "sent" for the
        candidate term "sentence", by default "doc".
    """

    def __init__(
        self,
        knowledge_source: KnowledgeSource,
        group_ct_on_synonyms: Optional[bool] = True,
        concept_max_distance: Optional[int] = None,
        scope: Optional[str] = "doc",
    ) -> None:
        """Initialise knowledge based relation extraction instance.

        Parameters
        ----------
        knowledge_source : KnowledgeSource
            The source of knowledge to use for relation matching.
        group_ct_on_synonyms: bool, optional
            Whether or not to group the candidate terms on synonyms before proceeding to the
            relation matching with the external source of knowledge, by default True.
        concept_max_distance: int, optional
            The maximum distance between the candidate term and the concept sought,
            by default 5.
        scope: str, optional
            Scope used to search concepts. Can be "doc" for the entire document or "sent" for the
            candidate term "sentence", by default "doc".
        """
        self.knowledge_source = knowledge_source
        self.group_ct_on_synonyms = group_ct_on_synonyms
        self.concept_max_distance = concept_max_distance
        self.scope = scope
        self._check_parameters()

        self._check_resources()

    def _check_parameters(self) -> None:
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

    def c_terms_texts_to_match(self, cr_group: Set[CandidateRelation]) -> Set[str]:
        """Extract from a set of candidate relations the strings to use for concept matching.

        Parameters
        ----------
        cr_group : Set[CandidateRelation]
            The set of candidate relations.

        Returns
        -------
        Set[str]
            The set of strings to use for relation matching.
        """
        c_term_texts = set()

        for cr in cr_group:
            c_term_texts.add(cr.label)
            if cr.enrichment:
                c_term_texts.update(cr.enrichment.synonyms)

        return c_term_texts

    def run(self, pipeline: Any) -> None:
        """Method that is responsible for the execution of the component.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline running.
        """
        concepts_labels_map = dict()
        for concept in pipeline.kr.concepts:
            concepts_labels_map[concept.label] = concept

        candidate_relations = cts_to_crs(
            pipeline.candidate_terms,
            concepts_labels_map,
            pipeline.spacy_model,
            self.concept_max_distance,
            self.scope,
        )

        if self.group_ct_on_synonyms:
            crs_groups = group_cts_on_synonyms(candidate_relations)
        else:
            crs_groups = [{cr} for cr in candidate_relations]

        for cr_group in crs_groups:
            cr_group_texts = self.c_terms_texts_to_match(cr_group)

            relation_uids = self.knowledge_source.match_external_concepts(
                matching_terms=cr_group_texts
            )

            if len(relation_uids) > 0:
                c_term_relation = crs_to_relation(cr_group)
                c_term_relation.external_uids.update(relation_uids)
                pipeline.kr.relations.add(c_term_relation)

        pipeline.candidate_terms = set()
