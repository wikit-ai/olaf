from typing import Any, Dict, List, Optional

import numpy as np

from ...pipeline_schema import Pipeline
from ....algorithm.agglomerative_clustering import AgglomerativeClustering
from ....commons.embedding_tools import sbert_embeddings
from ....commons.errors import ParameterError
from ....commons.logging_config import logger
from ....commons.relation_tools import crs_to_relation, cts_to_crs, group_cr_by_concepts
from ....data_container.knowledge_representation_schema import KnowledgeRepresentation
from ...pipeline_schema import Pipeline
from ..pipeline_component_schema import PipelineComponent


class AgglomerativeClusteringRelationExtraction(PipelineComponent):
    """Extract relation based on candidate terms with agglomerative clustering.

    Attributes
    ----------
    candidate_relations: List[CandidateRelations], optional
        List of candidate relations to extract relations from, by default None.
    nb_clusters: int, optional
        Number of clusters to find with the agglomerative clustering algorithm.
        It must be None if distance_threshold is not None, by default 2.
    metric: str, optional
        Metric used to compute the linkage.
        Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”, by default cosine.
    linkage: str, optional
        Distance to use between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion.
        Can be “ward”, “complete”, “average”, “single”, by default "average".
    distance_threshold: float, optional
        The linkage distance threshold at or above which clusters will not be merged.
        If not None, n_clusters must be None, by default None.
    embedding_model: str, optional
        Name of the embedding model to use.
        The list of available models can be found here : https://www.sbert.net/docs/pretrained_models.html,
        by default None.
    concept_max_distance: int, optional
        The maximum distance between the candidate term and the concept sought, by defautl 5.
    scope: str, optional
        Scope used to search concepts. Can be "doc" for the entire document or "sent" for the
        candidate term "sentence", by default "doc".
    """

    def __init__(
        self,
        nb_clusters: Optional[int] = None,
        metric: Optional[str] = None,
        linkage: Optional[str] = "average",
        distance_threshold: Optional[float] = None,
        embedding_model: Optional[str] = None,
        concept_max_distance: Optional[int] = None,
        scope: Optional[str] = "doc",
    ) -> None:
        """Initialise agglomerative clustering-based relation extraction instance.

        Parameters
        ----------
        candidate_relations: List[CandidateRelations], optional
            List of candidate relations to extract relations from, by default None.
        nb_clusters: int, optional
            Number of clusters to find with the agglomerative clustering algorithm.
            It must be None if distance_threshold is not None, by default 2.
        metric: str, optional
            Metric used to compute the linkage.
            Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”, by default cosine.
        linkage: str, optional
            Distance to use between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion.
            Can be “ward”, “complete”, “average”, “single”, by default "average".
        distance_threshold: float, optional
            The linkage distance threshold at or above which clusters will not be merged.
            If not None, n_clusters must be None, by default None.
        embedding_model: str, optional
            Name of the embedding model to use.
            The list of available models can be found here : https://www.sbert.net/docs/pretrained_models.html,
            by default all-mpnet-base-v2.
        concept_max_distance: int, optional
            The maximum distance between the candidate term and the concept sought, by defautl 5.
        scope: str, optional
            Scope used to search concepts. Can be "doc" for the entire document or "sent" for the
            candidate term "sentence", by default "sentence".
        """
        self.candidate_relations = None
        self._nb_clusters = nb_clusters
        self._metric = metric
        self._linkage = linkage
        self._distance_threshold = distance_threshold
        self._embedding_model = embedding_model
        self.concept_max_distance = concept_max_distance
        self.scope = scope
        self._check_parameters()

    def _check_parameters(self) -> None:
        """Check wether required parameters are given and correct. If this is not the case, suitable default ones are set or errors are raised.

        Raises
        ------
        ParameterError
            Exception raised when a required parameter is missing or a wrong value is provided.
        """

        if self._embedding_model:
            if not isinstance(self._embedding_model, str):
                raise ParameterError(
                    component_name=self.__class__,
                    param_name="embedding_model",
                    error_type="Wrong value type",
                )
        else:
            logger.warning(
                "No value given for embedding_model parameter, default will be set to all-mpnet-base-v2."
            )
            self._embedding_model = "all-mpnet-base-v2"

        if not self._nb_clusters:
            if not self._distance_threshold:
                logger.warning(
                    "No value given for nb_clusters option, default will be set to 2."
                )
                self._nb_clusters = 2
        elif self._distance_threshold:
            self._distance_threshold = None
            logger.warning(
                "both nb_clusters and distance_threshold options should be set, distance_threshold is ignored"
            )

        if not self._metric:
            logger.warning(
                "No value given for metric option, default will be set to cosine."
            )
            self._metric = "cosine"

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

    def optimise(self) -> None:
        """A method to optimise the pipeline component by tuning the options."""
        raise NotImplementedError

    def check_resources(self) -> None:
        """Method to check that the component has access to all its required resources."""
        logger.info(
            "Agglomerative clustering-based relation extraction pipeline component has no external resources to check."
        )

    def _compute_metrics(self) -> None:
        """A method to compute component performance metrics. It is used by the optimise method to update the options."""
        raise NotImplementedError

    def get_performance_report(self) -> Dict[str, Any]:
        """A getter for the pipeline component performance report.
            If the component has been optimised, it only returns the best performance.
            Otherwise, it returns the results obtained with the parameters set.

        Returns
        -------
        Dict[str, Any]
            The pipeline component performance report.
        """
        raise NotImplementedError

    def _create_relations(
        self, clustering_labels: List[int], kr: KnowledgeRepresentation
    ) -> None:
        """Create relations based on clusters produced by the agglomerative clustering.

        Parameters
        ----------
        clustering_labels : List[int]
            Labels of the clusters produced.
        kr : KnowledgeRepresentation
            Existing knowledge representation to update.
        """

        labels = set(clustering_labels)

        for label in labels:
            relation_indexes = np.where(clustering_labels == label)[0]
            candidate_relations = [
                self.candidate_relations[i] for i in relation_indexes
            ]
            cr_common_concepts = group_cr_by_concepts(candidate_relations)
            for cr_group in cr_common_concepts:
                relation = crs_to_relation(cr_group)
                kr.relations.add(relation)

    def run(self, pipeline: "Pipeline") -> None:
        """Execution of the agglomerative clustering algorithm on
        candidate terms embedded.
        Relations creation and candidate terms purge.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline running.
        """
        if len(pipeline.candidate_terms) <= 1:
            logger.warning(
                """No enough candidate terms found to run this component :
                agglomerative clustering-based relation extraction ignored.
                """
            )
        else:
            concepts_labels_map = {}
            for concept in pipeline.kr.concepts:
                concepts_labels_map[concept.label] = concept

            candidate_relations = cts_to_crs(
                pipeline.candidate_terms,
                concepts_labels_map,
                pipeline.spacy_model,
                self.concept_max_distance,
                self.scope,
            )

            self.candidate_relations = list(candidate_relations)

            embeddings = sbert_embeddings(
                self._embedding_model,
                [candidate.label for candidate in self.candidate_relations],
            )

            agglo_clustering = AgglomerativeClustering(
                embeddings,
                self._nb_clusters,
                self._metric,
                self._linkage,
                self._distance_threshold,
            )
            agglo_clustering.compute_agglomerative_clustering()

            self._create_relations(agglo_clustering.clustering_labels, pipeline.kr)

            pipeline.candidate_terms = set()
