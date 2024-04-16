from typing import Any, Dict, List, Optional

import numpy as np

from ....algorithm.agglomerative_clustering import AgglomerativeClustering
from ....commons.candidate_term_tools import cts_to_concept
from ....commons.embedding_tools import sbert_embeddings
from ....commons.errors import ParameterError
from ....commons.logging_config import logger
from ....data_container.knowledge_representation_schema import KnowledgeRepresentation
from ..pipeline_component_schema import PipelineComponent


class AgglomerativeClusteringConceptExtraction(PipelineComponent):
    """Extract concept based candidate terms with agglomerative clustering.

    Attributes
    ----------
    candidate_terms: List[CandidateTerm]
        List of candidate terms to extract concepts from.
    nb_clusters: int, optional
        Number of clusters to find with the agglomerative clustering algorithm.
        It must be None if distance_threshold is not None, by default 2.
    metric: str, optional
        Metric used to compute the linkage.
        Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”, by default "cosine".
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
    """

    def __init__(
        self,
        nb_clusters: Optional[int] = None,
        metric: Optional[str] = None,
        linkage: Optional[str] = "average",
        distance_threshold: Optional[float] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        """Initialise agglomerative clustering-based concept extraction instance.

        Parameters
        ----------
        nb_clusters: int, optional
            Number of clusters to find with the agglomerative clustering algorithm.
            It must be None if distance_threshold is not None, by default 2.
        metric: str, optional
            Metric used to compute the linkage.
            Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”, by default "cosine".
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
        """
        self.candidate_terms = None
        self._nb_clusters = nb_clusters
        self._metric = metric
        self._linkage = linkage
        self._distance_threshold = distance_threshold
        self._embedding_model = embedding_model
        self._check_parameters()

    def _check_parameters(self) -> None:
        """Check wether required parameters are given and correct. If this is not the case,
        suitable default ones are set or errors are raised.

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
                    "No value given for nb_clusters or distance_threshold, default to nb_clusters=2"
                )
                self._nb_clusters = 2
        elif self._distance_threshold:
            self._distance_threshold = None
            logger.warning(
                "Both nb_clusters and distance_threshold options cannot be set together, distance_threshold is ignored"
            )

        if not self._metric:
            logger.warning(
                "No value given for metric option, default will be set to cosine."
            )
            self._metric = "cosine"

    def optimise(self) -> None:
        # TODO
        """A method to optimise the pipeline component by tuning the options."""
        raise NotImplementedError

    def _check_resources(self) -> None:
        """Method to check that the component has access to all its required resources."""
        logger.info(
            "Agglomerative clustering-based concept extraction pipeline component has no external resources to check."
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

    def _create_concepts(
        self, clustering_labels: List[int], kr: KnowledgeRepresentation
    ) -> None:
        """Create concepts based on clusters produced by the agglomerative clustering.

        Parameters
        ----------
        clustering_labels : List[int]
            Labels of the clusters produced.
        kr : KnowledgeRepresentation
            Existing knowledge representation to update.
        """

        labels = set(clustering_labels)

        for label in labels:
            concept_indexes = np.where(clustering_labels == label)[0]
            concept_candidates = [self.candidate_terms[i] for i in concept_indexes]
            concept = cts_to_concept(concept_candidates)
            kr.concepts.add(concept)

    def run(self, pipeline: Any) -> None:
        """Execution of the agglomerative clustering algorithm on candidate terms embedded. Concepts creation and candidate terms purge.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline running.
        """

        self.candidate_terms = list(pipeline.candidate_terms)

        embeddings = sbert_embeddings(
            self._embedding_model,
            [candidate.label for candidate in self.candidate_terms],
        )

        agglo_clustering = AgglomerativeClustering(
            embeddings,
            self._nb_clusters,
            self._metric,
            self._linkage,
            self._distance_threshold,
        )
        agglo_clustering.compute_agglomerative_clustering()

        self._create_concepts(agglo_clustering.clustering_labels, pipeline.kr)

        pipeline.candidate_terms = set()
