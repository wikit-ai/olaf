from typing import Any, List, Optional

from sklearn import cluster


class AgglomerativeClustering:
    """Implementation of agglomerative clustering algorithm."""

    def __init__(
        self,
        training_instances: List[Any],
        nb_clusters: Optional[int] = 2,
        metric: Optional[str] = "cosine",
        linkage: Optional[str] = "average",
        distance_threshold: Optional[float] = None,
    ) -> None:
        """Initialise agglomerative clustering instance.

        Parameters
        ----------
        training_instances: List[Any]
            List of value to cluster.
        nb_clusters: Optional[int] = 2
            Number of cluster to find.
        metric: Optional[str] = "cosine",
            Metric used to compute similarity between values.
        linkage: Optional[str] = "average",
            Type of linkage used for the algorithm.
        distance_threshold: Optional[float] = None
            Distance threshold to stop the clustering.

        Raises
        ------
        AttributeError
            Exception raised when there is incompatible attributes combination.
        """
        self.training_instances = training_instances
        self.nb_clusters = nb_clusters
        self.metric = metric
        self.linkage = linkage
        self.distance_threshold = distance_threshold

        if not (self.nb_clusters) and not (distance_threshold):
            raise AttributeError(
                "Attributes nb_clusters and distance_threshold cannot be both set to None."
            )

        self.clustering = cluster.AgglomerativeClustering(
            n_clusters=self.nb_clusters,
            metric=self.metric,
            linkage=self.linkage,
            distance_threshold=self.distance_threshold,
        )

    def compute_agglomerative_clustering(self) -> None:
        """Method used to compute the agglomerative clustering on the training instances."""
        self.clustering.fit(self.training_instances)

    @property
    def clustering_labels(self) -> List[int]:
        """Getter to return the labels found for each training instance.

        Returns
        -------
        List[int]
            List of cluster labels found for each training instance.
        """
        return self.clustering.labels_
