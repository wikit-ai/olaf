from typing import List
from olaf.algorithm import AgglomerativeClustering

import pytest

@pytest.fixture(scope="session")
def agglo_clustering_test_data() -> List[List[int]]:
    return [[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]]


@pytest.fixture(scope="session")
def agglo_clustering_expected_output() -> List[int]:
    return [0, 0, 1, 0, 0, 1]

def test_agglomeratice_clustering_wrong_attributes(agglo_clustering_test_data):
    with pytest.raises(AttributeError):
        AgglomerativeClustering(agglo_clustering_test_data, nb_clusters=None, distance_threshold=None)

def test_agglomerative_clustering_computation(agglo_clustering_test_data, agglo_clustering_expected_output):
    agglo_clustering = AgglomerativeClustering(agglo_clustering_test_data)
    agglo_clustering.compute_agglomerative_clustering()
    assert len(agglo_clustering.clustering_labels) == len(agglo_clustering_expected_output)
    assert all([a == b for a, b in zip(agglo_clustering.clustering_labels, agglo_clustering_expected_output)])
