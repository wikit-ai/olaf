import pytest

from olaf.data_container.enrichment_schema import Enrichment


@pytest.fixture(scope="session")
def bike_enrichment() -> Enrichment:
    enrichment = Enrichment(synonyms={"bicycle", "cycle"}, antonyms={"not_bicycle"})
    return enrichment


@pytest.fixture(scope="session")
def wine_enrichment() -> Enrichment:
    enrichment = Enrichment(synonyms={"drink", "beer"}, antonyms={"water"})
    return enrichment


def test_enrichment_merge_with_enrichment(bike_enrichment, wine_enrichment) -> None:
    bike_enrichment.merge_with_enrichment(wine_enrichment)

    conditions = [syn in bike_enrichment.synonyms for syn in wine_enrichment.synonyms]

    conditions.extend(
        [syn in bike_enrichment.antonyms for syn in wine_enrichment.antonyms]
    )

    assert all(conditions)
