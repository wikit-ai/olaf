import pytest

from olaf.data_container import Concept, Metarelation, Relation


@pytest.fixture(scope="module")
def red_wine():
    return Concept(label="Red Wine")


@pytest.fixture(scope="module")
def white_wine():
    return Concept(label="White Wine")


@pytest.fixture(scope="module")
def grape():
    return Concept(label="Grape")


@pytest.fixture(scope="module")
def vineyard():
    return Concept(label="Vineyard")


@pytest.fixture(scope="module")
def wine_glass():
    return Concept(label="Wine Glass")


@pytest.fixture(scope="module")
def cork():
    return Concept(label="Cork")


@pytest.fixture(scope="module")
def sommelier():
    return Concept(label="Sommelier")


@pytest.fixture(scope="module")
def drink():
    return Concept(label="Drink")


@pytest.fixture(scope="module")
def alcoholic_drink():
    return Concept(label="Alcoholic Drink")


@pytest.fixture(scope="module")
def red_wine_an_alcoholic_drink(alcoholic_drink, red_wine):
    return Metarelation(
        label="is_generalised_by",
        source_concept=red_wine,
        destination_concept=alcoholic_drink,
    )


@pytest.fixture(scope="module")
def white_wine_an_alcoholic_drink(alcoholic_drink, white_wine):
    return Metarelation(
        label="is_generalised_by",
        source_concept=white_wine,
        destination_concept=alcoholic_drink,
    )


@pytest.fixture(scope="module")
def alcoholic_drink_a_drink(alcoholic_drink, drink):
    return Metarelation(
        label="is_generalised_by",
        source_concept=alcoholic_drink,
        destination_concept=drink,
    )


@pytest.fixture(scope="module")
def made_from(red_wine, grape):
    return Relation(
        label="Made From", source_concept=red_wine, destination_concept=grape
    )


@pytest.fixture(scope="module")
def produced_in(grape, vineyard):
    return Relation(
        label="Produced In", source_concept=grape, destination_concept=vineyard
    )


@pytest.fixture(scope="module")
def paired_with(red_wine, white_wine):
    return Relation(
        label="Paired With", source_concept=red_wine, destination_concept=white_wine
    )


@pytest.fixture(scope="module")
def aged_in(red_wine, vineyard):
    return Relation(
        label="Aged In", source_concept=red_wine, destination_concept=vineyard
    )


@pytest.fixture(scope="module")
def has_quality(red_wine, grape):
    return Metarelation(
        label="Has Quality", source_concept=red_wine, destination_concept=grape
    )


@pytest.fixture(scope="module")
def wine_concepts(
    red_wine,
    white_wine,
    grape,
    vineyard,
    wine_glass,
    cork,
    sommelier,
    drink,
    alcoholic_drink,
):
    return {
        red_wine,
        white_wine,
        grape,
        vineyard,
        wine_glass,
        cork,
        sommelier,
        drink,
        alcoholic_drink,
    }


@pytest.fixture(scope="module")
def wine_relations(made_from, produced_in, paired_with, aged_in):
    return {made_from, produced_in, paired_with, aged_in}


@pytest.fixture(scope="module")
def wine_metarelations(
    red_wine_an_alcoholic_drink, white_wine_an_alcoholic_drink, alcoholic_drink_a_drink
):
    return {
        red_wine_an_alcoholic_drink,
        white_wine_an_alcoholic_drink,
        alcoholic_drink_a_drink,
    }