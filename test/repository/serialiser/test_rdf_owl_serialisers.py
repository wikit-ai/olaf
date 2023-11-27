import pytest
from rdflib import OWL, RDF, RDFS, SKOS, XSD, Graph, Literal, URIRef

from olaf.data_container.concept_schema import Concept
from olaf.data_container.knowledge_representation_schema import KnowledgeRepresentation
from olaf.data_container.metarelation_schema import Metarelation
from olaf.data_container.relation_schema import Relation
from olaf.repository.serialiser.rdf_owl_serialisers import (
    BaseOWLSerialiser,
    BaseRDFserialiser,
    DomainRangeOWLSerialiser,
)


@pytest.fixture(scope="session")
def red_wine():
    return Concept(label="Red Wine")


@pytest.fixture(scope="session")
def white_wine():
    return Concept(label="White Wine")


@pytest.fixture(scope="session")
def grape():
    return Concept(label="Grape")


@pytest.fixture(scope="session")
def vineyard():
    return Concept(label="Vineyard")


@pytest.fixture(scope="session")
def wine_glass():
    return Concept(label="Wine Glass")


@pytest.fixture(scope="session")
def cork():
    return Concept(label="Cork")


@pytest.fixture(scope="session")
def sommelier():
    return Concept(label="Sommelier")


@pytest.fixture(scope="session")
def drink():
    return Concept(label="Drink")


@pytest.fixture(scope="session")
def alcoholic_drink():
    return Concept(label="Alcoholic Drink")


@pytest.fixture(scope="session")
def red_wine_an_alcoholic_drink(alcoholic_drink, red_wine):
    return Metarelation(
        label="is_generalised_by",
        source_concept=red_wine,
        destination_concept=alcoholic_drink,
    )


@pytest.fixture(scope="session")
def white_wine_an_alcoholic_drink(alcoholic_drink, white_wine):
    return Metarelation(
        label="is_generalised_by",
        source_concept=white_wine,
        destination_concept=alcoholic_drink,
    )


@pytest.fixture(scope="session")
def alcoholic_drink_a_drink(alcoholic_drink, drink):
    return Metarelation(
        label="is_generalised_by",
        source_concept=alcoholic_drink,
        destination_concept=drink,
    )


@pytest.fixture(scope="session")
def made_from(red_wine, grape):
    return Relation(
        label="Made From", source_concept=red_wine, destination_concept=grape
    )


@pytest.fixture(scope="session")
def produced_in(grape, vineyard):
    return Relation(
        label="Produced In", source_concept=grape, destination_concept=vineyard
    )


@pytest.fixture(scope="session")
def paired_with(red_wine, white_wine):
    return Relation(
        label="Paired With", source_concept=red_wine, destination_concept=white_wine
    )


@pytest.fixture(scope="session")
def aged_in(red_wine, vineyard):
    return Relation(
        label="Aged In", source_concept=red_wine, destination_concept=vineyard
    )


@pytest.fixture(scope="session")
def has_quality(red_wine, grape):
    return Metarelation(
        label="Has Quality", source_concept=red_wine, destination_concept=grape
    )


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def wine_relations(made_from, produced_in, paired_with, aged_in):
    return {made_from, produced_in, paired_with, aged_in}


@pytest.fixture(scope="session")
def wine_metarelations(
    red_wine_an_alcoholic_drink, white_wine_an_alcoholic_drink, alcoholic_drink_a_drink
):
    return {
        red_wine_an_alcoholic_drink,
        white_wine_an_alcoholic_drink,
        alcoholic_drink_a_drink,
    }


class TestBaseRDFSerialiser:
    @pytest.fixture(scope="class")
    def wine_knowledge_representation(
        self, wine_concepts, wine_relations, wine_metarelations
    ):
        return KnowledgeRepresentation(
            concepts=wine_concepts,
            relations=wine_relations,
            metarelations=wine_metarelations,
        )

    @pytest.fixture(scope="class")
    def serialiser(self):
        # Replace 'http://wine_example.org/' with your actual base URI
        return BaseRDFserialiser("http://wine_example.org/")

    def test_build_concept_uri(self, serialiser, red_wine):
        concept_uri = serialiser.build_concept_uri(red_wine)
        assert isinstance(concept_uri, URIRef)
        assert str(concept_uri) == "http://wine_example.org/red_wine"

    def test_build_relation_uri(self, serialiser, made_from):
        relation_uri = serialiser.build_relation_uri(made_from)
        assert isinstance(relation_uri, URIRef)
        assert str(relation_uri) == "http://wine_example.org/made_from"

    def test_build_metarelation_uri(self, serialiser, has_quality):
        metarelation_uri = serialiser.build_metarelation_uri(has_quality)
        assert isinstance(metarelation_uri, URIRef)
        assert str(metarelation_uri) == "http://wine_example.org/has_quality"

    def test_build_graph(self, serialiser, wine_knowledge_representation):
        serialiser.build_graph(wine_knowledge_representation)
        graph = serialiser.graph

        # Add assertions to check if the graph contains the expected triples.
        # Example assertions:
        assert (
            URIRef("http://wine_example.org/red_wine"),
            URIRef("http://wine_example.org/aged_in"),
            URIRef("http://wine_example.org/vineyard"),
        ) in graph
        assert (
            URIRef("http://wine_example.org/red_wine"),
            URIRef("http://wine_example.org/made_from"),
            URIRef("http://wine_example.org/grape"),
        ) in graph
        assert (
            URIRef("http://wine_example.org/red_wine"),
            RDF.type,
            OWL.Class,
        ) not in graph
        # Add more assertions for your specific use case.

    def test_export_graph(
        self, serialiser, wine_knowledge_representation, tmp_path_factory
    ):
        output_path = tmp_path_factory.mktemp("test_serialised_kr") / "output.ttl"
        serialiser.build_graph(wine_knowledge_representation)
        serialiser.export_graph(output_path, rdf_format="turtle")

        # Check if the exported graph file exists and is not empty
        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestBaseOWLSerialiser:
    @pytest.fixture(scope="class")
    def wine_knowledge_representation(
        self, wine_concepts, wine_relations, wine_metarelations
    ):
        return KnowledgeRepresentation(
            concepts=wine_concepts,
            relations=wine_relations,
            metarelations=wine_metarelations,
        )

    @pytest.fixture(scope="class")
    def serialiser(self):
        # Replace 'http://wine_example.org/' with your actual base URI
        return BaseOWLSerialiser("http://wine_example.org/")

    def test_build_concept_uri(self, serialiser, red_wine):
        concept_uri = serialiser.build_concept_uri(red_wine)
        assert isinstance(concept_uri, URIRef)
        assert str(concept_uri) == "http://wine_example.org/RedWine"

    def test_build_relation_uri(self, serialiser, made_from):
        relation_uri = serialiser.build_relation_uri(made_from)
        assert isinstance(relation_uri, URIRef)
        assert str(relation_uri) == "http://wine_example.org/madeFrom"

    def test_build_metarelation_uri(self, serialiser, has_quality):
        metarelation_uri = serialiser.build_metarelation_uri(has_quality)
        assert isinstance(metarelation_uri, URIRef)
        assert str(metarelation_uri) == "http://wine_example.org/hasQuality"

    def test_build_graph(self, serialiser, wine_knowledge_representation):
        serialiser.build_graph(wine_knowledge_representation)
        graph = serialiser.graph

        # Add assertions to check if the graph contains the expected triples.
        # Example assertions:
        assert (URIRef("http://wine_example.org/RedWine"), RDF.type, OWL.Class) in graph
        assert (
            URIRef("http://wine_example.org/RedWine"),
            RDFS.label,
            Literal("Red Wine", datatype=XSD.string),
        ) in graph
        assert (
            URIRef("http://wine_example.org/producedIn"),
            RDFS.label,
            Literal("Produced In", datatype=XSD.string),
        ) in graph
        assert (
            URIRef("http://wine_example.org/producedIn"),
            RDF.type,
            OWL.ObjectProperty,
        ) in graph
        assert (
            URIRef("http://wine_example.org/RedWine"),
            RDFS.subClassOf,
            URIRef("http://wine_example.org/AlcoholicDrink"),
        ) in graph
        # Add more assertions for your specific use case.

    def test_export_graph(
        self, serialiser, wine_knowledge_representation, tmp_path_factory
    ):
        output_path = tmp_path_factory.mktemp("test_serialised_kr") / "output.ttl"
        serialiser.build_graph(wine_knowledge_representation)
        serialiser.export_graph(output_path, rdf_format="turtle")

        # Check if the exported graph file exists and is not empty
        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestDomainRangeOWLSerialiser:
    @pytest.fixture(scope="class")
    def wine_knowledge_representation(
        self, wine_concepts, wine_relations, wine_metarelations
    ):
        return KnowledgeRepresentation(
            concepts=wine_concepts,
            relations=wine_relations,
            metarelations=wine_metarelations,
        )

    @pytest.fixture(scope="class")
    def serialiser(self):
        # Replace 'http://wine_example.org/' with your actual base URI
        return DomainRangeOWLSerialiser("http://wine_example.org/")

    def test_build_graph(self, serialiser, wine_knowledge_representation):
        serialiser.build_graph(wine_knowledge_representation)
        graph = serialiser.graph

        # Add assertions to check if the graph contains the expected triples.
        # Example assertions:
        assert (
            URIRef("http://wine_example.org/producedIn"),
            RDF.type,
            OWL.ObjectProperty,
        ) in graph
        assert (
            URIRef("http://wine_example.org/producedIn"),
            RDFS.domain,
            URIRef("http://wine_example.org/Grape"),
        ) in graph
        assert (
            URIRef("http://wine_example.org/producedIn"),
            RDFS.range,
            URIRef("http://wine_example.org/Vineyard"),
        ) in graph
        # Add more assertions for your specific use case.

    def test_export_graph(
        self, serialiser, wine_knowledge_representation, tmp_path_factory
    ):
        output_path = tmp_path_factory.mktemp("test_serialised_kr") / "output.ttl"
        serialiser.build_graph(wine_knowledge_representation)
        serialiser.export_graph(output_path, rdf_format="turtle")

        # Check if the exported graph file exists and is not empty
        assert output_path.exists()
        assert output_path.stat().st_size > 0
