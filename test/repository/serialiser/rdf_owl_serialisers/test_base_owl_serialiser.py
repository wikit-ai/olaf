import pytest
from rdflib import OWL, RDF, RDFS, XSD, Literal, URIRef

from olaf.data_container import KnowledgeRepresentation
from olaf.repository.serialiser import BaseOWLSerialiser


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

