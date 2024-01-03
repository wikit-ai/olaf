import pytest
from rdflib import OWL, RDF, RDFS, URIRef

from olaf.data_container import KnowledgeRepresentation
from olaf.repository.serialiser import DomainRangeOWLSerialiser


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