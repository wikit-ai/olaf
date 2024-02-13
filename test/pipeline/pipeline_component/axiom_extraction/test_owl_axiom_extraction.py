from typing import Callable, Set, Tuple

import pytest
from rdflib import Graph, URIRef

from olaf.commons.kr_to_rdf_tools import (
    concept_lrs_to_owl_individuals, kr_concepts_to_disjoint_classes,
    kr_concepts_to_owl_classes, kr_metarelations_to_owl,
    kr_relations_to_domain_range_obj_props, kr_relations_to_owl_obj_props)
from olaf.data_container import KnowledgeRepresentation
from olaf.pipeline.pipeline_component.axiom_extraction.owl_axiom_extraction import \
    OWLAxiomExtraction
from olaf.pipeline.pipeline_schema import Pipeline

# WARNING : These tests require a `.env` file with environment variables JAVA_EXE and ROBOT_JAR set.
# This should avoided using pytest monkeypatch but I did not manage to make it work.


@pytest.fixture(scope="module")
def american_pizza_pipline(american_pizza_ex_kr) -> Pipeline:
    
    pipeline = Pipeline(
            spacy_model=None,
            seed_kr=american_pizza_ex_kr,
            corpus=[]
        )
    return pipeline

@pytest.fixture(scope="module")
def american_pizza_axiom_generators_disjoint_classes() -> Set[Callable[[KnowledgeRepresentation, URIRef], Graph]]:

    axiom_generators = {
        kr_concepts_to_owl_classes,
        kr_relations_to_owl_obj_props,
        kr_relations_to_domain_range_obj_props,
        kr_metarelations_to_owl,
        kr_concepts_to_disjoint_classes
    }

    return axiom_generators

@pytest.fixture(scope="module")
def american_pizza_axiom_generators() -> Set[Callable[[KnowledgeRepresentation, URIRef], Graph]]:

    axiom_generators = {
        kr_concepts_to_owl_classes,
        kr_relations_to_owl_obj_props,
        kr_relations_to_domain_range_obj_props,
        kr_metarelations_to_owl,
        concept_lrs_to_owl_individuals
    }

    return axiom_generators

@pytest.fixture(scope="module")
def base_axiom_extract_comp(american_pizza_axiom_generators) -> OWLAxiomExtraction:
    
    axiom_extract_comp = OWLAxiomExtraction(
        owl_axiom_generators=american_pizza_axiom_generators
    )
    return axiom_extract_comp

@pytest.fixture(scope="module")
def class_disjoint_axiom_extract_comp(american_pizza_axiom_generators_disjoint_classes) -> OWLAxiomExtraction:

    axiom_extract_comp = OWLAxiomExtraction(
        owl_axiom_generators=american_pizza_axiom_generators_disjoint_classes,
    )
    return axiom_extract_comp

@pytest.fixture(scope="module")
def american_pizza_unsatisfiable_concept_uris() -> Set[str]:
    concept_uris = {
        "http://www.ms2.org/o/example#Mozzarella",
        "http://www.ms2.org/o/example#American",
        "http://www.ms2.org/o/example#Cheese",
        "http://www.ms2.org/o/example#America",
        "http://www.ms2.org/o/example#PepperoniSausage",
    }
    return concept_uris

@pytest.fixture(scope="module")
def american_pizza_owl_classes_frags() -> Set[Tuple[str]]:
    expected_class_fragments = {
            ("NonVegetarianPizza",), ("American",), ("Country",), ("Cheese",),
            ("Pizza",), ("CheesyPizza",), ("America",), ("Mozzarella",), ("Tomato",),
            ("Topping",), ("PepperoniSausage",)
        }
    return expected_class_fragments

@pytest.fixture(scope="module")
def american_pizza_owl_named_individuals_frags() -> Set[Tuple[str]]:
    expected_individuals_fragments = {
            ("_nonVegetarianPizza",), ("_american",), ("_country",), ("_cheese",),
            ("_pizza",), ("_cheesyPizza",), ("_america",), ("_mozzarella",), ("_tomato",),
            ("_topping",), ("_pepperoniSausage",)
        }
    return expected_individuals_fragments

@pytest.fixture(scope="module")
def american_pizza_owl_named_individuals_labels() -> Set[Tuple[str]]:
    expected_individuals_labels = {
            "non vegetarian pizza", "american", "country", "cheese",
            "pizza", "cheesy pizza", "america", "mozzarella", "tomato",
            "topping", "pepperoni sausage"
        }
    return expected_individuals_labels

@pytest.fixture(scope="module")
def american_pizza_owl_obj_props_frags() -> Set[Tuple[str]]:
    expected_rel_fragments = {("hasCountryOfOrigin",), ("hasIngredient",), ('hasKind',), ("hasBase",)}
    return expected_rel_fragments

@pytest.fixture(scope="module")
def american_pizza_domain_range_frags() -> Set[Tuple[str]]:
    expected_domain_range_fragments = {
            ("hasIngredient", "American", "Mozzarella"),
            ("hasIngredient", "American", "Tomato"),
            ("hasIngredient", "American", "PepperoniSausage"),
            ("hasCountryOfOrigin", "American", "America")
        }
    return expected_domain_range_fragments

@pytest.fixture(scope="module")
def american_pizza_subclasses_frags() -> Set[Tuple[str]]:
    expected_metaprop_fragments = {
            ("American", "CheesyPizza"),
            ("Mozzarella", "Cheese"),
            ("Mozzarella", "Topping"),
            ("PepperoniSausage", "Topping"),
            ("Cheese", "Topping"),
            ("America", "Country"),
            ("American", "Pizza")
        }
    return expected_metaprop_fragments

@pytest.fixture(scope="module")
def american_pizza_disjoint_classes_frags() -> Set[Tuple[str]]:
    disjoint_classes_fragments = {
            ("NonVegetarianPizza",), ("American",), ("Country",), ("Cheese",),
            ("Pizza",), ("CheesyPizza",), ("America",), ("Mozzarella",), ("Tomato",),
            ("Topping",), ("PepperoniSausage",)
        }
    return disjoint_classes_fragments

class TestOWLAxiomExtractionInit:

    def test_base_init(self, base_axiom_extract_comp) -> None:

        assert base_axiom_extract_comp.base_uri == URIRef("http://www.ms2.org/o/example#")
        assert base_axiom_extract_comp.reasoner == "ELK"

    def test_full_custom_init(self, american_pizza_axiom_generators) -> None:


        custom_base_uri = URIRef("http://www.ms2.org/o/example#")
        custom_reasoner = "hermit"

        axiom_extract_comp = OWLAxiomExtraction(
            owl_axiom_generators=american_pizza_axiom_generators,
            base_uri=custom_base_uri,
            reasoner=custom_reasoner,
        )

        assert axiom_extract_comp.base_uri == custom_base_uri
        assert axiom_extract_comp.reasoner == custom_reasoner

class Test_build_graph_without_owl_instances:

    @pytest.fixture(scope="class")
    def base_axiom_extract_comp_graph(self, base_axiom_extract_comp, american_pizza_ex_kr) -> Graph:
        graph = base_axiom_extract_comp.build_graph_without_owl_instances(kr=american_pizza_ex_kr)
        return graph
    
    @pytest.fixture(scope="class")
    def class_disjoint_axiom_extract_comp_graph(self, class_disjoint_axiom_extract_comp, american_pizza_ex_kr) -> Graph:
        graph = class_disjoint_axiom_extract_comp.build_graph_without_owl_instances(kr=american_pizza_ex_kr)
        return graph
    
    def test_owl_classes(self,
                         base_axiom_extract_comp_graph, 
                         class_disjoint_axiom_extract_comp_graph, 
                         get_sparql_q_res_fragments,
                         owl_classes_sparql_q,
                         ms2_ns,
                         american_pizza_ex_kr,
                         american_pizza_owl_classes_frags
                    ) -> None:
        base_axiom_extract_comp_class_fragments = get_sparql_q_res_fragments(
            sparql_q=owl_classes_sparql_q,
            graph=base_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )

        class_disjoint_axiom_extract_comp_class_fragments = get_sparql_q_res_fragments(
            sparql_q=owl_classes_sparql_q,
            graph=class_disjoint_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )
        
        nb_kr_concepts = len(american_pizza_ex_kr.concepts)

        assert len(base_axiom_extract_comp_class_fragments) == nb_kr_concepts
        assert len(class_disjoint_axiom_extract_comp_class_fragments) == nb_kr_concepts
        assert base_axiom_extract_comp_class_fragments == american_pizza_owl_classes_frags
        assert class_disjoint_axiom_extract_comp_class_fragments == american_pizza_owl_classes_frags

    def test_owl_obj_props(self,
                         base_axiom_extract_comp_graph, 
                         class_disjoint_axiom_extract_comp_graph, 
                         get_sparql_q_res_fragments,
                         owl_obj_props_sparql_q,
                         ms2_ns,
                         american_pizza_owl_obj_props_frags
                    ) -> None:
        
        base_axiom_extract_comp_prop_fragments = get_sparql_q_res_fragments(
                                    sparql_q=owl_obj_props_sparql_q,
                                    graph=base_axiom_extract_comp_graph,
                                    ns={"ms2": ms2_ns}
                                )
        
        class_disjoint_axiom_extract_comp_prop_fragments = get_sparql_q_res_fragments(
                                    sparql_q=owl_obj_props_sparql_q,
                                    graph=class_disjoint_axiom_extract_comp_graph,
                                    ns={"ms2": ms2_ns}
                                )
    
        assert base_axiom_extract_comp_prop_fragments == american_pizza_owl_obj_props_frags
        assert class_disjoint_axiom_extract_comp_prop_fragments == american_pizza_owl_obj_props_frags

    def test_domain_range(self,
                         base_axiom_extract_comp_graph, 
                         class_disjoint_axiom_extract_comp_graph, 
                         get_sparql_q_res_fragments,
                         domain_range_sparql_q,
                         ms2_ns,
                         american_pizza_domain_range_frags
                    ) -> None:

        base_axiom_extract_comp_domain_range_fragments = get_sparql_q_res_fragments(
            sparql_q=domain_range_sparql_q,
            graph=base_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )
        class_disjoint_axiom_extract_domain_range_fragments = get_sparql_q_res_fragments(
            sparql_q=domain_range_sparql_q,
            graph=class_disjoint_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )

        assert base_axiom_extract_comp_domain_range_fragments == american_pizza_domain_range_frags
        assert class_disjoint_axiom_extract_domain_range_fragments == american_pizza_domain_range_frags

    def test_owl_subclasses(self,
                         base_axiom_extract_comp_graph, 
                         class_disjoint_axiom_extract_comp_graph, 
                         get_sparql_q_res_fragments,
                         subclasses_sparql_q,
                         ms2_ns,
                         american_pizza_subclasses_frags
                    ) -> None:
        
        base_axiom_extract_comp_metaprop_fragments = get_sparql_q_res_fragments(
            sparql_q=subclasses_sparql_q,
            graph=base_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )

        class_disjoint_axiom_extract_comp_metaprop_fragments = get_sparql_q_res_fragments(
            sparql_q=subclasses_sparql_q,
            graph=class_disjoint_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )

        assert base_axiom_extract_comp_metaprop_fragments == american_pizza_subclasses_frags
        assert class_disjoint_axiom_extract_comp_metaprop_fragments == american_pizza_subclasses_frags


    def test_disjoint_classes(self,
                         base_axiom_extract_comp_graph, 
                         class_disjoint_axiom_extract_comp_graph, 
                         get_sparql_q_res_fragments,
                         disjoint_classes_sparql_q,
                         ms2_ns
                    ) -> None:
        

        base_axiom_extract_comp_disjoint_classes_fragments = get_sparql_q_res_fragments(
            sparql_q=disjoint_classes_sparql_q,
            graph=base_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )

        class_disjoint_axiom_extract_comp_disjoint_classes_fragments = get_sparql_q_res_fragments(
            sparql_q=disjoint_classes_sparql_q,
            graph=class_disjoint_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )
        
        assert len(base_axiom_extract_comp_disjoint_classes_fragments) == 0
        assert class_disjoint_axiom_extract_comp_disjoint_classes_fragments == {
            ("NonVegetarianPizza",), ("American",), ("Country",), ("Cheese",),
            ("Pizza",), ("CheesyPizza",), ("America",), ("Mozzarella",), ("Tomato",),
            ("Topping",), ("PepperoniSausage",)
        }

    # TODO: test OWL named individuals

class Test_build_full_graph:
    @pytest.fixture(scope="class")
    def base_axiom_extract_comp_graph(self, base_axiom_extract_comp, american_pizza_ex_kr) -> Graph:
        graph = base_axiom_extract_comp.build_full_graph(kr=american_pizza_ex_kr)
        return graph
    
    @pytest.fixture(scope="class")
    def class_disjoint_axiom_extract_comp_graph(self, class_disjoint_axiom_extract_comp, american_pizza_ex_kr) -> Graph:
        graph = class_disjoint_axiom_extract_comp.build_full_graph(kr=american_pizza_ex_kr)
        return graph
    
    def test_owl_classes(self,
                         base_axiom_extract_comp_graph, 
                         class_disjoint_axiom_extract_comp_graph, 
                         get_sparql_q_res_fragments,
                         owl_classes_sparql_q,
                         ms2_ns,
                         american_pizza_ex_kr,
                         american_pizza_owl_classes_frags
                    ) -> None:
        base_axiom_extract_comp_class_fragments = get_sparql_q_res_fragments(
            sparql_q=owl_classes_sparql_q,
            graph=base_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )

        class_disjoint_axiom_extract_comp_class_fragments = get_sparql_q_res_fragments(
            sparql_q=owl_classes_sparql_q,
            graph=class_disjoint_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )
        
        nb_kr_concepts = len(american_pizza_ex_kr.concepts)

        assert len(base_axiom_extract_comp_class_fragments) == nb_kr_concepts
        assert len(class_disjoint_axiom_extract_comp_class_fragments) == nb_kr_concepts
        assert base_axiom_extract_comp_class_fragments == american_pizza_owl_classes_frags
        assert class_disjoint_axiom_extract_comp_class_fragments == american_pizza_owl_classes_frags

    def test_owl_obj_props(self,
                         base_axiom_extract_comp_graph, 
                         class_disjoint_axiom_extract_comp_graph, 
                         get_sparql_q_res_fragments,
                         owl_obj_props_sparql_q,
                         ms2_ns,
                         american_pizza_owl_obj_props_frags
                    ) -> None:
        
        base_axiom_extract_comp_prop_fragments = get_sparql_q_res_fragments(
                                    sparql_q=owl_obj_props_sparql_q,
                                    graph=base_axiom_extract_comp_graph,
                                    ns={"ms2": ms2_ns}
                                )
        
        class_disjoint_axiom_extract_comp_prop_fragments = get_sparql_q_res_fragments(
                                    sparql_q=owl_obj_props_sparql_q,
                                    graph=class_disjoint_axiom_extract_comp_graph,
                                    ns={"ms2": ms2_ns}
                                )
    
        assert base_axiom_extract_comp_prop_fragments == american_pizza_owl_obj_props_frags
        assert class_disjoint_axiom_extract_comp_prop_fragments == american_pizza_owl_obj_props_frags

    def test_domain_range(self,
                         base_axiom_extract_comp_graph, 
                         class_disjoint_axiom_extract_comp_graph, 
                         get_sparql_q_res_fragments,
                         domain_range_sparql_q,
                         ms2_ns,
                         american_pizza_domain_range_frags
                    ) -> None:

        base_axiom_extract_comp_domain_range_fragments = get_sparql_q_res_fragments(
            sparql_q=domain_range_sparql_q,
            graph=base_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )
        class_disjoint_axiom_extract_domain_range_fragments = get_sparql_q_res_fragments(
            sparql_q=domain_range_sparql_q,
            graph=class_disjoint_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )

        assert base_axiom_extract_comp_domain_range_fragments == american_pizza_domain_range_frags
        assert class_disjoint_axiom_extract_domain_range_fragments == american_pizza_domain_range_frags

    def test_owl_subclasses(self,
                         base_axiom_extract_comp_graph, 
                         class_disjoint_axiom_extract_comp_graph, 
                         get_sparql_q_res_fragments,
                         subclasses_sparql_q,
                         ms2_ns,
                         american_pizza_subclasses_frags
                    ) -> None:
        
        base_axiom_extract_comp_metaprop_fragments = get_sparql_q_res_fragments(
            sparql_q=subclasses_sparql_q,
            graph=base_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )

        class_disjoint_axiom_extract_comp_metaprop_fragments = get_sparql_q_res_fragments(
            sparql_q=subclasses_sparql_q,
            graph=class_disjoint_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )

        assert base_axiom_extract_comp_metaprop_fragments == american_pizza_subclasses_frags
        assert class_disjoint_axiom_extract_comp_metaprop_fragments == american_pizza_subclasses_frags

    def test_disjoint_classes(self,
                         base_axiom_extract_comp_graph, 
                         class_disjoint_axiom_extract_comp_graph, 
                         get_sparql_q_res_fragments,
                         disjoint_classes_sparql_q,
                         ms2_ns,
                         american_pizza_disjoint_classes_frags
                    ) -> None:
        

        base_axiom_extract_comp_disjoint_classes_fragments = get_sparql_q_res_fragments(
            sparql_q=disjoint_classes_sparql_q,
            graph=base_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )

        class_disjoint_axiom_extract_comp_disjoint_classes_fragments = get_sparql_q_res_fragments(
            sparql_q=disjoint_classes_sparql_q,
            graph=class_disjoint_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )
        
        assert len(base_axiom_extract_comp_disjoint_classes_fragments) == 0
        assert class_disjoint_axiom_extract_comp_disjoint_classes_fragments == american_pizza_disjoint_classes_frags

    # TODO: test OWL named individuals
    def test_owl_named_individuals(self,
                                   base_axiom_extract_comp_graph, 
                                    owl_named_individuals_sparql_q,
                                    ms2_ns, get_sparql_q_res_fragments,
                                    owl_named_individuals_labels_sparql_q,
                                    get_sparql_q_label_res,
                                    american_pizza_owl_named_individuals_frags,
                                    american_pizza_owl_named_individuals_labels
                                ) -> None:

        individuals_fragments = get_sparql_q_res_fragments(
            sparql_q=owl_named_individuals_sparql_q,
            graph=base_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )

        individuals_labels = get_sparql_q_label_res(
            sparql_q=owl_named_individuals_labels_sparql_q,
            graph=base_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )

        assert individuals_fragments == american_pizza_owl_named_individuals_frags

        assert individuals_labels == american_pizza_owl_named_individuals_labels

# TODO
# class Test_check_owl_graph_consistency:
#     pass

def test_get_concept_uris_from_error_output(base_axiom_extract_comp, american_pizza_unsatisfiable_concept_uris) -> None:
    
    ex_error_output_unsatisfiable = """
        2024-01-31 08:59:33,850 ERROR org.obolibrary.robot.ReasonerHelper - There are 5 unsatisfiable classes in the ontology.
        2024-01-31 08:59:33,850 ERROR org.obolibrary.robot.ReasonerHelper -     unsatisfiable: http://www.ms2.org/o/example#Mozzarella
        2024-01-31 08:59:33,850 ERROR org.obolibrary.robot.ReasonerHelper -     unsatisfiable: http://www.ms2.org/o/example#American
        2024-01-31 08:59:33,850 ERROR org.obolibrary.robot.ReasonerHelper -     unsatisfiable: http://www.ms2.org/o/example#Cheese
        2024-01-31 08:59:33,850 ERROR org.obolibrary.robot.ReasonerHelper -     unsatisfiable: http://www.ms2.org/o/example#America
        2024-01-31 08:59:33,850 ERROR org.obolibrary.robot.ReasonerHelper -     unsatisfiable: http://www.ms2.org/o/example#PepperoniSausage
    """
    concept_uris = base_axiom_extract_comp._get_concept_uris_from_error_output(ex_error_output_unsatisfiable)

    assert concept_uris == american_pizza_unsatisfiable_concept_uris

class Test_update_unsatisfiable_kr_owl_graph:
    @pytest.fixture(scope="class")
    def base_axiom_extract_comp_graph(
                                self, 
                                base_axiom_extract_comp, 
                                american_pizza_ex_kr,
                                american_pizza_unsatisfiable_concept_uris
                            ) -> Graph:
        graph = base_axiom_extract_comp._update_unsatisfiable_kr_owl_graph(
                                                                kr=american_pizza_ex_kr,
                                                                unsatisfiable_concept_uris=american_pizza_unsatisfiable_concept_uris
                                                                )
        return graph
    
    @pytest.fixture(scope="class")
    def class_disjoint_axiom_extract_comp_graph(
                                        self, 
                                        class_disjoint_axiom_extract_comp, 
                                        american_pizza_ex_kr,
                                        american_pizza_unsatisfiable_concept_uris
                                    ) -> Graph:
        graph = class_disjoint_axiom_extract_comp._update_unsatisfiable_kr_owl_graph(
                                                                kr=american_pizza_ex_kr,
                                                                unsatisfiable_concept_uris=american_pizza_unsatisfiable_concept_uris
                                                                )
        return graph

    def test_base_axiom_extract_comp(
            self, 
            base_axiom_extract_comp_graph,
            get_sparql_q_res_fragments,
            owl_classes_sparql_q,
            owl_obj_props_sparql_q,
            domain_range_sparql_q,
            subclasses_sparql_q,
            disjoint_classes_sparql_q,
            ms2_ns,
            american_pizza_owl_classes_frags
        ) -> None:

        base_axiom_extract_comp_class_fragments = get_sparql_q_res_fragments(
                                        sparql_q=owl_classes_sparql_q,
                                        graph=base_axiom_extract_comp_graph,
                                        ns={"ms2": ms2_ns}
                                    )

        base_axiom_extract_comp_prop_fragments = get_sparql_q_res_fragments(
                                    sparql_q=owl_obj_props_sparql_q,
                                    graph=base_axiom_extract_comp_graph,
                                    ns={"ms2": ms2_ns}
                                )
        base_axiom_extract_comp_domain_range_fragments = get_sparql_q_res_fragments(
            sparql_q=domain_range_sparql_q,
            graph=base_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )
        base_axiom_extract_comp_metaprop_fragments = get_sparql_q_res_fragments(
            sparql_q=subclasses_sparql_q,
            graph=base_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )
        base_axiom_extract_comp_disjoint_classes_fragments = get_sparql_q_res_fragments(
            sparql_q=disjoint_classes_sparql_q,
            graph=base_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )

        assert base_axiom_extract_comp_class_fragments == american_pizza_owl_classes_frags
        assert len(base_axiom_extract_comp_prop_fragments) == 1
        assert len(base_axiom_extract_comp_domain_range_fragments) == 0
        assert len(base_axiom_extract_comp_metaprop_fragments) == 0
        assert len(base_axiom_extract_comp_disjoint_classes_fragments) == 0

    def test_class_disjoint_axiom_extract_comp(
            self, 
            class_disjoint_axiom_extract_comp_graph,
            get_sparql_q_res_fragments,
            owl_classes_sparql_q,
            owl_obj_props_sparql_q,
            domain_range_sparql_q,
            subclasses_sparql_q,
            disjoint_classes_sparql_q,
            ms2_ns,
            american_pizza_owl_classes_frags,
            american_pizza_disjoint_classes_frags
        ) -> None:

        class_disjoint_axiom_extract_comp_class_fragments = get_sparql_q_res_fragments(
                                        sparql_q=owl_classes_sparql_q,
                                        graph=class_disjoint_axiom_extract_comp_graph,
                                        ns={"ms2": ms2_ns}
                                    )

        class_disjoint_axiom_extract_comp_prop_fragments = get_sparql_q_res_fragments(
                                    sparql_q=owl_obj_props_sparql_q,
                                    graph=class_disjoint_axiom_extract_comp_graph,
                                    ns={"ms2": ms2_ns}
                                )
        class_disjoint_axiom_extract_comp_domain_range_fragments = get_sparql_q_res_fragments(
            sparql_q=domain_range_sparql_q,
            graph=class_disjoint_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )
        class_disjoint_axiom_extract_comp_metaprop_fragments = get_sparql_q_res_fragments(
            sparql_q=subclasses_sparql_q,
            graph=class_disjoint_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )
        class_disjoint_axiom_extract_comp_disjoint_classes_fragments = get_sparql_q_res_fragments(
            sparql_q=disjoint_classes_sparql_q,
            graph=class_disjoint_axiom_extract_comp_graph,
            ns={"ms2": ms2_ns}
        )

        assert class_disjoint_axiom_extract_comp_class_fragments == american_pizza_owl_classes_frags
        assert len(class_disjoint_axiom_extract_comp_prop_fragments) == 1
        assert len(class_disjoint_axiom_extract_comp_domain_range_fragments) == 0
        assert len(class_disjoint_axiom_extract_comp_metaprop_fragments) == 0
        assert class_disjoint_axiom_extract_comp_disjoint_classes_fragments == american_pizza_disjoint_classes_frags

def test_update_kr_external_uris(
        base_axiom_extract_comp, 
        american_pizza_ex_kr,
        american_pizza_owl_classes_frags
    ) -> None:
    base_axiom_extract_comp._update_kr_external_uris(kr=american_pizza_ex_kr)

    concept_ext_uris = set()
    for concept in american_pizza_ex_kr.concepts:
        concept_ext_uris.update(concept.external_uids)

    rel_ext_uris = set()
    for relation in american_pizza_ex_kr.relations:
        rel_ext_uris.update(relation.external_uids)

    base_uri_str = str(base_axiom_extract_comp.base_uri)
    rel_ext_uris_frags = {(uri.replace(base_uri_str, ""),) for uri in rel_ext_uris}
    concept_ext_uris_frags = {(uri.replace(base_uri_str, ""),) for uri in concept_ext_uris}

    assert concept_ext_uris_frags == american_pizza_owl_classes_frags
    assert rel_ext_uris_frags == {("hasCountryOfOrigin",), ("hasIngredient",), ("hasBase",)}

# TODO
# class Test_run:
#     pass