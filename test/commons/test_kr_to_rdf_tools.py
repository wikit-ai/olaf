import pytest
from rdflib import Graph

from olaf.commons.kr_to_rdf_tools import (
    kr_concepts_to_disjoint_classes, kr_concepts_to_owl_classes,
    kr_metarelations_to_owl, kr_relations_to_anonymous_only_parent,
    kr_relations_to_anonymous_some_equivalent,
    kr_relations_to_anonymous_some_parent,
    kr_relations_to_domain_range_obj_props, kr_relations_to_owl_obj_props,
    owl_class_uri, owl_instance_uri, owl_obj_prop_uri)


def test_owl_class_uri(ms2_base_uri) -> None:
    assert owl_class_uri(label="Mozzarella", base_uri=ms2_base_uri) == ms2_base_uri + "Mozzarella"
    assert owl_class_uri(label="mozzarella", base_uri=ms2_base_uri) == ms2_base_uri + "Mozzarella"
    assert owl_class_uri(label="Peperoni Sausage", base_uri=ms2_base_uri) == ms2_base_uri + "PeperoniSausage"
    assert owl_class_uri(label="Non Vegetarian Pizza", base_uri=ms2_base_uri) == ms2_base_uri + "NonVegetarianPizza"

def test_owl_obj_prop_uri(ms2_base_uri) -> None:
    assert owl_obj_prop_uri(label="has ingredient", base_uri=ms2_base_uri) == ms2_base_uri + "hasIngredient"
    assert owl_obj_prop_uri(label="Has Ingredient", base_uri=ms2_base_uri) == ms2_base_uri + "hasIngredient"
    assert owl_obj_prop_uri(label="Has", base_uri=ms2_base_uri) == ms2_base_uri + "has"
    assert owl_obj_prop_uri(label="has", base_uri=ms2_base_uri) == ms2_base_uri + "has"

def test_owl_instance_uri(ms2_base_uri) -> None:
    assert owl_instance_uri(label="Mozzarella", base_uri=ms2_base_uri) == ms2_base_uri + "_mozzarella"
    assert owl_instance_uri(label="mozzarella", base_uri=ms2_base_uri) == ms2_base_uri + "_mozzarella"
    assert owl_instance_uri(label="Peperoni Sausage", base_uri=ms2_base_uri) == ms2_base_uri + "_peperoniSausage"
    assert owl_instance_uri(label="Non Vegetarian Pizza", base_uri=ms2_base_uri) == ms2_base_uri + "_nonVegetarianPizza"

def test_kr_concepts_to_owl_classes(american_pizza_ex_kr, ms2_base_uri, owl_classes_sparql_q, ms2_ns, get_sparql_r_res) -> None:

    concept2classes_g = kr_concepts_to_owl_classes(kr=american_pizza_ex_kr, base_uri=ms2_base_uri)

    class_fragments = get_sparql_r_res(
        sparql_q=owl_classes_sparql_q,
        graph=concept2classes_g,
        ns={"ms2": ms2_ns}
    )
    
    assert len(class_fragments) == len(american_pizza_ex_kr.concepts)
    assert class_fragments == {
        ("NonVegetarianPizza",), ("American",), ("Country",), ("Cheese",),
        ("Pizza",), ("CheesyPizza",), ("America",), ("Mozzarella",), ("Tomato",),
        ("Topping",), ("PeperoniSausage",)
    }

def test_kr_relations_to_owl_obj_props(american_pizza_ex_kr, ms2_base_uri, owl_obj_props_sparql_q, ms2_ns, get_sparql_r_res) -> None:

    relations2objProps_g = kr_relations_to_owl_obj_props(kr=american_pizza_ex_kr, base_uri=ms2_base_uri)

    prop_fragments = get_sparql_r_res(
        sparql_q=owl_obj_props_sparql_q,
        graph=relations2objProps_g,
        ns={"ms2": ms2_ns}
    )
    
    assert len(prop_fragments) == len({rel.label for rel in american_pizza_ex_kr.relations})
    assert prop_fragments == {("hasCountryOfOrigin",), ("hasIngredient",)}

def test_kr_metarelations_to_owl_domain_range(
        american_pizza_ex_kr, ms2_base_uri,
        domain_range_sparql_q, owl_obj_props_sparql_q, ms2_ns,
        get_sparql_r_res
    ) -> None:

    
    relations2domainRangeObjProps_g = kr_relations_to_domain_range_obj_props(kr=american_pizza_ex_kr, base_uri=ms2_base_uri)

    prop_fragments = get_sparql_r_res(
        sparql_q=owl_obj_props_sparql_q,
        graph=relations2domainRangeObjProps_g,
        ns={"ms2": ms2_ns}
    )

    domain_range_fragments = get_sparql_r_res(
        sparql_q=domain_range_sparql_q,
        graph=relations2domainRangeObjProps_g,
        ns={"ms2": ms2_ns}
    )
    
    assert len(prop_fragments) == len({rel.label for rel in american_pizza_ex_kr.relations})
    assert prop_fragments == {("hasCountryOfOrigin",), ("hasIngredient",)}

    assert domain_range_fragments == {
        ("hasIngredient", "American", "Mozzarella"),
        ("hasIngredient", "American", "Tomato"),
        ("hasIngredient", "American", "PeperoniSausage"),
        ("hasCountryOfOrigin", "American", "America")
    }


class Test_kr_metarelations_to_owl:

    @pytest.fixture(scope="class")
    def metarelations2Props_g(self, american_pizza_ex_kr, ms2_base_uri) -> Graph:
        graph = kr_metarelations_to_owl(
                                kr=american_pizza_ex_kr,
                                base_uri=ms2_base_uri
                            )
        return graph
    
    def test_kr_metarelations_to_owl_obj_props(self, metarelations2Props_g, owl_obj_props_sparql_q, ms2_ns, get_sparql_r_res) -> None:
        
        prop_fragments = get_sparql_r_res(
            sparql_q=owl_obj_props_sparql_q,
            graph=metarelations2Props_g,
            ns={"ms2": ms2_ns}
        )
        
        assert prop_fragments == {("hasKind",)}

    def test_kr_metarelations_to_owl_subclasses(self, metarelations2Props_g, subclasses_sparql_q, ms2_ns, get_sparql_r_res) -> None:
        
        metaprop_fragments = get_sparql_r_res(
            sparql_q=subclasses_sparql_q,
            graph=metarelations2Props_g,
            ns={"ms2": ms2_ns}
        )
        
        assert metaprop_fragments == {
            ("American", "CheesyPizza"),
            ("Mozzarella", "Cheese"),
            ("Mozzarella", "Topping"),
            ("PeperoniSausage", "Topping"),
            ("Cheese", "Topping"),
            ("America", "Country"),
            ("American", "Pizza")
        }

class Test_kr_concepts_to_disjoint_classes:

    @pytest.fixture(scope="class")
    def concept2Disjointclasses_g(self, american_pizza_ex_kr, ms2_base_uri) -> Graph:
        graph = kr_concepts_to_disjoint_classes(
                                kr=american_pizza_ex_kr,
                                base_uri=ms2_base_uri
                            )
        return graph
    
    def test_kr_metarelations_to_owl_classes(self, american_pizza_ex_kr, concept2Disjointclasses_g, owl_classes_sparql_q, ms2_ns, get_sparql_r_res) -> None:
        
        class_fragments = get_sparql_r_res(
            sparql_q=owl_classes_sparql_q,
            graph=concept2Disjointclasses_g,
            ns={"ms2": ms2_ns}
        )
        
        assert len(class_fragments) == len(american_pizza_ex_kr.concepts)
        assert class_fragments == {
            ("NonVegetarianPizza",), ("American",), ("Country",), ("Cheese",),
            ("Pizza",), ("CheesyPizza",), ("America",), ("Mozzarella",), ("Tomato",),
            ("Topping",), ("PeperoniSausage",)
        }

    def test_kr_metarelations_to_disjoint_classes(self, american_pizza_ex_kr, concept2Disjointclasses_g, disjoint_classes_sparql_q, ms2_ns, get_sparql_r_res) -> None:
        
        disjoint_classes_fragments = get_sparql_r_res(
            sparql_q=disjoint_classes_sparql_q,
            graph=concept2Disjointclasses_g,
            ns={"ms2": ms2_ns}
        )
        
        assert len(disjoint_classes_fragments) == len(american_pizza_ex_kr.concepts)
        assert disjoint_classes_fragments == {
            ("NonVegetarianPizza",), ("American",), ("Country",), ("Cheese",),
            ("Pizza",), ("CheesyPizza",), ("America",), ("Mozzarella",), ("Tomato",),
            ("Topping",), ("PeperoniSausage",)
        }

class Test_kr_relations_to_anonymous_some_parent:

    @pytest.fixture(scope="class")
    def relations2subclassSome_g(self, american_pizza_ex_kr, ms2_base_uri) -> Graph:
        graph = kr_relations_to_anonymous_some_parent(
                                kr=american_pizza_ex_kr,
                                base_uri=ms2_base_uri
                            )
        return graph
    
    def test_kr_relations_to_anonymous_some_parent_classes(self, relations2subclassSome_g, owl_classes_sparql_q, ms2_ns, get_sparql_r_res) -> None:
        
        class_fragments = get_sparql_r_res(
            sparql_q=owl_classes_sparql_q,
            graph=relations2subclassSome_g,
            ns={"ms2": ms2_ns}
        )
        
        assert class_fragments == {("American",), ("America",), ("Mozzarella",), ("Tomato",), ("PeperoniSausage",)}

    def test_kr_relations_to_anonymous_some_parent_obj_props(self, american_pizza_ex_kr, relations2subclassSome_g, owl_obj_props_sparql_q, ms2_ns, get_sparql_r_res) -> None:
        
        prop_fragments = get_sparql_r_res(
            sparql_q=owl_obj_props_sparql_q,
            graph=relations2subclassSome_g,
            ns={"ms2": ms2_ns}
        )
    
        assert len(prop_fragments) == len({rel.label for rel in american_pizza_ex_kr.relations})
        assert prop_fragments == {("hasCountryOfOrigin",), ("hasIngredient",)}

    def test_kr_relations_to_anonymous_some_parent_restriction(self, american_pizza_ex_kr, relations2subclassSome_g, anonymous_some_parent_sparql_q, ms2_ns, get_sparql_r_res) -> None:
        
        restriction_fragments = get_sparql_r_res(
            sparql_q=anonymous_some_parent_sparql_q,
            graph=relations2subclassSome_g,
            ns={"ms2": ms2_ns}
        )
    
        assert len(restriction_fragments) == len(american_pizza_ex_kr.relations)
        assert restriction_fragments == {
            ("American", "hasIngredient", "Mozzarella"),
            ("American", "hasIngredient", "Tomato"),
            ("American", "hasCountryOfOrigin", "America"),
            ("American", "hasIngredient", "PeperoniSausage")
        }

class Test_kr_relations_to_anonymous_only_parent:

    @pytest.fixture(scope="class")
    def relations2subclassOnly_g(self, american_pizza_ex_kr, ms2_base_uri) -> Graph:
        graph = kr_relations_to_anonymous_only_parent(
                                kr=american_pizza_ex_kr,
                                base_uri=ms2_base_uri
                            )
        return graph
    
    def test_kr_relations_to_anonymous_only_parent_classes(self, relations2subclassOnly_g, owl_classes_sparql_q, ms2_ns, get_sparql_r_res) -> None:
        
        class_fragments = get_sparql_r_res(
            sparql_q=owl_classes_sparql_q,
            graph=relations2subclassOnly_g,
            ns={"ms2": ms2_ns}
        )
        
        assert class_fragments == {("American",), ("America",), ("Mozzarella",), ("Tomato",), ("PeperoniSausage",)}

    def test_kr_relations_to_anonymous_only_parent_obj_props(self, american_pizza_ex_kr, relations2subclassOnly_g, owl_obj_props_sparql_q, ms2_ns, get_sparql_r_res) -> None:
        
        prop_fragments = get_sparql_r_res(
            sparql_q=owl_obj_props_sparql_q,
            graph=relations2subclassOnly_g,
            ns={"ms2": ms2_ns}
        )
    
        assert len(prop_fragments) == len({rel.label for rel in american_pizza_ex_kr.relations})
        assert prop_fragments == {("hasCountryOfOrigin",), ("hasIngredient",)}

    def test_kr_relations_to_anonymous_only_parent_restriction(self, american_pizza_ex_kr, relations2subclassOnly_g, anonymous_only_parent_sparql_q, ms2_ns, get_sparql_r_res) -> None:
        
        restriction_fragments = get_sparql_r_res(
            sparql_q=anonymous_only_parent_sparql_q,
            graph=relations2subclassOnly_g,
            ns={"ms2": ms2_ns}
        )
    
        assert len(restriction_fragments) == len(american_pizza_ex_kr.relations)
        assert restriction_fragments == {
            ("American", "hasIngredient", "Mozzarella"),
            ("American", "hasIngredient", "Tomato"),
            ("American", "hasCountryOfOrigin", "America"),
            ("American", "hasIngredient", "PeperoniSausage")
        }

class Test_kr_relations_to_anonymous_some_equivalent:

    @pytest.fixture(scope="class")
    def relations2equivalentSome_g(self, american_pizza_ex_kr, ms2_base_uri) -> Graph:
        graph = kr_relations_to_anonymous_some_equivalent(
                                kr=american_pizza_ex_kr,
                                base_uri=ms2_base_uri
                            )
        return graph
    
    def test_kr_relations_to_anonymous_some_equivalent_classes(self, relations2equivalentSome_g, owl_classes_sparql_q, ms2_ns, get_sparql_r_res) -> None:
        
        class_fragments = get_sparql_r_res(
            sparql_q=owl_classes_sparql_q,
            graph=relations2equivalentSome_g,
            ns={"ms2": ms2_ns}
        )
        
        assert class_fragments == {("American",), ("America",), ("Mozzarella",), ("Tomato",), ("PeperoniSausage",)}

    def test_kr_relations_to_anonymous_some_equivalent_obj_props(self, american_pizza_ex_kr, relations2equivalentSome_g, owl_obj_props_sparql_q, ms2_ns, get_sparql_r_res) -> None:
        
        prop_fragments = get_sparql_r_res(
            sparql_q=owl_obj_props_sparql_q,
            graph=relations2equivalentSome_g,
            ns={"ms2": ms2_ns}
        )
    
        assert len(prop_fragments) == len({rel.label for rel in american_pizza_ex_kr.relations})
        assert prop_fragments == {("hasCountryOfOrigin",), ("hasIngredient",)}

    def test_kr_relations_to_anonymous_some_equivalent_restriction(self, american_pizza_ex_kr, relations2equivalentSome_g, anonymous_some_equivalent_sparql_q, ms2_ns, get_sparql_r_res) -> None:
        
        restriction_fragments = get_sparql_r_res(
            sparql_q=anonymous_some_equivalent_sparql_q,
            graph=relations2equivalentSome_g,
            ns={"ms2": ms2_ns}
        )
    
        assert len(restriction_fragments) == len(american_pizza_ex_kr.relations)
        assert restriction_fragments == {
            ("American", "hasIngredient", "Mozzarella"),
            ("American", "hasIngredient", "Tomato"),
            ("American", "hasCountryOfOrigin", "America"),
            ("American", "hasIngredient", "PeperoniSausage")
        }