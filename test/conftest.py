import os
import tempfile
from typing import Callable, Dict, Set, Tuple

import pytest
import spacy
from pytest import MonkeyPatch
from rdflib import Graph, Namespace, URIRef
from spacy.matcher import PhraseMatcher

from olaf import Pipeline
from olaf.data_container import (Concept, KnowledgeRepresentation,
                                 LinguisticRealisation, Metarelation, Relation)


@pytest.fixture(scope="session")
def test_data_path():
    with tempfile.TemporaryDirectory() as newpath:
        old_cwd = os.getcwd()
        os.chdir(newpath)
        mp = MonkeyPatch()
        mp.setenv("DATA_PATH", newpath)
        yield
        os.chdir(old_cwd)


@pytest.fixture(scope="session")
def en_sm_spacy_model():
    spacy_model = spacy.load("en_core_web_sm", exclude=["ner"])
    return spacy_model

@pytest.fixture(scope="session")
def en_md_spacy_model():
    spacy_model = spacy.load("en_core_web_md", exclude=["ner"])
    return spacy_model

@pytest.fixture(scope="session")
def american_cheesy_pizza_text() -> str:
    text = "American cheesy pizza is a quintessential delight that embodies the heart of American culinary tradition. This beloved pizza variety, with its generous toppings of gooey Mozzarella cheese, zesty tomato sauce, and savory pepperoni sausage, captures the essence of comfort food indulgence. Originating in America, this pizza kind epitomizes the nation's love for hearty, flavorful dishes. As a non vegetarian pizza, it has gained popularity for its rich ingredients and satisfying taste, making it a staple in pizzerias across the country. With its origins rooted in America, the American cheesy pizza continues to reign supreme as a symbol of deliciousness and innovation in the world of pizzas."

    return text

@pytest.fixture(scope="session")
def american_cheesy_pizza_doc(american_cheesy_pizza_text, en_sm_spacy_model) -> spacy.tokens.doc:
    doc = en_sm_spacy_model(american_cheesy_pizza_text)
    return doc

def get_concept_lrs(concept: Concept, doc: str, spacy_model) -> LinguisticRealisation:
    phrase_matcher = PhraseMatcher(spacy_model.vocab, attr="LOWER")
    phrase_matcher.add(concept.label, [spacy_model(concept.label)])
    matches = phrase_matcher(doc, as_spans=True)
    lr = LinguisticRealisation(
        label=concept.label.lower(), corpus_occurrences=set(matches)
    )
    return lr

@pytest.fixture(scope="module")
def american_pizza_ex_kr(american_cheesy_pizza_doc, en_sm_spacy_model) -> KnowledgeRepresentation:
    pizza = Concept(label="Pizza")
    topping = Concept(label="Topping")
    cheesy_pizza = Concept(label="Cheesy Pizza")
    cheese_topping = Concept(label="Cheese")
    american_pizza = Concept(label="American")
    mozza_topping = Concept(label="Mozzarella")
    peperoni_sausage_topping = Concept(label="Pepperoni Sausage")
    tomato_topping = Concept(label="Tomato")
    america_country = Concept(label="America")
    country = Concept(label="Country")
    non_veggie_pizza = Concept(label="Non Vegetarian Pizza")

    ex_concepts = {
        pizza, topping, cheesy_pizza, cheese_topping, american_pizza, mozza_topping,
        peperoni_sausage_topping, tomato_topping, america_country, country, non_veggie_pizza
    }

    ex_relations = {
        Relation(label="has ingredient", source_concept=american_pizza, destination_concept=mozza_topping),
        Relation(label="has ingredient", source_concept=american_pizza, destination_concept=peperoni_sausage_topping),
        Relation(label="has ingredient", source_concept=american_pizza, destination_concept=tomato_topping),
        Relation(label="has country of origin", source_concept=american_pizza, destination_concept=america_country),
        Relation(label="has base"),
    }

    ex_metarelations = {
        Metarelation(label="is_generalised_by", source_concept=american_pizza, destination_concept=pizza),
        Metarelation(label="is_generalised_by", source_concept=american_pizza, destination_concept=cheesy_pizza),
        Metarelation(label="is_generalised_by", source_concept=mozza_topping, destination_concept=topping),
        Metarelation(label="is_generalised_by", source_concept=mozza_topping, destination_concept=cheese_topping),
        Metarelation(label="is_generalised_by", source_concept=cheese_topping, destination_concept=topping),
        Metarelation(label="is_generalised_by", source_concept=peperoni_sausage_topping, destination_concept=topping),
        Metarelation(label="is_generalised_by", source_concept=america_country, destination_concept=country),
        Metarelation(label="has kind", source_concept=american_pizza, destination_concept=non_veggie_pizza),
    }

    ex_kr = KnowledgeRepresentation(
        concepts=ex_concepts,
        relations=ex_relations,
        metarelations=ex_metarelations
    )

    for concept in ex_kr.concepts:
        concept_lr = get_concept_lrs(concept=concept, doc=american_cheesy_pizza_doc, spacy_model=en_sm_spacy_model)
        concept.add_linguistic_realisation(concept_lr)

    return ex_kr

@pytest.fixture(scope="module")
def american_pizza_pipeline(en_sm_spacy_model, american_cheesy_pizza_doc) -> Pipeline:
    pipeline = Pipeline(
        spacy_model=en_sm_spacy_model,
        corpus=[american_cheesy_pizza_doc]
    )
    return pipeline

@pytest.fixture(scope="session")
def ms2_base_uri() -> URIRef:
    base_uri = URIRef("http://www.ms2.org/o/example#")
    return base_uri

@pytest.fixture(scope="session")
def ms2_ns(ms2_base_uri) -> Namespace:
    ns = Namespace(ms2_base_uri)
    return ns

@pytest.fixture(scope="session")
def owl_classes_sparql_q() -> Namespace:
    sparql_q = """
            SELECT ?class WHERE {
                ?class rdf:type owl:Class .
            }
        """
    return sparql_q

@pytest.fixture(scope="session")
def owl_obj_props_sparql_q() -> Namespace:
    sparql_q = """
            SELECT ?prop WHERE {
                ?prop rdf:type owl:ObjectProperty .
            }
        """
    return sparql_q

@pytest.fixture(scope="session")
def domain_range_sparql_q() -> Namespace:
    sparql_q = """
        SELECT ?prop ?domain ?range WHERE {
            ?prop rdfs:domain ?domain ;
                rdfs:range ?range .
        }
    """
    return sparql_q

@pytest.fixture(scope="session")
def subclasses_sparql_q() -> Namespace:
    sparql_q = """
            SELECT ?subclass ?class WHERE {
                ?subclass rdfs:subClassOf ?class .
            }
        """
    return sparql_q

@pytest.fixture(scope="session")
def anonymous_some_parent_sparql_q() -> Namespace:
    sparql_q = """
        SELECT ?class ?restriction_rel ?restriction_cls WHERE {
                ?class rdfs:subClassOf [
                    rdf:type owl:Restriction ;
                    owl:onProperty ?restriction_rel ;
                    owl:someValuesFrom ?restriction_cls
                ] .
        }
    """
    return sparql_q

@pytest.fixture(scope="session")
def anonymous_some_equivalent_sparql_q() -> Namespace:
    sparql_q = """
        SELECT ?class ?restriction_rel ?restriction_cls WHERE {
                ?class owl:equivalentClass [
                    rdf:type owl:Restriction ;
                    owl:onProperty ?restriction_rel ;
                    owl:someValuesFrom ?restriction_cls
                ] .
        }
    """
    return sparql_q

@pytest.fixture(scope="session")
def anonymous_only_parent_sparql_q() -> Namespace:
    sparql_q = """
        SELECT ?class ?restriction_rel ?restriction_cls WHERE {
                ?class rdfs:subClassOf [
                    rdf:type owl:Restriction ;
                    owl:onProperty ?restriction_rel ;
                    owl:allValuesFrom ?restriction_cls
                ] .
        }
    """
    return sparql_q

@pytest.fixture(scope="session")
def disjoint_classes_sparql_q() -> Namespace:
    sparql_q = """
        SELECT ?disjoint_cls WHERE {
                [] rdf:type owl:AllDisjointClasses ;
                owl:members/rdf:rest* ?node .
                ?node rdf:first ?disjoint_cls .
        }
    """
    return sparql_q

@pytest.fixture(scope="session")
def get_sparql_r_res() -> Callable[[str, Graph, Dict[str, Namespace]], Set[Tuple]]:
    
    def funct(sparql_q: str, graph: Graph, ns: Dict[str, Namespace]) -> Set[Tuple]:
        q_res = graph.query(sparql_q, initNs=ns)

        fragments = {tuple((item.fragment for item in res)) for res in q_res}

        return fragments
    return funct