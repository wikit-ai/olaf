import pytest
import spacy

from olaf.data_container.concept_schema import Concept
from olaf.data_container.linguistic_realisation_schema import LinguisticRealisation
from olaf.pipeline.pipeline_schema import Pipeline
from olaf.pipeline.pipeline_component.concept_relation_hierarchy.subsumption_hierarchisation import SubsumptionHierarchisation

@pytest.fixture(scope="session")
def spacy_model():
    spacy_model = spacy.load(
        "en_core_web_sm",
        exclude=["tok2vec", "tagger", "attribute_ruler", "lemmatizer", "ner"],
    )
    return spacy_model

@pytest.fixture(scope="session")
def corpus(spacy_model):
    text_corpus = [
        "First test sentence.",
        "I love doing tests.",
        "Help me write my first test sentences."
    ]
    corpus = list(spacy_model.pipe(text_corpus))
    return corpus

@pytest.fixture(scope="session")
def c1(corpus):
    c1 = Concept(
        label="test",
        linguistic_realisations={
            LinguisticRealisation(
                label="test",
                corpus_occurrences={corpus[0][1], corpus[2][5]}
            ),
            LinguisticRealisation(
                label="tests",
                corpus_occurrences={corpus[1][3]}
            )
        }
    )
    return c1

@pytest.fixture(scope="session")
def c2(corpus):
    c2 = Concept(
        label="sentence",
        linguistic_realisations={
            LinguisticRealisation(
                label="sentence",
                corpus_occurrences={corpus[0][2]}
            ),
            LinguisticRealisation(
                label="sentences",
                corpus_occurrences={corpus[2][6]}
            )
        }
    )
    return c2

@pytest.fixture(scope="session")
def c3(corpus):
    c3 = Concept(
        label="first test",
        linguistic_realisations={
            LinguisticRealisation(
                label="first test",
                corpus_occurrences = {corpus[0][:1], corpus[2][4:5]}
            )
        }
    )
    return c3

@pytest.fixture(scope="session")
def concepts(c1, c2, c3):
    concepts = set()
    concepts.add(c1)
    concepts.add(c2)
    concepts.add(c3)
    return concepts


@pytest.fixture(scope="session")
def pipeline(spacy_model, corpus, concepts):
    pipeline = Pipeline(spacy_model=spacy_model, corpus=corpus)
    pipeline.kr.concepts = concepts
    return pipeline

@pytest.fixture(scope="session")
def subsumption():
    options = {}
    subsumption = SubsumptionHierarchisation(options = options)
    return subsumption


def test_concept_occurrence_count(c1,c2,c3, subsumption):
    assert subsumption._concept_occurrence_count(c1) == 3
    assert subsumption._concept_occurrence_count(c2) == 2
    assert subsumption._concept_occurrence_count(c3) == 2

def test_concepts_cooccurrence_count(c1,c2,c3,subsumption):
    assert subsumption._concepts_cooccurrence_count(c1,c2) == 2
    assert subsumption._concepts_cooccurrence_count(c1,c3) == 2
    assert subsumption._concepts_cooccurrence_count(c2,c3) == 2

def test_compute_subsumption(subsumption):
    assert subsumption._compute_subsumption(1,2) == 0.5
    assert subsumption._compute_subsumption(0,2) == 0
    assert subsumption._compute_subsumption(2,0) == 0

def test_is_sub_hierarchy(subsumption):
    assert subsumption._is_sub_hierarchy(0.8,0.4)
    assert not(subsumption._is_sub_hierarchy(0.4,0.2))
    assert not(subsumption._is_sub_hierarchy(0.6,0.9))
    assert not(subsumption._is_sub_hierarchy(0.2,0.4))

def test_running_subsumption(subsumption,pipeline):
    subsumption.run(pipeline)
    assert len(pipeline.kr.metarelations) == 2
    for metarelation in pipeline.kr.metarelations:
        assert metarelation.destination_concept.label == "test"
        assert metarelation.source_concept.label in {"sentence", "first test"}
        assert metarelation.label == "is_generalised_by"