from typing import Any, Dict, Set

import pytest
import spacy

from olaf.data_container.candidate_term_schema import CandidateTerm
from olaf.data_container.concept_schema import Concept
from olaf.data_container.enrichment_schema import Enrichment
from olaf.data_container.knowledge_representation_schema import KnowledgeRepresentation
from olaf.data_container.linguistic_realisation_schema import LinguisticRealisation
from olaf.pipeline.pipeline_component.concept_relation_extraction.knowledge_based_relation_extraction import (
    KnowledgeBasedRelationExtraction,
)
from olaf.pipeline.pipeline_schema import Pipeline
from olaf.repository.knowledge_source.knowledge_source_schema import KnowledgeSource


@pytest.fixture(scope="session")
def spacy_model():
    spacy_model = spacy.load(
        "en_core_web_sm",
        exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
    )
    return spacy_model

@pytest.fixture(scope="session")
def corpus(spacy_model):
    texts = [
        "Cats eat mouses.",
        "Dogs can eat little mouses too.",
        "I do not know if cats can devour mouses.",
        "I will devour my pizza.",
        "You should not eat so fast",
        "I like bike.",
    ]
    corpus = list(spacy_model.pipe(texts))
    return corpus


@pytest.fixture(scope="session")
def candidate_terms(corpus) -> Set[CandidateTerm]:
    candidate_terms = set()

    candidate_terms.add(
        CandidateTerm(
            label="eat",
            corpus_occurrences={corpus[0][1:2], corpus[1][2:3], corpus[4][3:4]},
            enrichment=Enrichment({"devour"})
        )
    )
    candidate_terms.add(
        CandidateTerm(
            label="devour",
            corpus_occurrences={corpus[2][7:8], corpus[3][2:3]}
        )
    )
    return candidate_terms


@pytest.fixture(scope="session")
def c_cat() -> Concept:
    c_cat = Concept(
        label="cat", linguistic_realisations={LinguisticRealisation(label="cats")}
    )
    return c_cat


@pytest.fixture(scope="session")
def c_mouse() -> Concept:
    c_mouse = Concept(
        label="mouse", linguistic_realisations={LinguisticRealisation(label="mouses")}
    )
    return c_mouse


@pytest.fixture(scope="session")
def c_dog() -> Concept:
    c_dog = Concept(
        label="dog", linguistic_realisations={LinguisticRealisation(label="dogs")}
    )
    return c_dog


@pytest.fixture(scope="function")
def pipeline(spacy_model, candidate_terms, corpus, c_cat, c_dog, c_mouse) -> Pipeline:
    pipeline = Pipeline(spacy_model=spacy_model, corpus=corpus)
    pipeline.candidate_terms = candidate_terms
    pipeline.kr = KnowledgeRepresentation()
    pipeline.kr.concepts.update({c_cat, c_dog, c_mouse})
    return pipeline


class MockKnowledgeSource(KnowledgeSource):
    def __init__(self, parameters: Dict[str, Any] | None = None) -> None:
        super().__init__(parameters)

    def _check_parameters(self) -> None:
        """Check wether required parameters are given and correct. If this is not the case,
        suitable default ones are set.
        """
        raise NotImplementedError

    def _check_resources(self) -> None:
        pass

    def match_external_concepts(self, matching_terms: Set[str]) -> Set[str]:
        """Method to fetch concepts matching the candidate term.

        Parameters
        ----------
        term : CandidateTerm
            The candidate term the method should look concepts for.

        Returns
        -------
        Set[str]
            The concept(s) external UIDs found matching the candidate term.
        """
        concepts_uids = set()

        if "eat" in matching_terms:
            concepts_uids.update({"some_eat_uid", "another_eat_uid"})
        if "devour" in matching_terms:
            concepts_uids.add("devour_uid")

        return concepts_uids

    def fetch_terms_synonyms(self, terms: Set[str]) -> Set[str]:
        raise NotImplementedError

    def fetch_terms_antonyms(self, terms: Set[str]) -> Set[str]:
        raise NotImplementedError

    def fetch_terms_hypernyms(self, terms: Set[str]) -> Set[str]:
        raise NotImplementedError

    def fetch_terms_hyponyms(self, terms: Set[str]) -> Set[str]:
        raise NotImplementedError


@pytest.fixture(scope="session")
def mock_knowledge_source() -> KnowledgeSource:
    kg = MockKnowledgeSource()
    return kg


class TestKnowledgeBasedRelationExtraction:

    @pytest.fixture(scope="class")
    def kg_based_relation_extraction(
        self, mock_knowledge_source
    ) -> KnowledgeBasedRelationExtraction:
        relation_extraction = KnowledgeBasedRelationExtraction(mock_knowledge_source)
        return relation_extraction

    def test_pipeline_cts(self, kg_based_relation_extraction, pipeline) -> None:
        kg_based_relation_extraction.run(pipeline)

        relations_ext_udis = set()

        for relation in pipeline.kr.relations:            
            relations_ext_udis.update(relation.external_uids)

        conditions = [
            "some_eat_uid" in relations_ext_udis,
            "another_eat_uid" in relations_ext_udis,
            "devour_uid" in relations_ext_udis,
        ]

        assert len(pipeline.kr.relations) == 3
        assert all(conditions)

        assert len(pipeline.candidate_terms) == 0


class TestKnowledgeBasedRelationExtractionNoMerge:

    @pytest.fixture(scope="class")
    def kg_based_relation_extraction_no_merge_syn(
        self, mock_knowledge_source
    ) -> KnowledgeBasedRelationExtraction:
        params = {"group_ct_on_synonyms": False}
        relation_extraction = KnowledgeBasedRelationExtraction(
            mock_knowledge_source, parameters=params
        )
        return relation_extraction

    def test_pipeline_no_merge(
        self, kg_based_relation_extraction_no_merge_syn, pipeline
    ) -> None:
        kg_based_relation_extraction_no_merge_syn.run(pipeline)

        relations_ext_uids = set()

        for relation in pipeline.kr.relations:
            relations_ext_uids.update(relation.external_uids)

        conditions = [
            "some_eat_uid" in relations_ext_uids,
            "another_eat_uid" in relations_ext_uids,
            "devour_uid" in relations_ext_uids,
        ]

        assert len(pipeline.kr.relations) == 5
        assert all(conditions)

        assert len(pipeline.candidate_terms) == 0
