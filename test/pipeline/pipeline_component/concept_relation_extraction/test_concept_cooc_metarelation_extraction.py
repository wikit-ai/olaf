from typing import Callable, List, Set

import pytest
import spacy

from olaf.data_container.concept_schema import Concept
from olaf.data_container.linguistic_realisation_schema import LinguisticRealisation
from olaf.pipeline.pipeline_component.concept_relation_extraction.concept_cooc_metarelation_extraction import (
    ConceptCoocMetarelationExtraction,
)
from olaf.pipeline.pipeline_schema import Pipeline


@pytest.fixture(scope="session")
def raw_corpus() -> List[str]:
    corpus = [
        "A sentence with some spans that might be in common with other ones.",
        "Another sentence with some exiting spans!",
        "Also, and why not? A doc with a url http://super.truc.ad and some numbers 145 58.69!",
        "A two sentences doc about numbers and sentence. This second sentence should be about numbers.",
    ]
    return corpus


@pytest.fixture(scope="session")
def spacy_nlp():
    spacy_model = spacy.load(
        "en_core_web_sm",
        exclude=["tagger", "attribute_ruler", "lemmatizer", "ner"],
    )
    return spacy_model


@pytest.fixture(scope="session")
def corpus_docs(spacy_nlp, raw_corpus) -> List[spacy.tokens.Doc]:
    docs = [doc for doc in spacy_nlp.pipe(raw_corpus)]
    return docs


@pytest.fixture(scope="session")
def sentence_concept(corpus_docs) -> Concept:
    c_sentence = Concept(
        label="sentence",
        linguistic_realisations={
            LinguisticRealisation(
                "sentence",
                corpus_occurrences={
                    corpus_docs[0][1:2],
                    corpus_docs[1][1:2],
                    corpus_docs[3][2:3],
                    corpus_docs[3][7:8],
                    corpus_docs[3][11:12],
                },
            )
        },
    )
    return c_sentence


@pytest.fixture(scope="session")
def span_concept(corpus_docs) -> Concept:
    c_span = Concept(
        label="span",
        linguistic_realisations={
            LinguisticRealisation(
                "span", corpus_occurrences={corpus_docs[0][4:5], corpus_docs[1][5:6]}
            )
        },
    )
    return c_span


@pytest.fixture(scope="session")
def doc_concept(corpus_docs) -> Concept:
    c_doc = Concept(
        label="doc",
        linguistic_realisations={
            LinguisticRealisation(
                "doc", corpus_occurrences={corpus_docs[2][7:8], corpus_docs[3][3:4]}
            )
        },
    )
    return c_doc


@pytest.fixture(scope="session")
def num_concept(corpus_docs) -> Concept:
    c_num = Concept(
        label="number",
        linguistic_realisations={
            LinguisticRealisation(
                "number",
                corpus_occurrences={
                    corpus_docs[2][-4:-3],
                    corpus_docs[3][5:6],
                    corpus_docs[3][15:16],
                },
            )
        },
    )
    return c_num


@pytest.fixture(scope="session")
def with_concept(corpus_docs) -> Concept:
    c_with = Concept(
        label="with",
        linguistic_realisations={
            LinguisticRealisation(
                "with",
                corpus_occurrences={
                    corpus_docs[0][2:3],
                    corpus_docs[0][10:11],
                    corpus_docs[1][2:3],
                    corpus_docs[2][8:9],
                },
            )
        },
    )
    return c_with


@pytest.fixture(scope="session")
def value_concept(corpus_docs) -> Concept:
    c_value = Concept(
        label="d+",
        linguistic_realisations={
            LinguisticRealisation(
                "d+",
                corpus_occurrences={corpus_docs[2][-3:-2], corpus_docs[2][-2:-1]},
            )
        },
    )
    return c_value


@pytest.fixture(scope="session")
def kr_concepts(
    sentence_concept,
    span_concept,
    doc_concept,
    num_concept,
    with_concept,
    value_concept,
) -> Set[Concept]:
    concepts = {
        sentence_concept,
        span_concept,
        doc_concept,
        num_concept,
        with_concept,
        value_concept,
    }

    return concepts


class TestConceptCoocMetarelationExtractionDefault:
    @pytest.fixture(scope="class")
    def pipeline(self, kr_concepts, spacy_nlp, raw_corpus) -> Pipeline:
        custom_pipeline = Pipeline(
            spacy_model=spacy_nlp, corpus=[doc for doc in spacy_nlp.pipe(raw_corpus)]
        )
        custom_pipeline.kr.concepts = kr_concepts

        return custom_pipeline

    @pytest.fixture(scope="class")
    def default_c_cooc_rel_extract(self) -> ConceptCoocMetarelationExtraction:
        rel_extract = ConceptCoocMetarelationExtraction()

        return rel_extract

    def test_default_params_options(self, default_c_cooc_rel_extract) -> None:
        assert default_c_cooc_rel_extract.window_size is None
        assert default_c_cooc_rel_extract.threshold == 0
        assert default_c_cooc_rel_extract.metarelation_creation_metric(1)
        assert default_c_cooc_rel_extract.scope == "doc"
        assert default_c_cooc_rel_extract.metarelation_label == "RELATED_TO"
        assert default_c_cooc_rel_extract.create_symmetric_metarelation == False

    def test_fetch_concept_occurrences_fragments(
        self, default_c_cooc_rel_extract, with_concept, corpus_docs
    ) -> None:
        c_occ_fragments = (
            default_c_cooc_rel_extract._fetch_concept_occurrences_fragments(
                with_concept
            )
        )

        conditions = [
            corpus_docs[0] in c_occ_fragments,
            corpus_docs[1] in c_occ_fragments,
            corpus_docs[2] in c_occ_fragments,
        ]

        assert all(conditions)

    def test_count_concept_cooccurrence(
        self, default_c_cooc_rel_extract, sentence_concept, span_concept, value_concept
    ) -> None:
        sent_span_cooc_count = default_c_cooc_rel_extract._count_concept_cooccurrence(
            sentence_concept, span_concept
        )
        sent_value_cooc_count = default_c_cooc_rel_extract._count_concept_cooccurrence(
            sentence_concept, value_concept
        )

        assert sent_span_cooc_count == 2
        assert sent_value_cooc_count == 0

    def test_run(self, default_c_cooc_rel_extract, pipeline) -> None:
        pipeline.kr.metarelations = set()
        default_c_cooc_rel_extract.run(pipeline)

        assert len(pipeline.kr.metarelations) == 11

        metarelations_label = {rel.label for rel in pipeline.kr.metarelations}
        assert metarelations_label == {"RELATED_TO"}


class TestConceptCoocMetarelationExtractionCustom:
    @pytest.fixture(scope="class")
    def pipeline(self, kr_concepts, spacy_nlp, raw_corpus) -> Pipeline:
        custom_pipeline = Pipeline(
            spacy_model=spacy_nlp, corpus=[doc for doc in spacy_nlp.pipe(raw_corpus)]
        )
        custom_pipeline.kr.concepts = kr_concepts

        return custom_pipeline

    @pytest.fixture(scope="class")
    def custom_relation_creation_metric(self) -> Callable[[int], bool]:
        def custom_metric(count: int) -> bool:
            return (count * 2) > 3

        return custom_metric

    @pytest.fixture(scope="class")
    def c_cooc_rel_extract(
        self, custom_relation_creation_metric
    ) -> ConceptCoocMetarelationExtraction:
        params = {"scope": "sent", "metarelation_label": "custom relation"}
        opts = {"window_size": 4, "threshold": 3}
        rel_extract = ConceptCoocMetarelationExtraction(
            custom_relation_creation_metric, parameters=params, options=opts
        )

        return rel_extract

    def test_params_options(self, c_cooc_rel_extract) -> None:
        assert c_cooc_rel_extract.window_size is 4
        assert c_cooc_rel_extract.threshold == 3
        assert not c_cooc_rel_extract.metarelation_creation_metric(1)
        assert c_cooc_rel_extract.metarelation_creation_metric(2)
        assert c_cooc_rel_extract.scope == "sent"
        assert c_cooc_rel_extract.metarelation_label == "custom relation"
        assert c_cooc_rel_extract.create_symmetric_metarelation == False

    def test_fetch_concept_occurrences_fragments(
        self, c_cooc_rel_extract, sentence_concept, span_concept
    ) -> None:
        sentence_c_occ_fragments = (
            c_cooc_rel_extract._fetch_concept_occurrences_fragments(sentence_concept)
        )

        span_c_occ_fragments = c_cooc_rel_extract._fetch_concept_occurrences_fragments(
            span_concept
        )

        len_conditions = [len(frag) == 4 for frag in sentence_c_occ_fragments]
        assert all(len_conditions)

        token_conditions = [
            "sentence" in frag.text for frag in sentence_c_occ_fragments
        ]
        assert all(token_conditions)

        assert len(sentence_c_occ_fragments) == 12

        token_conditions = ["span" in frag.text for frag in span_c_occ_fragments]
        assert all(token_conditions)

        assert len(span_c_occ_fragments) == 6

    def test_count_concept_cooccurrence(
        self, c_cooc_rel_extract, sentence_concept, span_concept, value_concept
    ) -> None:
        sent_span_cooc_count = c_cooc_rel_extract._count_concept_cooccurrence(
            sentence_concept, span_concept
        )
        sent_value_cooc_count = c_cooc_rel_extract._count_concept_cooccurrence(
            sentence_concept, value_concept
        )

        assert sent_span_cooc_count == 1
        assert sent_value_cooc_count == 0

        span_sent_cooc_count = c_cooc_rel_extract._count_concept_cooccurrence(
            span_concept, sentence_concept
        )
        value_sent_cooc_count = c_cooc_rel_extract._count_concept_cooccurrence(
            value_concept, sentence_concept
        )

        assert span_sent_cooc_count == 1
        assert value_sent_cooc_count == 0

    def test_run(self, c_cooc_rel_extract, pipeline) -> None:
        c_cooc_rel_extract.run(pipeline)

        print(
            [
                (rel.source_concept.label, rel.destination_concept.label)
                for rel in pipeline.kr.metarelations
            ]
        )

        assert len(pipeline.kr.metarelations) == 7

        metarelations_label = {rel.label for rel in pipeline.kr.metarelations}
        assert metarelations_label == {"custom relation"}
