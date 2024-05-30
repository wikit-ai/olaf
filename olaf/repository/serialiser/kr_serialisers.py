import json
from collections.abc import Iterable
from os import PathLike

import spacy
from rdflib import Graph
from spacy.matcher import PhraseMatcher

from ...data_container import (
    Concept,
    KnowledgeRepresentation,
    LinguisticRealisation,
    Metarelation,
    Relation,
)
from ...pipeline.pipeline_schema import Pipeline


class KRJSONSerialiser:
    """JSON serialiser for KR objects."""

    def __init__(self) -> None: ...

    def serialise(self, kr: KnowledgeRepresentation, file_path: PathLike) -> None:
        """Serialise the KR object into a JSON formatted file.

        Parameters
        ----------
        kr : KnowledgeRepresentation
            The KR object to serialise.
        file_path : PathLike
            The path to the file to save the serialised KR object.
        """
        kr_json = {
            "concepts": self.get_concepts_json(kr=kr),
            "relations": self.get_relations_json(kr=kr),
            "metarelations": self.get_metarelations_json(kr=kr),
            "rdf_graph": (
                kr.rdf_graph.serialize(format="ttl") if len(kr.rdf_graph) else None
            ),
        }

        with open(file_path, "w", encoding="utf8") as json_file:
            json.dump(kr_json, json_file)

    def load(self, pipeline: Pipeline, file_path: PathLike) -> None:
        """Load a KR object from a JSON serialisation.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to use when loading the KR.
            It is used to access the corpus and spacy model
            to reconstruct the linguistic realisations.
        file_path : PathLike
            The path to the file containing the JSON serialised KR object.
        """
        with open(file_path, "r", encoding="utf8") as json_file:
            kr_json = json.load(json_file)

        concepts_index = {}
        concepts = self.load_concepts_from_json(
            concepts_json=kr_json.get("concepts", set()),
            concepts_idx=concepts_index,
            pipeline=pipeline,
        )

        relations = self.load_relations_from_json(
            relations_json=kr_json.get("relations", set()),
            concepts_idx=concepts_index,
            pipeline=pipeline,
        )

        metarelations = self.load_metarelations_from_json(
            metarelations_json=kr_json.get("metarelations", set()),
            concepts_idx=concepts_index,
            pipeline=pipeline,
        )

        rdf_graph = Graph()
        if kr_json["rdf_graph"] is not None:
            rdf_graph.parse(data=kr_json["rdf_graph"])

        kr = KnowledgeRepresentation(
            concepts=concepts,
            relations=relations,
            metarelations=metarelations,
            rdf_graph=rdf_graph,
        )

        pipeline.kr = kr

    def load_concepts_from_json(
        self,
        concepts_json: list[dict[str]],
        concepts_idx: dict[int, Concept],
        pipeline: Pipeline,
    ) -> set[Concept]:
        """Load concepts from the concepts portion of the KR JSON serialisation.

        Parameters
        ----------
        concepts_json : list[dict[str]]
            The concepts portion of the KR JSON serialisation.
        concepts_idx : dict[int, Concept]
            A dictionary object to store the concept index.
        pipeline : Pipeline
            The pipeline to use for reconstructing the linguistic realisations.

        Returns
        -------
        set[Concept]
            The set of Concepts.
        """
        concepts = set()
        for concept_json in concepts_json:
            concept_lrs = {
                LinguisticRealisation(
                    label=lr["label"],
                    corpus_occurrences=self.build_cos_from_strings(
                        co_texts=lr["co_texts"],
                        spacy_model=pipeline.spacy_model,
                        docs=pipeline.corpus,
                    ),
                )
                for lr in concept_json.get("lrs", [])
            }
            concept = Concept(
                label=concept_json["label"], linguistic_realisations=concept_lrs
            )
            concepts_idx[concept_json["concept_id"]] = concept
            concepts.add(concept)
        return concepts

    def load_relations_from_json(
        self,
        relations_json: list[dict[str]],
        concepts_idx: dict[int, Concept],
        pipeline: Pipeline,
    ) -> set[Relation]:
        """Load relations from the relations portion of the KR JSON serialisation.

        Parameters
        ----------
        relations_json : list[dict[str]]
            The relations portion of the KR JSON serialisation.
        concepts_idx : dict[int, Concept]
            The concept index mapping concept IDs to concept instances.
        pipeline : Pipeline
            The pipeline to use for reconstructing the linguistic realisations.

        Returns
        -------
        set[Relation]
            The set of Relations.
        """

        relations = set()
        for rel_json in relations_json:
            rel_lrs = {
                LinguisticRealisation(
                    label=lr["label"],
                    corpus_occurrences=self.build_cos_from_strings(
                        co_texts=lr["co_texts"],
                        spacy_model=pipeline.spacy_model,
                        docs=pipeline.corpus,
                    ),
                )
                for lr in rel_json.get("lrs", [])
            }

            rel_source_concept = (
                concepts_idx.get(rel_json.get("source_concept_id"))
                if rel_json.get("source_concept_id") is not None
                else None
            )

            rel_dest_concept = (
                concepts_idx.get(rel_json.get("destination_concept_id"))
                if rel_json.get("destination_concept_id") is not None
                else None
            )

            relations.add(
                Relation(
                    label=rel_json["label"],
                    source_concept=rel_source_concept,
                    destination_concept=rel_dest_concept,
                    linguistic_realisations=rel_lrs,
                )
            )
        return relations

    def load_metarelations_from_json(
        self,
        metarelations_json: list[dict[str]],
        concepts_idx: dict[int, Concept],
        pipeline: Pipeline,
    ) -> set[Metarelation]:
        """Load metarelations from the metarelations portion of the KR JSON serialisation.

        Parameters
        ----------
        metarelations_json : list[dict[str]]
            The metarelations portion of the KR JSON serialisation.
        concepts_idx : dict[int, Concept]
            The concept index mapping concept IDs to concept instances.
        pipeline : Pipeline
            The pipeline to use for reconstructing the linguistic realisations.

        Returns
        -------
        set[Metarelation]
            The set of metarelations.
        """

        metarelations = set()
        for rel_json in metarelations_json:
            rel_lrs = {
                LinguisticRealisation(
                    label=lr["label"],
                    corpus_occurrences=self.build_cos_from_strings(
                        co_texts=lr["co_texts"],
                        spacy_model=pipeline.spacy_model,
                        docs=pipeline.corpus,
                    ),
                )
                for lr in rel_json.get("lrs", [])
            }

            rel_source_concept = concepts_idx.get(rel_json["source_concept_id"])
            rel_dest_concept = concepts_idx.get(rel_json["destination_concept_id"])

            metarelations.add(
                Metarelation(
                    label=rel_json["label"],
                    source_concept=rel_source_concept,
                    destination_concept=rel_dest_concept,
                    linguistic_realisations=rel_lrs,
                )
            )
        return metarelations

    def get_concepts_json(self, kr: KnowledgeRepresentation) -> list[dict[str]]:
        """Construct the JSON serialisation of KR concepts.

        Parameters
        ----------
        kr : KnowledgeRepresentation
            The KR to serialise the concepts.

        Returns
        -------
        list[dict[str]]
            The serialised concepts JSON-like object.
        """
        concepts_list = []

        for concept in kr.concepts:
            new_concept = {"concept_id": id(concept)}
            new_concept["label"] = concept.label
            lrs = []
            for lr in concept.linguistic_realisations:
                new_lr = {}
                new_lr["label"] = lr.label
                new_lr["co_texts"] = list({co.text for co in lr.corpus_occurrences})
                lrs.append(new_lr)
            new_concept["lrs"] = lrs
            concepts_list.append(new_concept)

        return concepts_list

    def get_metarelations_json(self, kr: KnowledgeRepresentation) -> list[dict[str]]:
        """Construct the JSON serialisation of KR metarelations.

        Parameters
        ----------
        kr : KnowledgeRepresentation
            The KR to serialise the metarelations.

        Returns
        -------
        list[dict[str]]
            The serialised metarelations JSON-like object.
        """
        metarelations_list = []

        for meta in kr.metarelations:
            new_meta = {
                "source_concept_id": id(meta.source_concept),
                "destination_concept_id": id(meta.destination_concept),
            }
            new_meta["label"] = meta.label
            lrs = [
                {
                    "label": lr.label,
                    "co_texts": list({co.text for co in lr.corpus_occurrences}),
                }
                for lr in meta.linguistic_realisations
            ]

            new_meta["lrs"] = lrs

            metarelations_list.append(new_meta)

        return metarelations_list

    def get_relations_json(self, kr: KnowledgeRepresentation) -> list[dict[str]]:
        """Construct the JSON serialisation of KR relations.

        Parameters
        ----------
        kr : KnowledgeRepresentation
            The KR to serialise the relations.

        Returns
        -------
        list[dict[str]]
            The serialised relations JSON-like object.
        """
        relations_list = []

        for relation in kr.relations:

            new_relation = {
                "source_concept_id": (
                    id(relation.source_concept) if relation.source_concept else None
                ),
                "destination_concept_id": (
                    id(relation.destination_concept)
                    if relation.destination_concept
                    else None
                ),
            }

            new_relation["label"] = relation.label

            lrs = [
                {
                    "label": lr.label,
                    "co_texts": list({co.text for co in lr.corpus_occurrences}),
                }
                for lr in relation.linguistic_realisations
            ]

            new_relation["lrs"] = lrs

            relations_list.append(new_relation)

        return relations_list

    def build_cos_from_strings(
        self,
        co_texts: Iterable[str],
        spacy_model: spacy.language.Language,
        docs: list[spacy.tokens.Doc],
    ) -> set[spacy.tokens.Span]:
        """Create corpus occurrences from a set of strings label and a corpus.

        Parameters
        ----------
        co_texts : Iterable[str]
            The strings to use for corpus occurrences extraction.
        spacy_model : spacy.language.Language
            The spaCy model to retrieve the corpus occurrences.
        docs : list[spacy.tokens.Doc]
            The corpus in which to find the corpus occurrences.

        Returns
        -------
        set[CandidateTerm]
            The set of corpus occurrences.
        """

        phrase_matcher = PhraseMatcher(spacy_model.vocab, attr="LOWER")

        for label in co_texts:
            phrase_matcher.add(label, [spacy_model(label)])

        corpus_occurrences = set()

        for doc in docs:
            matches = phrase_matcher(doc, as_spans=True)
            corpus_occurrences.update(set(matches))

        return corpus_occurrences
