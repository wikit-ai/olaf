from collections import defaultdict
from itertools import product
from typing import Dict, Set

import spacy
from spacy.matcher import PhraseMatcher

from ..data_container.candidate_term_schema import CandidateRelation, CandidateTerm
from ..data_container.concept_schema import Concept
from ..data_container.linguistic_realisation_schema import RelationLR
from ..data_container.relation_schema import Relation


def crs_to_relation(candidate_relations: Set[CandidateRelation]) -> Relation:
    """Convert a set of candidate relations to a new relation.
    Each candidate relation represents a different linguistic realisation in the relation created.

    Parameters
    ----------
    candidate_relations : Set[CandidateRelation]
        Set of candidate relations to convert into a relation.

    Returns
    -------
    Relation
        The relation created from the candidate relations.
    """
    candidates = list(candidate_relations)
    new_relation = Relation(
        label=candidates[0].label,
        source_concept=candidates[0].source_concept,
        destination_concept=candidates[0].destination_concept,
    )
    for candidate in candidates:
        candidate_lr = RelationLR(
            label=candidate.label, corpus_occurrences=candidate.corpus_occurrences
        )
        new_relation.add_linguistic_realisation(candidate_lr)
    return new_relation


def cts_to_crs(
    candidate_terms: Set[CandidateTerm],
    concepts_labels_map: Dict[str, Concept],
    spacy_model: spacy.language.Language,
    concept_max_distance: int,
    scope: str,
) -> Set[CandidateRelation]:
    """Convert candidate terms into candidate relations.
    Concepts are searched around the candidate term within a given distance.
    If source and destination concepts are found, candidate relation as triple is created.
    Otherwise, candidate relation has no source and destination concepts.

    Parameters
    ----------
    candidate_terms : Set[CandidateTerm]
        Set of candidate terms to convert into candidate relations.
    concepts_labels_map : Dict[str,Concept]
        Dictionary with concept labels as keys and concepts corresponding as values.
    spacy_model :  spacy.language.Language
        SpaCy model to use.
    concept_max_distance : int
        The maximum distance between the candidate term and the concept sought.
    scope : str
        Scope used to search concepts. Can be "doc" for the entire document or "sent" for
        the candidate term sentence.

    Returns
    -------
    Set[CandidateRelation]
        Set of candidate relations found from the candidate terms.
    """

    matcher = PhraseMatcher(spacy_model.vocab, attr="LOWER")
    for concept in concepts_labels_map.values():
        matcher.add(
            concept.label,
            [spacy_model(lr.label) for lr in concept.linguistic_realisations],
        )

    candidate_relations = set()
    for ct in candidate_terms:
        co_concept = defaultdict(set)

        for co in ct.corpus_occurrences:
            content = co.sent if scope == "sent" else co.doc
            matches = matcher(content)
            source_concepts = set()
            destination_concepts = set()

            for match_id, start, end in matches:
                # Check for source concept
                if (start < co.start) and (
                    start >= max(0, co.start - concept_max_distance)
                ):
                    source_concepts.add(
                        concepts_labels_map.get(spacy_model.vocab.strings[match_id])
                    )

                # Check for destination concept
                elif (start >= co.end) and (
                    end <= min(len(content), co.end + concept_max_distance)
                ):
                    destination_concepts.add(
                        concepts_labels_map.get(spacy_model.vocab.strings[match_id])
                    )

            # Relation without concepts
            if len(source_concepts) == 0 or len(destination_concepts) == 0:
                co_concept[(None, None)].add(co)
            # Relation with all possible concept pairs
            else:
                concept_pairs = list(product(source_concepts, destination_concepts))
                for source, destination in concept_pairs:
                    co_concept[(source, destination)].add(co)

        # Creation of candidate relation
        for concept_tuple, rel_occ in co_concept.items():
            cr = CandidateRelation(
                label=ct.label,
                corpus_occurrences=rel_occ,
                source_concept=concept_tuple[0],
                destination_concept=concept_tuple[1],
                enrichment=ct.enrichment,
            )
            candidate_relations.add(cr)
    return candidate_relations
