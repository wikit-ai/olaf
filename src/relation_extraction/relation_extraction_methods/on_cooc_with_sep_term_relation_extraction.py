from collections import Counter, defaultdict
import spacy.language
import spacy.matcher
import spacy.tokens
from typing import Any, Dict, List, Tuple
import uuid

from commons.ontology_learning_schema import KR, MetaRelation
from commons.spacy_processing_tools import build_concept_matcher
from config import logging_config


class OnCoocWithSepTermMetaRelationExtraction:

    def __init__(self,
                 corpus: List[spacy.tokens.doc.Doc],
                 kr: KR,
                 spacy_nlp: spacy.language.Language,
                 options: Dict[str, Any]
                 ) -> None:
        """Initializer for the OnCoocWithSepTermMetaRelationExtraction class.

        Parameters
        ----------
        corpus : List[spacy.tokens.doc.Doc]
            The corpus of documents
        kr : KR
            The Knowledge Representation object containing the concepts.
        spacy_nlp : spacy.language.Language
            The Language model used to process the corpus.
        term_relation_map : Dict[str, str]
            A mapping of the terms to find in the documents and the corresponding meta relations.
        options : Dict[str, Any]
            The class options.
        """
        self.corpus = corpus
        self.kr = kr
        self.spacy_nlp = spacy_nlp
        self.options = options
        self.concept_spans = None
        self.concept_cooc_content_map = None
        self.concept_cooc_counts = None

        try:
            assert self.options.get('cooc_scope') in {"doc", "sentence"}
            assert isinstance(self.options.get('cooc_treshold'), int)
            assert isinstance(self.options.get('term_relation_map'), dict)
        except KeyError as e:
            logging_config.logger.error(
                f"""Config information missing for relation extraction based on co-occurrence and term. Make sure you provided the configuration fields:
                    - relation_extraction.on_occurrence_with_sep_term.cooc_scope ("doc" or "sentence")
                    - relation_extraction.on_occurrence_with_sep_term.cooc_treshold
                    Trace : {e}
                """)

        self.term_relation_map = self.options["term_relation_map"]

        self.concept_matcher = build_concept_matcher(self.kr, self.spacy_nlp)

        if self.options.get("cooc_scope", "doc") == "sentence":
            if not spacy.tokens.span.Span.has_extension('concepts'):
                spacy.tokens.span.Span.set_extension('concepts', default=[])
            self._tag_concepts_in_corpus_sents()
        else:
            if not spacy.tokens.doc.Doc.has_extension('concepts'):
                spacy.tokens.doc.Doc.set_extension('concepts', default=[])
            self._tag_concepts_in_corpus_docs()

        self._build_concept_cooc_count()

    def _get_concept_pairs(self, concept_matches: List[spacy.tokens.span.Span]) -> List[Tuple[str, str]]:
        """Extract relevant concept pairs from the concepts (Spacy Spans) matched in a document.

        Parameters
        ----------
        concept_matches : List[spacy.tokens.span.Span]
            The concepts Spans found in a document.

        Returns
        -------
        Iterable
            An iterable of concept pairs (concept IDs).
        """
        # we want all cooccurrences, even duplicates,
        # so that the cooc_content_counts takes into account duplicate cooccurrences in a content
        concept_pairs = list()
        concept_distance_limit = self.options.get(
            "concept_distance_limit", 1000)

        for i in range(len(concept_matches) - 1):
            concept_source = concept_matches[i]
            concept_dest = concept_matches[i+1]

            if (concept_dest.start - concept_source.end) <= concept_distance_limit:
                concept_pairs.append(
                    (concept_source.label_, concept_dest.label_)
                )

        return concept_pairs

    def _tag_concepts_in_corpus_docs(self) -> None:
        """Method specific to Documents. Match concepts to documents contained in the Corpus.
            Matched concepts are added to the "concepts" custom attribute.

            Also build the self.concept_spans, self.concept_cooc_content_map indices attributes.
        """
        concept_spans = list()
        concept_cooc_content_map = defaultdict(list)

        for doc in self.corpus:

            doc_concept_matches = self.concept_matcher(doc, as_spans=True)
            doc_concept_matches.sort(key=lambda e: e.start)

            concept_spans.extend(doc_concept_matches)

            if len(set(doc_concept_matches)) > 1:

                all_concept_combinations = self._get_concept_pairs(
                    doc_concept_matches)

                for concept_pair in all_concept_combinations:

                    concept_cooc_content_map[concept_pair].append(doc)

            doc._.concepts.extend(doc_concept_matches)

        self.concept_spans = concept_spans
        self.concept_cooc_content_map = concept_cooc_content_map

    def _tag_concepts_in_corpus_sents(self) -> None:
        """Method specific to sentences in documents. Match concepts to sentences in documents contained in the corpus.
            Matched concepts are added to the "concepts" custom attribute.

            Also build the self.concept_spans, self.concept_cooc_content_map indices attributes.
        """
        concept_spans = list()
        concept_cooc_content_map = defaultdict(list)

        for doc_idx, doc in enumerate(self.corpus):

            if not doc.has_annotation('SENT_START'):
                logging_config.logger.warn(f"""
                    Option relation_extraction.on_occurrence_with_sep_term.cooc_scope set to "sentence" but "{doc.text}" (corpus index {doc_idx}) has no sentence. 
                    Document indexed {doc_idx} in the corpus will be ignored.
                """)
                continue
            else:
                for sentence in doc.sents:
                    sent_concept_matches = self.concept_matcher(
                        sentence, as_spans=True)
                    sent_concept_matches.sort(key=lambda e: e.start)

                    concept_spans.extend(sent_concept_matches)

                    if len(set(sent_concept_matches)) > 1:

                        all_concept_combinations = self._get_concept_pairs(
                            sent_concept_matches)

                        for concept_pair in all_concept_combinations:

                            concept_cooc_content_map[concept_pair].append(
                                sentence)

                    sentence._.concepts.extend(sent_concept_matches)

        self.concept_spans = concept_spans
        self.concept_cooc_content_map = concept_cooc_content_map

    def _build_concept_cooc_count(self) -> None:
        """Build the index of concepts pairs occurrence counts in documents.
            The attribute self.concept_cooc_content_map must have been set 
            (by the methods self._tag_concepts_in_corpus_docs, or self._tag_concepts_in_corpus_sents)

            Sets the attribute self.concept_cooc_counts.
        """

        concept_cooc_counts = dict()

        if self.concept_cooc_content_map is not None:
            for concept_pair, contents in self.concept_cooc_content_map.items():
                concept_cooc_counts[concept_pair] = len(contents)

            self.concept_cooc_counts = concept_cooc_counts
        else:
            logging_config.logger.error(f"""
                    Attribute self.concept_cooc_content_map has not been set.
                    It is required to build the self.concept_cooc_counts index.
                    Make sure you ran either method self._tag_concepts_in_corpus_docs, or self._tag_concepts_in_corpus_sents.
                """)

    def _select_relation_type(self, relation_types: List[str]) -> str:
        """Select a the most relevant relation type in a List of relation types.
            Currently the most relevant means the one occuring the most.
            Default is "related_to".

        Parameters
        ----------
        relation_types : List[str]
            The list of relations types

        Returns
        -------
        str
            The selected relation type
        """
        relation_type = "related_to"
        unique_relation_types = set(relation_types)

        if (len(unique_relation_types) > 1):
            non_related_to_rel_types = [
                rel for rel in relation_types if rel != "related_to"]
            counts_rel_types = list(Counter(non_related_to_rel_types).items())
            counts_rel_types.sort(key=lambda e: e[1], reverse=True)
            relation_type = counts_rel_types[0][0]
        elif len(unique_relation_types) == 1:
            relation_type = unique_relation_types.pop()

        return relation_type

    def _get_meta_relation_type(self, source_concept_id: str, dest_concept_id: str) -> str:
        """Extract the meta relation type from a document or sentence and its tagged concepts.
            If not tokens text in between the concepts match in the self.term_relation_map, the
            default meta relation is "related_to".

        Parameters
        ----------
        source_concept_id : str
            The source concept ID.
        dest_concept_id : str
            The destination concept ID.

        Returns
        -------
        str
            The meta relation type string.
        """
        concept_distance_limit = self.options.get(
            "concept_distance_limit", 1000)

        relation_types = list()

        cooc_concerned_contents = set(self.concept_cooc_content_map[(
            source_concept_id, dest_concept_id)])

        for content in cooc_concerned_contents:

            content_concepts_spans = content._.concepts

            source_concept_spans = [
                span for span in content_concepts_spans if span.label_ == source_concept_id]
            dest_concept_spans = [
                span for span in content_concepts_spans if span.label_ == dest_concept_id]

            if (len(source_concept_spans) == 0) or (len(dest_concept_spans) == 0):
                logging_config.logger.error(f"""
                        There has been an issue while extracting concept "{source_concept_id}" or "{dest_concept_id}"
                        in content "{content.text}" for relation type extraction.
                        """)

            token_sequences_between_concepts = list()

            for source_concept_span in source_concept_spans:
                for dest_concept_span in dest_concept_spans:
                    if (dest_concept_span.start - source_concept_span.end) <= concept_distance_limit:
                        token_sequences_between_concepts.append(
                            [token for token in content[source_concept_span.end:dest_concept_span.start]]
                        )

            for token_sequences in token_sequences_between_concepts:
                for token in token_sequences:
                    if self.options.get("use_lemma", False):
                        if self.term_relation_map.get(token.lemma_) is not None:
                            relation_types.append(
                                self.term_relation_map.get(token.lemma_))
                    else:
                        if self.term_relation_map.get(token.text) is not None:
                            relation_types.append(
                                self.term_relation_map.get(token.text))
                relation_types.append("related_to")

        relation_type = self._select_relation_type(relation_types)

        return relation_type

    def _set_meta_relation(self, concept_pair: Tuple[str, str], meta_relation: str) -> None:
        """Add a meta relation in the Knowledge Representation object
            This method updates the self.kr attribute.

        Parameters
        ----------
        concept_pair : Tuple[str, str]
            The ordered pair of concept IDs
        meta_relation : str
            The meta relation type
        """
        self.kr.meta_relations.add(MetaRelation(
            uid=str(uuid.uuid4()),
            source_concept_id=concept_pair[0],
            destination_concept_id=concept_pair[1],
            relation_type=meta_relation
        ))

        if meta_relation == "related_to":
            self.kr.meta_relations.add(MetaRelation(
                uid=str(uuid.uuid4()),
                source_concept_id=concept_pair[1],
                destination_concept_id=concept_pair[0],
                relation_type=meta_relation
            ))

    def on_cooc_with_sep_term_map_meta_rel_extraction(self) -> None:
        """Main method of the class. Create the meta relations based on the concepts 
            coocurrences in the corpus and the terms in the middle.
        """
        cooc_treshold = self.options.get("cooc_treshold", 0)

        for concept_pair, count in self.concept_cooc_counts.items():

            if count > cooc_treshold:

                relation_type = self._get_meta_relation_type(concept_pair[0],
                                                             concept_pair[1])
                self._set_meta_relation(concept_pair, relation_type)
