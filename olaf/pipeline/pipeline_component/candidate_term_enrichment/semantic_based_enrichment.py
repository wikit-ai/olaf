from typing import Any, Dict, Optional

import numpy as np
from spacy.language import Language

from ....commons.logging_config import logger
from ....data_container.candidate_term_schema import CandidateTerm
from ....data_container.enrichment_schema import Enrichment
from ..pipeline_component_schema import PipelineComponent


class SemanticBasedEnrichment(PipelineComponent):
    """Pipeline component to enrich candidate terms based on semantic meaning
    computed from embeddings similarity.
    The most similar words in the vocabulary are added as synonyms.

    Attributes
    ----------
    threshold : Optional[int]
        The threshold defines the minimum similarity score required to be synonymous.
        By default the threshold is set to 0.9.
    """

    def __init__(self, threshold: Optional[int] = 0.9) -> None:
        """Initialise semantic based term enrichment instance.

        Parameters
        ----------
        threshold : Optional[int]
            The threshold defines the minimum similarity score required to be synonymous.
            By default the threshold is set to 0.9.
        """
        super().__init__()

        self.threshold = threshold
        self._check_resources()

    def _check_resources(self) -> None:
        """Method to check that the component has access to all its required resources."""
        logger.info(
            "Semantic based enrichment pipeline component has no external resources to check."
        )

    def optimise(self) -> None:
        # TODO
        """A method to optimise the pipeline component by tuning the options."""
        raise NotImplementedError

    def _compute_metrics(self) -> None:
        """A method to compute component performance metrics. It is used by the optimise
        method to update the options.
        """
        raise NotImplementedError

    def get_performance_report(self) -> Dict[str, Any]:
        """A getter for the pipeline component performance report.
        If the component has been optimised, it only returns the best performance.
        Otherwise, it returns the results obtained with the set parameters.

        Returns
        -------
        Dict[str, Any]
            The pipeline component performance report.
        """
        raise NotImplementedError

    def enrich_term(self, c_term: CandidateTerm, spacy_model: Language) -> None:
        """Enrich candidate term synonyms based on most similar words in the vocabulary.
        Similarity is computed based on vectors cosine similarity measure.
        """
        synonyms = set()
        if spacy_model.vocab.has_vector(c_term.label):
            most_similar_words = spacy_model.vocab.vectors.most_similar(
                np.array([spacy_model.vocab.get_vector(c_term.label)]), n=10
            )
            most_similar_words = tuple(
                zip(
                    most_similar_words[0][0],
                    most_similar_words[1][0],
                    most_similar_words[2][0],
                )
            )
            for word_key, _, similarity_score in most_similar_words:
                if similarity_score > self.threshold:
                    synonyms.add(spacy_model.vocab.strings[word_key])
                else:
                    break
        else:
            logger.info(
                "%{c_term.label} has no vector, semantic enrichment can't be executed."
            )
        if len(synonyms) > 0:
            if c_term.enrichment is None:
                c_term.enrichment = Enrichment()
            c_term.enrichment.add_synonyms(synonyms)

    def run(self, pipeline: Any) -> None:
        """Method responsible for the component execution.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline running.
        """
        if not (pipeline.spacy_model.vocab.has_vector("test")):
            logger.error(
                """No vectors loaded with the spaCy model. 
                Consider use another model or another enrichment component."""
            )
        else:
            for c_term in pipeline.candidate_terms:
                self.enrich_term(c_term, pipeline.spacy_model)
