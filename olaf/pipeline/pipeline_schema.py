from typing import List, Optional

import spacy

from ..commons.errors import PipelineCorpusInitialisationError
from ..data_container.knowledge_representation_schema import \
    KnowledgeRepresentation
from .data_preprocessing.data_preprocessing_schema import \
    DataPreprocessing
from .pipeline_component.pipeline_component_schema import \
    PipelineComponent
from ..repository.corpus_loader.corpus_loader_schema import CorpusLoader


class Pipeline():
    """A Pipeline is the library main class. It orchestrates the pipeline starting 
    from raw texts to build the final knowledge representation.

    The corpus loader is responsible for the conversion for raw text to spacy document.
    We separate data preprocessing to explicitly enable pipelines without preprocessing.

    Parameters
    ----------
    spacy_model: spacy.language.Language
        The spacy model used to represent text corpus.
    pipeline_components: List[PipelineComponent]
        The ontology learning pipeline components that build the knowledge representation from the corpus.
    preprocessing_components: List[DataPreprocessing]
        The pipeline components specific to preprocessing.
    corpus_loader: CorpusLoader
        The component that loads the text corpus in the format used by the framework, i.e., a List[spacy.tokens.doc.Doc].
    corpus: List[spacy.tokens.doc.Doc]
        The preprocessed corpus the knowledge representation is built from.
    kr: KnowledgeRepresentation
        The knowledge extracted from the corpus.
    candidate_terms: Set[CandidateTerms]
        The candidate terms extracted and processed to create concept and relations.
    """

    def __init__(self,
                 spacy_model: spacy.language.Language,
                 pipeline_components: Optional[List[PipelineComponent]] = None,
                 preprocessing_components: Optional[List[DataPreprocessing]] = None,
                 corpus_loader: Optional[CorpusLoader] = None,
                 corpus: Optional[List[spacy.tokens.doc.Doc]] = None,
                 seed_kr: Optional[KnowledgeRepresentation] = None
                 ) -> None:
        """Initialise Pipeline instance.

        Parameters
        ----------
        spacy_model: spacy.language.Language
            The spacy model used to represent text corpus.
        pipeline_components: Optional[List[PipelineComponent]] 
            The ontology learning pipeline components that build the knowledge representation from the corpus.
        preprocessing_components: Optional[List[DataPreprocessing]]
            The pipeline components specific to preprocessing.
        corpus_loader: CorpusLoader
            The component that loads the text corpus in the format used by the framework, i.e., a List[spacy.tokens.doc.Doc].
        corpus: Optional[List[spacy.tokens.doc.Doc]]
            The preprocessed corpus the knowledge representation is built from.
        seed_kr: Optional[KnowledgeRepresentation]
            An initial knowledge representation to work with.
        """
        self.pipeline_components = pipeline_components
        self.preprocessing_components = preprocessing_components
        self.spacy_model = spacy_model
        self.corpus_loader = corpus_loader
        self.corpus = corpus
        self.kr = seed_kr
        self.candidate_terms = set()

        if self.preprocessing_components is None:
            self.preprocessing_components = []
        
        if self.pipeline_components is None :
            self.pipeline_components = []

        if self.corpus is None:
            if self.corpus_loader is None:
                raise PipelineCorpusInitialisationError
            else:
                self.corpus = self.corpus_loader(self.spacy_model)

        if self.kr is None:
            self.kr = KnowledgeRepresentation()

    def build(self) -> None:
        """Effectively build the pipeline, making the instance runnable.
            This method check each components and the constrained order.
        """
        # TODO : Check that the order of the pipeline components is valid.

        for component in self.pipeline_components : 
            component._check_resources()

    def add_preprocessing_component(self, preprocessing_component: DataPreprocessing) -> None:
        """Add a preprocessing component to the pipeline.

        Parameters
        ----------
        preprocessing_component : DataPreprocessing
            The preprocessing pipeline component to add.
        """
        self.preprocessing_components.append(preprocessing_component)

    def remove_preprocessing_component(self, preprocessing_component: DataPreprocessing) -> None:
        """Remove a preprocessing component from the pipeline.

        Parameters
        ----------
        preprocessing_component : DataPreprocessing
            The preprocessing pipeline component to remove.
        """
        self.preprocessing_components.remove(preprocessing_component)

    def add_pipeline_component(self, pipeline_component: PipelineComponent) -> None:
        """Add a component to the pipeline.

        Parameters
        ----------
        pipeline_component : PipelineComponent
            The pipeline component to add.
        """
        self.pipeline_components.append(pipeline_component)

    def remove_pipeline_component(self, pipeline_component: PipelineComponent) -> None:
        """Remove a component from the pipeline.

        Parameters
        ----------
        pipeline_component : PipelineComponent
            The pipeline component to remove.
        """
        self.pipeline_components.remove(pipeline_component)

    def run(self) -> None:
        """Run the pipeline. The method hence run each pipeline components in 
            the determined order filling the Knowledge Representation.
        """
        for component in self.preprocessing_components:
            component.run(self)
            
        for component in self.pipeline_components :
            component.run(self)
