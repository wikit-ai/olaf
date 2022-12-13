import spacy
import uuid

from commons.ontology_learning_schema import Concept, KR
from concept_hierarchy.concept_hierarchy_service import TermSubsumption

corpus = [
    "Les propriétaires d’équipe gèrent certains paramètres de l’organisation ainsi que ses utilisateurs.",
    "Ils ajoutent et suppriment des membres, ajoutent des invités, modifient les paramètres d’une équipe et gèrent les tâches administratives.",
    "Les membres d'équipe peuvent discuter avec les autres utilisateurs de l'organisation.",
    "Les invités sont des personnes extérieures à votre organisation qu’un propriétaire d’équipe invite, par exemple des partenaires ou des consultants, à rejoindre le groupe.",
    "Les invités ont moins de fonctionnalités que les membres d’équipe ou les propriétaires d’équipe, mais ils peuvent cependant faire beaucoup de choses."
]
corpus_preprocessed = []
spacy_model = spacy.load("fr_core_news_md")
for spacy_document in spacy_model.pipe(corpus):
    corpus_preprocessed.append(spacy_document)

options = {
    "algo_type": "MEAN",
    "use_span": False,
    "subsumption_threshold": 0.8,
    "mean_high_threshold": 0.6,
    "mean_low_threshold": 0.4,
    "use_lemma": True
}

kr = KR()
kr.concepts.add(Concept(str(uuid.uuid4()), {"équipe", "organisation"}))
kr.concepts.add(Concept(str(uuid.uuid4()), {"utilisateur", "membre"}))
kr.concepts.add(Concept(str(uuid.uuid4()), {"propriétaire"}))

print("\nKnowledge graph before concept hierarchisation : \n")
print(kr)

term_sub = TermSubsumption(corpus_preprocessed, kr, options)
term_sub.term_subsumtion()

print("\n\nKnowledge graph after concept hierarchisation : \n")

print(term_sub.kr)