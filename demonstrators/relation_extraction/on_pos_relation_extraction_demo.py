import spacy
import uuid

from commons.ontology_learning_schema import Concept, KR
from relation_extraction.relation_extraction_service import RelationExtraction

def main() -> None:

    corpus = [
        "Les propriétaires d’équipe gèrent certains paramètres de l’organisation ainsi que ses utilisateurs.",
        "Ils ajoutent et suppriment des membres, ajoutent des invités, modifient les paramètres d’une équipe et gèrent les tâches administratives.",
        "Les membres d'équipe peuvent discuter avec les autres utilisateurs de l'organisation.",
        "Les invités sont des personnes extérieures à votre organisation qu’un propriétaire d’équipe invite, par exemple des partenaires ou des consultants, à rejoindre le groupe.",
        "Les invités ont moins de fonctionnalités que les membres d’équipe ou les propriétaires d’équipe, mais ils peuvent cependant faire beaucoup de choses.",
        "Seul le propriétaire supprime des membres d'un groupe. Seul le propriétaire supprime des membres d'un groupe."
    ]

    corpus_preprocessed = []
    spacy_model = spacy.load("fr_core_news_md")
    for spacy_document in spacy_model.pipe(corpus):
        corpus_preprocessed.append(spacy_document)

    kr = KR()
    kr.concepts.add(Concept(str(uuid.uuid4()), {"équipe", "organisation", "groupe"}))
    kr.concepts.add(Concept(str(uuid.uuid4()), {"paramètre", "fonctionnalité"}))
    kr.concepts.add(Concept(str(uuid.uuid4()), {"invité"}))
    kr.concepts.add(Concept(str(uuid.uuid4()), {"utilisateur", "membre"}))
    kr.concepts.add(Concept(str(uuid.uuid4()), {"propriétaire"}))

    relation_extraction = RelationExtraction(corpus_preprocessed, kr)
    relation_extraction.on_pos_relation_extraction()

    print("\n\nKnowledge representation with relations extracted : \n")
    print(relation_extraction.kr)

if __name__ == "__main__" :
    main()