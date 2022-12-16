import spacy
from timeit import default_timer

from commons.ontology_learning_schema import Concept, KR
from relation_extraction.relation_extraction_service import RelationExtraction


def main() -> None:

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

    kr = KR()
    kr.concepts.add(Concept("id_c_equipe", {"équipe", "équipes"}))
    kr.concepts.add(Concept("id_c_org", {"organisation", "organisations"}))
    kr.concepts.add(Concept("id_c_utilisateur", {
                    "utilisateur", "membre", "utilisateurs", "membres"}))
    kr.concepts.add(Concept("id_c_proprio", {"propriétaire", "propriétaires"}))
    kr.concepts.add(Concept("id_c_invite", {"invités", "invité"}))
    kr.concepts.add(
        Concept("id_c_fonct", {"fonctionnalité", "fonctionnalités"}))

    relation_extraction = RelationExtraction(
        corpus=corpus_preprocessed,
        kr=kr,
        configuration={
            "on_occurrence_with_sep_term": {
                "term_relation_map": {
                    "ses": "is_part_of",
                    "ont": "is_part_of",
                },
                "use_lemma": False,
                "cooc_scope": "sentence",
                "cooc_treshold": 0,
                "concept_distance_limit": 1000
            }
        }
    )

    relation_extraction.on_cooc_with_sep_term_map_meta_rel_extraction(
        spacy_nlp=spacy_model)

    print("\n\nKnowledge representation with related_to meta relation extracted : \n")

    print(
        "==============================================================================")
    print(
        "=======                  CONCEPTS                        =====================")
    print(
        "==============================================================================")

    for concept in kr.concepts:
        print(str(concept))

    print("\n")
    print(
        "==============================================================================")
    print(
        "=======                  META RELATIONS                  =====================")
    print(
        "==============================================================================")

    for relation in kr.meta_relations:
        if relation.relation_type != "related_to":
            print(str(relation))


if __name__ == "__main__":
    start = default_timer()
    main()
    end = default_timer()
    print(f"Processing Time: {end - start}")
