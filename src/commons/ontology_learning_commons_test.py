import os.path
import unittest

from axiom_extraction.axiom_extraction_methods.owl_axiom_induction import OWLconceptRestriction
from commons.ontology_learning_repository import load_KR_from_text
from config.core import DATA_PATH


class TestConceptNetTermEnrichment(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        pass

    def test_load_KR_from_text(self) -> None:
        kr = load_KR_from_text(kr_file=os.path.join(
            DATA_PATH, "demo_data", "axiom_extraction_test_kr.txt"))

        kr_concept_ids = {concept.uid for concept in kr.concepts}
        kr_meta_realtions_ids = {
            meta_relation.uid for meta_relation in kr.meta_relations}

        self.assertEqual(len(kr.relations), 0)
        self.assertIn("979f8b5c-a4ba-4a43-8ced-490dd6282bd8", kr_concept_ids)
        self.assertIn("d50095de-88db-46c6-ab57-204b32a416b6",
                      kr_meta_realtions_ids)


if __name__ == '__main__':
    unittest.main()
