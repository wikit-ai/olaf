import os.path
import os
import unittest

from axiom_extraction.axiom_extraction_methods.owl_axiom_induction import OWLConceptRestriction
from commons.ontology_learning_repository import load_KR_from_text
from config.core import DATA_PATH


class TestOWLConceptRestriction(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        self.kr = load_KR_from_text(kr_file=os.path.join(
            DATA_PATH, "demo_data", "axiom_extraction_test_kr.txt"))

        self.options = {
            "owl_onto_saving_file": os.path.join(DATA_PATH, "data_files", "axiom_extraction_test_valid.owl"),
            "reasoner": "ELK",
            "java_exe": "C:\\Program Files\\Common Files\\Oracle\\Java\\javapath\\java.exe",
            "robot_jar": "C:\\Users\\msesboue\\Documents\\robot\\robot.jar"
        }

        self.owl_concept_restriction = OWLConceptRestriction(
            self.kr, self.options)

        self.owl_concept_restriction._create_owl_file()
        self.robot_output = self.owl_concept_restriction._check_consistency()

    def test_create_owl_file(self) -> None:
        self.owl_concept_restriction._create_owl_file()

        self.assertTrue(os.path.exists(self.owl_concept_restriction.temp_file))

    def test_check_consistency(self) -> None:
        output = self.owl_concept_restriction._check_consistency()

        self.assertIn("unsatisfiable", output.split())

    def test_get_concept_ids_from_error_output(self) -> None:
        bad_concept_ids = self.owl_concept_restriction._get_concept_ids_from_error_output(
            self.robot_output)

        expected_bad_concept_ids = {
            "00be1e6f-6bc4-4458-b65a-71a5f12d424e",
            "46bf762b-c0be-4f7c-adaf-a3e2812ae914"
        }

        self.assertSetEqual(bad_concept_ids, expected_bad_concept_ids)

    def test_update_kr_fix_unsatisfiable_concept_axioms(self) -> None:
        pass

    def test_create_owl_concept_restriction_file(self) -> None:
        self.owl_concept_restriction.create_owl_concept_restriction_file()

        self.assertTrue(os.path.exists(
            self.options.get("owl_onto_saving_file")))

    def tearDown(self) -> None:
        if os.path.exists(self.owl_concept_restriction.temp_file):
            os.remove(self.owl_concept_restriction.temp_file)


if __name__ == '__main__':
    unittest.main()
