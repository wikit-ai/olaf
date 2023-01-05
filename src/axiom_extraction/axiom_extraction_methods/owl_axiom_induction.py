import os.path
import subprocess
from typing import Any, Dict, Set
import urllib.parse

from commons.ontology_learning_repository import KR2OWL_restriction_on_concepts
from commons.ontology_learning_schema import KR
from config.core import DATA_PATH
import config.logging_config as logging_config


class OWLConceptRestriction:
    """A class to make axiom asumptions and validate them by checking the knowledge representation consistency with OWL reasoning.
        The axiom hypothesis are:
        - When a relation or a meta relation (other than hierarchical) between a source and a destination concept is found, we hypothesise that a 
        subset of the source concept instances are bound to an instance of the destination concept by the latter relation. 
        This is modelled in OWL by having an OWL SomeValuesFrom property restriction on the relation and the destination concept as a 
        subclass of the source concept class.
        - When a relation or a meta relation (other than hierarchical, or related to) is found between two concepts, the two concepts are disjoint.

        Attributes
        ----------
        kr : KR
            The Knowledge Representation instance.
        options : Dict[str, Any]
            The options for the OWLConceptRestriction process
        temp_file : str
            The file path to store intermediate OWL ontologies.
        owl_onto_saving_path : str
            The file path to store the final axiomatised and consistent OWL ontology.
    """

    def __init__(self, kr: KR, options: Dict[str, Any]) -> None:
        self.options = options
        self.kr = kr
        self.temp_file = os.path.join(
            DATA_PATH, "temp_axiom_extraction.owl")
        self.owl_onto_saving_path = self.options.get(
            "owl_onto_saving_path ", DATA_PATH)

        try:
            assert self.options.get("reasoner", "ELK") in [
                "ELK", "jfact", "hermit", "whelk", "emr", "structural"]
            assert self.options["java_exe"] is not None
            assert self.options["robot_jar"] is not None
            assert self.options["robot_jar"].split(".")[-1] == "jar"
            assert self.options["owl_onto_saving_file"] is not None
        except AssertionError as e:
            logging_config.logger.error(
                f"""Config information missing or wrong for axiom extraction OWL restriction on concepts. Make sure you provided the right configuration fields:
                    - axiom_extraction.owl_restriction_on_concepts.reasoner is one of ["ELK", "jfact", "hermit", "whelk", "emr", "structural"]
                    - axiom_extraction.owl_restriction_on_concepts.java_exe is a valid path to the java executable
                    - axiom_extraction.owl_restriction_on_concepts.robot_jar is a valid path to the robot .jar file.
                    Trace : {e}
                """)

    def _create_owl_file(self) -> None:
        """Create the OWL ontology file with the specific axioms.
        """
        KR2OWL_restriction_on_concepts(
            self.kr, format="xml", saving_file=self.temp_file)

    def _check_consistency(self) -> str:
        """Check the generated OWL ontology consistency.
            To validate the ontology, we use the robot CLI (http://robot.obolibrary.org/reason).
            Java must be installed on the local machine along with the robot.jar file.

        Returns
        -------
        str
            The checking process return message.
        """
        error_output = ""

        robot_command = [
            self.options.get("java_exe"),
            "-jar",
            self.options.get("robot_jar"),
            "reason",
            "--reasoner",
            self.options.get("reasoner"),
            "--annotate-inferred-axioms",
            "true",
            "--input",
            self.temp_file,
            "--output",
            self.temp_file
        ]

        try:
            output = subprocess.check_output(
                robot_command, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            error_output = e.output.decode()

        return error_output

    def _get_concept_ids_from_error_output(self, error_output: str) -> Set[str]:
        """Extract the concept IDs from the robot CLI program error message.

        Parameters
        ----------
        error_output : str
            The robot CLI program error message

        Returns
        -------
        Set[str]
            The extracted concept IDs
        """
        splitted_output = error_output.split()
        unsatisfiable_concept_ids = set()

        for idx, txt in enumerate(splitted_output):
            if "unsatisfiable:" in txt:
                concept_uri = splitted_output[idx + 1]
                concept_id = urllib.parse.unquote(concept_uri).split("#")[-1]
                unsatisfiable_concept_ids.add(concept_id)

        return unsatisfiable_concept_ids

    def _update_kr_fix_unsatisfiable_concept_axioms(self, unsatisfiable_concept_ids: Set[str]) -> None:
        """Update the Knowledge Representation instance to remove inconsistencies.
            The strategie adopted here is to remove all non-hierarchical relations 
            involving the concepts leading to inconsistencies.

        Parameters
        ----------
        unsatisfiable_concept_ids : Set[str]
            The set of concept IDs of the concepts identified as generating the inconsistencies.
        """
        new_relations = set()
        new_meta_relations = set()

        for relation in self.kr.relations:
            conditions = [
                relation.source_concept_id in unsatisfiable_concept_ids,
                relation.destination_concept_id in unsatisfiable_concept_ids
            ]
            if not any(conditions):
                new_relations.add(relation)

        for meta_relation in self.kr.meta_relations:
            conditions = [
                meta_relation.source_concept_id in unsatisfiable_concept_ids,
                meta_relation.destination_concept_id in unsatisfiable_concept_ids
            ]
            if not any(conditions):
                new_meta_relations.add(meta_relation)

        self.kr.relations = new_relations
        self.kr.meta_relations = new_meta_relations

    def create_owl_concept_restriction_file(self) -> None:
        """Create the OWL file containing the valid axiomatised Knowledge Representation instance.
        """
        reasoner_output = "start testing"

        while reasoner_output != "":

            self._create_owl_file()
            reasoner_output = self._check_consistency()

            if reasoner_output != "":
                unsatisfiable_concept_ids = self._get_concept_ids_from_error_output(
                    reasoner_output)
                self._update_kr_fix_unsatisfiable_concept_axioms(
                    unsatisfiable_concept_ids)

        KR2OWL_restriction_on_concepts(
            self.kr, format="xml", saving_file=self.options.get("owl_onto_saving_file"))

        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
