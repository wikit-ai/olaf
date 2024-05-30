import os
import re
import subprocess
from typing import Any, Callable, Dict, Optional, Set, Union, TYPE_CHECKING

from rdflib import Graph, URIRef


if TYPE_CHECKING:
    from ...pipeline_schema import Pipeline
from ....commons.errors import MissingEnvironmentVariable
from ....commons.kr_to_rdf_tools import (
    all_individuals_different,
    concept_lrs_to_owl_individuals,
    owl_class_uri,
    owl_obj_prop_uri,
)
from ....commons.logging_config import logger
from ....data_container import KnowledgeRepresentation
from ....data_container.metarelation_schema import METARELATION_RDFS_OWL_MAP
from ..pipeline_component_schema import PipelineComponent


class OWLAxiomExtraction(PipelineComponent):
    """The OWL axiom extraction component inductively construct OWL axioms from the knowledge
    representation based on some OWL ontology modeling patterns.

    The component generates the OWL ontology based on the provided patterns, test its semantic
    consistence with an OWL reasoner and remove axioms until the ontology is consistent.

    This component requires the environment variables:
    - "JAVA_EXE"
    - "ROBOT_JAR"

    Attributes
    ----------
    owl_axiom_generators : Set[Callable[[KnowledgeRepresentation, URIRef], Graph]]
        The function to generate the OWL axioms.
    base_uri : Union[str, URIRef], optional
        The base URI to use when creating the concepts and relations URIs,
        by default "http://www.ms2.org/o/example#".
    reasoner : str, optional
        The reasoner to use, by default "ELK".
        Reasoning is performed using the ROBOT CLI: <https://robot.obolibrary.org/reason>.
        Hence, possible values are: "hermit", "jfact", "whelk", "emr", "structural"
    java_exe: str
        Environment variable path to the Java executable.
    robot_jar: str
        Environment variable path to the ROBOT CLI .jar file.
    graph_to_test_temp_file: PathLike
        Temporary file path to store intermediate RDF graphs for consistency check.
    tested_graph_temp_file: PathLike
        Temporary file path for the inferred triples.
    _pattern_nb_classes: str
        Regex pattern to match the number of unsatisfiable classes in the ROBOT CLI error message.
    _pattern_unsatisfiable_classes: str
        Regex pattern to match the unsatisfiable classes URIs in the ROBOT CLI error message.
    individuals_axiom_generators: Set[Set[Callable[[KnowledgeRepresentation, URIRef], Graph]]]
        A set of the possible OWL axiom generators creating OWL named individuals.
    """

    def __init__(
        self,
        owl_axiom_generators: Set[Callable[[KnowledgeRepresentation, URIRef], Graph]],
        base_uri: Optional[Union[str, URIRef]] = None,
        reasoner: Optional[str] = "ELK",
    ) -> None:
        """Initialiser for the OWL axiom extraction component.

        Parameters
        ----------
        owl_axiom_generators : Set[Callable[[KnowledgeRepresentation, URIRef], Graph]]
            The function to generate the OWL axioms.
        base_uri : Union[str, URIRef], optional
            The base URI to use when creating the concepts and relations URIs,
            by default "http://www.ms2.org/o/example#".
        reasoner : str, optional
            The reasoner to use, by default "ELK".
            Reasoning is performed using the ROBOT CLI: <https://robot.obolibrary.org/reason>.
            Hence, possible values are: "ELK", "hermit", "jfact", "whelk", "emr", "structural".
        """

        self.owl_axiom_generators = owl_axiom_generators

        if base_uri is None:
            self.base_uri = URIRef("http://www.ms2.org/o/example#")
            logger.warning(
                """No value given for base_uri parameter, default will be set to http://www.ms2.org/o/example#.
                """
            )
        elif isinstance(base_uri, URIRef):
            self.base_uri = base_uri
        else:
            self.base_uri = URIRef(base_uri)

        self.reasoner = reasoner

        self.java_exe = os.getenv("JAVA_EXE")
        self.robot_jar = os.getenv("ROBOT_JAR")

        self.graph_to_test_temp_file = os.path.join(
            os.getenv("DATA_PATH"), "kr_owl_to_check.owl"
        )
        self.tested_graph_temp_file = os.path.join(
            os.getenv("DATA_PATH"), "kr_owl_consistency_check.owl"
        )

        self.check_resources()

        self._pattern_nb_classes = re.compile(
            "There are (?P<nb_classes>\\d+) unsatisfiable classes in the ontology\\."
        )
        self._pattern_unsatisfiable_classes = re.compile(
            "unsatisfiable: (?P<class_uri>.+)\\n"
        )

        self.individuals_axiom_generators = {
            concept_lrs_to_owl_individuals,
            all_individuals_different,
        }

    def check_resources(self) -> None:
        """Method to check that the component has access to all its required resources.

        The OWL axiom extraction component requires the environment variables:
        - "JAVA_EXE"
        - "ROBOT_JAR"
        """
        if not self.java_exe:
            raise MissingEnvironmentVariable(
                component_name="OWLaxiomExtraction", env_var_name="JAVA_EXE"
            )

        if not self.robot_jar:
            raise MissingEnvironmentVariable(
                component_name="OWLaxiomExtraction", env_var_name="ROBOT_JAR"
            )

    def optimise(
        self, validation_terms: Set[str], option_values_map: Set[float]
    ) -> None:
        # TODO: how far in the grid search do we go?
        # scikitlearn grid search
        # default to grid search and log a warning
        # enable user defined optimisation function alternative
        """A method to optimise the pipeline component by tuning the options."""
        raise NotImplementedError

    def _compute_metrics(self) -> None:
        """A method to compute component performance metrics.
        It is used by the optimise method to update the options.
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

    def build_graph_without_owl_instances(self, kr: KnowledgeRepresentation) -> Graph:
        """Build the RDF graph based on the provided OWL axiom generators omitting
        the ones creating OWL named instances.

        Parameters
        ----------
        kr : KnowledgeRepresentation
            _description_

        Returns
        -------
        Graph
            _description_
        """
        graph = Graph()

        for axiom_generator in self.owl_axiom_generators:
            if axiom_generator not in self.individuals_axiom_generators:
                graph += axiom_generator(kr, self.base_uri)

        return graph

    def build_full_graph(self, kr: KnowledgeRepresentation) -> Graph:
        """Build the complete RDF graph based on the provided OWL axiom generators.

        Parameters
        ----------
        kr : KnowledgeRepresentation
            The knowledge representation to use when build the graph.

        Returns
        -------
        Graph
            The constructed RDF graph.
        """
        full_graph = Graph()

        full_graph = self.build_graph_without_owl_instances(kr)

        for axiom_generator in self.owl_axiom_generators:
            if axiom_generator in self.individuals_axiom_generators:
                full_graph += axiom_generator(kr, self.base_uri)

        return full_graph

    def _check_owl_graph_consistency(self, graph: Graph) -> Union[str, None]:
        """Check the generated OWL ontology consistency.
            To validate the ontology, we use the robot CLI (http://robot.obolibrary.org/reason).
            Java must be installed on the local machine along with the robot.jar file.

        Parameters
        ----------
        graph: Graph
            The RDF graph to check.

        Returns
        -------
        Union[str, None]
            The ROBOT CLI error message if any, else None.
        """

        graph.serialize(destination=self.graph_to_test_temp_file, format="xml")

        robot_command = [
            self.java_exe,
            "-jar",
            self.robot_jar,
            "reason",
            "--reasoner",
            self.reasoner,
            "--input",
            self.graph_to_test_temp_file,
            "--output",
            self.tested_graph_temp_file,
        ]

        error_output = None

        try:
            _ = subprocess.check_output(robot_command, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            error_output = e.output.decode()

        return error_output

    def _get_concept_uris_from_error_output(self, error_output: str) -> Set[str]:
        """Extract the concept URIs from the robot CLI error message.

        Parameters
        ----------
        error_output : str
            The robot CLI error message

        Returns
        -------
        Set[str]
            The extracted concept URIs.
        """

        nb_unsatisfiable_concepts = int(
            self._pattern_nb_classes.search(error_output).group("nb_classes")
        )
        unsatisfiable_concept_uris = set(
            self._pattern_unsatisfiable_classes.findall(error_output)
        )

        if len(unsatisfiable_concept_uris) != nb_unsatisfiable_concepts:
            logger.warning(
                """OWL axiom extraction: Something might have gone wrong while extracting unsatisfiable concept IDs.
                                We found %i unsatisfiable concepts but only got %i concept IDs.
                            """,
                nb_unsatisfiable_concepts,
                len(unsatisfiable_concept_uris),
            )

        return unsatisfiable_concept_uris

    def _update_unsatisfiable_kr_owl_graph(
        self, kr: KnowledgeRepresentation, unsatisfiable_concept_uris: Set[str]
    ) -> Graph:
        """Update the Knowledge Representation OWL graph inconsistencies.
            The strategy adopted here is to skip OWL axioms generation for concepts leading
            to unsatisfiable classes.

        Parameters
        ----------
        unsatisfiable_concept_uris : Set[str]
            The concept URIs of the concepts identified as generating the unsatisfiable classes.
        """
        new_relations = set()
        new_meta_relations = set()
        new_concepts = set()

        for relation in kr.relations:
            # Potential danger there:
            # URIs should be generated exactly the same as it is done in OWl axiom generators
            source_concept_uri = (
                str(
                    owl_class_uri(
                        label=relation.source_concept.label, base_uri=self.base_uri
                    )
                )
                if relation.source_concept
                else None
            )
            dest_concept_uri = (
                str(
                    owl_class_uri(
                        label=relation.destination_concept.label, base_uri=self.base_uri
                    )
                )
                if relation.destination_concept
                else None
            )

            conditions = [
                source_concept_uri in unsatisfiable_concept_uris,
                dest_concept_uri in unsatisfiable_concept_uris,
            ]
            if not any(conditions):
                new_relations.add(relation)
                if relation.source_concept:
                    new_concepts.add(relation.source_concept)
                if relation.destination_concept:
                    new_concepts.add(relation.destination_concept)

        for relation in kr.metarelations:
            # Potential danger there:
            # URIs should be generated exactly the same as it is done in OWl axiom generators
            source_concept_uri = str(
                owl_class_uri(
                    label=relation.source_concept.label, base_uri=self.base_uri
                )
            )
            dest_concept_uri = str(
                owl_class_uri(
                    label=relation.destination_concept.label, base_uri=self.base_uri
                )
            )

            conditions = [
                source_concept_uri in unsatisfiable_concept_uris,
                dest_concept_uri in unsatisfiable_concept_uris,
            ]
            if not any(conditions):
                new_meta_relations.add(relation)
                new_concepts.add(relation.source_concept)
                new_concepts.add(relation.destination_concept)

        kr_for_owl_graph = KnowledgeRepresentation(
            concepts=kr.concepts.union(new_concepts),
            relations=new_relations,
            metarelations=new_meta_relations,
        )

        updated_kr_owl_graph = self.build_full_graph(kr=kr_for_owl_graph)

        return updated_kr_owl_graph

    def _update_kr_external_uris(self, kr: KnowledgeRepresentation) -> None:
        """Update the knowledge representation to add the created RDF graph concept
        and relations URIs.

        Parameters
        ----------
        kr : KnowledgeRepresentation
            The knowledge representation to update.
        """
        for concept in kr.concepts:
            concept.external_uids.add(
                str(owl_class_uri(label=concept.label, base_uri=self.base_uri))
            )

        for relation in kr.relations:
            relation.external_uids.add(
                str(owl_obj_prop_uri(label=relation.label, base_uri=self.base_uri))
            )

        for relation in kr.metarelations:
            metarel_uri = METARELATION_RDFS_OWL_MAP.get(
                relation.label,
                str(owl_obj_prop_uri(label=relation.label, base_uri=self.base_uri)),
            )
            relation.external_uids.add(metarel_uri)

    def run(self, pipeline: "Pipeline") -> None:
        """Create the OWL file containing the valid axiomatised Knowledge Representation instance."""
        reasoner_output = "start testing"

        kr_owl_graph = self.build_full_graph(kr=pipeline.kr)

        kr_graph_trial = 0

        while (reasoner_output) and (kr_graph_trial < 5):

            kr_graph_trial += 1

            reasoner_output = self._check_owl_graph_consistency(graph=kr_owl_graph)

            if reasoner_output is not None:
                # An OWL ontology is either inconsistent or it contains some unsatisfiable classes.
                if "The ontology is inconsistent." in reasoner_output:
                    logger.warning(msg="Inconsistent ontology")
                    logger.warning(
                        """
                                    Reasoner output: 
                                   %s.
                                """,
                        reasoner_output,
                    )
                    # An inconsistent OWL ontology is often due to instances of unsatisfiable classes.
                    kr_owl_graph = self.build_graph_without_owl_instances(
                        kr=pipeline.kr
                    )

                elif "unsatisfiable classes in the ontology." in reasoner_output:
                    logger.warning(msg="Unsatisfiable ontology")
                    logger.warning(
                        """
                                    Reasoner output: 
                                   %s.
                                """,
                        reasoner_output,
                    )
                    unsatisfiable_concept_uris = (
                        self._get_concept_uris_from_error_output(reasoner_output)
                    )
                    kr_owl_graph = self._update_unsatisfiable_kr_owl_graph(
                        kr=pipeline.kr,
                        unsatisfiable_concept_uris=unsatisfiable_concept_uris,
                    )

        if kr_graph_trial == 5:
            logger.warning(
                msg="""
                        The OWL axiom extractor did not manage to make the ontology consistent.
                        We will create the full inconsistent ontology.
                    """
            )
            kr_owl_graph = self.build_full_graph(kr=pipeline.kr)
        else:
            logger.info("The constructed ontology is consistent.")

        self._update_kr_external_uris(kr=pipeline.kr)
        pipeline.kr.rdf_graph = kr_owl_graph

        if os.path.exists(self.graph_to_test_temp_file):
            os.remove(self.graph_to_test_temp_file)

        if os.path.exists(self.tested_graph_temp_file):
            os.remove(self.tested_graph_temp_file)
