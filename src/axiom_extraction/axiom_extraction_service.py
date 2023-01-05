from typing import Any, Dict

from axiom_extraction.axiom_extraction_methods.owl_axiom_induction import OWLConceptRestriction
from commons.ontology_learning_schema import KR
from config.core import config


class AxiomExtraction:

    def __init__(self, kr: KR, configuration: Dict[str, Any] = None) -> None:
        self.kr = kr
        if configuration is None:
            self.config = config['axiom_extraction']
        else:
            self.config = configuration

    def owl_concept_restriction(self) -> None:
        """Create the OWL file containing the valid axiomatised Knowledge Representation instance.

            The axiom hypothesis are:
            - When a relation or a meta relation (other than hierarchical) between a source and a destination concept is found, we hypothesise that a 
            subset of the source concept instances are bound to an instance of the destination concept by the latter relation. 
            This is modelled in OWL by having an OWL SomeValuesFrom property restriction on the relation and the destination concept as a 
            subclass of the source concept class.
            - When a relation or a meta relation (other than hierarchical, or related to) is found between two concepts, the two concepts are disjoint.
        """
        owl_concept_restriction = OWLConceptRestriction(
            self.kr, self.config.get("owl_restriction_on_concepts"))
        owl_concept_restriction.create_owl_concept_restriction_file()
