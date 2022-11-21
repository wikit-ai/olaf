from dataclasses import dataclass

@dataclass
class RepresentativeTerm:
    """Dataclass which represents the most representative string of a given concept.

    Parameters
    ----------
    value : str
        String most representative of a concept.
    concept_id : str
        UID of the given concept.
    """
    value: str
    concept_id: str

    def __hash__(self):
        return hash((self.value, self.concept_id))