from typing import List, Set
import uuid

from commons.ontology_learning_schema import CandidateTerm, Concept, KR


class GroupBySynonyms():

    def __init__(self, candidate_terms: List[CandidateTerm], kr: KR) -> None:
        """Initialisation.

        Parameters
        ----------
        candidate_terms : List[CandidateTerm]
            List of candidate ters to study.
        kr : KR
            The knowledge representation.
        """
        self.candidate_terms = candidate_terms
        self.kr = kr

    def __call__(self) -> None:
        """The method directly update the knowledge representation.
        """
        self.group_terms_by_synonyms()

    def _find_cterms_with_particular_synonym(self, synonym: str, candidate_terms: List[CandidateTerm]) -> Set[CandidateTerm]:
        """Find candidate terms that contain the same value or synonym as the input term value.

        Parameters
        ----------
        synonym : str
            String searched in the candidate terms.
        candidate_terms : List[CandidateTerm]
            List of candidate terms to analyze.

        Returns
        -------
        Set[CandidateTerm]
            Set of candidate terms with values or synonyms that contains a particular term.
        """
        candidates_of_same_concept = set()
        candidates_from_value = set(
            filter(lambda candidate: candidate.value == synonym, candidate_terms))
        candidates_from_syn = set(
            filter(lambda candidate: synonym in candidate.synonyms, candidate_terms))
        candidates_of_same_concept.update(candidates_from_value)
        candidates_of_same_concept.update(candidates_from_syn)
        return candidates_of_same_concept

    def _find_cterms_with_shared_synonyms(self, current_candidate: CandidateTerm, candidate_terms: List[CandidateTerm]) -> Set[CandidateTerm]:
        """Find candidate terms that contain the same value or synonym as the input candidate.

        Parameters
        ----------
        current_candidate : CandidateTerm
            Candidate term under study.
        candidate_terms : List[CandidateTerm]
            List of candidate terms to analyze.

        Returns
        -------
        Set[CandidateTerm]
            Set of candidate terms with values or synonyms that contains common values with the term under study.
        """
        candidates_of_same_concept = set()
        candidates_of_same_concept.update(self._find_cterms_with_particular_synonym(
            current_candidate.value, candidate_terms))
        for syn in current_candidate.synonyms:
            candidates_of_same_concept.update(
                self._find_cterms_with_particular_synonym(syn, candidate_terms))
        return candidates_of_same_concept

    def _build_concept(self, candidates_of_same_concept: Set[CandidateTerm]) -> None:
        """Create a new concept based on candidate terms and add it on the knowledge representation.

        Parameters
        ----------
        candidates_of_same_concept : Set[CandidateTerm]
            Candidate terms to merge in a unique concept.
        """
        new_concept = Concept(str(uuid.uuid4()))
        for candidate in candidates_of_same_concept:
            new_concept.terms.add(candidate.value)
            new_concept.terms.update(candidate.synonyms)
        self.kr.concepts.add(new_concept)

    def group_terms_by_synonyms(self) -> None:
        """Go throught list of candidate terms and merged candidates with common value or synonyms into concepts.
        """
        index_remaining_candidates = list(range(len(self.candidate_terms)))
        while not (len(index_remaining_candidates) == 0):
            candidate_of_same_concepts = set()
            current_term_index = index_remaining_candidates[0]
            current_term = self.candidate_terms[current_term_index]
            candidate_of_same_concepts.add(current_term)
            index_remaining_candidates.remove(current_term_index)
            other_candidates = [self.candidate_terms[i]
                                for i in index_remaining_candidates]
            candidates_proposal = set()
            candidates_proposal.update(
                self._find_cterms_with_shared_synonyms(current_term, other_candidates))
            index_candidates_proposal = [self.candidate_terms.index(
                candidate) for candidate in candidates_proposal]
            while not (len(index_candidates_proposal) == 0):
                index_looking_candidate = index_candidates_proposal[0]
                looking_candidate = self.candidate_terms[index_looking_candidate]
                index_candidates_proposal.remove(index_looking_candidate)
                index_remaining_candidates.remove(index_looking_candidate)
                candidate_of_same_concepts.add(looking_candidate)
                new_candidates = self._find_cterms_with_shared_synonyms(
                    looking_candidate, [self.candidate_terms[i] for i in index_remaining_candidates])
                for candidate in new_candidates:
                    if not (self.candidate_terms.index(candidate) in index_candidates_proposal):
                        index_candidates_proposal.append(
                            self.candidate_terms.index(candidate))
                candidates_proposal.update(new_candidates)

            self._build_concept(candidate_of_same_concepts)
