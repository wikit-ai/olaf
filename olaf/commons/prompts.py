from typing import Dict, List


def openai_prompt_concept_term_extraction(context: str) -> List[Dict[str, str]]:
    """Prompt template for concept term extraction with ChatCompletion OpenAI model.

    Parameters
    ----------
    context: str
        The context to add in the prompt template.

    Returns
    -------
    List[Dict[str, str]]
        ChatCompletion prompt template.
    """
    prompt_template = [
        {
            "role": "system",
            "content": "You are an helpful assistant helping building an ontology.",
        },
        {
            "role": "user",
            "content": "Extract the most meaningful keywords of the following text. Keep only keywords that could be concepts and not relations. Write them as a python list of string with double quotes.",
        },
        {
            "role": "user",
            "content": 'Here is an example. Text: This python package is about ontology learning. I do not know a lot about this field.\n["python package", "ontology learning", "field"]',
        },
        {"role": "user", "content": f"Text: {context}"},
    ]
    return prompt_template


def hf_prompt_concept_term_extraction(context: str) -> str:
    """Prompt template for concept term extraction with Hugging Face inference API.

    Parameters
    ----------
    context: str
        The context to add in the prompt template.

    Returns
    -------
    str
        Completion prompt template.
    """
    prompt_template = f"""You are an helpful assistant helping building an ontology.
    Extract the most meaningful keywords of the following text. Keep only keywords that could be concepts and not relations. Write them as a python list of string with double quotes.
    Here is an example. Text: This python package is about ontology learning. I do not know a lot about this field.
    ["python package", "ontology learning", "field"]
    Text: {context}"""
    return prompt_template


def openai_prompt_relation_term_extraction(context: str) -> List[Dict[str, str]]:
    """Prompt template for relation term extraction with ChatCompletion OpenAI model.

    Parameters
    ----------
    context: str
        The context to add in the prompt template.

    Returns
    -------
    List[Dict[str, str]]
        ChatCompletion prompt template.
    """
    prompt_template = [
        {
            "role": "system",
            "content": "You are an helpful assistant helping building an ontology.",
        },
        {
            "role": "user",
            "content": "Extract the most meaningful words describing actions or states in the following text. Keep only words that could be relations and not concepts. Write them as a python list of string with double quotes.",
        },
        {
            "role": "user",
            "content": 'Here is an example. Text: I plan to eat pizza tonight. I am looking for advice.\n["plan", "eat", "looking for"]',
        },
        {"role": "user", "content": f"Text: {context}"},
    ]
    return prompt_template


def hf_prompt_relation_term_extraction(context: str) -> str:
    """Prompt template for relation term extraction with Hugging Face inference API.

    Parameters
    ----------
    context: str
        The context to add in the prompt template.

    Returns
    -------
    str
        Completion prompt template.
    """
    prompt_template = f"""You are an helpful assistant helping building an ontology.
    Extract the most meaningful words describing actions or states in the following text. Keep only words that could be relations and not concepts. Write them as a python list of string with double quotes.
    Here is an example. Text: I plan to eat pizza tonight. I am looking for advice.
    ["plan", "eat", "looking for"]
    Text: {context}"""
    return prompt_template


def openai_prompt_term_enrichment(context: str) -> List[Dict[str, str]]:
    """Prompt template for term enrichment with ChatCompletion OpenAI model.

    Parameters
    ----------
    context: str
        The context to add in the prompt template.

    Returns
    -------
    List[Dict[str, str]]
        ChatCompletion prompt template.
    """
    prompt_template = [
        {
            "role": "system",
            "content": "You are an helpful assistant helping building an ontology.",
        },
        {
            "role": "user",
            "content": 'Give synonyms, hypernyms, hyponyms and antonyms of the following term. The output should be in json format with "synonyms", "hypernyms", "hyponyms" and "antonyms" as keys and a list a string as values. ',
        },
        {
            "role": "user",
            "content": """Here is an example. Term : dog
            {"synonyms": ["hound", "mutt"], "hypernyms":["animal", "mammal", "canine"], "hyponyms": ["labrador", "dalmatian"],"antonyms": []}""",
        },
        {"role": "user", "content": f"Term: {context}"},
    ]
    return prompt_template


def hf_prompt_term_enrichment(context: str) -> str:
    """Prompt template for term enrichment with Hugging Face inference API.

    Parameters
    ----------
    context: str
        The context to add in the prompt template.

    Returns
    -------
    str
        Completion prompt template.
    """
    prompt_template = f"""You are an helpful assistant helping building an ontology.
    Give synonyms, hypernyms, hyponyms and antonyms of the given term. The output should be in json format with "synonyms", "hypernyms", "hyponyms" and "antonyms" as keys and a list a string as values.
    Do it only for one term, do not add other ones.
    Here is an example. Term : dog
    {{"synonyms": ["hound", "mutt"], "hypernyms":["animal", "mammal", "canine"], "hyponyms": ["labrador", "dalmatian"],"antonyms": []}}
    Term: {context}"""
    return prompt_template


def openai_prompt_concept_extraction(
    doc_context: str, ct_labels: str
) -> List[Dict[str, str]]:
    """Prompt template for concept extraction with ChatCompletion OpenAI model.

    Parameters
    ----------
    doc_context: str
        Extract of document contents to use as context.
    ct_labels: str
        The candidate terms to group into concepts.

    Returns
    -------
    List[Dict[str, str]]
        ChatCompletion prompt template.
    """
    prompt_template = [
        {
            "role": "system",
            "content": "You are an helpful assistant helping building an ontology.",
        },
        {
            "role": "user",
            "content": "Based on the context given, group together the words listed below where each group should correspond to one concept. The result should be given as a python list of list of string with double quotes.",
        },
        {"role": "user", "content": f"Context: {doc_context} \nWords : {ct_labels}"},
    ]
    return prompt_template


def hf_prompt_concept_extraction(doc_context: str, ct_labels: str) -> str:
    """Prompt template for concept extraction with Hugging Face inference API.

    Parameters
    ----------
    doc_context: str
        Extract of document contents to use as context.
    ct_labels: str
        The candidate terms to group into concepts.

    Returns
    -------
    str
        Completion prompt template.
    """
    prompt_template = f"""You are an helpful assistant helping building an ontology.
    Based on the context given, group together the words listed below where each group should correspond to one concept.
    The result should be given as a python list of list of string with double quotes.
    Context: {doc_context}
    Words : {ct_labels}"""
    return prompt_template


def openai_prompt_relation_extraction(
    doc_context: str, ct_labels: str
) -> List[Dict[str, str]]:
    """Prompt template for relation extraction with ChatCompletion OpenAI model.

    Parameters
    ----------
    doc_context: str
        Extract of document contents to use as context.
    ct_labels: str
        The candidate terms to group as relations.

    Returns
    -------
    List[Dict[str, str]]
        ChatCompletion prompt template.
    """
    prompt_template = [
        {
            "role": "system",
            "content": "You are an helpful assistant helping building an ontology.",
        },
        {
            "role": "user",
            "content": "Based on the context given, group together the words listed below where each group should express the same relation. The result should be given as a python list of list of string with double quotes.",
        },
        {"role": "user", "content": f"Context: {doc_context} \nWords : {ct_labels}"},
    ]
    return prompt_template


def hf_prompt_relation_extraction(doc_context: str, ct_labels: str) -> str:
    """Prompt template for relation extraction with Hugging Face inference API.

    Parameters
    ----------
    doc_context: str
        Extract of document contents to use as context.
    ct_labels: str
        The candidate terms to group into relations.

    Returns
    -------
    str
        Completion prompt template.
    """
    prompt_template = f"""You are an helpful assistant helping building an ontology.
    Based on the context given, group together the words listed below where each group should express the same relation.
    The result should be given as a python list of list of string with double quotes.
    Context: {doc_context}
    Words : {ct_labels}"""
    return prompt_template


def openai_prompt_hierarchisation(
    doc_context: str, concepts_description: str
) -> List[Dict[str, str]]:
    """Prompt template for hierarchisation with ChatCompletion OpenAI model.

    Parameters
    ----------
    doc_context: str
        Extract of document contents where concepts appear to use as context.
    concepts_description: str
        Textual description of the concepts.

    Returns
    -------
    List[Dict[str, str]]
        ChatCompletion prompt template.
    """
    prompt_template = [
        {
            "role": "system",
            "content": "You are an helpful assistant helping building an ontology.",
        },
        {
            "role": "user",
            "content": """Based on the context given, define if there is a hierarchy between the listed concepts.
            The result should be given as a python list of list of string with double quotes.""",
        },
        {
            "role": "user",
            "content": """Here is an example. Concepts: animal, mammal, dog(canine), flower
            [["mammal","is_generalised_by","animal"], ["dog","is_generalised_by","mammal"], ["dog","is_generalised_by","animal"]]""",
        },
        {"role": "user", "content": f"Context: {doc_context}"},
        {"role": "user", "content": concepts_description},
    ]
    return prompt_template


def hf_prompt_hierarchisation(doc_context: str, concepts_description: str) -> str:
    """Prompt template for hierarchisation with Hugging Face inference API.

    Parameters
    ----------
    doc_context: str
        Extract of document contents where concepts appear to use as context.
    concepts_description: str
        Textual description of the concepts.

    Returns
    -------
    str
        Completion prompt template.
    """
    prompt_template = f"""You are an helpful assistant helping building an ontology.
    Based on the context given, define if there is a hierarchy between the listed concepts.
    The result should be given as a python list of list of string with double quotes.

    Here is an example. Concepts: animal, mammal, dog(canine), flower
    [["mammal","is_generalised_by","animal"], ["dog","is_generalised_by","mammal"], ["dog","is_generalised_by","animal"]]
    
    Context: {doc_context}
    {concepts_description}"""
    return prompt_template


def hf_prompt_owl_axiom_extraction(
    doc_context: str, kr_description: str, name_space: str
) -> str:
    """Prompt template for axiomatisation with Hugging Face inference API.

    Parameters
    ----------
    doc_context: str
        Extract of popular document contents.
    kr_description: str
        Textual description of the knowledge representation.

    Returns
    -------
    str
        Completion prompt template.
    """
    prompt_template = f"""You are a helpful assistant in building an ontology. You are fluent in the W3C Semantic Web stack and in the RDF, RDFS, and OWL languages.
    Use the following text to construct an OWL ontology in the Turtle format based on the given concepts and relations.
    Use the following namespace: {name_space}.
    Include the RDF, RDFS, and OWL prefixes.
    Return only the turtle file.

    Text: {doc_context}

    {kr_description}
    """
    return prompt_template


def openai_prompt_owl_axiom_extraction(
    doc_context: str, kr_description: str, name_space: str
) -> List[Dict[str, str]]:
    """Prompt template for axiomatisation with ChatCompletion OpenAI model.

    Parameters
    ----------
    doc_context: str
        Extract of popular document contents.
    kr_description: str
        Textual description of the knowledge representation.

    Returns
    -------
    List[Dict[str, str]]
        ChatCompletion prompt template.
    """
    prompt_template = [
        {
            "role": "system",
            "content": "You are a helpful assistant in building an ontology. You are fluent in the W3C Semantic Web stack and in the RDF, RDFS, and OWL languages..",
        },
        {
            "role": "user",
            "content": f"""Use the following text to construct an OWL ontology in the Turtle format based on the given concepts and relations.
            Use the following namespace: {name_space}.
            Include the RDF, RDFS, and OWL prefixes.
            Return only the turtle file.""",
        },
        {"role": "user", "content": f"Text: {doc_context}"},
        {"role": "user", "content": kr_description},
    ]
    return prompt_template
