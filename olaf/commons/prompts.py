from typing import Dict, List


def prompt_concept_term_extraction(context: str) -> List[Dict[str, str]]:
    """Prompt template for concept term extraction with ChatCompletion OpenAI model.

    Parameters
    ----------
    context: str
        The context to add in the prompt template.
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


def prompt_relation_term_extraction(context: str) -> List[Dict[str, str]]:
    """Prompt template for relation term extraction with ChatCompletion OpenAI model.

    Parameters
    ----------
    context: str
        The context to add in the prompt template.
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


def prompt_term_enrichment(context: str) -> List[Dict[str, str]]:
    """Prompt template for term enrichment with ChatCompletion OpenAI model.

    Parameters
    ----------
    context: str
        The context to add in the prompt template.
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
                {
                    "synonyms": ["hound", "mutt"],
                    "hypernyms":["animal", "mammal", "canine"]
                    "hyponyms": ["labrador", "dalmatian"],
                    "antonyms": [""]
                }""",
        },
        {"role": "user", "content": f"Term: {context}"},
    ]
    return prompt_template
