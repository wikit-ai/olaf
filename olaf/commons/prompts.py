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
