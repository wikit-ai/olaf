from typing import Dict, List

import pytest

from olaf.commons.prompts import prompt_concept_term_extraction


@pytest.fixture(scope="function")
def context() -> str:
    return "This is a text."


def test_prompt_concept_term_extraction(context) -> None:
    prompt = prompt_concept_term_extraction(context)
    assert isinstance(prompt, List)
    for elem in prompt:
        assert isinstance(elem, Dict)
    assert prompt[-1]["content"] == f"Text: {context}"
