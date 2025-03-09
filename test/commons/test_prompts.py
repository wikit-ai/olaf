from typing import Dict, List

import pytest

from olaf.commons.prompts import (
    hf_prompt_concept_term_extraction,
    hf_prompt_relation_term_extraction,
    hf_prompt_term_enrichment,
    openai_prompt_concept_term_extraction,
    openai_prompt_relation_term_extraction,
    openai_prompt_term_enrichment,
    deepseek_prompt_concept_term_extraction,
    deepseek_prompt_relation_term_extraction,
    deepseek_prompt_term_enrichment,
)


@pytest.fixture(scope="function")
def context() -> str:
    return "This is a text."


def test_openai_prompt_concept_term_extraction(context: str) -> None:
    prompt = openai_prompt_concept_term_extraction(context)
    assert isinstance(prompt, List)
    for elem in prompt:
        assert isinstance(elem, Dict)
    assert prompt[-1]["content"] == f"Text: {context}"


def test_hf_prompt_concept_term_extraction(context: str) -> None:
    prompt = hf_prompt_concept_term_extraction(context)
    assert isinstance(prompt, str)
    assert f"Text: {context}" in prompt

def test_deepseek_prompt_concept_term_extraction(context: str) -> None:
    prompt = deepseek_prompt_concept_term_extraction(context)
    assert isinstance(prompt, List)
    for elem in prompt:
        assert isinstance(elem, Dict)
    assert prompt[-1]["content"] == f"Text: {context}"


def test_openai_prompt_relation_term_extraction(context: str) -> None:
    prompt = openai_prompt_relation_term_extraction(context)
    assert isinstance(prompt, List)
    for elem in prompt:
        assert isinstance(elem, Dict)
    assert prompt[-1]["content"] == f"Text: {context}"


def test_hf_prompt_relation_term_extraction(context: str) -> None:
    prompt = hf_prompt_relation_term_extraction(context)
    assert isinstance(prompt, str)
    assert f"Text: {context}" in prompt

def test_deepseek_prompt_relation_term_extraction(context: str) -> None:
    prompt = deepseek_prompt_relation_term_extraction(context)
    assert isinstance(prompt, List)
    for elem in prompt:
        assert isinstance(elem, Dict)
    assert prompt[-1]["content"] == f"Text: {context}"

def test_openai_prompt_term_enrichment(context: str) -> None:
    prompt = openai_prompt_term_enrichment(context)
    assert isinstance(prompt, List)
    for elem in prompt:
        assert isinstance(elem, Dict)
    assert prompt[-1]["content"] == f"Term: {context}"


def test_hf_prompt_term_enrichment(context) -> None:
    prompt = hf_prompt_term_enrichment(context)
    assert isinstance(prompt, str)
    assert f"Term: {context}" in prompt

def test_deepseek_prompt_term_enrichment(context: str) -> None:
    prompt = deepseek_prompt_term_enrichment(context)
    assert isinstance(prompt, List)
    for elem in prompt:
        assert isinstance(elem, Dict)
    assert prompt[-1]["content"] == f"Term: {context}"