import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import openai


class LLMGenerator(ABC):
    """Text generator based on LLM."""

    def __init__(self) -> None:
        """Initialise LLM generator."""

    @abstractmethod
    def generate_text(self, prompt: Any) -> str:
        """Method that generates a textual output based on a prompt with a LLM."""


class OpenAIGenerator(LLMGenerator):
    """Text generator based on OpenAI gpt-3.5-turbo model."""

    def generate_text(self, prompt: List[Dict[str, str]]) -> str:
        """Generate text based on a chat completion prompt for the OpenAI gtp-3.5-turbo model."""
        llm_output = ""
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=prompt,
        )
        llm_output = response.choices[0].message.content

        return llm_output
