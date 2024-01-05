import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
)

from ..commons.errors import MissingEnvironmentVariable
from ..commons.logging_config import logger


class LLMGenerator(ABC):
    """Text generator based on LLM."""

    def __init__(self) -> None:
        """Initialise LLM generator."""

    @abstractmethod
    def check_resources(self) -> None:
        """Check that the resources needed to use the LLM Generator are available."""

    @abstractmethod
    def generate_text(self, prompt: Any) -> str:
        """Method that generates a textual output based on a prompt with a LLM."""


class OpenAIGenerator(LLMGenerator):
    """Text generator based on OpenAI gpt-3.5-turbo model."""

    def check_resources(self) -> None:
        """Check that the resources needed to use the OpenAI Generator are available."""
        if "OPENAI_API_KEY" not in os.environ:
            raise MissingEnvironmentVariable(self.__class__, "OPENAI_API_KEY")

    def generate_text(self, prompt: List[Dict[str, str]]) -> str:
        """Generate text based on a chat completion prompt for the OpenAI gtp-3.5-turbo model."""

        @retry(
            stop=stop_after_delay(15) | stop_after_attempt(3),
            retry=(
                retry_if_exception_type(
                    openai.APIConnectionError
                    | openai.APITimeoutError
                    | openai.RateLimitError
                    | openai.InternalServerError
                )
            ),
            reraise=True,
        )
        def openai_call():
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0,
                messages=prompt,
            )
            return response

        llm_output = ""
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        try:
            response = openai_call()
        except Exception as e:
            logger.error(
                """Exception %s still occurred after retries on OpenAI API.
                         Skipping document %s...""",
                e,
                prompt[-1]["content"][5:100],
            )
        llm_output = response.choices[0].message.content

        return llm_output
