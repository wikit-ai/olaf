import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import openai
import requests
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


class HuggingFaceGenerator(LLMGenerator):
    """Text generator base on Hugging Face inference API."""

    def __init__(
            self, 
            api_url: Optional[str] = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
        ) -> None:
        self.api_url = api_url

    def check_resources(self) -> None:
        """Check that the resources needed to use the HuggingFace Generator are available."""
        if "HF_API_KEY" not in os.environ:
            raise MissingEnvironmentVariable(self.__class__, "HF_API_KEY")

    def generate_text(self, prompt: str) -> str:
        """Generate text based on a chat completion prompt for an hugging face model."""
        headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 1024, "temperature": 0.1},
        }
        response = requests.post(
            self.api_url, headers=headers, json=payload, timeout=60
        )
        answer = ""
        try:
            answer = response.json()[0]["generated_text"]
            answer = answer.replace(prompt, "")
        except KeyError:
            logger.error(
                """Something went wrong the the HuggingFace API call.\n Message : %s""",
                response.text,
            )

        return answer


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
            llm_output = response.choices[0].message.content
        except Exception as e:
            logger.error(
                """Exception %s still occurred after retries on OpenAI API.
                         Skipping document %s...""",
                e,
                prompt[-1]["content"][5:100],
            )

        return llm_output


class MistralAIGenerator(LLMGenerator):
    """Text generator based on MiastralAI models."""

    def __init__(self, model_name: Optional[str] = "mistral-tiny") -> None:
        self.model_name = model_name 
        self.api_url = "https://api.mistral.ai/v1/chat/completions"

    def check_resources(self) -> None:
        """Check that the resources needed to use the MistralAI Generator are available."""
        if "MISTRAL_API_KEY" not in os.environ:
            raise MissingEnvironmentVariable(self.__class__, "MISTRAL_API_KEY")

    def generate_text(self, prompt: List[Dict[str, str]]) -> str:
        """Generate text based on a chat completion prompt for MistralAI model."""

        @retry(
            stop=stop_after_delay(15) | stop_after_attempt(3),
            reraise=True,
        )
        def mistralai_call():
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer " + os.getenv("MISTRAL_API_KEY"),
            }
            json_data = {
                "model": self.model_name,
                "messages": prompt,
                "temperature": 0.0,
            }
            response = requests.post(
                self.api_url, headers=headers, json=json_data, timeout=30
            )

            return response

        llm_output = ""
        try:
            response = mistralai_call()
        except Exception as e:
            logger.error(
                """Exception %s still occurred after retries on MistralAI API.
                         Skipping document %s...""",
                e,
                prompt[-1]["content"][5:100],
            )
        try:
            llm_output = response.json()["choices"][0]["message"]["content"]
        except KeyError:
            logger.error(
                """Something went wrong the the MistralAI API call.\n Message : %s""",
                response.json()["message"],
            )

        return llm_output

class DeepSeekGenerator(LLMGenerator):
    """Text generator based on DeepSeek models."""

    def __init__(self, model_name: Optional[str] = "deepseek-chat") -> None:
        self.model_name = model_name 
        self.api_url = "https://api.deepseek.com/chat/completions"

    def check_resources(self) -> None:
        """Check that the resources needed to use the DeepSeek Generator are available."""
        if "DEEPSEEK_API_KEY" not in os.environ:
            raise MissingEnvironmentVariable(self.__class__, "DEEPSEEK_API_KEY")

    def generate_text(self, prompt: List[Dict[str, str]]) -> str:
        """Generate text based on a chat completion prompt for DeepSeek model."""

        @retry(
            stop=stop_after_delay(15) | stop_after_attempt(3),
            reraise=True,
        )
        def deepseek_call():
            response = client.chat.completions.create(
                model="deepseek-chat",
                temperature=0,
                messages=prompt,
                stream=False
            )
            return response

        llm_output = ""
        client = openai.OpenAI(
            base_url="https://api.deepseek.com",
            api_key=os.getenv("DEEPSEEK_API_KEY")
            )
        try:
            response = deepseek_call()

        except Exception as e:
            logger.error(
                """Exception %s still occurred after retries on DeepSeek API.
                         Skipping document %s...""",
                e,
                prompt[-1]["content"][5:100],
            )
        try:
            llm_output = response.choices[0].message.content
        except KeyError:
            logger.error(
                """Something went wrong the the DeepSeek API call.\n Message : %s""",
                response.json()["message"],
            )

        return llm_output