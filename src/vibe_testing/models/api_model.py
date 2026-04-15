"""Implementation for models accessed via APIs (e.g., OpenAI, Anthropic)."""

import logging
import os
from typing import Any, Dict, Optional

from openai import OpenAI

from src.vibe_testing.env import load_project_dotenv

from .base import BaseModel
from .generation_failure_tracker import (
    record_generation_failure,
    record_generation_success,
)

logger = logging.getLogger(__name__)


class APIModel(BaseModel):
    """
    A concrete implementation for language models that are accessed through a
    web API.
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initializes the API model client.

        Args:
            model_name (str): The name of the model.
            config (Dict[str, Any]): Configuration dictionary containing API
                                     endpoint, keys, etc.
        """
        super().__init__(model_name, config)
        dotenv_path = load_project_dotenv(override=False)
        api_key_env_var = self.config.get("api_key_env_var")
        if not api_key_env_var:
            raise ValueError(
                "Model config is missing required key 'api_key_env_var' "
                "(e.g., 'OPENAI_API_KEY')."
            )
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            env_hint = ""
            if dotenv_path is None:
                env_hint = (
                    " No .env file was found/loaded for this process. "
                    "On Slurm, .env is typically not sourced automatically; "
                    "either export the variable in the job environment or ensure "
                    "the working directory contains a readable .env file."
                )
            else:
                env_hint = (
                    f" A .env file was loaded from '{dotenv_path}', but it did not "
                    f"set '{api_key_env_var}'."
                )
            raise ValueError(
                f"API key environment variable '{api_key_env_var}' is not set."
                f"{env_hint}"
            )

        self._client = OpenAI(api_key=api_key)
        logger.info("Initialized APIModel: %s", self.model_name)

    def generate(
        self, prompt: str, json_schema: Optional[Dict[str, Any]] = None, **kwargs
    ) -> str:
        """
        Generates a response by calling the configured API endpoint.

        Args:
            prompt (str): The input prompt for the model.
            json_schema (Optional[Dict[str, Any]]): If provided, enables JSON mode.
            **kwargs: Additional generation parameters to pass to the API.

        Returns:
            str: The model's generated output from the API.
        """
        logger.debug(
            "Generating response for model '%s' (prompt prefix: '%s')",
            self.model_name,
            prompt[:50].replace("\n", " "),
        )

        # Combine default config params with any runtime kwargs
        generation_params = {
            "temperature": self.config.get("temperature", 0.7),
            "max_tokens": self.config.get("max_tokens", 1024),
            **kwargs,
        }

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Respond in JSON format.",
            },
            {"role": "user", "content": prompt},
        ]

        if json_schema:
            generation_params["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **generation_params,
            )
            text = response.choices[0].message.content.strip()
            record_generation_success(self.model_name)
            return text
        except Exception as exc:  # noqa: BLE001
            record_generation_failure(self.model_name, exc)
            raise
